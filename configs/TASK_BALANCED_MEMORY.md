# 逐任务平衡记忆：固定保护秩 + 独立历史摘要

固定RainH边界、补seed7/2026的配对复验入口与汇总，参见 [TASK_MEMORY_REPLICATES.md](TASK_MEMORY_REPLICATES.md)。

新入口：`src/scripts/train_task_balanced.py`。
只改GPM记忆构造，不改stable_v1骨干、尺度秩、LoRA插入位置、优化损失或推理公式。
关闭分支B、回放、教师、蒸馏与范数上限，不同时叠加其他改进。
默认共同源：`outputs/stable_v1_b64/task_0_final.pth`。

## 从混合历史池改为逐任务摘要

以前有损合并后，历史方向不再保留任务身份，很难知道哪个旧任务被挤出。
本轮每个任务在恢复当前任务验证最优权重后，仅用自己的TRAIN输入构造独立摘要。
每层最多保存一个`d×32`的因子F，近似本任务采样的未中心化协方差：`C≈F Fᵀ`。
用流式截断SVD汇总6批、每层每批最多512列的真实空间patch；不保存图像、GT或完整激活。
摘要随后冻结，不随下一任务更新，也不重新读取旧TRAIN刷新摘要。

流式截断本身有损且依赖批次顺序；报告保留能量与原始采样总能量的比值。
原始保留能量比值因float32分解舍入可能略高于1，诊断不将其静默改写为精确100%。
每个任务的摘要按自身保留能量归一化到trace=1，避免图像数量或特征幅值决定任务权重。
已丢弃的能量不能被归一化“恢复”，覆盖率只指**保留的历史摘要**。

## 两个严格匹配组

| 模式 | 从同一套逐任务摘要构造共享GPM |
| --- | --- |
| `mean_pca`对照 | 各任务归一化协方差等权平均，取主要方向；优先优化平均覆盖 |
| `balanced`实验 | 每次找当前摘要覆盖率最低的任务，补充其最强残差方向；优先补短板 |

两组均保留同样的任务摘要，区别仅是选保护方向的规则，不与旧无摘要的结果混作严格对照。
共有方向可同时保护多个任务，不机械地将32个方向按任务数切成互不重叠的小块。
每层实际共享保护基最多`min(32,d)`列，取两组相同的可行秩；不是每任务各32个保护基并行投影。
本轮在摘要联合空间中填充可行秩，不沿用旧逐批更新的99%提前停止规则，故旧对照不能直接复用。
逐任务32维是摘要上限，与LoRA私有基础秩16是两个不同概念。

balanced是贪心启发式，不是严格全局max-min最优解；可能牺牲平均覆盖或新任务可塑性。
摘要来自各任务结束时的旧激活，上游特征随后仍会漂移；高摘要覆盖不保证旧输出保持不变。
摘要尾部截断、有限32维容量、非线性传播问题仍然存在，不能宣称已经解决遗忘。

## 存储与公平性

用于正交投影的GPM仍只有一套，但**额外任务摘要随任务数量线性增长**。
两个组都支付相同摘要开销；不能与不存摘要的方法宣称同总记忆预算。
每个epoch检查点保存CPU摘要、任务身份和SHA256；恢复缺失或篡改摘要会拒绝。
诊断文件区分`sketch_bytes`、`protected_basis_bytes`和总张量字节数。
这些是原始张量大小，不含优化器、Python对象、checkpoint格式或临时SVD工作空间开销。
临时因子与历史因子的联合空间分解不构建完整`d×d`协方差矩阵。

## 实验顺序

从同一份正式LSRW权重分支，在任务0记忆边界各重建一次LSRW摘要和GPM，不重训400轮。
之后训练RainH→Haze4K，各20轮，batch64、patch128、严格FP32、AdamW、lr2e-4、L1+0.1 LPIPS。
构造任务0摘要不优化权重。进入RainH后不再访问LSRW TRAIN；进入Haze4K后不访问旧TRAIN。
旧验证集仅诊断，不进入loss或选模。每5轮及末轮验证，仅按当前任务验证PSNR选最佳权重。
任务末更新记忆不改变当前模型前向，合并仍使用原有误差检查。

只有一个LSRW摘要时，两种模式走完全相同的选择路径，因此RainH的共同保护起点一致。
两组首次有意义的差异发生在RainH边界：组合LSRW和RainH摘要，影响随后Haze4K训练。
不能用以前的B、fused或stable_v1 RainH断点直接换模式接续，因为它们没有保存所需LSRW摘要。
三任务验证通过后，再适配五任务入口；本入口当前只运行前三任务，未自动加入去噪/GOPRO。

## 正式训练

```bash
conda activate odr-lora
cd /media/zhappy-701/新加卷/odr-lora/odr-lora-restoration-main
export PYTHONNOUSERSITE=1

# 逐任务补短板组
CUDA_VISIBLE_DEVICES=1 python -u src/scripts/train_task_balanced.py \
  --source outputs/stable_v1_b64/task_0_final.pth \
  --mode balanced --epochs 20 --batch-size 64 \
  --output-dir outputs/task_balanced_b64

# 匹配对照：同样保留全部摘要，等权均值PCA
CUDA_VISIBLE_DEVICES=1 python -u src/scripts/train_task_balanced.py \
  --source outputs/stable_v1_b64/task_0_final.pth \
  --mode mean_pca --epochs 20 --batch-size 64 \
  --output-dir outputs/task_mean_pca_b64
```

如要先停在RainH，两组均加`--stop-after rainh`，然后分别用各自`task_1_final.pth`恢复，
去掉stop-after再进入Haze4K。两组建议顺序运行于空闲显卡；实施验证时GPU0有其他训练，没有占用。

## 断点恢复

```bash
CUDA_VISIBLE_DEVICES=1 python -u src/scripts/train_task_balanced.py \
  --source outputs/stable_v1_b64/task_0_final.pth \
  --mode balanced --epochs 20 --batch-size 64 \
  --resume outputs/task_balanced_b64/latest.pth \
  --output-dir outputs/task_balanced_b64
```

按完整epoch恢复优化器、scheduler、随机状态和任务摘要，未完成epoch重跑。
mode/source/batch/预算/划分都不能中途更换。首次bootstrap未保存latest前断线，请换新目录重启。
任务末采集摘要期间若断线，从已有epoch断点重做本任务边界，不保存半成品摘要。
源LSRW断点和数据划分清单须保留。旧骨干实验可让两组统一改用clean_lsrw源及新的输出目录。

## 查看结果

- `task_0_memory_bootstrap.json`：任务0源模型前向与非记忆张量指纹不变，逐层采样预算。
- `task_{0,1,2}_task_memory.json`：每层逐任务摘要覆盖率、最差/平均覆盖率、选取顺序、
  保留能量比例、摘要SHA、摘要与GPM存储字节数。
- `task_{1,2}_protection.json`：实际LoRA完整更新的正交泄漏。
- `history.json`：新旧任务逐验证点PSNR/SSIM/LPIPS及原始输出异常。
- `continual_metrics.json`：任务边界测试、任务平均、BWT和遗忘。

先核对两组LSRW与RainH阶段网络权重/摘要采样一致，再比较Haze4K后两个旧任务的跌幅及新任务性能。
不要把摘要覆盖率当真实卷积输入当前覆盖率，也不要仅凭其改善就宣称减少遗忘。
没有自动启动正式20轮训练。

## 已完成验证（2026-11-10）

- 全部75项unittest通过。新增测试覆盖低秩摘要有界、协方差近似、任务等权归一化、
  最差任务贪心选择、共有方向复用、五任务合成输入、摘要不被后续任务改写、
  相同RainH物理权重/采样列数、无旧TRAIN读取、摘要篡改拒绝与逐张量精确epoch恢复。
- 两种模式的真实stable_v1/LSRW源→RainH→Haze4K冒烟均通过，使用真实NAFNet和VGG。
  每任务只做2步、小批量及每评估集1张图，不可用作质量证据或正式训练源。
  产物：`outputs/task_balanced_smoke_20261110`和`outputs/task_mean_pca_smoke_20261110`。
- 冒烟中每任务摘要约8 MiB，三任务约24 MiB，额外共享GPM约8.3 MiB；
  正式样本有效秩可能不同，请以正式诊断文件为准，不能将这些容量数字当作质量结果。
- 两组真实batch64单步及任务边界容量检查通过：RainH和Haze4K训练峰值allocated约17.53 GiB，
  reserved约19.67 GiB，摘要采集/选择边界峰值allocated约4.82 GiB。
  两组初始物理模型、RainH任务末物理模型、任务摘要SHA及逐层列数一致，源LSRW断点SHA不变。
  仅使用每边界1批统计/校准、每新任务1步更新，不代表正式6批摘要或20轮质量结果。
  记录：`outputs/task_balanced_capacity_20261110/verification.json`。
