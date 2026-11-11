# B：跳跃融合的分支感知正交保护

入口：`src/scripts/train_skip_branch.py`。默认源为已完成的
`outputs/stable_v1_b64/task_0_final.pth`。不再重训LSRW400轮。
只改任务末保护记忆的构造，保持源骨干、现有尺度秩、LoRA位置、损失与前向不变。
无回放、教师、蒸馏或任务路由；不同时实施秩分配方案A。

## 两个匹配组

| 组别 | 跳跃融合后第一个conv1的GPM统计 | 其余层 |
| --- | --- | --- |
| `fused`对照 | 原有真实融合输入patch | 原有空间统计 |
| `branch_v1`实验 | 同预算内混合融合输入和两条分支贡献 | 与对照一致 |

五级骨干只修改4个位置：`backbone.dec_blocks.{1,2,3,4}.0.conv1`。
不是整个解码器块，也不修改块内残差连接或图像级残差连接。
保留原`is_skip_fusion`标记的旧语义，避免改变既有实验；新目标通过真实融合位置单独选择。

## 必须在真实卷积输入坐标中分解

融合为`z=d+s`，第一层卷积实际接收的是`u=norm1(z)`。
**不能使用`norm1(d)+norm1(s)`替代u**，因为归一化不是可加函数。

本实现用融合输入z的共同方差与各分支均值构造贡献：

```text
u_d = weight * (d - mean(d)) / sqrt(var(d+s) + eps) + bias/2
u_s = norm1(d+s) - u_d
u_d + u_s ≈ u
```

第二项采用真实归一化输出减去第一项，以吸收数值舍入；数学上等于同一统计下的s贡献。
stable_v1逐像素按通道统计，legacy的GroupNorm(1,C)按整张图的通道和空间统计。
二者都不改变实际网络前向，只在任务末记忆采集时额外计算贡献。
该分解用于线性卷积输入方向保护，不是两条独立归一化网络，也不保证上游特征不漂移。

## 相同容量与采样预算

每层仍最多`min(32,d)`个共享GPM方向，没有额外的两套分支记忆。
每批每层总采样列数与fused对照一致，沿用原有最多512列的预算。
在目标位置，总列数N分为：`floor(N/4)`解码器贡献、`floor(N/4)`跳跃贡献，
剩余列给真实融合输入；N=512时为256融合+128解码器+128跳跃。
使用确定性均匀空间采样，不引入新的随机抽样；统计按实际幅值使用，不额外做分支能量归一化。
混合统计会减少目标层纯融合输入样本数，这本身属于实验因素，需要报告，不能宣称“免费增加保护”。
所有层仍使用同一个`bounded_svd`更新规则；B构造与完整更新硬投影不变。
有效基仍可能由于保护空间不同而有不同有效秩，但分配的参数形状与基础尺度秩不变。

## 共同起点与无回放边界

两组读取同一个正式LSRW任务末权重，清空GPM，再固定相同种子、相同6批×64个
LSRW训练裁剪重建任务0记忆。这里不优化任何权重，属于共同任务0终点的记忆构造分支。
前向误差必须为0，所有非记忆张量SHA保持一致，源模型SHA前后核对。

随后只训练RainH→Haze4K，各20轮，batch64、patch128。
进入RainH后不再访问LSRW训练loader；进入Haze4K后不再访问LSRW/RainH训练loader。
旧验证集仅诊断，不参与损失或选模。每5轮及末轮记录已见任务指标，
只按当前新任务验证PSNR选最佳权重，再更新GPM和合并，最后报告正式测试。

此前stable_v1的RainH/Haze4K断点不能直接换mode接续，必须让两组从共同任务0边界重建记忆。
bootstrap阶段如果断开但尚未保存latest，请换新输出目录重启。

## 训练命令

```bash
conda activate odr-lora
cd /media/zhappy-701/新加卷/odr-lora/odr-lora-restoration-main
export PYTHONNOUSERSITE=1

# B组：依次训练RainH、Haze4K，各20轮，batch64
CUDA_VISIBLE_DEVICES=1 python -u src/scripts/train_skip_branch.py \
  --source outputs/stable_v1_b64/task_0_final.pth \
  --mode branch_v1 --epochs 20 --batch-size 64 \
  --output-dir outputs/skip_branch_b64

# 匹配对照：使用同一源、预算与采样规则，只关闭分支混合
CUDA_VISIBLE_DEVICES=1 python -u src/scripts/train_skip_branch.py \
  --source outputs/stable_v1_b64/task_0_final.pth \
  --mode fused --epochs 20 --batch-size 64 \
  --output-dir outputs/skip_fused_control_b64
```

若希望先停在RainH，在命令末加`--stop-after rainh`；之后用同一命令加
`--resume outputs/skip_branch_b64/task_1_final.pth`（对照用自身路径），去掉stop-after即可。
如希望在旧骨干上做B对照，两组的source统一改为`outputs/clean_lsrw/task_0_final.pth`，
并使用另外的新目录；不要把不同骨干的两组混作匹配对照。

## 断点恢复

```bash
CUDA_VISIBLE_DEVICES=1 python -u src/scripts/train_skip_branch.py \
  --source outputs/stable_v1_b64/task_0_final.pth \
  --mode branch_v1 --epochs 20 --batch-size 64 \
  --resume outputs/skip_branch_b64/latest.pth \
  --output-dir outputs/skip_branch_b64
```

完整epoch边界恢复优化器、scheduler与随机状态；未完成epoch重跑。
mode、source、epochs、batchsize、划分等均不得中途改变。请保留源断点和分区清单。
source的`epochs_by_task`被本实验显式移除，防止400/20/20设置覆盖本次20轮预算。

## 结果检查

- `task_0_memory_bootstrap.json`：共同模型指纹、前向误差、逐层采样列数。
- `task_{0,1,2}_skip_memory.json`：4个目标位置、实际融合/分支采样量、重构误差与GPM秩。
- `task_{1,2}_protection.json`：完整LoRA更新的正交泄漏。
- `task_{1,2}_gpm_transition.json`：记忆压缩与合并误差。
- `history.json`：逐验证点新旧任务PSNR/SSIM/LPIPS与原始输出异常。
- `continual_metrics.json`：任务边界测试、平均性能与遗忘。

首先确认两组任务0非记忆指纹及所有层列数一致；没有新增可学习参数，推理公式不变。
随后比较Haze4K前后LSRW与RainH各自下降量，不能只挑一个任务提升来宣称整体改进。
该方法仍有32维压缩、旧特征漂移、分支采样权重等限制，尚无质量收益保证。
没有自动启动正式20轮训练。

## 已完成验证（2026-11-09）

- 全部69项unittest通过；新增5项测试覆盖两种归一化坐标、抵消分支信号、
  精确目标位置、相同列数/参数形状、非目标层记忆逐张量一致、硬正交、
  无旧TRAIN访问、精确epoch恢复以及跨mode恢复拒绝。
- 真实stable_v1/LSRW源的两步RainH→Haze4K冒烟流程通过，包括任务0记忆重建、
  校准、反向、验证/测试、GPM更新和合并。仅每任务两步、小批量、每评估集1张图，
  不能作为性能证据。产物：`outputs/skip_branch_smoke_20261109`。
- 两组真实batch64的任务0单批记忆重建，非记忆模型指纹和逐层采样列数完全一致；
  源LSRW断点SHA保持不变。
- B组真实batch64、patch128、VGG损失单步容量检查：RainH峰值allocated约17.61 GiB，
  Haze4K约17.54 GiB，reserved峰值约19.77 GiB；RainH的batch64分支记忆更新与合并通过。
  这些检查仅1批校准/GPM与每任务1次优化，未导出正式训练模型，请使用空闲显卡。
  记录：`outputs/skip_branch_v1_capacity_20261109/capacity.json`。
