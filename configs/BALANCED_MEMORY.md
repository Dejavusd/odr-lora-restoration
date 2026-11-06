# 固定32维记忆：历史空间与当前任务残差的配额对照

沿用 `src/scripts/train_memory_retention.py`，新增 `--merge-rule balanced --old-fraction`。
仍从 `outputs/memory_raw32_b64` 的同一对RainH best/final分支，只训练Haze4K。
没有LSRW/RainH重新优化，没有回放、教师、蒸馏、输出范围损失或范数上限，LoRA完整硬正交保持开启。
正式20轮尚未启动，目前通过的是实现与容量检查，不能宣称配额方案已经改善质量。

## 分配方式

总容量仍为每层 `min(32,输入维度)`，以 `old_fraction=0.5` 为例：

1. 保留最多一半容量的旧子空间方向；满32维的层为16个。
2. 在这些保留方向的正交补里，从当前RainH统计提取最多16个新主方向。
3. 只保留非微小特征值方向，达到99%当前统计能量阈值后可提前停止，实际秩不会强行填满。
4. 旧空间不足配额时，剩余容量可交给当前任务；如果当前统计完全无能量则保留旧记忆。

旧方向不是直接取原矩阵前16列：原记忆已经经过合并，没有可靠的逐方向历史重要性排序。
本实验在旧子空间内部按当前RainH能量旋转，优先保留当前任务能量较低的方向，
假设这些方向代表较少被当前任务覆盖的历史信息；再让当前任务残差PCA处理主要当前信息及重叠部分。
这是明确的互补性启发式，**不能声称选中了最重要的LSRW方向**。
保留比例表示预留方向数量，不是历史性能、能量或图像数量的比例。
新增方向也可能包含部分被让出的旧空间，所以最终旧空间重合率不必恰等于old_fraction。

在本次RainH边界，旧空间来自LSRW，当前统计来自RainH；这是检验两类历史信息分配的局部实验。
后续若历史池混合多个任务，本实现没有单独的每任务配额或任务重要性估计。
Haze4K任务末也会执行同样规则，为未来任务准备记忆；任务末更新不改变已学模型的当前前向输出。

## 汇总后分配，避免批次顺序反复淘汰

先用与前轮相同的6批×64个RainH训练裁剪、相同种子与逐层列预算累计 `Σ H Hᵀ`，
期间保留原记忆不变。全部批次完成后，一次性分配旧配额和当前残差配额。
协方差经归一化后，在CPU float64上做精确对称特征分解。
每层记录原始汇总协方差SHA，可核对不同配额组实际输入统计完全一致。
完整协方差为任务末临时统计，不写入模型checkpoint，分配完成即释放。
断点恢复不需要保存训练图像；任务末尚未完成的统计构造需重做。

**必须使用同样汇总式的100%旧配额作为匹配对照。**
旧 `--merge-rule preserve` 是逐批补充规则；不能将汇总方式与配额的影响混在一起，
所以它先前17.88/12.81/19.97的测试成绩仅作背景参考。

## 训练命令：先做50%与100%两组

准备：

```bash
conda activate odr-lora
cd /media/zhappy-701/新加卷/odr-lora/odr-lora-restoration-main
export PYTHONNOUSERSITE=1
```

50%历史预留组（满层最多16旧+16当前）：

```bash
CUDA_VISIBLE_DEVICES=1 python -u src/scripts/train_memory_retention.py \
  --run-dir outputs/memory_raw32_b64 \
  --merge-rule balanced --old-fraction 0.5 \
  --epochs 20 --batch-size 64 \
  --output-dir outputs/memory_balanced50_b64
```

100%历史预留匹配对照（满层不再加入当前方向）：

```bash
CUDA_VISIBLE_DEVICES=1 python -u src/scripts/train_memory_retention.py \
  --run-dir outputs/memory_raw32_b64 \
  --merge-rule balanced --old-fraction 1.0 \
  --epochs 20 --batch-size 64 \
  --output-dir outputs/memory_balanced100_b64
```

两组都不重训LSRW/RainH，固定同一个RainH网络权重，只重建任务末记忆并训练Haze4K20轮。
Haze4K仅当前TRAIN用于优化，旧训练图像不再读取，旧验证仅作诊断。
每5轮及末轮完整验证，仅按Haze4K验证PSNR选模；保持同一种子/预算/容量/采样/学习率。
如果两组显示明显权衡，再将old_fraction改为0.75、新目录 `outputs/memory_balanced75_b64`
做24旧+最多8当前方向的中间对照，不必在最初盲目扩大扫描。

## 恢复

```bash
CUDA_VISIBLE_DEVICES=1 python -u src/scripts/train_memory_retention.py \
  --run-dir outputs/memory_raw32_b64 \
  --merge-rule balanced --old-fraction 0.5 \
  --epochs 20 --batch-size 64 \
  --resume outputs/memory_balanced50_b64/latest.pth \
  --output-dir outputs/memory_balanced50_b64
```

保持所有参数与首跑一致，原始run-dir的best/final源断点须保留。
不允许在同一个任务中途切换配额；每个完整epoch保存优化器、scheduler、随机状态。

## 读结果

`task_1_balanced_memory.json`记录每层：old_quota、retained_old_rank、current_residual_rank、
rank/cap、current_energy_coverage、old_span_residual、old_span_overlap_fraction、
columns和covariance_sha256。
两组任务1的协方差SHA应该一致。Haze4K训练后网络已不同，任务2的协方差不要求一致。
`rainh_boundary_rebuild.json`确认物理权重与原RainH终点一致。
注意其added_rank仅表示总秩变化，不代表换入了多少新方向；分配量应读current_residual_rank。
`history.json`含三任务验证PSNR/SSIM/LPIPS及原始输出异常指标；
`continual_metrics.json`给出最终测试指标、平均性能和遗忘。

优先检查50%组是否恢复RainH保持，同时避免LSRW重新大幅退化，以及Haze4K是否学得动。
单独提升任一旧任务不构成整体改进，配额不保证输出稳定，也不保证能解决原始输出极端值。

## 实现验证

- 新单元测试覆盖配额、容量、正交性、汇总与批次次序、旧基旋转不变性、零统计与异常值拒绝。
- 两配额的边界协方差哈希相同，网络起点逐张量一致；改配额拒绝恢复，中断恢复逐张量一致。
- 完整模型真实两步冒烟通过，产物：`outputs/balanced_half_smoke_20261105`，不是质量证据。
- 真实6批×64的RainH边界及每组1个Haze4K batch64前向/反向/AdamW检查通过；
  两配额协方差逐层SHA一致，峰值allocated约14.15 GiB，正交泄漏最大值约1.07e-7。
  该检查只使用1个B校准batch，不代表正式20轮效果。
- 容量记录：`outputs/balanced_capacity_20261105/verification.json`。
