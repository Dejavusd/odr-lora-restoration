# 下一步：固定RainH边界的配对种子复验

目标：检验`balanced`相对`mean_pca`约0.1214 dB的单种子收益是否稳定。
不叠加新模块，不改骨干、尺度秩、GPM容量、任务摘要预算、损失、选模或训练轮数。
已有seed42结果仅作为已观察的探索性参考；新增种子预先固定为7和2026，不根据测试结果挑种子。

## 共同边界与实验范围

两种模式统一读取`outputs/task_balanced_b64/task_1_final.pth`：

- 复用相同RainH物理权重、LSRW/RainH逐任务摘要与阶段测试记录。
- 不重新训练LSRW/RainH，不读取旧TRAIN图像，不从旧图重建摘要。
- 仅从已有摘要重新计算所选模式的共享保护基，验证物理权重SHA、摘要SHA、合成输入前向均不变。
- 进入Haze4K后才读取当前TRAIN，训练20轮、batch64；每5轮与末轮验证，仅按Haze4K验证PSNR选模。
- 变化的是Haze4K采样/增强/校准/优化的训练种子。沿用原代码偏移，实际种子为`--seed + 20000`。
  因此7、2026、原42分别对应20007、22026、20042。
- 任务末摘要采集仍使用原有固定规则`experiment.seed + 900002`（当前为900044），两个模式一致。

这是**固定前两个任务后的条件复验**，不是从模型初始化开始整条序列的多种子实验。
不代表LSRW/RainH训练随机性、不同任务顺序或不同划分的稳健性。
mean_pca模式不会直接沿用源的balanced保护基，而是用同一套摘要重选；只切模式字段是不够的。
新入口在已完成RainH边界显式建立新实验，不放宽现有任务中途恢复规则。

## 一条命令依次完成4次正式训练

```bash
conda activate odr-lora
cd /media/zhappy-701/新加卷/odr-lora/odr-lora-restoration-main
export PYTHONNOUSERSITE=1

CUDA_VISIBLE_DEVICES=1 python -u src/scripts/run_task_replicates.py \
  --source outputs/task_balanced_b64/task_1_final.pth \
  --seeds 7 2026 --epochs 20 --batch-size 64 \
  --output-root outputs/task_memory_replicates
```

执行顺序：balanced_seed7 → mean_pca_seed7 → balanced_seed2026 → mean_pca_seed2026。
它们顺序使用同一张GPU，不并发抢显存。实施验证时GPU0已有其他训练，未占用GPU0。
运行前确认GPU1仍有足够空闲显存。可在tmux中运行，断开后重新连接会话。

本次只自动运行了隔离的短程冒烟检查，没有启动上述20轮正式训练。

## 中断恢复

重新运行同一条批量命令即可：已完成且协议匹配的任务自动跳过，有latest的未完成任务自动恢复。
每个完整epoch恢复优化器、scheduler、随机状态和任务摘要；未完成epoch重跑。
如果初次重选边界时失败且还没有latest，会拒绝覆盖失败目录，请保留它并指定新output-root。
不能中途更换seed、mode、源断点、batch、轮数、划分或代码实现。

也可单独恢复某个任务：

```bash
CUDA_VISIBLE_DEVICES=1 python -u src/scripts/train_task_replicate.py \
  --source outputs/task_balanced_b64/task_1_final.pth \
  --mode balanced --seed 7 --epochs 20 --batch-size 64 \
  --resume outputs/task_memory_replicates/balanced_seed7/latest.pth \
  --output-dir outputs/task_memory_replicates/balanced_seed7
```

旧`train_task_balanced.py`任务0/1/2断点不能直接作为`--resume`，只有共同原RainH断点能通过`--source`分支。
应保留共同RainH源、逐任务摘要和划分清单。两个模式都使用同一个source，不分别指定不同来源。

## 完成后的配对汇总

```bash
python src/scripts/summarize_task_replicates.py \
  --source outputs/task_balanced_b64/task_1_final.pth \
  --root outputs/task_memory_replicates --seeds 7 2026 \
  --include-existing \
  --output-dir outputs/task_memory_replicates/summary
```

`--include-existing`加入原`outputs/task_balanced_b64`和`outputs/task_mean_pca_b64`的seed42结果。
不加该参数则只汇总两个新增种子。重复种子或缺少配对会拒绝，不把一次重复运行当成新种子。
默认任何任务缺失、未完成、属于smoke、共同物理权重/摘要不同或核心代码不同都会拒绝汇总。

输出：

- `summary.json`：每种子两组指标、配对差值、差值均值/样本标准差、来源文件SHA与代码SHA。
- `paired_deltas.csv`：每个种子的`balanced - mean_pca`差值。
- `all_metrics.csv`：全部运行的各任务PSNR/SSIM/LPIPS与宏平均。
- `summary.md`：可直接阅读的逐种子对比和差值均值/样本标准差表。

重点看平均PSNR差值是否在新增种子上同向，以及是否以牺牲LSRW或Haze4K为代价。
遗忘和LPIPS差值越低越好；PSNR/SSIM差值越高越好。正差数量不是统计显著性检验。
报告全部种子，不挑选最好种子；均值±标准差不等于置信区间，三个种子也不足以证明普遍有效。
源代码中已有seed42与新入口逐张量一致的单元测试，既有结果能在共同边界协议下纳入参考。

## 诊断产物

每个新目录的`rainh_boundary_reselect.json`记录重选前后物理模型与摘要SHA、前向误差和逐层覆盖。
`task_1_final.pth`是重选后、尚未训练Haze4K的恢复边界；`task_2_final.pth`为选中Haze4K模型。
`history.json`记录新旧任务验证曲线，`continual_metrics.json`给出阶段测试与遗忘。
摘要覆盖率仍不是实际旧任务输出稳定性的保证；本次不据旧验证/测试调参或选保护方向。

## 实施验证（2026-11-10）

- 全部79项unittest通过；新增测试验证源/摘要完整性、无旧TRAIN读取、
  seed42与原入口逐张量一致、种子/方法更换拒绝、精确epoch恢复与配对统计/缺失/代码变化拒绝。
- 两模式的实际RainH源→Haze4K冒烟通过，各只有1步更新与每评估集1张图，不能作为质量证据。
  产物：`outputs/task_replicate_smoke_20261110`。
- 重跑相同批量冒烟命令，已验证两组完成任务均被自动跳过。
- 原正式seed42配对的共同模型、摘要和协议验证通过，平均PSNR差值为+0.12140155 dB。
  审计：`outputs/task_memory_replicates/baseline_seed42_audit.json`。
- 额外batch64单步容量复查因持续文件访问等待而停止，没有新容量结果；
  此前同骨干/任务记忆配置已实测allocated约17.53 GiB、reserved约19.67 GiB，
  仅作为显存估计，不能冒充本入口新的容量验证。正式4次20轮训练未启动。
