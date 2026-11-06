# 固定容量、无回放的旧记忆方向保留实验

后续的汇总式旧/当前方向配额对照见 [BALANCED_MEMORY.md](BALANCED_MEMORY.md)，
使用 `--merge-rule balanced --old-fraction`；其100%组与本页逐批preserve不能视为完全相同算法。

2026-11-05完成机制诊断与实现验证，正式20轮质量实验未启动。

## 合并前后覆盖率诊断

固定同一份 RainH 结束模型的真实激活，对比 `memory_raw32_b64/task_1_best.pth`
中的合并前GPM与 `task_1_final.pth` 中的合并后GPM。
32张固定LSRW训练留出验证图像、完整分辨率，每层均匀抽128个真实空间patch。
指标是各层覆盖率的算术均值，并非全网络能量总覆盖率。

| 记忆合并规则 | 合并前覆盖率 | 合并后覆盖率 | 旧子空间相对残差层均值 |
|---|---:|---:|---:|
| 原有压缩 |81.253%|67.324%|0.5831|
| 保留旧方向 |81.253%|81.294%|8.80e-8|

两份模型的非GPM张量逐项一致，所以这个局部差异来自记忆本身，不是两组网络权重不同。
覆盖率改善不等于最终输出稳定，前向非线性与旧任务特征漂移仍然存在。
原始结果：`outputs/memory_transition_20261105/audit.json` 和
`outputs/memory_transition_preserved_20261105/audit.json`。

## 实验范围

不重训LSRW或RainH，固定上一轮空间组已选中的RainH权重。
新入口加载匹配的 `outputs/memory_raw32_b64/task_1_best.pth`（尚未任务末更新/合并）
和 `task_1_final.pth`（用于确认选中epoch、物理权重及已有阶段结果）。
仅重做RainH任务末记忆更新，再训练Haze4K20轮。
任务末使用RainH TRAIN输入拟合记忆、不更新网络；进入Haze4K训练后不再读取RainH或LSRW训练图。
这属于从旧任务结束前的边界状态分支，不是训练Haze4K时回放旧图像。

两组保持真实空间统计、每层容量上限32、相同抽样预算、相同RainH物理权重、
LoRA校准方式、学习率、batch64、20轮和仅新任务验证选模。无教师、蒸馏、范数上限或输出范围损失。

- `compressed`：沿用有损SVD合并。
- `preserve`：对旧基QR正交化，在旧空间正交补中提取当前任务残差方向，只填剩余容量，禁止丢弃旧方向。

旧记忆更新规则标记为 `bounded_preserving`，仍使用与对照相同的列数。
它调用已有 `DualGPM.update_preserving`，容量保持32，不扩大记忆、不保存旧图像。
51项unittest与50个独立函数测试通过，包括冻结权重、旧空间保留、同列预算、容量边界、
只在RainH边界访问其训练数据、改变规则拒绝恢复，以及中断恢复逐张量一致。
实际GPU检查还修正了从CPU源断点加载动态GPM缓冲区后迁移到训练设备的处理。

## 关键限制：多数层已满

162层中，RainH合并前已有127层满容量，平均每层仅剩1.895个空位。
这些满层若严格保留旧空间，就不能再加入RainH方向。它可能保护LSRW，却削弱RainH在下一任务中的保护。
因此这是检验“丢弃历史方向是否导致遗忘”的机制对照，不是一个保证解决长期多任务记忆的方案。
不要仅看LSRW改善就判定成功，必须一起看RainH、Haze4K和平均性能。

## 训练命令

在项目根目录运行，先跑保留旧方向组：

```bash
conda activate odr-lora
cd /media/zhappy-701/新加卷/odr-lora/odr-lora-restoration-main
export PYTHONNOUSERSITE=1

CUDA_VISIBLE_DEVICES=1 python -u src/scripts/train_memory_retention.py \
  --run-dir outputs/memory_raw32_b64 \
  --merge-rule preserve --epochs 20 --batch-size 64 \
  --output-dir outputs/memory_preserve32_b64
```

入口自动匹配best/final并重新执行RainH记忆边界。正式边界使用原实验的6批×64输入，
固定seed=42+900001，与原实验相同。随后Haze4K独立重置增量种子。
不要手动把task_1_final当成尚未更新记忆的状态，也不要用其他旧实验的best代替。

如需要通过新入口重新跑压缩对照：

```bash
CUDA_VISIBLE_DEVICES=1 python -u src/scripts/train_memory_retention.py \
  --run-dir outputs/memory_raw32_b64 \
  --merge-rule compressed --epochs 20 --batch-size 64 \
  --output-dir outputs/memory_compressed32_b64
```

已核验按原预算重建的压缩组任务末checkpoint，所有模型张量（含GPM）与原
`memory_raw32_b64/task_1_final.pth`逐项完全一致。原空间组Haze4K结果可作为已完成参考；
新入口的压缩组用于需要再次确认端到端重现时运行。两组都仅按Haze4K验证PSNR选模。

## 恢复

```bash
CUDA_VISIBLE_DEVICES=1 python -u src/scripts/train_memory_retention.py \
  --run-dir outputs/memory_raw32_b64 \
  --merge-rule preserve --epochs 20 --batch-size 64 \
  --resume outputs/memory_preserve32_b64/latest.pth \
  --output-dir outputs/memory_preserve32_b64
```

保留原run-dir中的best/final源文件，恢复时校验源SHA、数据划分和完整协议。
RainH边界完成后即保存新task_1_final与latest，此后恢复不再拟合RainH记忆。
每个完整Haze4K epoch保存优化器、scheduler、随机状态；未保存的epoch重跑。
边界构造中断且没有latest时，保留失败目录，在新目录重启。

## 检查产物

- `rainh_boundary_rebuild.json`：网络权重一致性、各层旧/新秩、空位、旧方向残差。
- `task_1_gpm_transition.json`：边界GPM与合并前向一致性。
- `history.json`：三任务验证曲线与原始输出范围，选模只使用Haze4K分数。
- `continual_metrics.json`：最终测试指标、平均性能、BWT/遗忘。

两个真实batch64单步已完成（原预算6个RainH边界batch，Haze4K单步前后向）：
峰值allocated约14.16 GiB，两组当前更新的正交泄漏最大值约1e-7，网络起点一致。
记录：`outputs/retention_capacity_20261105/verification.json`；
此检查只用一个B校准batch，不代表正式训练质量。
两步GPU冒烟：`outputs/retention_preserve_smoke_v2_20261105`，不能用作正式训练起点。

离线诊断复现：

```bash
CUDA_VISIBLE_DEVICES=1 python -u src/scripts/audit_memory_transition.py \
  --run-dir outputs/memory_raw32_b64 --output-dir outputs/memory_transition_new
```
