# 干净配对重训 LSRW，再复验 RainH → Haze4K

从balanced50三任务断点继续去噪/GOPRO的五任务实验，参见 [FIVE_TASKS.md](FIVE_TASKS.md)。

历史空间/当前任务残差的配额对照，参见 [BALANCED_MEMORY.md](BALANCED_MEMORY.md)。

固定容量保留历史方向的无回放对照，参见 [MEMORY_RETENTION.md](MEMORY_RETENTION.md)。

频域/真实空间 GPM 的无回放统计来源单变量训练，参见 [MEMORY_SOURCE_STUDY.md](MEMORY_SOURCE_STUDY.md)。

频域/空间域统计的离线核对结果，参见 [GPM_COORDINATE_AUDIT.md](GPM_COORDINATE_AUDIT.md)。

无回放正交LoRA的相对更新范数约束实验，参见 [BOUNDED_LORA.md](BOUNDED_LORA.md)。

2026-11-04 的下一步短程全图回放/输出一致性对照，参见 [STABILITY_CONTROL.md](STABILITY_CONTROL.md)。

新入口：`src/scripts/train_clean_sequence.py`。配置：`configs/clean_lsrw_rainh_haze4k.yaml`。
旧 `train_incremental.py` 的行为没有被替换；本轮正式复验请使用新入口。
这里提供正式训练命令，验证阶段仅启动了短程冒烟与恢复测试，没有后台启动 400 轮长跑。

## 1. 数据准备（当前工作区已完成）

在仓库根目录、`odr-lora` 环境执行；所有后续命令也在此目录运行：

```bash
conda activate odr-lora
cd /media/zhappy-701/新加卷/odr-lora/odr-lora-restoration-main
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
```

仅首次没有划分清单时运行：

```bash
python src/scripts/prepare_clean_splits.py --config configs/clean_lsrw_rainh_haze4k.yaml
```

现有 `outputs/clean_splits_v1/manifest.json` 已生成，不要覆盖或重新随机划分。
可选完整文件 SHA256 复查（会重新读取所有图像）：

```bash
python src/scripts/prepare_clean_splits.py --config configs/clean_lsrw_rainh_haze4k.yaml --check
```

| 数据集 | 原训练对数 | 实际训练 | 留出验证 | 原测试 |
|---|---:|---:|---:|---:|
| LSRW | 5600 | 5036 | 564 | 50 |
| RainH | 1254 | 1128 | 126 | 100 |
| Haze4K | 3000 | 2700 | 300 | 1000 |

按完整文件名安全配对，GT 尺寸不一致直接报错，不再静默缩放。
按解码后 GT 内容 SHA256 分组，重复真实图像的不同退化版本不会跨训练/验证。
LSRW 有 5553 个不同训练 GT；RainH 209 个；Haze4K 750 个，因此验证对数不一定恰好是原图数的 10%。
审计没有发现训练/测试完全相同的 GT，实际划分也没有相同输入文件跨分区。
这不等于保证近重复场景完全隔离；原测试集此前已参与开发，不能称为全新未触碰测试集。
清单保存绝对路径、文件哈希和文件状态；原图不移动、不删除、不改名。

## 2. 从头重训共同的 LSRW 起点

先用 `nvidia-smi` 确认可见 GPU 有足够余量。以下只使用物理 GPU 1：

```bash
CUDA_VISIBLE_DEVICES=1 python src/scripts/train_clean_sequence.py \
  --config configs/clean_lsrw_rainh_haze4k.yaml \
  --output-dir outputs/clean_lsrw \
  --stop-after lowlight
```

**不要添加 `--resume outputs/lsrw/...`**：这是从零训练，不接续历史错误配对的模型。
当前默认 batch=64、128 裁剪、最多 400 个 epoch、LR=2e-4 的 cosine、L1+0.1 LPIPS。
2026-11-03 已完成的 clean 断点使用 batch=16；对这些历史实验精确恢复时需使用 batch=16 的配置。
主训练与评估使用严格 FP32（TF32 关闭），每 10 epoch 用完整训练留出集验证，末轮也验证。
不据旧测试集调参或选模；同分优先较早 epoch。warmup 显式为 0，不像旧配置那样写了但未使用。
任务结束时恢复验证最优权重，再仅用实际训练分区初始化共享基并更新 GPM，最后合并、评估测试集。
所以 `task_0_final.pth` 不一定来自第 400 轮权重，而是验证最优模型加上可继续训练的记忆。

共同起点是：`outputs/clean_lsrw/task_0_final.pth`。
`task_0_best.pth` 是尚未执行任务末记忆更新的中途状态，不作为不同方法的分支起点。
`merged.pth` 仅用于推理，不能续训。

## 3. effective-r16 接续 RainH，再到 Haze4K

等上一步确实完成，再执行：

```bash
CUDA_VISIBLE_DEVICES=1 python src/scripts/train_clean_sequence.py \
  --config configs/clean_lsrw_rainh_haze4k.yaml \
  --resume outputs/clean_lsrw/task_0_final.pth \
  --output-dir outputs/clean_effective_r16 \
  --method effective_r16
```

这条命令自动依次训练 **RainH → Haze4K**，不会重新训练 LSRW。
两任务分别最多 400 epoch；相似度估计、基校准、优化和任务末 GPM 更新均只读取实际训练分区。
新任务全部 A/B 更新方向都受到完整投影保护；旧基座和旧分支在任务内冻结并验证指纹。
仅增大后续新任务的私有基础秩到 16，旧任务分支形状不变。

若希望先停下检查 RainH，在上述命令末尾加 `--stop-after rainh`；之后接 Haze4K：

```bash
CUDA_VISIBLE_DEVICES=1 python src/scripts/train_clean_sequence.py \
  --config configs/clean_lsrw_rainh_haze4k.yaml \
  --resume outputs/clean_effective_r16/task_1_final.pth \
  --output-dir outputs/clean_effective_r16 \
  --method effective_r16
```

## 4. 公平复验：三种方法共用同一个干净 task-0

只跑候选并不能确认增益。另运行完整保护基线，以及等参数的 effective-r8；不要分别重训三个不同 task-0 混淆比较。

```bash
CUDA_VISIBLE_DEVICES=0 python src/scripts/train_clean_sequence.py \
  --resume outputs/clean_lsrw/task_0_final.pth \
  --output-dir outputs/clean_current_basis --method current_basis
```

```bash
CUDA_VISIBLE_DEVICES=1 python src/scripts/train_clean_sequence.py \
  --resume outputs/clean_lsrw/task_0_final.pth \
  --output-dir outputs/clean_effective_r8 --method effective_r8
```

| 方法 | 基构造 | 后续任务私有基础秩 |
|---|---|---:|
| current_basis | legacy + 完整 B 投影 | 8 |
| effective_r8 | 投影后有效单位基 | 8 |
| effective_r16 | 投影后有效单位基 | 16 |

三者都固定 frequency GPM-32、网络、数据划分、训练预算、损失和选模协议。
r16 多参数，不是等参数消融。只有两卡都空闲时才并行跑两个独立实验；本入口不是 DDP 双卡训练。
重复增量种子可加 `--incremental-seed 7` 或 `123`，并改成新的输出目录，例如 `outputs/clean_effective_r16_s7`。
各方法用相同增量种子配对比较；这仍是共用同一个预训练起点的增量多种子，不是完整从零多种子。

## 5. 断点恢复与防止断线

每个完整 epoch 原子保存 `latest.pth`，包含模型、AdamW、scheduler、CPU/CUDA/Python/NumPy RNG、
best 指标/路径、历史曲线、数据划分 SHA 和完整实验协议。未保存完的 epoch 会重跑，不承诺批次中途恢复。
丢失验证最优文件会报错，不会悄悄用末轮代替。保留完整输出目录，不只复制 latest。

LSRW 中断后：

```bash
CUDA_VISIBLE_DEVICES=1 python src/scripts/train_clean_sequence.py \
  --resume outputs/clean_lsrw/latest.pth \
  --output-dir outputs/clean_lsrw --stop-after lowlight
```

后续两任务中断后：

```bash
CUDA_VISIBLE_DEVICES=1 python src/scripts/train_clean_sequence.py \
  --resume outputs/clean_effective_r16/latest.pth \
  --output-dir outputs/clean_effective_r16 --method effective_r16
```

若原来指定了其他 `--incremental-seed`，恢复时保持相同参数。
不能在同一个任务中途更换方法、种子、数据、batch、轮数或学习率；旧历史模型/probe/smoke/inference export 都不能混入正式训练。
`latest` 若已是任务完成状态，会自动进入下一个任务。

从已完成任务分支并显式改为当前配置的 batch=64，需要新输出目录和
`--allow-batch-size-change`。例如仅重跑 Haze4K：

```bash
CUDA_VISIBLE_DEVICES=1 python -u src/scripts/train_clean_sequence.py \
  --config configs/clean_lsrw_rainh_haze4k.yaml \
  --resume outputs/clean_effective_r16/task_1_final.pth \
  --output-dir outputs/clean_effective_r16_b64_haze4k \
  --method effective_r16 --allow-batch-size-change
```

该选项只放宽 batch size 和显存额度比较，其余协议检查继续执行；任务中途禁止使用。
batch64 + VGG 的真实单步检查超过旧 50% 显存额度，当前配置已将 memory_fraction 提至 0.9。
分支记录在新目录的 run_manifest.json 中。64 相比 16 的每轮优化步数更少，
新实验不属于等更新次数对照；改变 batch size 本身不保证减少遗忘。

## 旧任务输出稳定性审计

使用两个正式任务末断点，在全部旧任务训练留出验证图像上逐张比较输出：

```bash
CUDA_VISIBLE_DEVICES=1 python -u src/scripts/audit_output_stability.py \
  --before outputs/clean_effective_r16/task_1_final.pth \
  --after outputs/clean_effective_r16/task_2_final.pth \
  --output-dir outputs/output_stability_new
```

输出 stability.json 包含每图 PSNR/SSIM、原始及 clamp 后输出漂移、变化方向、
各层真实卷积输入的 GPM 覆盖率和局部增量响应。覆盖率只抽样每任务前 8 张验证图，
每层每图最多 128 个真实卷积 patch；输出指标使用全部验证图像。
审计读取历史训练结果，不代表 batch=64 已重新训练；源断点 SHA256 在运行前后检查。

远程连接不稳定，建议先在 `tmux new -s odr-clean` 中执行训练命令；断开后用
`tmux attach -t odr-clean` 回到同一会话。另一个终端查看：

```bash
tail -f outputs/clean_lsrw/trainer.log
```

候选阶段改看 `outputs/clean_effective_r16/trainer.log`。

## 6. 结果与解释边界

- `history.json`：各任务 loss、LR 与验证 PSNR/SSIM/LPIPS。
- `task_i_best.pth`：验证最优但尚未做任务末更新的状态；仅同方法中途恢复使用。
- `task_i_final.pth`：最优权重 → 训练数据更新记忆 → 合并后的正式续训起点。
- `continual_metrics.json`：每阶段完整测试矩阵、平均 PSNR、BWT 与非负 forgetting。
- `task_i_protection.json`：训练内完整更新对旧 GPM 的泄漏；`task_i_gpm_transition.json`：
  任务末记忆压缩的旧方向残差，以及更新记忆/合并前后的输出误差。
- `merged.pth`：去除训练记忆与分支后的推理模型，不可续训。
- `run_manifest.json`：配置/划分/源码哈希；非空目录不允许意外从头覆盖。

本轮没有把 GPM-32 改成“无限保留所有旧方向”：频带记忆合并压缩可能丢弃旧方向，
这是后续三任务遗忘必须记录和分析的因素。新入口会告警并保存逐层诊断，而不声称压缩后全部旧方向仍受保护。
不能把 `更新 GPM` 理解为已经解决所有遗忘。也不要直接与旧错误配对/测试选模的结果作等协议比较。

仅检查流程是否连通，可以用独立目录：

```bash
CUDA_VISIBLE_DEVICES=1 python src/scripts/train_clean_sequence.py \
  --output-dir outputs/clean_smoke_new --smoke-test
```

这只做每任务 2 次更新、小样本验证/测试，不能据此判断恢复质量，也不能续入正式训练。
默认 4-worker 的真实 GPU 恢复一致性检查：

```bash
CUDA_VISIBLE_DEVICES=1 python src/scripts/check_clean_resume.py \
  --output-dir outputs/clean_resume_audit_new
```
