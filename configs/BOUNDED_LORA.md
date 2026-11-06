# 无回放正交 LoRA：逐层相对更新范数上限

入口 `src/scripts/train_bounded_lora.py`，从无回放的 clean RainH task_1_final 分支，只训练 Haze4K。
本实验仅在已有完整更新硬正交保护上增加幅度约束；不创建旧数据回放池、教师或蒸馏损失，
不添加输出范围损失、不改变网络结构。LPIPS的固定VGG仍用于当前任务的原有感知损失。

## 约束定义

对每个 ODRLoRAConv2d 层，当前任务增量为 ΔW=A×B。
每次 AdamW 更新后，统一梯度裁剪已执行，再计算：

```text
r = ||A B||_F / max(||W_start||_F, 1e-12)
if r > rho:
    A <- A * (rho / r) * (1 - 1e-6)
```

其中 W_start 是进入当前任务时的累计主干权重，训练期间冻结。
rho 默认0.05，表示每层当前任务增量的 Frobenius 范数不超过基座范数约5%；
它不是每个权重元素最大改5%，也不是限制梯度范数。
rho=0 表示关闭此上限，作为匹配对照，而不是禁止所有更新。
缩放只修改当前A，B/旧参数/GPM保持不变，因此保留既有零空间约束。
AdamW一阶、二阶矩不随缩放修改；每步后重新施加硬上限。
该约束不保证非线性网络的旧输出严格不变，也不会恢复源断点已经遗忘的内容。
它约束本任务相对起点的增量，不是跨全部任务的总范数上限。

初始0.05仅为待验证超参数。历史20轮无回放对照的选中epoch5模型中，逐层相对增量
min/median/p90/max约为0.0061/0.0245/0.0352/0.0723，说明0.05可能首先限制少数大更新层。
此统计只使用模型权重，没有据测试图像搜索阈值；它不是0.05最优的证据。

## 数据与选模

训练、基校准、频率统计和任务末GPM更新仅使用 Haze4K 实际TRAIN分区。
默认 batch64、128裁剪、20轮（每轮43次，共860次optimizer更新），其他模型/保护设置继承共同源。
每5轮和末轮验证。仅按 **Haze4K验证PSNR** 选最优模型。
旧任务验证图像只作前向诊断，不参与loss、梯度、早停或选模。
最终测试仍报告所有已见任务。源断点/划分SHA与完整实验配置写入新目录。
此选模规则与之前 `train_stability_control.py` 的三个任务平均验证选模不同，
必须使用本入口重新跑关闭上限的对照，不能直接复用旧对照结果作等条件比较。

## 训练

```bash
conda activate odr-lora
cd /media/zhappy-701/新加卷/odr-lora/odr-lora-restoration-main
export PYTHONNOUSERSITE=1
```

约束组：

```bash
CUDA_VISIBLE_DEVICES=1 python -u src/scripts/train_bounded_lora.py \
  --source outputs/clean_effective_r16/task_1_final.pth \
  --max-relative-update 0.05 --epochs 20 --batch-size 64 \
  --output-dir outputs/bounded_lora_r005_b64
```

匹配对照（关闭上限，其余一致；可依次跑，或在GPU0空闲时改为0并行）：

```bash
CUDA_VISIBLE_DEVICES=1 python -u src/scripts/train_bounded_lora.py \
  --source outputs/clean_effective_r16/task_1_final.pth \
  --max-relative-update 0 --epochs 20 --batch-size 64 \
  --output-dir outputs/bounded_lora_control_b64
```

不要接续回放组、旧Haze4K末断点或smoke模型。此入口显式构造独立预算分支，
无需 `--allow-batch-size-change`。正式20轮训练未由代码验证自动启动。

## 恢复与日志

```bash
CUDA_VISIBLE_DEVICES=1 python -u src/scripts/train_bounded_lora.py \
  --source outputs/clean_effective_r16/task_1_final.pth \
  --resume outputs/bounded_lora_r005_b64/latest.pth \
  --max-relative-update 0.05 --epochs 20 --batch-size 64 \
  --output-dir outputs/bounded_lora_r005_b64
```

参数与首跑一致；源断点必须保留。每个完整epoch保存优化器/scheduler/随机状态，
未完成epoch重跑。恢复中途不能更换rho、轮数、batch、数据或源断点。

```bash
tail -f outputs/bounded_lora_r005_b64/trainer.log
```

`UPDATE_BOUND` / `history.json` 每轮记录 cap、投影前/后最大比值及 clipped_layer_steps。
clipped_layer_steps是所有步中超限层数的累加，不是不同层的数量。
若一直为0，说明阈值尚未实际限制训练；不能将该实验解释为幅度约束已产生效果。
`validation.psnr`只代表Haze4K；`validation.tasks`包含三任务PSNR/SSIM/LPIPS及原始输出异常统计。
任务末 `task_2_update_bound.json` 保存选中模型合并前各层比值，`task_2_protection.json`检查正交泄漏。
`task_2_final.pth`包含选中模型更新后的GPM；`merged.pth`仅推理。

优先比较两组相同轮次的旧任务保持与Haze4K学习，随后比较各自选中模型的最终测试指标。
若遗忘改善但Haze4K明显变差，属于稳定性/可塑性权衡，不能仅据旧任务指标宣称整体改进。

## 验证范围

已运行相关单元测试，覆盖超限投影、正交性与冻结张量、关闭上限、零基座、非有限值拒绝、
不访问旧任务训练loader、只按新任务选模、修改上限拒绝恢复，以及中断恢复逐张量一致。
真实NAFNet/VGG的两步冒烟使用极小rho=1e-6强制触发缩放，
产物在 `outputs/bounded_lora_smoke_20261104`，不属于质量证据。
另有完整batch64、两步更新与强制投影的容量检查，结果在
`outputs/bounded_lora_batch64_smoke_20261104/verification.json`。
这些检查不代表默认rho=0.05已通过正式质量实验。
