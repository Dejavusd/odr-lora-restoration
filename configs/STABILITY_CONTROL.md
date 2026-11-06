# 全图输出稳定性对照

2026-11-04 已完成代码实现、异常样本激活追踪、真实 GPU 冒烟、batch64 单步容量检查。
尚未启动正式20轮对照，尚无该方法能提升整体指标的质量结论。

## 异常定位证据

追踪三个阶段断点上的 LSRW 验证索引97及 RainH 验证索引123，比较完整图像和中心128裁剪。
原始结果：`outputs/extreme_trace_20261104/trace.json`。

| 阶段/样本 | 全图输出范围 | 中心128输出范围 | 按前向调用顺序首个绝对值超过1000的模块 |
|---|---|---|---|
| LSRW完成 / LSRW97 |[-0.006,0.826]|[-0.011,0.630]|无|
| RainH完成 / LSRW97 |[-4900.77,7757.81]|[-0.042,0.602]|dec_blocks.3.3.sca|
| Haze4K完成 / LSRW97 |[-25407.15,29168.84]|[-0.119,0.249]|dec_blocks.3.3.gate|
| RainH完成 / RainH123 |[-0.346,1.152]|[-0.043,1.069]|无|
| Haze4K完成 / RainH123 |[-2622.30,2033.89]|[-0.087,1.153]|dec_blocks.0.0.sca|

阈值只用于定位数值放大的位置，不表示该模块就是唯一根因；某些内部大激活未导致异常最终输出。
中心裁剪正常不能证明全部小裁剪正常。这里的骨干采用 GroupNorm(1,C)、全局池化注意力和乘法门控，
全图统计与小裁剪不同。现有证据支持优先验证全图输出约束，不能声称已证明全部根因。
本次未改网络归一化、注意力、门控或推理输出公式。
RainH123 异常全图另用FP64复查，输出仍为约[-2622.32,2033.90]，与FP32输出相对L2差约2.61e-5。
该样本的极端值并未因提高精度消失，不能简单归因于TF32或FP32舍入。
记录：`outputs/extreme_trace_20261104/precision_check.json`。

## 对照设计

两组都从同一个 `outputs/clean_effective_r16/task_1_final.pth` 开始，只训练 Haze4K。
相同种子、20轮、每轮全部2700张新任务训练图、batch64、128裁剪、AdamW、cosine、lr2e-4、GPM32。
每轮43次新任务更新（最后一个 batch=12），每组共860次优化器更新。
两组每5轮及末轮用完整 LSRW564/RainH126/Haze4K300 验证集评估。
使用三个任务验证PSNR的算术均值选模；测试集只在任务结束时使用。
选模规则与旧的仅新任务验证选模不同，所以必须比较这两组，不能直接对比历史400轮宣称改进。

- `control`：原有新任务 L1 + 0.1 LPIPS。
- `replay_distill`：新任务损失 + 0.1范围惩罚 + 两个旧任务回放损失的平均值。

每个旧任务回放损失为：

```text
L1(student_raw, old_GT)
+ L1(student_raw, clamp(teacher_raw, 0, 1))
+ 0.1 * mean(relu(-student_raw) + relu(student_raw - 1))
```

教师为固定的共同 RainH 起点，eval模式且所有参数不求梯度。
旧教师存在已知全图极端值，所以只对教师目标截断；学生输出保留原始值并接受梯度。
真实旧GT同时提供监督，避免将已遗忘或异常的教师输出视为唯一正确目标。
这是“监督回放+输出蒸馏+范围约束”的组合对照，不能用于分辨各组件的独立贡献。

每个旧任务从实际 TRAIN 分区固定抽64张输入，按GT组去重，总128张。
每次优化器更新从两个池中各抽一张**完整图像**，不裁剪、不缩放；验证和测试图像不进入池。
选图清单、SHA、分组和TRAIN索引存入 `replay_manifest.json`。
新任务与两个旧图分别backward，再统一梯度裁剪/optimizer.step，避免同时保留全部图。
相比纯新任务组，回放组使用额外旧数据和计算，须在报告中明确标注。

## 训练命令

在仓库根目录：

```bash
conda activate odr-lora
cd /media/zhappy-701/新加卷/odr-lora/odr-lora-restoration-main
export PYTHONNOUSERSITE=1
```

先运行回放组：

```bash
CUDA_VISIBLE_DEVICES=1 python -u src/scripts/train_stability_control.py \
  --source outputs/clean_effective_r16/task_1_final.pth \
  --mode replay_distill --epochs 20 --batch-size 64 --memory-per-task 64 \
  --output-dir outputs/stability_replay_b64
```

另运行匹配对照组（可依次使用GPU1；两卡空闲时也可将该命令的GPU改为0并行）：

```bash
CUDA_VISIBLE_DEVICES=1 python -u src/scripts/train_stability_control.py \
  --source outputs/clean_effective_r16/task_1_final.pth \
  --mode control --epochs 20 --batch-size 64 --memory-per-task 64 \
  --output-dir outputs/stability_control_b64
```

本入口从源断点继承模型、GPM、数据划分和保护设置，显式建立新预算/损失分支，不需要额外
`--allow-batch-size-change`。默认显存额度0.9。不要加 `--smoke-test` 做质量实验。

回放组断线恢复（所有参数与首跑一致）：

```bash
CUDA_VISIBLE_DEVICES=1 python -u src/scripts/train_stability_control.py \
  --source outputs/clean_effective_r16/task_1_final.pth \
  --resume outputs/stability_replay_b64/latest.pth \
  --mode replay_distill --epochs 20 --batch-size 64 --memory-per-task 64 \
  --output-dir outputs/stability_replay_b64
```

每个完整epoch保存latest，包含优化器、scheduler、随机状态和训练历史；教师/回放池从固定源重建，
源SHA、数据划分和完整新协议必须一致。未完成epoch重跑。输出目录非空时禁止新开覆盖。

## 查看结果

```bash
tail -f outputs/stability_replay_b64/trainer.log
```

`history.json` 的每个验证点有三个任务PSNR/SSIM，以及原始输出最大值、越界比例；
旧任务还记录与教师的clamp后MAE、学生原始输出对有界教师的MAE。
`validation.psnr` 是三个验证PSNR的平均值，不是仅Haze4K的PSNR。
`last_step_losses` 是该轮最后一步分项，不是整轮均值。
最终 `continual_metrics.json` 给出标准测试集PSNR/SSIM/LPIPS、ACC/BWT/遗忘；
`task_2_final.pth` 是验证均值最优模型更新GPM并合并后的断点，`merged.pth`仅用于推理。

先看相同验证轮次下旧任务跌幅、输出漂移与极端值是否变小，再看Haze4K及三任务平均表现。
若旧任务稳定但新任务明显学不动，才能进一步对回放/蒸馏权重做单变量对照。

## 已完成的实现验证

- 17项相关测试通过，包括教师目标有界且无梯度、回放仅取TRAIN、训练中断恢复一致。
- 两组真实NAFNet/VGG/数据单步冒烟通过；smoke产物不能用于正式实验恢复。
- 完整batch64 + 两张完整旧图回放完成真实前向/反向/AdamW更新：allocated峰值14.24 GiB、
  reserved峰值16.20 GiB，教师无梯度，源断点未修改。
- 单步容量检查仅用一个校准batch，不代表正式质量测试；其他完整图像尺寸可能增加峰值。
- GPU冒烟产物：`outputs/stability_replay_smoke_20261104`、`outputs/stability_control_smoke_20261104`。
- 容量记录：`outputs/stability_batch64_capacity_20261104/capacity.json`。
