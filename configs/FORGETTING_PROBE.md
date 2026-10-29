# LSRW → RainH 短程遗忘对照

后续审计发现并修复了 LSRW 训练配对的补零编号碰撞。这里的对照固定同一个历史
task-0 checkpoint，并未重新训练任务0；不应视为数据修正后的从头基准。
分组验证与更多解释边界见 [BASIS_VALIDATION.md](BASIS_VALIDATION.md)。

运行前激活 odr-lora 环境，并在项目根目录执行：

```bash
CUDA_VISIBLE_DEVICES=1 python src/scripts/run_forgetting_probe.py \
  --output-dir outputs/forgetting_probe_new \
  --checkpoint outputs/lsrw/task_0_final.pth \
  --config configs/lsrw_rainh.yaml \
  --rounds 5 --steps-per-round 40 --batch-size 8
```

输出目录必须尚不存在，以保护之前的实验。默认仅使用可见的第一张卡，
PyTorch 显存分配上限为该卡总显存的 32%（不代表预留显存）。
中断后可在同一命令末尾加入 `--resume-experiment`，从保存的完整轮继续。
恢复时校验原配置、checkpoint SHA256 与主要实验参数；各轮按固定种子
重新采样。尚未保存的那一轮会重新执行。

两组从相同的 LSRW 完整训练 checkpoint 开始，均冻结非当前任务参数：

- baseline：保留原来的共享/私有基校准和软投影。
- protected：校准完成后，把共享、私有的整个 B 投影到旧 GPM 的正交补，
  B 随后冻结。对任意可训练 A，实际更新 ΔW=A B 均满足 ΔW M≈0。

投影两次以减少浮点残差，不重新归一化零行。旧 GPM 固定到本次实验结束，
不在训练结束时更换投影矩阵，以保证前向与合并函数一致。实验不增大 GPM
容量，便于分离“约束泄漏”和“保护子空间覆盖不足”两个问题。

两组数据与随机种子相同；各轮重新设定随机种子，使采样顺序、裁剪及增强
一致。默认 5 轮×40 步×8 张裁剪，并非 5 个完整数据 epoch。
学习率固定为配置 train.lr，损失仍为 L1+0.1 LPIPS（无回放/蒸馏）。
旧任务测试集只用于诊断，不进入优化器，不据此早停或挑选模型。

指标及产物：

- manifest.json：checkpoint SHA256、参数、随机种子及测试子集索引。
- coverage_lowlight.json / coverage_rainh.json：旧 GPM 对真实卷积输入的
  能量覆盖率。每集固定 8 张中心裁剪、每层每张最多 256 个输入 patch；
  mean 是逐层非加权平均，不能解释为全数据或整体模型的覆盖率。
- baseline/curve.json、protected/curve.json：每轮固定 LSRW 20 张、RainH
  30 张的 PSNR/SSIM/LPIPS，以及每层实际更新的旧子空间重叠指标。
- initial_full_metrics.json、各组 final_full_metrics.json、summary.json：
  完整 LSRW 50 张、RainH 100 张的起点与终点指标。
- curves.png：固定子集曲线；最终完整集指标以 summary.json 为准。
- probe_latest.pth / probe_merged.pth：仅用于此次实验检查。未更新 GPM，
  不应作为正式后续任务的训练起点，也不应拿轮数当正式 epoch 恢复。

默认整图评估，对应原有实验。可用 --tile-size 512 降低前向峰值显存，
但结果可能改变，两组应使用相同设置。实验只运行单个随机种子，适合
筛查实现问题；不能代替完整训练、多种子的持续学习评估。

普通训练也支持在 odr_lora 配置里显式加入 full_update_protection: true。
默认保持 false，避免修改已有实验的算法定义。启用后仍需检查 GPM 覆盖，
并注意保护空间满秩的层将没有可学习的正交补方向。
