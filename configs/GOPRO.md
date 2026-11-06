# GOPRO 去模糊

当前无回放五任务实验使用新入口 `train_five_tasks.py`，参见 [FIVE_TASKS.md](FIVE_TASKS.md)。
下文为旧入口；新实验采用录像分组验证与分块评估协议。

保留 `data/GOPRO/{train,test}/序列名/{blur,blur_gamma,sharp}/` 原始目录。
配置 `type: gopro` 使用序列名和完整文件名配对，默认输入为 `blur`，
GT 为 `sharp`。可将 `input_folder` 改为 `blur_gamma`，但训练与测试
会统一使用同一个版本。不会同时使用两个模糊版本。

发现缺失配对、缺失序列目录时立即报错，不丢弃未配对样本。
当前数据为训练 2103 对、测试 1111 对，RGB、1280×720。
训练采用同步随机裁剪和增强；验证保留完整尺寸。评估保存示例时
文件名包含序列名，避免不同序列的同名帧覆盖。

单独训练：

```bash
python src/scripts/train_incremental.py --config configs/gopro.yaml
```

在 LSRW → RainH → Haze4K → 去噪完成之后继续，先确认 checkpoint 存在：

```bash
python src/scripts/train_incremental.py \
  --config configs/lsrw_rainh_haze4k_denoise_gopro.yaml \
  --resume outputs/lsrw_rainh_haze4k_denoise/task_3_final.pth
```

默认 patch_size=128、batch_size=8，评估采用 512 大小、64 重叠的分块。
分块和整图评估的上下文不同，指标可能不同；比较时保持设置一致。
如需整图评估，将 `eval.tile_size` 设为 0。
