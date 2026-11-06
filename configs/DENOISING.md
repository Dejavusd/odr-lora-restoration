# DIV2K + Flickr2K 去噪

当前无回放五任务实验使用新入口 `train_five_tasks.py`，参见 [FIVE_TASKS.md](FIVE_TASKS.md)。
下文为旧入口协议，噪声种子及评估方式不要与新协议混用比较。

保留原始干净图像目录，不需要生成或复制 input/gt：

```text
data/DIV2K+Flickr2K/
  train/   # DIV2K 800 张 + Flickr2K 2650 张
  test/    # DIV2K 100 张
  BSD68/   # 灰度图像 68 张
```

`type: gaussian_denoising` 对每张干净图在线生成 AWGN 输入。默认
`noise_sigma: 25` 表示 0–255 尺度上的标准差，归一化后为 25/255。
训练时先随机裁剪及增强，再独立采样噪声；测试不裁剪，并使用
`noise_seed + 图像索引` 固定噪声。噪声输入不截断到 [0,1]。
这是合成高斯去噪实验，不代表真实传感器噪声。

RGB 图像的各通道独立加噪。BSD68 单通道图像先复制为 RGB，并使用
同一份单通道噪声；评估前将模型输出的三个通道求平均再复制为 RGB。
PSNR/SSIM 等价于该灰度结果上的计算；LPIPS 使用复制的三通道图像。

测试结果包含 `datasets.DIV2K` 和 `datasets.BSD68` 两组指标与样本数。
任务顶层指标按全部 168 张图片等权平均，而不是两个测试集等权平均。
验证和独立评估均使用配置中的 `eval.tile_size: 512`、
`tile_overlap: 64`，重叠区域求平均。分块可降低模型前向峰值显存，
但由于局部上下文和通道注意力，结果可能与整图推理不同；比较实验时
保持该设置一致。设置 `tile_size: 0` 可改为整图推理。

单独训练：

```bash
python src/scripts/train_incremental.py --config configs/denoise.yaml
```

在完成的 Haze4K 任务之后继续（须先确认 task_2_final.pth 存在）：

```bash
python src/scripts/train_incremental.py \
  --config configs/lsrw_rainh_haze4k_denoise.yaml \
  --resume outputs/lsrw_rainh_haze4k/task_2_final.pth
```

需要其他噪声强度时修改 `noise_sigma`，建议使用新的输出目录，避免
覆盖不同噪声设置的结果。当前实现每次训练采用固定 sigma。
