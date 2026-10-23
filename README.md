# ODR-LoRA: 空间-频率-尺度感知正交动态低秩适配

面向图像复原持续学习（Continual Image Restoration）的工程实现。

ODR-LoRA 在**完整 NAFNet** 骨干上，把标准卷积替换为 `ODRLoRAConv2d`，通过**共享基座 + 动态低秩分支 + 空间-频率联合正交**，实现：

- 任务数可动态增加，不受固定 `T` 限制；
- 每个新任务只优化当前低秩系数（首任务用于建立基座）；
- 每个任务结束后立即合并回累积基座，推理时不增加 FLOPs；
- 用每层一个、秩上限受控的 DualGPM 激活子空间限制旧任务输出漂移；
- 多级 Haar DWT 与空间分块用于改善子空间采样覆盖，不引入位置相关推理分支。

当前代码是可验证的研究原型。“减少遗忘”需要通过多任务顺序、多随机种子
与强基线实验证明，不应在实验前解读为“无遗忘”。



## 环境依赖
```bash
conda env create -f environment.yml
conda activate odr-lora
```

已有 Python 3.10 环境时也可使用 `pip install -r requirements.txt`。
主要依赖：

- torch 2.5.1（当前环境使用 CUDA 11.8 构建）
- torchvision 0.20.1
- numpy
- scipy
- opencv-python
- PyYAML
- tqdm
- lpips（可选，缺失时回退到 L1）
- scikit-image（可选，缺失时使用内置 PSNR/SSIM）



## 数据集：

数据集准备
使用五个图像复原任务
任务	数据集
去雨	Rain100H
去雾	Haze4K
去模糊	GoPro
去噪	DIV2K + BSD68（合并为 DIV2K_BSD68）
低光增强	LSRW

odr-lora-restoration/
data/
├── Rain100H/
│   ├── train/input/  gt/
│   └── test/input/   gt/
├── Haze4K/
│   ├── train/input/  gt/
│   └── test/input/   gt/
├── GoPro/
│   ├── train/input/  gt/
│   └── test/input/   gt/
├── DIV2K_BSD68/
│   ├── train/input/  gt/
│   └── test/input/   gt/
└── LSRW/
    ├── train/input/  gt/
    └── test/input/   gt/

检查目录结构：
```bash
python src/scripts/download_data.py --data_root ./data --check_only
```


# 快速开始
1. 增量训练
```bash
python src/scripts/train_incremental.py --config configs/default.yaml
```

指定输出目录：
```bash
python src/scripts/train_incremental.py \
    --config configs/default.yaml \
    --output_dir ./outputs/exp1
```

从 checkpoint 恢复：
```bash
python src/scripts/train_incremental.py \
    --config configs/default.yaml \
    --resume ./outputs/task_2_final.pth
```

训练器会在每个任务合并后用同一累积模型评估所有已见任务，并将
`R(i,j)` 矩阵、平均 PSNR、BWT 和 forgetting 写入
`outputs/continual_metrics.json`。

2. 评估
评估合并后的累积模型（所有任务使用同一权重）：
```bash
python src/scripts/evaluate.py \
    --config configs/default.yaml \
    --ckpt ./outputs/merged.pth \
    --merged
```

评估训练中保存的 `task_*_best.pth`（当前任务分支尚未合并）：
```bash
python src/scripts/evaluate.py \
    --config configs/default.yaml \
    --ckpt ./outputs/task_4_best.pth
```

`task_*_final.pth` 和 `merged.pth` 都是已合并状态，不依赖任务 ID。

保存部分复原图像：
```bash
python src/scripts/evaluate.py \
    --config configs/default.yaml \
    --ckpt ./outputs/merged.pth \
    --merged \
    --save_images
```

3. 消融实验
运行全部消融：
```bash
python src/scripts/run_ablations.py
```

只运行某个消融：
```bash
python src/scripts/run_ablations.py --only no_spatial
```

跳过已完成消融：
```bash
python src/scripts/run_ablations.py --resume
```

4. 单元测试
```bash
python tests/test_odr_lora.py
python tests/test_dual_gpm.py
python tests/test_orthogonality.py
python tests/test_frequency.py
python tests/test_training_lifecycle.py
```

## 引用
如果本项目对你有帮助，请参考相关论文：

- CLFD: Continual Learning in the Frequency Domain
- InFoRA: Interference-Free Low-Rank Adaptation for Continual Learning
- NAFNet: Simple Baselines for Image Restoration
