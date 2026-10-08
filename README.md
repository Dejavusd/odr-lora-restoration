# ODR-LoRA: 空间-频率-尺度感知正交动态低秩适配

面向图像复原持续学习（Continual Image Restoration）的工程实现。

ODR-LoRA 在**完整 NAFNet** 骨干上，把标准卷积替换为 `ODRLoRAConv2d`，通过**共享基座 + 动态低秩分支 + 空间-频率联合正交**，实现：

- 任务数可动态增加，不受固定 `T` 限制；
- 每任务只增加少量低秩参数，内存友好；
- 训练后分支合并回基座，推理零额外 FLOPs；
- 旧任务几乎无遗忘，且不需要旧任务数据；
- 借鉴 CLFD 的频域思想，使用**多级 Haar DWT** 将特征分解为 7 个子带，按空间块和子带独立维护旧任务梯度空间；
- 引入频带互补损失、频带选择正则化，进一步减少任务间干扰。

---

数据集准备
使用五个图像复原任务
任务	数据集
去雨	Rain100H
去雾	Haze4K
去模糊	GoPro
去噪	DIV2K + BSD68（合并为 DIV2K_BSD68）
低光增强	LSRW

目录结构：
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
python src/scripts/download_data.py --data_root ./data --check_only

快速开始
1. 增量训练
bash
python src/scripts/train_incremental.py --config configs/default.yaml
指定输出目录：

bash
python src/scripts/train_incremental.py \
    --config configs/default.yaml \
    --output_dir ./outputs/exp1
从 checkpoint 恢复：

bash
python src/scripts/train_incremental.py \
    --config configs/default.yaml \
    --resume ./outputs/task_2_final.pth
2. 评估
评估合并后的模型：

bash
python src/scripts/evaluate.py \
    --config configs/default.yaml \
    --ckpt ./outputs/merged.pth \
    --merged
评估未合并模型（按任务 ID）：

bash
python src/scripts/evaluate.py \
    --config configs/default.yaml \
    --ckpt ./outputs/task_4_final.pth
保存部分复原图像：

bash
python src/scripts/evaluate.py \
    --config configs/default.yaml \
    --ckpt ./outputs/merged.pth \
    --merged \
    --save_images
3. 消融实验
运行全部消融：

bash
python src/scripts/run_ablations.py
只运行某个消融：

bash
python src/scripts/run_ablations.py --only no_spatial
跳过已完成消融：

bash
python src/scripts/run_ablations.py --resume
4. 单元测试
bash
python tests/test_odr_lora.py
python tests/test_dual_gpm.py
python tests/test_orthogonality.py
python tests/test_frequency.py