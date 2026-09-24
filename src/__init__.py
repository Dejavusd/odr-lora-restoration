"""
ODR-LoRA: 空间-频率-尺度感知正交动态低秩适配。

面向图像复原持续学习的工程实现。

主要模块：
    - src.models: ODR-LoRA 卷积层、DualGPM、NAFNet 骨干、复原网络
    - src.data: 数据集加载与变换
    - src.losses: 复原损失、正交损失、频带损失、共享损失
    - src.trainers: 增量训练器与基线训练器
    - src.utils: SVD 工具、指标、日志、checkpoint
    - src.scripts: 训练、评估、消融脚本
"""

__version__ = "0.1.0"
__author__ = "ODR-LoRA Team"

__all__ = [
    "__version__",
    "__author__",
]