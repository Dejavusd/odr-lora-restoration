import argparse
import os
import sys
import random
import numpy as np
import torch
import yaml

# 保证可以导入 src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.trainers.incremental_trainer import IncrementalTrainer
from src.utils.logger import get_logger
from src.utils.checkpoint import load_checkpoint


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="ODR-LoRA 增量训练脚本")
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                        help='从 checkpoint 恢复训练')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='覆盖配置中的输出目录')
    args = parser.parse_args()

    # 加载配置
    config_path = os.path.abspath(args.config)
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # 覆盖输出目录
    if args.output_dir is not None:
        cfg['experiment']['output_dir'] = args.output_dir
    os.makedirs(cfg['experiment']['output_dir'], exist_ok=True)

    # 设置随机种子
    seed = cfg['experiment'].get('seed', 42)
    set_seed(seed)

    # 日志
    logger = get_logger('main', cfg['experiment']['output_dir'])
    logger.info(f"配置文件: {config_path}")
    logger.info(f"随机种子: {seed}")
    logger.info(f"设备: {cfg['experiment'].get('device', 'cuda')}")

    # 打印关键配置
    odr_cfg = cfg.get('odr_lora', {})
    freq_level = odr_cfg.get('freq_level', 2)
    freq_bands = odr_cfg.get('freq_bands', 7)
    logger.info(f"频带级别 freq_level: {freq_level}")
    logger.info(f"频带数量 freq_bands: {freq_bands}")
    logger.info(f"空间分块 spatial_blocks: {odr_cfg.get('spatial_blocks', [2, 2])}")
    logger.info(f"共享子空间 share_subspace: {odr_cfg.get('share_subspace', True)}")
    logger.info(f"自适应正交 lambda: "
                f"[{odr_cfg.get('adaptive_lambda_min', 0.5)}, "
                f"{odr_cfg.get('adaptive_lambda_max', 1.0)}]")

    # 打印任务顺序
    task_order = [t['name'] for t in cfg['data']['tasks']]
    logger.info(f"任务顺序: {' -> '.join(task_order)}")

    # 构建训练器
    trainer = IncrementalTrainer(cfg)

    # 恢复训练
    if args.resume is not None:
        if os.path.isfile(args.resume):
            logger.info(f"从 checkpoint 恢复: {args.resume}")
            epoch, task_idx = load_checkpoint(
                trainer.model, args.resume,
                optimizer=None,
                map_location=str(trainer.device)
            )
            logger.info(f"恢复完成: epoch={epoch}, task_idx={task_idx}")
        else:
            logger.warning(f"checkpoint 不存在: {args.resume}，从头开始训练。")

    # 开始训练
    trainer.run()

    logger.info(f"训练完成。输出目录: {cfg['experiment']['output_dir']}")
    logger.info(f"合并后的模型: {os.path.join(cfg['experiment']['output_dir'], 'merged.pth')}")


if __name__ == '__main__':
    main()