import argparse
import os
import sys
import json
import glob
import numpy as np
import yaml

# 保证可以导入 src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.trainers.incremental_trainer import IncrementalTrainer
from src.utils.logger import get_logger


def load_metrics(metrics_path):
    """
    读取训练器保存的 metrics.json，提取每个任务的最终指标和平均值。
    """
    if not os.path.isfile(metrics_path):
        return None
    with open(metrics_path, 'r') as f:
        records = json.load(f)

    # 每个任务取最后一次记录
    latest = {}
    for r in records:
        task = r['task']
        if task not in latest or r['epoch'] > latest[task]['epoch']:
            latest[task] = r

    if len(latest) == 0:
        return None

    psnrs = [v['psnr'] for v in latest.values()]
    ssims = [v['ssim'] for v in latest.values()]
    lpipses = [v['lpips'] for v in latest.values()]

    return {
        'per_task': {
            k: {
                'psnr': v['psnr'],
                'ssim': v['ssim'],
                'lpips': v['lpips'],
                'epoch': v['epoch'],
            }
            for k, v in latest.items()
        },
        'average': {
            'psnr': float(np.mean(psnrs)),
            'ssim': float(np.mean(ssims)),
            'lpips': float(np.mean(lpipses)),
        }
    }


def main():
    parser = argparse.ArgumentParser(description="ODR-LoRA 消融实验批量运行")
    parser.add_argument('--config_dir', type=str, default='configs/ablations',
                        help='消融配置文件目录')
    parser.add_argument('--output_root', type=str, default='./outputs/ablations',
                        help='消融实验输出根目录')
    parser.add_argument('--only', type=str, default=None,
                        help='只运行指定消融，例如 no_spatial。默认全部运行')
    parser.add_argument('--resume', action='store_true',
                        help='若输出目录已存在 metrics.json，则跳过训练')
    args = parser.parse_args()

    config_dir = os.path.abspath(args.config_dir)
    output_root = os.path.abspath(args.output_root)
    os.makedirs(output_root, exist_ok=True)

    logger = get_logger('ablations', output_root)
    logger.info(f"消融配置目录: {config_dir}")
    logger.info(f"输出根目录: {output_root}")

    # 发现所有 yaml
    yaml_files = sorted(glob.glob(os.path.join(config_dir, '*.yaml')))
    if args.only is not None:
        yaml_files = [f for f in yaml_files if args.only in os.path.basename(f)]

    if len(yaml_files) == 0:
        logger.error("未找到任何消融配置文件。")
        return

    logger.info(f"待运行消融: {[os.path.basename(f) for f in yaml_files]}")

    summary = {}

    for yaml_path in yaml_files:
        name = os.path.splitext(os.path.basename(yaml_path))[0]
        logger.info(f"===== 运行消融: {name} =====")

        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)

        # 覆盖输出目录
        out_dir = os.path.join(output_root, name)
        cfg['experiment']['output_dir'] = out_dir
        os.makedirs(out_dir, exist_ok=True)

        # 打印关键配置
        odr_cfg = cfg.get('odr_lora', {})
        logger.info(f"  freq_level: {odr_cfg.get('freq_level', 2)}")
        logger.info(f"  freq_bands: {odr_cfg.get('freq_bands', 7)}")
        logger.info(f"  spatial_blocks: {odr_cfg.get('spatial_blocks', [2, 2])}")
        logger.info(f"  adaptive_lambda: "
                    f"[{odr_cfg.get('adaptive_lambda_min', 0.5)}, "
                    f"{odr_cfg.get('adaptive_lambda_max', 1.0)}]")
        logger.info(f"  share_subspace: {odr_cfg.get('share_subspace', True)}")
        logger.info(f"  skip_orthogonal: {odr_cfg.get('skip_orthogonal', True)}")

        metrics_path = os.path.join(out_dir, 'metrics.json')

        # 若已存在且指定 resume，则跳过训练
        if args.resume and os.path.isfile(metrics_path):
            logger.info(f"  发现已有 metrics.json，跳过训练。")
        else:
            trainer = IncrementalTrainer(cfg)
            trainer.run()

        # 读取指标
        metrics = load_metrics(metrics_path)
        if metrics is None:
            logger.warning(f"  未找到有效 metrics.json，跳过汇总。")
            continue

        summary[name] = {
            'config': yaml_path,
            'output_dir': out_dir,
            'average': metrics['average'],
            'per_task': metrics['per_task'],
        }
        logger.info(
            f"  {name} 平均: "
            f"PSNR={metrics['average']['psnr']:.4f}, "
            f"SSIM={metrics['average']['ssim']:.4f}, "
            f"LPIPS={metrics['average']['lpips']:.4f}"
        )

    # 保存汇总
    summary_path = os.path.join(output_root, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"消融汇总已保存: {summary_path}")

    # 打印表格
    logger.info("=" * 80)
    logger.info(f"{'消融':<20} {'PSNR':>10} {'SSIM':>10} {'LPIPS':>10}")
    logger.info("-" * 80)
    for name, info in summary.items():
        avg = info['average']
        logger.info(
            f"{name:<20} {avg['psnr']:>10.4f} {avg['ssim']:>10.4f} {avg['lpips']:>10.4f}"
        )
    logger.info("=" * 80)


if __name__ == '__main__':
    main()