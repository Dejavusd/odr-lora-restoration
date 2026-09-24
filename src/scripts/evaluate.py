import argparse
import os
import sys
import json
import numpy as np
import torch
import yaml

# 保证可以导入 src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.restoration_net import RestorationNet
from src.data.datasets import build_task_loaders
from src.utils.metrics import compute_psnr, compute_ssim, LPIPSMetric
from src.utils.logger import get_logger
from src.utils.checkpoint import load_checkpoint


def save_image(tensor, path):
    """将 [0,1] 范围的 tensor 保存为 PNG。"""
    try:
        import torchvision.transforms.functional as TF
        from PIL import Image
        tensor = tensor.detach().cpu().clamp(0, 1)
        if tensor.dim() == 4:
            tensor = tensor[0]
        img = TF.to_pil_image(tensor)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        img.save(path)
    except Exception as e:
        print(f"保存图像失败: {e}")


@torch.no_grad()
def evaluate_task(model, loader, task_idx, device, lpips_fn, save_dir=None, max_save=5):
    """
    评估单个任务。
    返回 dict: {'psnr': float, 'ssim': float, 'lpips': float, 'num': int}
    """
    model.eval()
    psnrs, ssims, lpipses = [], [], []
    saved = 0

    for batch in loader:
        x = batch['input'].to(device)
        gt = batch['gt'].to(device)
        pred = model(x, task_idx)

        psnrs.append(compute_psnr(pred, gt))
        ssims.append(compute_ssim(pred, gt))
        lpipses.append(lpips_fn(pred, gt))

        if save_dir is not None and saved < max_save:
            name = os.path.splitext(os.path.basename(batch['input_path'][0]))[0]
            save_image(pred, os.path.join(save_dir, f"{name}_pred.png"))
            save_image(gt, os.path.join(save_dir, f"{name}_gt.png"))
            save_image(x, os.path.join(save_dir, f"{name}_input.png"))
            saved += 1

    return {
        'psnr': float(np.mean(psnrs)) if psnrs else 0.0,
        'ssim': float(np.mean(ssims)) if ssims else 0.0,
        'lpips': float(np.mean(lpipses)) if lpipses else 0.0,
        'num': len(psnrs),
    }


def main():
    parser = argparse.ArgumentParser(description="ODR-LoRA 评估脚本")
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='配置文件路径')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='checkpoint 路径，例如 merged.pth 或 task_0_final.pth')
    parser.add_argument('--merged', action='store_true',
                        help='是否评估合并后的模型。若指定，则所有任务统一用 task_idx=None')
    parser.add_argument('--output', type=str, default=None,
                        help='评估结果 JSON 路径，默认保存到 ckpt 同目录下 eval_results.json')
    parser.add_argument('--save_images', action='store_true',
                        help='是否保存部分复原结果')
    parser.add_argument('--max_save', type=int, default=5,
                        help='每个任务最多保存多少张图像')
    args = parser.parse_args()

    # 加载配置
    config_path = os.path.abspath(args.config)
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        cfg['experiment'].get('device', 'cuda')
        if torch.cuda.is_available() else 'cpu'
    )

    # 日志
    ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))
    logger = get_logger('eval', ckpt_dir)
    logger.info(f"配置文件: {config_path}")
    logger.info(f"checkpoint: {args.ckpt}")
    logger.info(f"设备: {device}")
    logger.info(f"评估模式: {'合并模型' if args.merged else '未合并模型（按任务 ID）'}")

    # 打印频带配置
    odr_cfg = cfg.get('odr_lora', {})
    logger.info(f"freq_level: {odr_cfg.get('freq_level', 2)}")
    logger.info(f"freq_bands: {odr_cfg.get('freq_bands', 7)}")
    logger.info(f"spatial_blocks: {odr_cfg.get('spatial_blocks', [2, 2])}")

    # 构建模型
    model = RestorationNet(cfg).to(device)

    # 加载 checkpoint
    if not os.path.isfile(args.ckpt):
        logger.error(f"checkpoint 不存在: {args.ckpt}")
        return
    state = torch.load(args.ckpt, map_location=device)
    if 'model' in state:
        model.load_state_dict(state['model'], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.eval()
    logger.info("模型加载完成。")

    # 评估
    lpips_fn = LPIPSMetric(device)
    task_order = [t['name'] for t in cfg['data']['tasks']]
    results = {}

    for idx, task_name in enumerate(task_order):
        logger.info(f"===== 评估任务 {idx}: {task_name} =====")
        _, val_loader = build_task_loaders(cfg, task_name)

        if args.merged:
            task_idx = None
        else:
            task_idx = idx

        save_dir = None
        if args.save_images:
            save_dir = os.path.join(ckpt_dir, f'eval_images/{task_name}')

        metrics = evaluate_task(
            model, val_loader, task_idx, device, lpips_fn,
            save_dir=save_dir, max_save=args.max_save
        )
        results[task_name] = metrics
        logger.info(
            f"{task_name}: PSNR={metrics['psnr']:.4f}, "
            f"SSIM={metrics['ssim']:.4f}, "
            f"LPIPS={metrics['lpips']:.4f}, "
            f"num={metrics['num']}"
        )

    # 平均值
    avg_psnr = np.mean([r['psnr'] for r in results.values()])
    avg_ssim = np.mean([r['ssim'] for r in results.values()])
    avg_lpips = np.mean([r['lpips'] for r in results.values()])
    results['average'] = {
        'psnr': float(avg_psnr),
        'ssim': float(avg_ssim),
        'lpips': float(avg_lpips),
    }
    logger.info(
        f"平均: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}, LPIPS={avg_lpips:.4f}"
    )

    # 保存结果
    output_path = args.output
    if output_path is None:
        output_path = os.path.join(ckpt_dir, 'eval_results.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"评估结果已保存: {output_path}")


if __name__ == '__main__':
    main()