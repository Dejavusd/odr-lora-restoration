"""Export predetermined examples without tuning or cherry-picking test images."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from src.scripts.run_forgetting_probe import selected_indices, write_json
from src.scripts.validate_effective_basis import (
    ARMS, arm_config, datasets_from_manifest, make_model,
)
from src.utils.metrics import compute_psnr


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    output = Path(args.output_dir)
    manifest = json.loads((output / 'manifest.json').read_text())
    if args.seed not in manifest['seeds']:
        raise ValueError('Choose a seed in the experiment manifest')
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.15)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')
    _, _, datasets = datasets_from_manifest(manifest)
    original, _ = make_model(arm_config(manifest['config'], ARMS[0]),
                             manifest['source_checkpoint'], device)
    models = {'task0': original.eval()}
    selected_steps = {}
    for arm in ARMS:
        run_dir = output / arm / f'seed_{args.seed}'
        result = json.loads((run_dir / 'result.json').read_text())
        selected_steps[arm] = result['selected_step']
        model, _ = make_model(arm_config(manifest['config'], arm),
                              str(run_dir / 'best_validation.pth'), device)
        models[arm] = model.eval()
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 13)
    width, height, caption = 240, 180, 40
    sheet = Image.new('RGB', (width*5, (height+caption)*6), 'white')
    draw = ImageDraw.Draw(sheet)
    examples = []
    row_index = 0
    for task, dataset in datasets.items():
        for index in selected_indices(len(dataset), 3):
            item = dataset[index]
            x, gt = item['input'].unsqueeze(0).to(device), item['gt'].unsqueeze(0).to(device)
            columns = [('Input', x)]
            metrics = {}
            for arm, model in models.items():
                prediction = model(x, -1 if arm == 'task0' else 1)
                metrics[arm] = compute_psnr(prediction, gt)
                columns.append((f'{arm}: {metrics[arm]:.2f} dB', prediction))
            columns.append(('GT', gt))
            for column_index, (label, tensor) in enumerate(columns):
                pixels = (tensor[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()*255).round().astype(np.uint8)
                image = Image.fromarray(pixels)
                image.thumbnail((width, height), Image.Resampling.LANCZOS)
                left, top = column_index*width, row_index*(height+caption)
                draw.text((left+4, top+3), f'{task} #{index}', fill='black', font=font)
                draw.text((left+4, top+20), label, fill='black', font=font)
                sheet.paste(image, (left+(width-image.width)//2, top+caption+(height-image.height)//2))
            examples.append(dict(task=task, index=index, input_path=item['input_path'],
                                 gt_path=item['gt_path'], psnr=metrics))
            row_index += 1
    path = output / f'fixed_examples_seed_{args.seed}.png'
    sheet.save(path)
    write_json(output / f'fixed_examples_seed_{args.seed}.json', dict(
        protocol='Three evenly spaced existing-test indices per task, declared before inspection; '
                 'full-image PSNR, only display resized/clamped, no exposure correction',
        seed=args.seed, selected_steps=selected_steps, examples=examples))
    print(path.resolve(), flush=True)


if __name__ == '__main__':
    main()
