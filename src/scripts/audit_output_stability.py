"""Read-only paired old-task validation audit before/after a completed new task."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import torch.nn.functional as F

from src.data.clean_protocol import load_manifest, make_loader, sha256_file
from src.models.odr_lora import ODRLoRAConv2d
from src.models.restoration_net import RestorationNet
from src.utils.checkpoint import load_checkpoint
from src.utils.metrics import compute_psnr, compute_ssim
from src.utils.training_state import atomic_json


def read_model(path, device):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if (state.get('clean_protocol_version') != 1 or state.get('in_progress')
            or state.get('smoke_test') or not state.get('gpm_updated')):
        raise ValueError('审计需要正式 clean task-final 断点')
    cfg = state['run_protocol']
    model = RestorationNet(cfg).to(device)
    load_checkpoint(model, str(path), map_location='cpu')
    model.load_state_dict(state['model'], strict=True)
    model.set_current_task(-1)
    return model.to(device).eval().requires_grad_(False), state


def coverage_hook(name, delta, totals):
    def hook(layer, args):
        x = args[0]
        # Sample actual convolution patches without materializing a full unfold.
        padded = F.pad(x, (layer.padding,)*4)
        windows = padded.unfold(2, layer.kernel_size, layer.stride).unfold(
            3, layer.kernel_size, layer.stride)
        height, width = windows.shape[2:4]
        index = torch.linspace(0, height*width-1, min(128, height*width),
                               device=x.device).long()
        h = windows[0, :, index//width, index%width].permute(1,0,2,3).reshape(len(index), -1).T
        basis = layer.dual_gpms['global'].basis
        energy = h.square().sum()
        captured = (basis.T @ h).square().sum() if basis is not None else energy*0
        base = layer.weight.flatten(1) @ h
        change = delta @ h
        values = [float(energy), float(captured), float(base.square().sum()),
                  float(change.square().sum())]
        totals[name] = [a+b for a,b in zip(totals.get(name, [0.]*4), values)]
    return hook


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', required=True)
    parser.add_argument('--after', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--coverage-images', type=int, default=8)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(.8)
    before, a = read_model(args.before, device)
    after, b = read_model(args.after, device)
    if a['split_sha256'] != b['split_sha256'] or a['task_idx'] >= b['task_idx']:
        raise ValueError('需要相同数据划分上的有序训练阶段')
    cfg = a['run_protocol']
    if sha256_file(cfg['data']['split_manifest']) != a['split_sha256']:
        raise ValueError('划分哈希发生改变')
    manifest = load_manifest(cfg)
    target_layers = dict(after.named_modules())
    deltas = {name: (target_layers[name].weight-layer.weight).flatten(1)
              for name,layer in before.named_modules() if isinstance(layer, ODRLoRAConv2d)}
    result = dict(before=str(Path(args.before).resolve()), after=str(Path(args.after).resolve()),
        before_sha256=sha256_file(args.before), after_sha256=sha256_file(args.after),
        split_sha256=a['split_sha256'], precision='strict FP32', partition='full grouped TRAIN validation',
        trained_batch_size=[t['batch_size'] for t in b['run_protocol']['data']['tasks']],
        note='Historical completed checkpoints; NOT a batch64 retraining. Coverage samples at most '
             '128 convolution patches/layer/image on the first coverage-images validation images. '
             'Coverage and local delta response use BEFORE activations; end-to-end drift uses actual outputs.',
        tasks={})
    for task in cfg['data']['tasks'][:a['task_idx']+1]:
        name = task['name']
        totals, rows = {}, []
        handles = [layer.register_forward_pre_hook(coverage_hook(key, deltas[key], totals))
                   for key,layer in before.named_modules() if key in deltas]
        try:
            for i, batch in enumerate(make_loader(cfg, manifest, name, 'validation')):
                if i == args.coverage_images:
                    for handle in handles:
                        handle.remove()
                    handles = []
                x, gt = batch['input'].to(device), batch['gt'].to(device)
                p, q = before(x, -1), after(x, -1)
                if not torch.isfinite(p).all() or not torch.isfinite(q).all():
                    raise FloatingPointError('Nonfinite prediction')
                drift = q-p
                row = dict(index=i, before_psnr=compute_psnr(p,gt), after_psnr=compute_psnr(q,gt),
                    before_ssim=compute_ssim(p,gt), after_ssim=compute_ssim(q,gt),
                    raw_mae=float(drift.abs().mean()), raw_rmse=float(drift.square().mean().sqrt()),
                    raw_bias=float(drift.mean()), raw_max_abs=float(drift.abs().max()),
                    before_min=float(p.min()), before_max=float(p.max()),
                    after_min=float(q.min()), after_max=float(q.max()),
                    clamped_mae=float((q.clamp(0,1)-p.clamp(0,1)).abs().mean()))
                rows.append(row)
                if (i+1)%50 == 0:
                    print(f'{name}: {i+1}', flush=True)
        finally:
            for handle in handles:
                handle.remove()
        layers = {key: dict(coverage=v[1]/max(v[0],1e-30),
                    local_delta_to_base_rms=(v[3]/max(v[2],1e-30))**.5)
                  for key,v in totals.items()}
        means = {key: float(np.mean([r[key] for r in rows])) for key in rows[0] if key!='index'}
        drops = [r['after_psnr']-r['before_psnr'] for r in rows]
        result['tasks'][name] = dict(num=len(rows), means=means,
            psnr_change=means['after_psnr']-means['before_psnr'],
            images_losing_over_1db=sum(d < -1 for d in drops),
            coverage_images=min(args.coverage_images,len(rows)),
            layer_mean_coverage=float(np.mean([v['coverage'] for v in layers.values()])),
            layers=layers, rows=rows)
        atomic_json(output/'stability.json', result)
        print(name, json.dumps(means), flush=True)
    # Verify read-only checkpoint usage.
    if (sha256_file(args.before)!=result['before_sha256']
            or sha256_file(args.after)!=result['after_sha256']):
        raise RuntimeError('Source checkpoint changed during audit')
    result['sources_unchanged'] = True
    atomic_json(output/'stability.json', result)
    print(f'完成：{output}', flush=True)


if __name__ == '__main__':
    main()
