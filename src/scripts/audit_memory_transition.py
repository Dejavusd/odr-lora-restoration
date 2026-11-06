"""Measure GPM-only coverage loss at fixed post-RainH weights."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from src.scripts.audit_output_stability import read_model
from src.scripts.audit_gpm_coordinates import sampled_patches,choose_records
from src.data.clean_protocol import load_manifest,sha256_file
from src.data.datasets import RestorationDataset
from src.models.odr_lora import ODRLoRAConv2d
from src.utils.training_state import atomic_json
from src.trainers.memory_source_trainer import non_memory_state_fingerprint


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',default='outputs/memory_raw32_b64')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--after',help='Optional rebuilt task_1_final with the same selected physical weights')
    p.add_argument('--images',type=int,default=32)
    args=p.parse_args();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    if args.images<1:raise ValueError('images必须为正')
    torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    root=Path(args.run_dir);before=torch.load(root/'task_1_best.pth',map_location='cpu',weights_only=False)
    after_path=Path(args.after) if args.after else root/'task_1_final.pth'
    model,state=read_model(after_path,'cuda')
    if args.after:
        original=torch.load(root/'task_1_final.pth',map_location='cpu',weights_only=False)
        if non_memory_state_fingerprint(original['model'])!=non_memory_state_fingerprint(state['model']):
            raise ValueError('比较的物理权重不同，不能归因于记忆变化')
    cfg=state['run_protocol'];manifest=load_manifest(cfg)
    if before['split_sha256']!=state['split_sha256'] or before['epoch']!=state['selected_epoch']:
        raise ValueError('Selected checkpoints do not match')
    rows=manifest['tasks']['lowlight']['train_records']
    indices=choose_records(manifest['tasks']['lowlight'],'validation',args.images,20261105)
    ds=RestorationDataset([r['input'] for r in rows],[r['gt'] for r in rows],augment=False,strict_size=True)
    layers={n:m for n,m in model.named_modules() if isinstance(m,ODRLoRAConv2d)}
    old={n:before['model'][n+'.dual_gpms.global._basis'].cuda() for n in layers}
    statistics={};values={n:[] for n in layers}
    for n,m in layers.items():
        a=old[n];b=m.dual_gpms['global'].basis
        statistics[n]=dict(rank_before=a.shape[1],rank_after=b.shape[1],cap=m.dual_gpms['global'].max_basis,
            available_slots_before=m.dual_gpms['global'].max_basis-a.shape[1],
            old_span_residual=float((a-b@(b.T@a)).norm()/a.norm().clamp_min(1e-20)))
    def hook(n):
        def collect(m,inputs):
            h=sampled_patches(m,inputs[0],128);den=h.square().sum().clamp_min(1e-30)
            a=old[n];b=m.dual_gpms['global'].basis
            values[n].append((float((a.T@h).square().sum()/den),float((b.T@h).square().sum()/den)))
        return collect
    handles=[m.register_forward_pre_hook(hook(n)) for n,m in layers.items()]
    try:
        for i in indices:model(ds[i]['input'][None].cuda(),-1)
    finally:
        for h in handles:h.remove()
    for n,v in values.items():
        statistics[n].update(coverage_before=float(np.mean([r[0] for r in v])),coverage_after=float(np.mean([r[1] for r in v])))
    summary={k:float(np.mean([r[k] for r in statistics.values()])) for k in
        ('coverage_before','coverage_after','old_span_residual','available_slots_before')}
    summary['full_layers_before']=sum(r['available_slots_before']==0 for r in statistics.values())
    atomic_json(out/'audit.json',dict(summary=summary,layers=statistics,images=len(indices),indices=indices,
        fixed_weights=str(after_path),before_sha256=sha256_file(root/'task_1_best.pth'),
        after_sha256=sha256_file(after_path),note='Same post-RainH activations for both memories; LSRW holdout only, 128 spatial patches/layer/image; no training'))
    print(summary,flush=True)


if __name__=='__main__':main()
