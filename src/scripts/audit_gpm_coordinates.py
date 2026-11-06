"""Offline, equal-rank/equal-column audit of frequency vs actual convolution patches.

Old TRAIN inputs fit diagnostic bases only; held-out validation inputs score them.
No optimizer, replay, teacher, model mutation or resumable checkpoint is produced.
"""
import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import torch.nn.functional as F

from src.data.clean_protocol import load_manifest, make_loader, sha256_file
from src.data.datasets import RestorationDataset
from src.models.odr_lora import ODRLoRAConv2d
from src.scripts.audit_output_stability import read_model
from src.utils.training_state import atomic_json


def sampled_patches(layer, x, columns):
    """Same C,dy,dx layout/padding/stride as conv2d, without a full unfold copy."""
    padded = F.pad(x, (layer.padding,)*4)
    windows = padded.unfold(2, layer.kernel_size, layer.stride).unfold(
        3, layer.kernel_size, layer.stride)
    height, width = windows.shape[2:4]
    index = torch.linspace(0,height*width-1,min(columns,height*width),device=x.device).long()
    h = windows[0,:,index//width,index%width].permute(1,0,2,3).reshape(len(index),-1).T
    return h


def equal_columns(layer, x, columns):
    matrices = [h for _,_,h in layer._extract_statistics(x) if h.numel()]
    freq = torch.cat(matrices,dim=1)
    raw = sampled_patches(layer,x,min(columns,freq.shape[1]))
    index = torch.linspace(0,freq.shape[1]-1,raw.shape[1],device=x.device).long()
    return raw.float(),freq[:,index].float()


def fit_basis(h, rank):
    if not torch.isfinite(h).all():
        raise FloatingPointError('Nonfinite fit features')
    # Full deterministic SVD; scale avoids large-magnitude activation issues.
    u,_,_ = torch.linalg.svd(h/h.norm().clamp_min(1e-20),full_matrices=False)
    return u[:,:min(rank,u.shape[1])]


def coverage(basis,h):
    return float((basis.T@h).square().sum()/h.square().sum().clamp_min(1e-30))


def choose_records(item,role,count,seed):
    indices = list(item[f'{role}_indices'])
    random.Random(seed).shuffle(indices)
    chosen,groups = [],set()
    for i in indices:
        group = item['train_records'][i]['gt_group']
        if group not in groups:
            chosen.append(i)
            groups.add(group)
        if len(chosen)>=count:
            break
    return chosen


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='outputs/clean_effective_r16/task_1_final.pth')
    parser.add_argument('--after',default='outputs/bounded_lora_control_b64/task_2_final.pth')
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--train-per-task',type=int,default=8)
    parser.add_argument('--validation-per-task',type=int,default=16)
    parser.add_argument('--columns',type=int,default=32)
    args = parser.parse_args()
    if min(args.train_per_task,args.validation_per_task,args.columns)<1:
        raise ValueError('Sampling budgets must be positive')
    output = Path(args.output_dir)
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.cuda.set_per_process_memory_fraction(.8)
    model,state = read_model(args.source,'cuda')
    after,final = read_model(args.after,'cuda')
    cfg = state['run_protocol']
    if state['task_idx']!=1 or final['task_idx']!=2 or state['split_sha256']!=final['split_sha256']:
        raise ValueError('Need matching formal RainH/Haze4K task-final checkpoints')
    if sha256_file(cfg['data']['split_manifest'])!=state['split_sha256']:
        raise ValueError('Split SHA changed')
    manifest = load_manifest(cfg)
    layers = {n:m for n,m in model.named_modules() if isinstance(m,ODRLoRAConv2d)}
    target = dict(after.named_modules())
    delta = {n:(target[n].weight-m.weight).flatten(1).detach() for n,m in layers.items()}
    del after,final,target
    torch.cuda.empty_cache()
    records = {}
    datasets = {}
    for task in cfg['data']['tasks'][:2]:
        name=task['name'];item=manifest['tasks'][name]
        records[name] = {role:choose_records(item,role,count,20261104)
            for role,count in [('train',args.train_per_task),('validation',args.validation_per_task)]}
        # The validation dataset contains TRAIN holdout rows; train dataset below
        # uses the same full-image transform, not augmentation or random crops.
        datasets[name] = RestorationDataset([r['input'] for r in item['train_records']],
            [r['gt'] for r in item['train_records']],augment=False,strict_size=True)
    atomic_json(output/'sampling.json',dict(source_sha256=sha256_file(args.source),
        after_sha256=sha256_file(args.after),records=records,columns_per_layer_image=args.columns,
        fit_geometry='center128 TRAIN inputs',score_geometry=['center128','full'],
        gt_used=False,note='Indices refer to immutable train_records; unique GT groups per subset'))
    fit={n:{'raw':[],'freq':[]} for n in layers}
    def fit_hook(name):
        def hook(layer,inputs):
            raw,freq=equal_columns(layer,inputs[0],args.columns)
            fit[name]['raw'].append(raw.cpu());fit[name]['freq'].append(freq.cpu())
        return hook
    handles=[m.register_forward_pre_hook(fit_hook(n)) for n,m in layers.items()]
    try:
        for name in records:
            for i in records[name]['train']:
                x=datasets[name][i]['input'][None].cuda();h,w=x.shape[-2:]
                model(x[...,(h-128)//2:(h+128)//2,(w-128)//2:(w+128)//2],-1)
            print('FIT collected',name,flush=True)
    finally:
        for handle in handles:handle.remove()
    banks={};geometry={}
    for k,(name,layer) in enumerate(layers.items()):
        raw=torch.cat(fit[name]['raw'],1).cuda();freq=torch.cat(fit[name]['freq'],1).cuda()
        stored=layer.dual_gpms['global'].basis
        if stored is None:raise ValueError('Missing old GPM')
        rank=min(stored.shape[1],raw.shape[1],freq.shape[1])
        br,bf=fit_basis(raw,rank),fit_basis(freq,rank)
        banks[name]={'stored':stored,'raw_fit':br,'freq_fit':bf}
        geometry[name]=dict(rank=rank,stored_rank=stored.shape[1],d=layer.d,kernel=layer.kernel_size,
            fit_columns=raw.shape[1],subspace_overlap=float((br.T@bf).square().sum()/rank),
            stored_on_fit_raw=coverage(stored,raw),stored_on_fit_freq=coverage(stored,freq),
            raw_fit_on_fit_raw=coverage(br,raw),freq_fit_on_fit_raw=coverage(bf,raw))
        del fit[name]
        if (k+1)%30==0:print('SVD',k+1,'/',len(layers),flush=True)
    result=dict(source=args.source,after=args.after,geometry=geometry,scores={},
        note='Equal feature-column budgets and equal fresh PCA rank. Saved memory has historical '
             'compression, fresh banks do not. No fitted basis installed into any training model.')
    for task in records:
        for mode in ('center128','full'):
            totals={n:[] for n in layers}
            def score_hook(name):
                def hook(layer,inputs):
                    raw,freq=equal_columns(layer,inputs[0],args.columns)
                    d=delta[name];base=layer.weight.flatten(1)@raw
                    row={}
                    for bank,m in banks[name].items():
                        projected=d-(d@m)@m.T
                        row[bank+'_raw_coverage']=coverage(m,raw)
                        row[bank+'_freq_coverage']=coverage(m,freq)
                        row[bank+'_projected_response']=float((projected@raw).norm()/base.norm().clamp_min(1e-20))
                        row[bank+'_retained_delta_norm']=float(projected.norm()/d.norm().clamp_min(1e-20))
                    row['actual_response']=float((d@raw).norm()/base.norm().clamp_min(1e-20))
                    row['actual_stored_leakage']=float((d@banks[name]['stored']).norm()/d.norm().clamp_min(1e-20))
                    totals[name].append(row)
                return hook
            handles=[m.register_forward_pre_hook(score_hook(n)) for n,m in layers.items()]
            try:
                for i in records[task]['validation']:
                    x=datasets[task][i]['input'][None].cuda();h,w=x.shape[-2:]
                    if mode=='center128':x=x[...,(h-128)//2:(h+128)//2,(w-128)//2:(w+128)//2]
                    model(x,-1)
            finally:
                for handle in handles:handle.remove()
            means={n:{key:float(np.mean([r[key] for r in rows])) for key in rows[0]}
                   for n,rows in totals.items()}
            groups={}
            for group in ('all','kernel1','kernel3'):
                names=[n for n in layers if group=='all' or geometry[n]['kernel']==int(group[-1])]
                groups[group]={key:float(np.mean([means[n][key] for n in names])) for key in next(iter(means.values()))}
            result['scores'][task+'_'+mode]=dict(images=len(records[task]['validation']),groups=groups,layers=means)
            atomic_json(output/'audit.json',result)
            print('SCORE',task,mode,groups['all'],flush=True)
    sampling=json.loads((output/'sampling.json').read_text())
    result['sources_unchanged']=(sha256_file(args.source)==sampling['source_sha256']
        and sha256_file(args.after)==sampling['after_sha256'])
    atomic_json(output/'audit.json',result)
    if not result['sources_unchanged']:raise RuntimeError('Source SHA changed')


if __name__=='__main__':main()
