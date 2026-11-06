"""Append denoise/deblur to a completed balanced three-task checkpoint."""
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.data.clean_protocol import sha256_file
from src.data.five_task_protocol import new_tasks,extension_loader,load_extension
from src.trainers.memory_retention_trainer import MemoryRetentionTrainer
from src.trainers.clean_sequence_trainer import CleanSequenceTrainer
from src.utils.checkpoint import load_checkpoint
from src.utils.inference import predict_image
from src.utils.metrics import compute_psnr,compute_ssim


def five_task_config(source,manifest_path,output,epochs=20,batch_size=64,smoke_test=False):
    if min(epochs,batch_size)<1:raise ValueError('预算必须为正')
    state=torch.load(source,map_location='cpu',weights_only=False)
    if (state.get('memory_retention_version')!=1 or state.get('task_idx')!=2
            or state.get('in_progress') or state.get('smoke_test') or not state.get('gpm_updated')
            or state['run_protocol']['memory_retention'].get('merge_rule')!='balanced'
            or state['run_protocol']['memory_retention'].get('old_fraction')!=.5
            or state.get('five_task_version')):
        raise ValueError('需要正式balanced50三任务 task_2_final.pth')
    cfg=copy.deepcopy(state['run_protocol'])
    old_path=cfg['data']['split_manifest'];parent=json.loads(Path(old_path).read_text())
    if sha256_file(old_path)!=state['split_sha256']:raise ValueError('父划分SHA变化')
    cfg['data']['tasks']+=new_tasks(batch_size)
    cfg['data']['split_manifest']=str(manifest_path)
    manifest=load_extension(cfg,parent)
    if manifest['parent_sha256']!=state['split_sha256']:raise ValueError('扩展划分父SHA不匹配')
    cfg['experiment']['output_dir']=str(output);cfg['train']['epochs_per_task']=epochs
    cfg['five_task']=dict(source=str(Path(source).resolve()),source_sha256=sha256_file(source),
        parent_split_sha256=state['split_sha256'],selection='current_task_validation_psnr',
        noise='AWGN25, unclipped; fixed eval seed by clean-file SHA',
        new_task_lpips='nonoverlapping 512 patches, area weighted, short edges replicate-padded to32',
        note='Old task partitions/metrics unchanged; balanced50 historical-pool/current allocation, not per-task quotas')
    cfg['smoke_test']=smoke_test
    if smoke_test:
        cfg['train'].update(epochs_per_task=1,max_steps_per_epoch=1,calibration_batches=1,gpm_batches=1)
        cfg['eval'].update(validation_interval=1,max_images=1)
        cfg['data']['num_workers']=0
        for task in cfg['data']['tasks'][3:]:task['batch_size']=min(batch_size,2)
    return cfg


def patch_lpips(fn,pred,gt,tile):
    if not tile:return float(fn(pred*2-1,gt*2-1).mean())
    weighted=0.;area=0
    for y in range(0,pred.shape[-2],tile):
        for x in range(0,pred.shape[-1],tile):
            a=pred[...,y:y+tile,x:x+tile];b=gt[...,y:y+tile,x:x+tile]
            h,w=a.shape[-2:];pixels=h*w
            pad=(0,max(0,32-w),0,max(0,32-h))
            if any(pad):a=F.pad(a,pad,mode='replicate');b=F.pad(b,pad,mode='replicate')
            weighted+=float(fn(a*2-1,b*2-1).mean())*pixels;area+=pixels
    return weighted/area


class FiveTaskTrainer(MemoryRetentionTrainer):
    def loader(self,task,role):return extension_loader(self.cfg,self.manifest,task,role)

    def payload(self,*args,**kwargs):
        state=super().payload(*args,**kwargs);state['five_task_version']=1;return state

    def restore(self,path):
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state.get('five_task_version')==1:return CleanSequenceTrainer.restore(self,path)
        if sha256_file(path)!=self.cfg['five_task']['source_sha256']:raise ValueError('三任务源SHA不匹配')
        if state['split_sha256']!=self.cfg['five_task']['parent_split_sha256']:raise ValueError('源划分不匹配')
        _,_,loaded=load_checkpoint(self.model,str(path),map_location=self.device,return_state=True)
        self.model.load_state_dict(loaded['model'],strict=True)
        self.restore_metadata(loaded);self.history=[]
        return loaded

    @torch.no_grad()
    def evaluate_task(self,loader,task_idx):
        self.model.eval()
        task=next(t for t in self.cfg['data']['tasks'] if t['name']==loader.task_name)
        rows=[];groups={}
        for i,batch in enumerate(loader):
            limit=self.cfg['eval'].get('max_images')
            if limit is not None and i>=limit:break
            x,gt=batch['input'].to(self.device),batch['gt'].to(self.device)
            pred=predict_image(self.model,x,task_idx,task.get('tile_size',self.cfg['eval'].get('tile_size',0)),task.get('tile_overlap',64))
            if not torch.isfinite(pred).all():raise FloatingPointError('评估输出非有限')
            if bool(batch.get('grayscale',[False])[0]):pred=pred.mean(1,keepdim=True).expand_as(pred)
            row=dict(psnr=compute_psnr(pred,gt),ssim=compute_ssim(pred,gt),
                lpips=patch_lpips(self.loss_fn.perceptual.lpips,pred,gt,task.get('lpips_tile_size',0)),
                raw_abs_max=float(pred.abs().max()),out_of_range_fraction=float(((pred<0)|(pred>1)).float().mean()))
            rows.append(row)
            groups.setdefault(batch.get('dataset_name',[task['name']])[0],[]).append(row)
        if not rows:raise ValueError('空评估集')
        def average(items):return {**{k:float(np.mean([r[k] for r in items])) for k in items[0]},
                                   'num':len(items),'worst_raw_abs_max':max(r['raw_abs_max'] for r in items)}
        result=average(rows)
        if task['name']=='denoise':result['datasets']={n:average(items) for n,items in groups.items()}
        return result

    @torch.no_grad()
    def validation_for_selection(self,loader,idx):
        metrics={n:self.evaluate_task(self.loader(n,'validation'),idx) for n in self.task_order[:idx+1]}
        return dict(psnr=metrics[self.task_order[idx]]['psnr'],selection='new_task_validation_psnr',tasks=metrics)
