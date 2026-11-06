"""Branch at selected RainH weights: change memory merging only, no replay."""
import copy
from pathlib import Path

import torch

from src.data.clean_protocol import sha256_file
from src.models.odr_lora import ODRLoRAConv2d
from src.trainers.clean_sequence_trainer import CleanSequenceTrainer
from src.trainers.bounded_lora_trainer import BoundedLoRATrainer
from src.trainers.memory_source_trainer import non_memory_fingerprint,non_memory_state_fingerprint
from src.utils.checkpoint import load_checkpoint
from src.utils.training_state import seed_all,atomic_json


def retention_config(run_dir,output,merge_rule='preserve',epochs=20,batch_size=64,smoke_test=False,old_fraction=.5):
    import math
    if merge_rule not in ('preserve','compressed','balanced') or min(epochs,batch_size)<1:
        raise ValueError('Invalid rule or budget')
    if not math.isfinite(old_fraction) or not 0<=old_fraction<=1:raise ValueError('old_fraction必须在[0,1]')
    root=Path(run_dir)
    best_path=root/'task_1_best.pth';final_path=root/'task_1_final.pth'
    best=torch.load(best_path,map_location='cpu',weights_only=False)
    final=torch.load(final_path,map_location='cpu',weights_only=False)
    if (best.get('memory_source_version')!=1 or final.get('memory_source_version')!=1
            or best.get('task_idx')!=1 or final.get('task_idx')!=1
            or not best.get('in_progress') or final.get('in_progress')
            or best.get('smoke_test') or final.get('smoke_test')
            or not final.get('gpm_updated') or best.get('gpm_updated')
            or best['epoch']!=final.get('selected_epoch')
            or best['split_sha256']!=final['split_sha256']
            or best['run_protocol']!=final['run_protocol']
            or final['run_protocol']['odr_lora']['gpm_statistics']!='raw'
            or final['run_protocol']['odr_lora']['gpm_update_rule']!='bounded_svd'):
        raise ValueError('需要匹配的正式 memory_raw32 RainH best/final 断点')
    cfg=copy.deepcopy(final['run_protocol'])
    cfg['experiment'].update(output_dir=str(output),memory_fraction=.9)
    cfg['train']['epochs_per_task']=epochs
    cfg['eval']['validation_interval']=5
    for task in cfg['data']['tasks']:task['batch_size']=batch_size
    cfg['odr_lora']['gpm_update_rule']='bounded_preserving' if merge_rule=='preserve' else 'bounded_svd'
    cfg['memory_retention']=dict(best_source=str(best_path.resolve()),final_source=str(final_path.resolve()),
        best_sha256=sha256_file(best_path),final_sha256=sha256_file(final_path),merge_rule=merge_rule,
        selected_rainh_epoch=best['epoch'],boundary_batch_size=final['run_protocol']['data']['tasks'][1]['batch_size'],
        boundary_batches=final['run_protocol']['train']['gpm_batches'],
        rule_scope='rebuild RainH boundary memory, then Haze4K only',replay=False,teacher=False)
    if merge_rule=='balanced':
        cfg['odr_lora']['gpm_update_rule']='bounded_balanced'
        cfg['memory_retention'].update(old_fraction=old_fraction,
            aggregate='once per boundary; same covariance across quota arms',
            selector='least current-energy modes inside old span')
    cfg['smoke_test']=smoke_test
    if smoke_test:
        cfg['train'].update(epochs_per_task=1,max_steps_per_epoch=2,calibration_batches=1,gpm_batches=1)
        cfg['eval'].update(validation_interval=1,max_images=1)
        cfg['data']['num_workers']=0
        for task in cfg['data']['tasks']:task['batch_size']=min(batch_size,2)
        cfg['memory_retention'].update(boundary_batch_size=2,boundary_batches=1)
    return cfg


class MemoryRetentionTrainer(CleanSequenceTrainer):
    validation_for_selection=BoundedLoRATrainer.validation_for_selection

    def payload(self,*args,**kwargs):
        result=super().payload(*args,**kwargs);result['memory_retention_version']=1
        return result

    def _finish_task(self,loader,idx):
        if self.cfg['memory_retention']['merge_rule']=='balanced':
            for m in self.model.modules():
                if isinstance(m,ODRLoRAConv2d):
                    m.dual_gpms['global']._balanced_covariance=None
                    m.dual_gpms['global']._balanced_columns=0
        return super()._finish_task(loader,idx)

    @torch.no_grad()
    def finalize_memory(self,idx):
        settings=self.cfg['memory_retention']
        if settings['merge_rule']!='balanced':return
        layers={}
        for n,m in self.model.named_modules():
            if not isinstance(m,ODRLoRAConv2d):continue
            layers[n]=m.dual_gpms['global'].finalize_balanced(settings['old_fraction'])
            if len(layers)%30==0:self.logger.info(f'BALANCED memory {len(layers)}/162')
        atomic_json(Path(self.output_dir)/f'task_{idx}_balanced_memory.json',dict(
            old_fraction=settings['old_fraction'],capacity=32,layers=layers))

    @torch.no_grad()
    def rebuild_rainh_boundary(self,reference):
        settings=self.cfg['memory_retention']
        seed_all(self.cfg['experiment']['seed']+900001)
        old={n:m.dual_gpms['global'].basis.clone() for n,m in self.model.named_modules()
             if isinstance(m,ODRLoRAConv2d)}
        # Reproduce the original boundary data/budget even if Haze4K batch differs.
        task=next(t for t in self.cfg['data']['tasks'] if t['name']=='rainh')
        old_batch=task['batch_size'];old_budget=self.cfg['train']['gpm_batches']
        task['batch_size']=settings['boundary_batch_size']
        self.cfg['train']['gpm_batches']=settings['boundary_batches']
        try:
            loader=self.loader('rainh','train')
            self.model.set_current_task(1)
            self._finish_task(loader,1)
        finally:
            task['batch_size']=old_batch;self.cfg['train']['gpm_batches']=old_budget
        same_weights=non_memory_fingerprint(self.model)==non_memory_state_fingerprint(reference['model'])
        if not same_weights:raise RuntimeError('分支的RainH物理权重与原任务末权重不同')
        layers={}
        for n,m in self.model.named_modules():
            if not isinstance(m,ODRLoRAConv2d):continue
            a=old[n];b=m.dual_gpms['global'].basis;cap=m.dual_gpms['global'].max_basis
            residual=float((a-b@(b.T@a)).norm()/a.norm().clamp_min(1e-20))
            if settings['merge_rule']=='preserve' and residual>1e-5:
                raise RuntimeError('旧子空间未被保留')
            layers[n]=dict(rank_before=a.shape[1],rank_after=b.shape[1],capacity=cap,
                available_before=cap-a.shape[1],added_rank=b.shape[1]-a.shape[1],old_span_residual=residual)
        atomic_json(Path(self.output_dir)/'rainh_boundary_rebuild.json',dict(
            merge_rule=settings['merge_rule'],same_non_memory_tensors=same_weights,
            boundary_batches=settings['boundary_batches'],boundary_batch_size=settings['boundary_batch_size'],
            full_layers_before=sum(r['available_before']==0 for r in layers.values()),layers=layers))
        self.history=[];self.best_psnr=self.best_epoch=self.best_path=None
        self.save(Path(self.output_dir)/'task_1_final.pth',1,0,in_progress=False,gpm_updated=True,
                  selected_epoch=settings['selected_rainh_epoch'],memory_rebuilt=True)
        self.save(Path(self.output_dir)/'latest.pth',1,0,in_progress=False,gpm_updated=True,
                  selected_epoch=settings['selected_rainh_epoch'],memory_rebuilt=True)

    def restore(self,path):
        saved=torch.load(path,map_location='cpu',weights_only=False)
        if saved.get('memory_retention_version')==1:return super().restore(path)
        settings=self.cfg['memory_retention']
        if sha256_file(path)!=settings['best_sha256']:raise ValueError('必须从指定RainH best分支')
        if saved['split_sha256']!=self.split_sha:raise ValueError('数据划分SHA不同')
        reference=torch.load(settings['final_source'],map_location='cpu',weights_only=False)
        if sha256_file(settings['final_source'])!=settings['final_sha256']:raise ValueError('参考源变化')
        load_checkpoint(self.model,str(path),map_location=self.device)
        self.model.load_state_dict(saved['model'],strict=True)
        # Dynamic GPM buffers are resized from the CPU source state above.
        self.model.to(self.device)
        self.restore_metadata(reference)
        self.rebuild_rainh_boundary(reference)
        return self.payload(1,0,in_progress=False,gpm_updated=True)
