"""Matched replay-free frequency/raw GPM statistics from common clean task0 weights."""
import copy
import hashlib
from pathlib import Path

import torch

from src.data.clean_protocol import sha256_file
from src.models.odr_lora import ODRLoRAConv2d
from src.trainers.bounded_lora_trainer import BoundedLoRATrainer
from src.trainers.clean_sequence_trainer import CleanSequenceTrainer
from src.utils.checkpoint import load_checkpoint
from src.utils.training_state import atomic_json,seed_all


def memory_source_config(source,output,statistics='raw',epochs=20,batch_size=64,smoke_test=False):
    if statistics not in ('raw','frequency') or min(epochs,batch_size)<1:
        raise ValueError('Invalid memory source or budget')
    state=torch.load(source,map_location='cpu',weights_only=False)
    if (state.get('clean_protocol_version')!=1 or state.get('task_idx')!=0
            or state.get('in_progress') or not state.get('gpm_updated')
            or state.get('smoke_test') or state.get('probe_only') or state.get('inference_only')
            or any(k in state.get('run_protocol',{}) for k in ('stability','update_bound','memory_source'))):
        raise ValueError('共同起点必须为正式 clean LSRW task_0_final.pth')
    cfg=copy.deepcopy(state['run_protocol'])
    if not cfg['odr_lora'].get('full_update_protection'):
        raise ValueError('需要完整正交保护')
    cfg['experiment'].update(output_dir=str(output),memory_fraction=.9)
    cfg['train']['epochs_per_task']=epochs
    cfg['eval']['validation_interval']=5
    for task in cfg['data']['tasks']:task['batch_size']=batch_size
    cfg['odr_lora'].update(gpm_statistics=statistics,gpm_update_rule='bounded_svd',
                            gpm_max_basis=32,gpm_sample_columns=512)
    cfg['memory_source']=dict(source=str(Path(source).resolve()),source_sha256=sha256_file(source),
        statistics=statistics,calibration_statistics='unchanged frequency',
        bootstrap_seed=20261104,bootstrap='reset GPM only; rebuild on current LSRW TRAIN before RainH',
        selection='new_task_validation_psnr',replay=False,teacher=False,update_bound=False)
    cfg['smoke_test']=smoke_test
    if smoke_test:
        cfg['train'].update(epochs_per_task=1,max_steps_per_epoch=2,calibration_batches=1,gpm_batches=1)
        cfg['eval'].update(validation_interval=1,max_images=1)
        cfg['data']['num_workers']=0
        for task in cfg['data']['tasks']:task['batch_size']=min(batch_size,2)
    return cfg


def non_memory_fingerprint(model):
    return non_memory_state_fingerprint(model.state_dict())


def non_memory_state_fingerprint(state):
    digest=hashlib.sha256()
    for name,value in state.items():
        if '.dual_gpms.' in name:continue
        digest.update(name.encode())
        digest.update(value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


class MemorySourceTrainer(CleanSequenceTrainer):
    # Same diagnostic metrics, but current-task PSNR alone selects weights.
    validation_for_selection=BoundedLoRATrainer.validation_for_selection

    def payload(self,*args,**kwargs):
        result=super().payload(*args,**kwargs)
        result['memory_source_version']=1
        return result

    def memory_counts(self):
        return {n:list(m.gpm_column_history) for n,m in self.model.named_modules()
                if isinstance(m,ODRLoRAConv2d)}

    def clear_memory_counts(self):
        for m in self.model.modules():
            if isinstance(m,ODRLoRAConv2d):m.gpm_column_history=[]

    @torch.no_grad()
    def bootstrap_memory(self):
        seed_all(self.cfg['memory_source']['bootstrap_seed'])
        self.model.eval()
        self.model.set_current_task(-1)
        before_fingerprint=non_memory_fingerprint(self.model)
        for m in self.model.modules():
            if isinstance(m,ODRLoRAConv2d):
                m.dual_gpms['global'].basis=None
                m.dual_gpms['global'].singular_values=None
        loader=self.loader('lowlight','train')
        sample=next(iter(loader))['input'][:1].to(self.device)
        before=self.model(sample,-1)
        for i,batch in enumerate(loader):
            self.model.update_gpm(batch['input'].to(self.device),0)
            self.logger.info(f'BOOTSTRAP memory {i+1}/{self.cfg["train"]["gpm_batches"]}')
            if i+1>=self.cfg['train']['gpm_batches']:break
        error=float((self.model(sample,-1)-before).abs().max())
        after_fingerprint=non_memory_fingerprint(self.model)
        if error!=0 or before_fingerprint!=after_fingerprint:
            raise RuntimeError('构造记忆改变了共同模型')
        for m in self.model.modules():
            if isinstance(m,ODRLoRAConv2d) and m.dual_gpms['global'].basis is None:
                raise RuntimeError('任务0记忆为空')
        atomic_json(Path(self.output_dir)/'task_0_memory_bootstrap.json',dict(
            statistics=self.cfg['memory_source']['statistics'],rule='bounded_svd',capacity=32,
            before_non_memory_sha256=before_fingerprint,after_non_memory_sha256=after_fingerprint,
            forward_max_error=error,columns=self.memory_counts(),
            note='LSRW TRAIN at task0 memory boundary only; no optimizer or weight update'))
        self.history=[]
        self.best_psnr=self.best_epoch=self.best_path=None
        self.save(Path(self.output_dir)/'task_0_final.pth',0,0,in_progress=False,gpm_updated=True,
                  memory_rebuilt=True)
        self.save(Path(self.output_dir)/'latest.pth',0,0,in_progress=False,gpm_updated=True,
                  memory_rebuilt=True)

    def restore(self,path):
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state.get('memory_source_version')==1:
            return super().restore(path)
        if sha256_file(path)!=self.cfg['memory_source']['source_sha256']:
            raise ValueError('初始源断点不匹配')
        if state['split_sha256']!=self.split_sha:
            raise ValueError('源断点数据划分 SHA256 不匹配')
        _,_,state=load_checkpoint(self.model,str(path),map_location=self.device,return_state=True)
        self.model.load_state_dict(state['model'],strict=True)
        self.restore_metadata(state)
        self.bootstrap_memory()
        return state

    def _finish_task(self,loader,idx):
        self.clear_memory_counts()
        super()._finish_task(loader,idx)
        atomic_json(Path(self.output_dir)/f'task_{idx}_memory_columns.json',dict(
            statistics=self.cfg['memory_source']['statistics'],rule='bounded_svd',
            capacity=32,columns=self.memory_counts()))
