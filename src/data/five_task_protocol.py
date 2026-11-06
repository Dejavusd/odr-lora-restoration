"""Append-only clean splits for online AWGN and sequence-grouped GOPRO."""
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader

from src.data.clean_protocol import record_pair,group_indices,sha256_file,load_manifest,make_loader
from src.data.datasets import pair_gopro_sequences,RestorationDataset
from src.data.transforms import Compose,RandomCrop,RandomFlip,RandomRotate,ToTensor
from src.utils.training_state import atomic_json


def new_tasks(batch_size=64):
    return [dict(name='denoise',dataset='DIV2K+Flickr2K',type='gaussian_denoising',
                 batch_size=batch_size,patch_size=128,noise_sigma=25,noise_seed=42,
                 tile_size=512,tile_overlap=64,lpips_tile_size=512),
            dict(name='deblur',dataset='GOPRO',type='gopro',input_folder='blur',
                 batch_size=batch_size,patch_size=128,tile_size=512,tile_overlap=64,lpips_tile_size=512)]


def clean_record(path):
    path=Path(path);before=[path.stat().st_size,path.stat().st_mtime_ns]
    data=path.read_bytes()
    with Image.open(path) as im:
        group=hashlib.sha256(str(im.size).encode()+im.convert('RGB').tobytes()).hexdigest();size=list(im.size)
    if before!=[path.stat().st_size,path.stat().st_mtime_ns]:raise ValueError('审计期间文件变化')
    sha=hashlib.sha256(data).hexdigest()
    return dict(input=str(path.resolve()),gt=str(path.resolve()),size=size,gt_group=group,
                input_sha256=sha,gt_sha256=sha,file_stats=[before,before.copy()])


def gopro_group_split(rows,fraction,seed):
    # All clips of a source recording remain together in TRAIN/holdout.
    parent={r['sequence_group']:r['sequence_group'] for r in rows}
    def root(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    seen={}
    for row in rows:
        g=row['sequence_group'];previous=seen.setdefault(row['gt_group'],g)
        parent[root(g)]=root(previous)
    groups={}
    for i,row in enumerate(rows):groups.setdefault(root(row['sequence_group']),[]).append(i)
    if len(groups)<2:raise ValueError('GOPRO 至少需要两个来源组')
    names=sorted(groups);random.Random(seed).shuffle(names)
    sums={0:[]}
    for name in names:
        for n,chosen in list(sums.items()):
            sums.setdefault(n+len(groups[name]),chosen+[name])
    target=len(rows)*fraction
    total=min((n for n in sums if 0<n<len(rows)),key=lambda n:(abs(n-target),n))
    heldout={i for n in sums[total] for i in groups[n]}
    return [i for i in range(len(rows)) if i not in heldout],sorted(heldout)


def prepare_extension(parent_path,destination,workers=4):
    destination=Path(destination)
    if destination.exists():raise FileExistsError('扩展划分已存在，不覆盖')
    old=json.loads(Path(parent_path).read_text());result=copy.deepcopy(old)
    if list(old['tasks'])!=['lowlight','rainh','dehaze']:raise ValueError('父划分必须为原三任务')
    result.update(extension_version=1,parent_sha256=sha256_file(parent_path))
    root=Path(old['root'])
    def audit(paths,fn,label):
        rows=[]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i,row in enumerate(pool.map(fn,paths),1):
                rows.append(row)
                if i%200==0 or i==len(paths):print(f'AUDIT {label}: {i}/{len(paths)}',flush=True)
        return rows
    extensions={'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}
    for task in new_tasks():
        name=task['name'];base=root/task['dataset']
        if name=='denoise':
            paths=sorted(p for p in (base/'train').iterdir() if p.suffix.lower() in extensions)
            train=audit(paths,clean_record,'denoise TRAIN');test=[]
            for label,folder in [('DIV2K','test'),('BSD68','BSD68')]:
                paths=sorted(p for p in (base/folder).iterdir() if p.suffix.lower() in extensions)
                rows=audit(paths,clean_record,'denoise '+label)
                for row in rows:row['dataset_name']=label
                test+=rows
            a,b=group_indices(train,old['validation_fraction'],old['split_seed'])
        else:
            splits={}
            for split in ('train','test'):
                inputs,gts=pair_gopro_sequences(str(base/split),'blur')
                rows=audit(list(zip(inputs,gts)),record_pair,'GOPRO '+split)
                for row in rows:
                    sequence=Path(row['input']).parent.parent.name
                    row['sequence_group']='_'.join(sequence.split('_')[:2])
                    row['dataset_name']='GOPRO'
                splits[split]=rows
            train,test=splits['train'],splits['test']
            a,b=gopro_group_split(train,old['validation_fraction'],old['split_seed'])
        if not train or not test:raise ValueError('空数据集')
        if {r['gt_group'] for r in train}&{r['gt_group'] for r in test}:
            raise ValueError(f'{name} TRAIN/TEST存在相同GT，需先处理')
        result['tasks'][name]=dict(dataset=task['dataset'],train_records=train,test_records=test,
            train_indices=a,validation_indices=b,total_train_pairs=len(train),train_pairs=len(a),
            validation_pairs=len(b),test_pairs=len(test),unique_train_gt=len({r['gt_group'] for r in train}),
            unique_test_gt=len({r['gt_group'] for r in test}))
        print('SPLIT',name,len(a),len(b),len(test),flush=True)
    atomic_json(destination,result)
    return result


class ManifestNoiseDataset(Dataset):
    def __init__(self,records,task,train):
        self.records=records;self.task=task;self.train=train
        self.transform=Compose(([RandomCrop(task['patch_size']),RandomFlip(),RandomRotate()] if train else [])+[ToTensor()])
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        row=self.records[index]
        with Image.open(row['gt']) as im:
            gray=im.mode in ('1','L','I','F','I;16');clean,_=self.transform(im.convert('RGB'),None)
        generator=None
        if not self.train:
            seed=int(hashlib.sha256(f'{self.task["noise_seed"]}:{row["gt_sha256"]}'.encode()).hexdigest()[:16],16)%(2**63)
            generator=torch.Generator().manual_seed(seed)
        shape=(1,*clean.shape[1:]) if gray else clean.shape
        noise=torch.randn(shape,generator=generator)*(self.task['noise_sigma']/255.)
        return dict(input=clean+noise,gt=clean,grayscale=gray,
                    dataset_name=row.get('dataset_name','DF2K_holdout'))


def extension_loader(cfg,manifest,name,role):
    if name not in ('denoise','deblur'):
        loader=make_loader(cfg,manifest,name,role)
    else:
        task=next(t for t in cfg['data']['tasks'] if t['name']==name);item=manifest['tasks'][name]
        rows=(item['test_records'] if role=='test' else
              [item['train_records'][i] for i in item['train_indices' if role=='train' else 'validation_indices']])
        ds=(ManifestNoiseDataset(rows,task,role=='train') if name=='denoise' else
            RestorationDataset([r['input'] for r in rows],[r['gt'] for r in rows],patch_size=task['patch_size'],augment=role=='train',strict_size=True))
        loader=DataLoader(ds,batch_size=task['batch_size'] if role=='train' else 1,
            shuffle=role=='train',drop_last=False,num_workers=cfg['data']['num_workers'],
            pin_memory=cfg['experiment']['device'].startswith('cuda'))
    loader.task_name=name
    return loader


def load_extension(cfg,parent_manifest=None):
    manifest=load_manifest(cfg)
    if manifest.get('extension_version')!=1:raise ValueError('需要五任务扩展划分')
    if parent_manifest is not None:
        for name in ('lowlight','rainh','dehaze'):
            if manifest['tasks'][name]!=parent_manifest['tasks'][name]:raise ValueError('原任务划分被修改')
    item=manifest['tasks']['deblur']
    a={item['train_records'][i]['sequence_group'] for i in item['train_indices']}
    b={item['train_records'][i]['sequence_group'] for i in item['validation_indices']}
    if a&b:raise ValueError('GOPRO来源组跨训练/验证泄漏')
    return manifest
