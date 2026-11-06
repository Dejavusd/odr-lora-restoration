"""Continue balanced50: denoise (DF2K/AWGN25) then deblur (GOPRO)."""
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from src.data.five_task_protocol import load_extension
from src.data.clean_protocol import sha256_file
from src.trainers.five_task_trainer import five_task_config,FiveTaskTrainer
from src.trainers.clean_sequence_trainer import run_protocol,validate_resume
from src.utils.training_state import seed_all,atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='outputs/memory_balanced50_b64/task_2_final.pth')
    p.add_argument('--manifest',default='outputs/five_task_splits_v1/manifest.json')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--resume')
    p.add_argument('--stop-after',choices=('denoise','deblur'),default='deblur')
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--smoke-test',action='store_true')
    a=p.parse_args();cfg=five_task_config(a.source,a.manifest,a.output_dir,a.epochs,a.batch_size,a.smoke_test)
    if not torch.cuda.is_available():raise RuntimeError('需要CUDA')
    manifest=load_extension(cfg);sha=sha256_file(a.manifest)
    if a.resume:
        saved=torch.load(a.resume,map_location='cpu',weights_only=False)
        if saved.get('five_task_version')!=1:raise ValueError('--resume只接受五任务入口断点')
        validate_resume(cfg,saved,sha)
        if saved['task_idx']>['lowlight','rainh','dehaze','denoise','deblur'].index(a.stop_after):raise ValueError('不能倒退任务')
    out=Path(a.output_dir);run=out/'run_manifest.json'
    if out.exists() and any(out.iterdir()):
        if not a.resume or not run.is_file():raise FileExistsError('使用新目录或指定--resume')
        old=json.loads(run.read_text())
        if old['protocol']!=run_protocol(cfg) or old['split_sha256']!=sha:raise ValueError('不同实验目录')
    seed_all(cfg['experiment']['seed']);torch.set_num_threads(4)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.9)
    atomic_json(run,dict(protocol=run_protocol(cfg),split_sha256=sha,resume=a.resume,
        source_code_sha256={f:sha256_file(f) for f in ('src/data/five_task_protocol.py',
            'src/trainers/five_task_trainer.py','src/models/dual_gpm.py','src/scripts/train_five_tasks.py')},
        note='Append-only five-task continuation; first three partitions/metrics unchanged. '
             'New tasks tiled inference and area-weighted patch LPIPS; AWGN25 fixed evaluation noise; no replay.'))
    t=FiveTaskTrainer(cfg,manifest);t.run(stop_after=a.stop_after,resume_path=a.resume or a.source)
    if sha256_file(a.source)!=cfg['five_task']['source_sha256']:raise RuntimeError('源变化')
    print(f'完成：{out.resolve()}（smoke_test={a.smoke_test}）',flush=True)


if __name__=='__main__':main()
