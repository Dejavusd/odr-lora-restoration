"""No-replay, fixed-capacity test of preserving old GPM directions after RainH."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from src.data.clean_protocol import load_manifest,sha256_file
from src.trainers.memory_retention_trainer import retention_config,MemoryRetentionTrainer
from src.trainers.clean_sequence_trainer import run_protocol,validate_resume
from src.utils.training_state import seed_all,atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',default='outputs/memory_raw32_b64')
    p.add_argument('--merge-rule',choices=('preserve','compressed','balanced'),default='preserve')
    p.add_argument('--old-fraction',type=float,help='Balanced rule: fraction of capacity reserved for old span; default .5')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--resume')
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--smoke-test',action='store_true')
    args=p.parse_args()
    if args.old_fraction is not None and args.merge_rule!='balanced':p.error('--old-fraction只适用于balanced')
    cfg=retention_config(args.run_dir,args.output_dir,args.merge_rule,args.epochs,args.batch_size,args.smoke_test,
                         .5 if args.old_fraction is None else args.old_fraction)
    if not torch.cuda.is_available():raise RuntimeError('需要CUDA')
    manifest=load_manifest(cfg);split_sha=sha256_file(cfg['data']['split_manifest'])
    if args.resume:
        saved=torch.load(args.resume,map_location='cpu',weights_only=False)
        if saved.get('memory_retention_version')!=1:raise ValueError('--resume仅接受本入口断点')
        validate_resume(cfg,saved,split_sha)
    output=Path(args.output_dir);run=output/'run_manifest.json'
    if output.exists() and any(output.iterdir()):
        if not args.resume or not run.is_file():raise FileExistsError('请使用新目录或指定--resume')
        previous=json.loads(run.read_text())
        if previous['protocol']!=run_protocol(cfg) or previous['split_sha256']!=split_sha:raise ValueError('不同实验目录')
    seed_all(cfg['experiment']['seed']);torch.set_num_threads(4)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.9)
    atomic_json(run,dict(protocol=run_protocol(cfg),split_sha256=split_sha,resume=args.resume,
        source_code_sha256={f:sha256_file(f) for f in (
            'src/models/odr_lora.py','src/models/dual_gpm.py','src/trainers/memory_retention_trainer.py',
            'src/scripts/train_memory_retention.py')},
        note='Same selected RainH physical weights and pre-merge memory; only GPM merge changes. '
             'RainH TRAIN used at its memory boundary only; Haze4K training uses no old images.'))
    trainer=MemoryRetentionTrainer(cfg,manifest)
    trainer.run(stop_after='dehaze',resume_path=args.resume or cfg['memory_retention']['best_source'])
    for key,path_key in [('best_sha256','best_source'),('final_sha256','final_source')]:
        if sha256_file(cfg['memory_retention'][path_key])!=cfg['memory_retention'][key]:raise RuntimeError('源断点变化')
    print(f'完成：{output.resolve()}（smoke_test={args.smoke_test}）',flush=True)


if __name__=='__main__':main()
