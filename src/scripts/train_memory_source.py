"""Matched GPM source experiment: common LSRW weights -> RainH -> Haze4K, no replay."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch

from src.data.clean_protocol import load_manifest,sha256_file
from src.trainers.memory_source_trainer import memory_source_config,MemorySourceTrainer
from src.trainers.clean_sequence_trainer import run_protocol,validate_resume
from src.utils.training_state import seed_all,atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='outputs/clean_lsrw/task_0_final.pth')
    parser.add_argument('--statistics',choices=('raw','frequency'),default='raw')
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--resume')
    parser.add_argument('--stop-after',choices=('rainh','dehaze'),default='dehaze')
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--smoke-test',action='store_true')
    args=parser.parse_args()
    cfg=memory_source_config(args.source,args.output_dir,args.statistics,args.epochs,args.batch_size,args.smoke_test)
    if not torch.cuda.is_available():raise RuntimeError('训练需要 CUDA')
    manifest=load_manifest(cfg)
    split_sha=sha256_file(cfg['data']['split_manifest'])
    if args.resume:
        saved=torch.load(args.resume,map_location='cpu',weights_only=False)
        if saved.get('memory_source_version')!=1:raise ValueError('--resume只接受本入口断点')
        validate_resume(cfg,saved,split_sha)
        if saved['task_idx']>['lowlight','rainh','dehaze'].index(args.stop_after):
            raise ValueError('断点已超过指定停止阶段')
    output=Path(args.output_dir);run_path=output/'run_manifest.json'
    if output.exists() and any(output.iterdir()):
        if not args.resume or not run_path.is_file():raise FileExistsError('使用新目录或指定本实验--resume')
        previous=json.loads(run_path.read_text())
        if previous['protocol']!=run_protocol(cfg) or previous['split_sha256']!=split_sha:
            raise ValueError('目标目录属于不同实验')
    seed_all(cfg['experiment']['seed']);torch.set_num_threads(4)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(cfg['experiment']['memory_fraction'])
    atomic_json(run_path,dict(protocol=run_protocol(cfg),split_sha256=split_sha,resume=args.resume,
        source_code_sha256={p:sha256_file(p) for p in (
            'src/models/odr_lora.py','src/models/restoration_net.py',
            'src/trainers/memory_source_trainer.py','src/trainers/clean_sequence_trainer.py',
            'src/scripts/train_memory_source.py')},
        note='Same task0 weights; rebuild task0 GPM with LSRW TRAIN at boundary. '
             'Then only current-task TRAIN used; no replay/teacher/bound. '
             'Same capped SVD merge and matched column counts; calibration stays frequency.'))
    trainer=MemorySourceTrainer(cfg,manifest)
    trainer.run(stop_after=args.stop_after,resume_path=args.resume or args.source)
    if sha256_file(args.source)!=cfg['memory_source']['source_sha256']:raise RuntimeError('源断点变化')
    print(f'完成：{output.resolve()}（smoke_test={args.smoke_test}）',flush=True)


if __name__=='__main__':main()
