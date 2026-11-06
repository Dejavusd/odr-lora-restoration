"""Append denoising/GOPRO splits, preserving all existing three-task records."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.data.five_task_protocol import prepare_extension


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',default='outputs/clean_splits_v1/manifest.json')
    p.add_argument('--output',default='outputs/five_task_splits_v1/manifest.json')
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();prepare_extension(a.parent,a.output,a.workers)
