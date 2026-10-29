"""Audit original pairs and build fixed grouped train holdouts, without moving images."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import yaml
from src.data.clean_protocol import prepare_splits, load_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/clean_lsrw_rainh_haze4k.yaml')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--check', action='store_true', help='Verify an existing manifest, including file SHA256')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.check:
        load_manifest(cfg, verify_hashes=True)
        print('All files and grouped partitions verified.')
    else:
        prepare_splits(cfg, args.workers)


if __name__ == '__main__':
    main()
