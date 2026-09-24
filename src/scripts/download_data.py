import argparse
import os
import shutil
import glob


# ============================================================
# 数据集信息
# ============================================================

DATASETS = {
    'Rain100H': {
        'task': '去雨',
        'desc': 'Rain100H 去雨数据集',
        'links': [
            'https://www.icst.pku.edu.cn/struct/Projects/Rain100H.html',
            'https://github.com/nnUyi/DerainZoo',
        ],
        'structure': {
            'train': ('input', 'gt'),
            'test': ('input', 'gt'),
        },
    },
    'Haze4K': {
        'task': '去雾',
        'desc': 'Haze4K 去雾数据集',
        'links': [
            'https://github.com/zhilin007/FFA-Net',
            'https://www.kaggle.com/datasets/',
        ],
        'structure': {
            'train': ('input', 'gt'),
            'test': ('input', 'gt'),
        },
    },
    'GoPro': {
        'task': '去模糊',
        'desc': 'GoPro 去模糊数据集',
        'links': [
            'https://seungjunnah.github.io/Datasets/gopro.html',
            'https://github.com/SeungjunNah/DeepDeblur-PyTorch',
        ],
        'structure': {
            'train': ('input', 'gt'),
            'test': ('input', 'gt'),
        },
    },
    'DIV2K_BSD68': {
        'task': '去噪',
        'desc': 'DIV2K 训练 + BSD68 测试的去噪数据集',
        'links': [
            'https://data.vision.ee.ethz.ch/cvl/DIV2K/',
            'https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/',
        ],
        'structure': {
            'train': ('input', 'gt'),
            'test': ('input', 'gt'),
        },
    },
    'LSRW': {
        'task': '低光增强',
        'desc': 'LSRW 低光增强数据集',
        'links': [
            'https://github.com/JianghaiSCU/R2RNet',
            'https://github.com/zhangbaijin/LSRW',
        ],
        'structure': {
            'train': ('input', 'gt'),
            'test': ('input', 'gt'),
        },
    },
}


# ============================================================
# 工具函数
# ============================================================

def check_structure(root, dataset_name, info, verbose=True):
    """
    检查数据集目录结构是否正确。
    返回 True/False。
    """
    dataset_dir = os.path.join(root, dataset_name)
    if not os.path.isdir(dataset_dir):
        if verbose:
            print(f"  [缺失] {dataset_name}: 目录不存在 {dataset_dir}")
        return False

    ok = True
    for split, (input_sub, gt_sub) in info['structure'].items():
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.isdir(split_dir):
            if verbose:
                print(f"  [缺失] {dataset_name}/{split}: 目录不存在")
            ok = False
            continue

        input_dir = os.path.join(split_dir, input_sub)
        gt_dir = os.path.join(split_dir, gt_sub)
        if not os.path.isdir(input_dir) or not os.path.isdir(gt_dir):
            if verbose:
                print(f"  [缺失] {dataset_name}/{split}: 需要 {input_sub}/ 和 {gt_sub}/")
            ok = False
            continue

        input_files = glob.glob(os.path.join(input_dir, '*'))
        gt_files = glob.glob(os.path.join(gt_dir, '*'))
        if len(input_files) == 0 or len(gt_files) == 0:
            if verbose:
                print(f"  [空] {dataset_name}/{split}: input/gt 中无文件")
            ok = False
        else:
            if verbose:
                print(f"  [OK] {dataset_name}/{split}: "
                      f"input={len(input_files)}, gt={len(gt_files)}")
    return ok


def auto_organize(root, dataset_name, info, verbose=True):
    """
    尝试自动整理数据集：
      - 如果 split 目录下只有一层，且文件名相同，则创建 input/ 和 gt/ 软链接或复制。
      - 如果 split 目录下已有 input/gt，则跳过。
    仅做简单处理，复杂情况建议手动整理。
    """
    dataset_dir = os.path.join(root, dataset_name)
    if not os.path.isdir(dataset_dir):
        return

    for split, (input_sub, gt_sub) in info['structure'].items():
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.isdir(split_dir):
            continue

        input_dir = os.path.join(split_dir, input_sub)
        gt_dir = os.path.join(split_dir, gt_sub)

        if os.path.isdir(input_dir) and os.path.isdir(gt_dir):
            continue

        # 尝试：如果 split 下同时存在 "input" 和 "gt" 之外的同名文件
        # 这里仅打印提示，不自动复制，避免误操作。
        if verbose:
            print(f"  [提示] {dataset_name}/{split} 需要手动整理为 "
                  f"{input_sub}/ 和 {gt_sub}/")


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="检查 / 准备 ODR-LoRA 所需的五个图像复原数据集"
    )
    parser.add_argument('--data_root', type=str, default='./data',
                        help='数据根目录，默认 ./data')
    parser.add_argument('--check_only', action='store_true',
                        help='仅检查目录结构，不打印下载链接')
    parser.add_argument('--organize', action='store_true',
                        help='尝试自动整理目录结构（谨慎使用）')
    args = parser.parse_args()

    root = os.path.abspath(args.data_root)
    os.makedirs(root, exist_ok=True)

    print("=" * 70)
    print(f"ODR-LoRA 数据集准备")
    print(f"数据根目录: {root}")
    print("=" * 70)

    all_ok = True
    for name, info in DATASETS.items():
        print(f"\n[{info['task']}] {name}: {info['desc']}")
        ok = check_structure(root, name, info)
        all_ok = all_ok and ok

        if not args.check_only:
            print("  下载链接:")
            for link in info['links']:
                print(f"    - {link}")
            print(f"  期望结构:")
            for split, (input_sub, gt_sub) in info['structure'].items():
                print(f"    {name}/{split}/{input_sub}/")
                print(f"    {name}/{split}/{gt_sub}/")

        if args.organize:
            auto_organize(root, name, info)

    print("\n" + "=" * 70)
    if all_ok:
        print("所有数据集目录结构检查通过。")
    else:
        print("部分数据集缺失或结构不正确，请按上述提示准备。")
    print("=" * 70)


if __name__ == '__main__':
    main()