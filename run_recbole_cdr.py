# @Time   : 2022/3/11
# @Author : Zihan Lin
# @Email  : zhlin@ruc.edu.cn

import argparse
from pathlib import Path

import torch

from recbole_cdr.quick_start import run_recbole_cdr


# Dataset selection. Change only DATASET_PRESET to switch a complete set of
# dataset properties (names, fields, filters, separators, and link files):
#   'movielens'     -> ml-1m -> ml-100k
#   'amazon'        -> AmazonBooks -> AmazonMov
#   'book_crossing' -> Book-Crossing -> Librarything
#   'douban'        -> DoubanBook -> DoubanMovie
# You can also override it without editing this file, for example:
#   python run_recbole_cdr.py --dataset_preset amazon
DATASET_PRESET = 'book_crossing'

# This must be an absolute path, but the datasets may live outside the project.
# The selected preset expects its two dataset directories under this root.
# DATASET_ROOT = Path('/content/drive/MyDrive/CDR-2026/recbole_cdr/dataset_example')
# DATASET_ROOT = Path('/content/drive/MyDrive/CDR-2026-Data/dataset_example')
DATASET_ROOT = Path('/Users/jing/Documents/PhD-CDR/20260828/RecBole-CDR/recbole_cdr/dataset_example')

DATASET_PRESET_FILES = {
    'movielens': 'movielens.yaml',
    'amazon': 'amazon.yaml',
    'book_crossing': 'book_crossing.yaml',
    'douban': 'douban.yaml',
}
DATASET_CONFIG_DIR = (
    Path(__file__).resolve().parent / 'recbole_cdr' / 'properties' / 'dataset'
)

# Compute configuration:
#   'auto' -> use a CUDA GPU when available; otherwise use CPU
#   'cuda' -> require a CUDA GPU (useful for Colab experiments)
#   'cpu'  -> always use CPU (useful for local debugging)
COMPUTE_DEVICE = 'auto'
GPU_ID = '0'


def resolve_dataset_preset(dataset_preset, dataset_root):
    """Return the selected YAML file and its validated absolute data root."""
    if dataset_preset not in DATASET_PRESET_FILES:
        raise ValueError(
            'unknown dataset preset {!r}; choose from {}'.format(
                dataset_preset, ', '.join(sorted(DATASET_PRESET_FILES))
            )
        )

    dataset_root = Path(dataset_root).expanduser()
    if not dataset_root.is_absolute():
        raise ValueError(
            'dataset_root must be an absolute path, for example '
            "'/data/recommendation-datasets'"
        )
    dataset_root = dataset_root.resolve()
    preset_file = DATASET_CONFIG_DIR / DATASET_PRESET_FILES[dataset_preset]
    if not preset_file.is_file():
        raise FileNotFoundError('dataset preset not found: {}'.format(preset_file))
    return preset_file, dataset_root


def build_compute_config(compute_device='auto', gpu_id='0'):
    """Select CUDA automatically while keeping an explicit CPU fallback."""
    compute_device = compute_device.lower()
    if compute_device not in {'auto', 'cuda', 'cpu'}:
        raise ValueError("compute_device must be one of: 'auto', 'cuda', 'cpu'")

    cuda_available = torch.cuda.is_available()
    if compute_device == 'cuda' and not cuda_available:
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot detect a CUDA GPU. "
            "Use --compute_device auto or --compute_device cpu."
        )

    use_gpu = cuda_available if compute_device == 'auto' else compute_device == 'cuda'
    return {
        'use_gpu': use_gpu,
        'gpu_id': str(gpu_id),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='AttentionDTCDR', help='name of models')
    parser.add_argument('--config_files', type=str, default=None, help='config files')
    parser.add_argument('--dataset_preset', choices=tuple(DATASET_PRESET_FILES),
                        default=DATASET_PRESET,
                        help='complete source/target dataset configuration')
    parser.add_argument('--dataset_root', type=Path, default=DATASET_ROOT,
                        help='directory containing all dataset subdirectories')
    parser.add_argument('--compute_device', choices=('auto', 'cuda', 'cpu'),
                        default=COMPUTE_DEVICE,
                        help='auto-detect CUDA, require CUDA, or force CPU')
    parser.add_argument('--gpu_id', type=str, default=GPU_ID,
                        help='CUDA GPU index used when a GPU is selected')

    args, _ = parser.parse_known_args()

    preset_file, dataset_root = resolve_dataset_preset(
        args.dataset_preset, args.dataset_root
    )
    config_file_list = [str(preset_file)]
    if args.config_files:
        config_file_list.extend(args.config_files.strip().split(' '))

    runtime_config = build_compute_config(args.compute_device, args.gpu_id)
    config = {
        'data_path': str(dataset_root),
        'source_domain': {'data_path': str(dataset_root)},
        'target_domain': {'data_path': str(dataset_root)},
        **runtime_config,
    }
    run_recbole_cdr(
        model=args.model,
        config_file_list=config_file_list,
        config_dict=config,
    )
