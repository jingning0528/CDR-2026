# @Time   : 2022/3/11
# @Author : Zihan Lin
# @Email  : zhlin@ruc.edu.cn

import argparse
from pathlib import Path

import torch

from recbole_cdr.quick_start import run_recbole_cdr


# ---------------------------------------------------------------------------
# Dataset configuration
#
# DATASET_ROOT must be an ABSOLUTE path. It may point anywhere on the machine;
# datasets do not need to be inside this project. It must contain one
# subdirectory per dataset, for example:
#
#   DATASET_ROOT/
#   |-- Book-Crossing/Book-Crossing.inter
#   `-- Librarything/Librarything.inter
#
# Link files are resolved relative to DATASET_ROOT. Set either link file to
# None when that type of cross-domain mapping is not used.
#
# Current pair: Book-Crossing -> Librarything
#   DATASET_ROOT = Path('/absolute/path/to/datasets')
#   SOURCE_DATASET = 'Book-Crossing'
#   TARGET_DATASET = 'Librarything'
#   SOURCE_ITEM_ID_FIELD = 'ISBN'
#   TARGET_ITEM_ID_FIELD = 'book_name'
#   ITEM_LINK_FILE = 'Book-Crossing_Librarything.link'
#
# To use MovieLens ml-1m -> ml-100k:
#   DATASET_ROOT = Path('/absolute/path/to/movielens-datasets')
#   SOURCE_DATASET = 'ml-1m'
#   TARGET_DATASET = 'ml-100k'
#   SOURCE_ITEM_ID_FIELD = 'item_id'
#   TARGET_ITEM_ID_FIELD = 'item_id'
#   ITEM_LINK_FILE = None
#
# To use Amazon Books -> Amazon Movies:
#   DATASET_ROOT = Path('/absolute/path/to/amazon-datasets')
#   SOURCE_DATASET = 'AmazonBooks'
#   TARGET_DATASET = 'AmazonMov'
#   SOURCE_ITEM_ID_FIELD = 'item_id'
#   TARGET_ITEM_ID_FIELD = 'item_id'
#   ITEM_LINK_FILE = None
#
# For the two examples above, each .inter file must use the columns
# user_id, item_id, rating (timestamp may also be present but is not loaded).
# ---------------------------------------------------------------------------
# DATASET_ROOT = Path('/content/drive/MyDrive/CDR-2026-Data')
DATASET_ROOT = Path('/Users/jing/Documents/PhD-CDR/20260828/RecBole-CDR/recbole_cdr/dataset_example')
SOURCE_DATASET = 'Book-Crossing'
TARGET_DATASET = 'Librarything'
SOURCE_ITEM_ID_FIELD = 'ISBN'
TARGET_ITEM_ID_FIELD = 'book_name'
USER_LINK_FILE = None
ITEM_LINK_FILE = 'Book-Crossing_Librarything.link'

# Compute configuration:
#   'auto' -> use a CUDA GPU when available; otherwise use CPU
#   'cuda' -> require a CUDA GPU (useful for Colab experiments)
#   'cpu'  -> always use CPU (useful for local debugging)
COMPUTE_DEVICE = 'auto'
GPU_ID = '0'


def _resolve_optional_path(path, dataset_root):
    """Resolve a link path against the configured dataset root."""
    if path is None:
        return None
    path = Path(path).expanduser()
    return str(path if path.is_absolute() else dataset_root / path)


def build_dataset_config(dataset_root, source_dataset, target_dataset,
                         source_item_id_field='item_id',
                         target_item_id_field='item_id',
                         user_link_file=None, item_link_file=None):
    """Build one consistent configuration for both dataset domains."""
    dataset_root = Path(dataset_root).expanduser()
    if not dataset_root.is_absolute():
        raise ValueError(
            'dataset_root must be an absolute path, for example '
            "'/data/recommendation-datasets'"
        )
    dataset_root = dataset_root.resolve()
    return {
        'source_domain': {
            'dataset': source_dataset,
            'data_path': str(dataset_root),
            'USER_ID_FIELD': 'user_id',
            'ITEM_ID_FIELD': source_item_id_field,
            'RATING_FIELD': 'rating',
            'load_col': {
                'inter': ['user_id', source_item_id_field, 'rating'],
            },
        },
        'target_domain': {
            'dataset': target_dataset,
            'data_path': str(dataset_root),
            'USER_ID_FIELD': 'user_id',
            'ITEM_ID_FIELD': target_item_id_field,
            'RATING_FIELD': 'rating',
            'load_col': {
                'inter': ['user_id', target_item_id_field, 'rating'],
            },
        },
        'user_link_file_path': _resolve_optional_path(user_link_file, dataset_root),
        'item_link_file_path': _resolve_optional_path(item_link_file, dataset_root),
    }


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
    parser.add_argument('--model', '-m', type=str, default='DTCDR', help='name of models')
    parser.add_argument('--config_files', type=str, default=None, help='config files')
    parser.add_argument('--dataset_root', type=Path, default=DATASET_ROOT,
                        help='directory containing all dataset subdirectories')
    parser.add_argument('--source_dataset', type=str, default=SOURCE_DATASET,
                        help='source dataset directory and file prefix')
    parser.add_argument('--target_dataset', type=str, default=TARGET_DATASET,
                        help='target dataset directory and file prefix')
    parser.add_argument('--source_item_id_field', type=str, default=SOURCE_ITEM_ID_FIELD,
                        help='item ID column in the source interaction file')
    parser.add_argument('--target_item_id_field', type=str, default=TARGET_ITEM_ID_FIELD,
                        help='item ID column in the target interaction file')
    parser.add_argument('--user_link_file', type=str, default=USER_LINK_FILE,
                        help='user link file, absolute or relative to dataset_root')
    parser.add_argument('--item_link_file', type=str, default=ITEM_LINK_FILE,
                        help='item link file, absolute or relative to dataset_root')
    parser.add_argument('--compute_device', choices=('auto', 'cuda', 'cpu'),
                        default=COMPUTE_DEVICE,
                        help='auto-detect CUDA, require CUDA, or force CPU')
    parser.add_argument('--gpu_id', type=str, default=GPU_ID,
                        help='CUDA GPU index used when a GPU is selected')

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    dataset_config = build_dataset_config(
        dataset_root=args.dataset_root,
        source_dataset=args.source_dataset,
        target_dataset=args.target_dataset,
        source_item_id_field=args.source_item_id_field,
        target_item_id_field=args.target_item_id_field,
        user_link_file=args.user_link_file,
        item_link_file=args.item_link_file,
    )
    runtime_config = build_compute_config(args.compute_device, args.gpu_id)
    config = {**dataset_config, **runtime_config}
    run_recbole_cdr(
        model=args.model,
        config_file_list=config_file_list,
        config_dict=config,
    )
