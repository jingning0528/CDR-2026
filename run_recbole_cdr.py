# @Time   : 2022/3/11
# @Author : Zihan Lin
# @Email  : zhlin@ruc.edu.cn

import argparse
import copy
from datetime import datetime
from pathlib import Path

import torch

from recbole_cdr.quick_start import run_recbole_cdr


# Change this default, or pass --dataset_preset on the command line:
#   movielens     -> ml-1m -> ml-100k
#   amazon        -> AmazonBooks -> AmazonMov
#   book_crossing -> Book-Crossing -> Librarything
#   douban        -> DoubanBook -> DoubanMovie
DATASET_PRESET = 'amazon'

# Dataset roots are absolute and may be outside this repository. Colab's
# Google Drive path is selected when it exists; otherwise the local path is
# used. --dataset_root explicitly overrides either path.
COLAB_DATASET_ROOT = Path(
    '/content/drive/MyDrive/CDR-2026-Data/dataset_example'
)
LOCAL_DATASET_ROOT = Path(
    '/Users/jing/Documents/PhD-CDR/20260828/RecBole-CDR/'
    'recbole_cdr/dataset_example'
)
DATASET_ROOT = (
    COLAB_DATASET_ROOT if COLAB_DATASET_ROOT.is_dir() else LOCAL_DATASET_ROOT
)

DATASET_PRESET_FILES = {
    'movielens': 'movielens.yaml',
    'amazon': 'amazon.yaml',
    'book_crossing': 'book_crossing.yaml',
    'douban': 'douban.yaml',
}
DATASET_CONFIG_DIR = (
    Path(__file__).resolve().parent / 'recbole_cdr' / 'properties' / 'dataset'
)

COMPUTE_DEVICE = 'auto'
GPU_ID = '0'
DP_SEEDS = (2022)
DP_EPSILONS = (0.1, 1.0, 10.0, 100.0)


def resolve_dataset_preset(dataset_preset, dataset_root):
    """Return the selected preset and validated absolute dataset root."""
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
            'CUDA was requested, but PyTorch cannot detect a CUDA GPU. '
            'Use --compute_device auto or --compute_device cpu.'
        )

    use_gpu = cuda_available if compute_device == 'auto' else compute_device == 'cuda'
    return {'use_gpu': use_gpu, 'gpu_id': str(gpu_id)}


def format_number_for_filename(value):
    """Format a numeric DP setting without unnecessary trailing zeros."""
    return '{:g}'.format(value)


def as_grid_values(value):
    """Allow a grid default to be either one value or a sequence of values."""
    if isinstance(value, (list, tuple)):
        return value
    return (value,)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='AttentionDTCDR',
                        help='name of model')
    parser.add_argument('--config_files', type=str, default=None,
                        help='additional config files')
    parser.add_argument('--dataset_preset', choices=tuple(DATASET_PRESET_FILES),
                        default=DATASET_PRESET,
                        help='complete source/target dataset configuration')
    parser.add_argument('--dataset_root', type=Path, default=DATASET_ROOT,
                        help='directory containing dataset subdirectories')
    parser.add_argument('--compute_device', choices=('auto', 'cuda', 'cpu'),
                        default=COMPUTE_DEVICE,
                        help='auto-detect CUDA, require CUDA, or force CPU')
    parser.add_argument('--gpu_id', type=str, default=GPU_ID,
                        help='CUDA GPU index used when a GPU is selected')
    parser.add_argument('--dp_grid', action='store_true',
                        help='run the configured DP seed/epsilon grid')
    parser.add_argument('--dp_seeds', type=int, nargs='+',
                        default=as_grid_values(DP_SEEDS),
                        help='DP projection seeds used with --dp_grid')
    parser.add_argument('--dp_epsilons', type=float, nargs='+',
                        default=as_grid_values(DP_EPSILONS),
                        help='epsilon values used with --dp_grid')
    parser.add_argument('--dp_log_dir', type=Path, default=Path('log/dp_grid'),
                        help='directory for explicitly named DP-grid logs')

    args, _ = parser.parse_known_args()
    preset_file, dataset_root = resolve_dataset_preset(
        args.dataset_preset, args.dataset_root
    )
    config_file_list = [str(preset_file)]
    if args.config_files:
        config_file_list.extend(args.config_files.strip().split(' '))

    config = {
        'data_path': str(dataset_root),
        'source_domain': {'data_path': str(dataset_root)},
        'target_domain': {'data_path': str(dataset_root)},
        **build_compute_config(args.compute_device, args.gpu_id),
    }
    if not args.dp_grid:
        run_recbole_cdr(
            model=args.model,
            config_file_list=config_file_list,
            config_dict=config,
        )
    else:
        for dp_seed in args.dp_seeds:
            for dp_epsilon in args.dp_epsilons:
                epsilon_name = format_number_for_filename(dp_epsilon)
                run_time = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
                log_name = '{}-{}-{}-{}-{}.log'.format(
                    args.dataset_preset,
                    args.model,
                    dp_seed,
                    epsilon_name,
                    run_time,
                )
                run_config = copy.deepcopy(config)
                run_config.update({
                    'source_dp_enabled': True,
                    'dp_seed': dp_seed,
                    'dp_epsilon': dp_epsilon,
                })
                run_recbole_cdr(
                    model=args.model,
                    config_file_list=config_file_list,
                    config_dict=run_config,
                    log_file_path=args.dp_log_dir / log_name,
                )
