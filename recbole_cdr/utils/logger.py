"""RecBole logging adapter with model-and-dataset run names."""

import re

from recbole.utils import init_logger as _recbole_init_logger


def _safe_component(value):
    """Convert a model or dataset name into a safe path component."""
    component = re.sub(r'[^A-Za-z0-9_.-]+', '-', str(value)).strip('-')
    return component or 'unknown'


def get_log_run_name(config):
    """Return ``source-target-model`` for the current experiment."""
    model = _safe_component(config['model'])
    source = _safe_component(config['source_domain']['dataset'])
    target = _safe_component(config['target_domain']['dataset'])
    return '{}-{}-{}'.format(source, target, model)


class _LoggingConfig:
    """Expose logging-only names without mutating the training config."""

    def __init__(self, config):
        self._config = config
        self.run_name = get_log_run_name(config)
        source = _safe_component(config['source_domain']['dataset'])
        target = _safe_component(config['target_domain']['dataset'])
        self.dataset_name = '{}-{}'.format(source, target)

    def __getitem__(self, key):
        if key == 'model':
            return self.run_name
        if key == 'dataset':
            return self.dataset_name
        return self._config[key]

    def __getattr__(self, name):
        return getattr(self._config, name)


def init_logger(config):
    """Initialize RecBole logging under ``log/source-target-model/``."""
    _recbole_init_logger(_LoggingConfig(config))
