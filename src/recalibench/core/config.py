"""Compose the Hydra config tree.

The CLI owns argv (Typer), so configs are built with Hydra's compose API
instead of `@hydra.main`. Extra config dirs (e.g. a private client package, or
the toy plug-ins in tests/) are appended to Hydra's search path, so their
groups merge with the built-in ones.
"""

from collections.abc import Sequence
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

DEFAULT_CONF_DIR = Path(__file__).resolve().parents[3] / "conf"
CONFIG_NAME = "config"


def searchpath_overrides(extra_conf_dirs: Sequence[Path]) -> list[str]:
    if not extra_conf_dirs:
        return []
    for d in extra_conf_dirs:
        if not d.is_dir():
            raise FileNotFoundError(f"config dir not found: {d}")
    dirs = ",".join(f"file://{d.resolve()}" for d in extra_conf_dirs)
    return [f"hydra.searchpath=[{dirs}]"]


def load_config(
    overrides: Sequence[str] = (),
    extra_conf_dirs: Sequence[Path] = (),
    conf_dir: Path = DEFAULT_CONF_DIR,
) -> DictConfig:
    """Compose the main config from `conf_dir` plus any extra dirs, with overrides."""
    all_overrides = searchpath_overrides(extra_conf_dirs) + list(overrides)
    with initialize_config_dir(config_dir=str(conf_dir.resolve()), version_base=None):
        return compose(config_name=CONFIG_NAME, overrides=all_overrides)
