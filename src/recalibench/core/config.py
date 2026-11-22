"""Compose the Hydra config tree.

The CLI owns argv (Typer), so configs are built with Hydra's compose API
instead of `@hydra.main`.
"""

from collections.abc import Sequence
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

DEFAULT_CONF_DIR = Path(__file__).resolve().parents[3] / "conf"


def load_config(
    overrides: Sequence[str] = (),
    conf_dir: Path = DEFAULT_CONF_DIR,
    config_name: str = "config",
) -> DictConfig:
    """Compose `config_name` from `conf_dir` with Hydra-style overrides."""
    with initialize_config_dir(config_dir=str(conf_dir.resolve()), version_base=None):
        return compose(config_name=config_name, overrides=list(overrides))
