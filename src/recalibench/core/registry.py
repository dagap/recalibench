"""Plug-in discovery and instantiation through Hydra config groups.

Each group (dataset, task, model, calibrator) is a registry: one YAML file per
plug-in, whose `_target_` names the class (tasks are config only). Pipeline code
asks the Registry for plug-ins by name and never imports a concrete class.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from recalibench.core.config import (
    CONFIG_NAME,
    DEFAULT_CONF_DIR,
    load_config,
    searchpath_overrides,
)
from recalibench.core.protocols import Calibrator, Dataset, Model
from recalibench.core.types import TaskSpec

Group = Literal["dataset", "task", "model", "calibrator"]
GROUPS: tuple[Group, ...] = ("dataset", "task", "model", "calibrator")

T = TypeVar("T")


class PluginError(Exception):
    """A plug-in config or class is missing, broken, or doesn't meet its protocol."""


class Registry:
    def __init__(
        self,
        extra_conf_dirs: Sequence[Path] = (),
        overrides: Sequence[str] = (),
        conf_dir: Path = DEFAULT_CONF_DIR,
    ) -> None:
        self.extra_conf_dirs = tuple(extra_conf_dirs)
        self.overrides = tuple(overrides)
        self.conf_dir = conf_dir

    def names(self, group: Group) -> list[str]:
        """Every option in `group` across the built-in and extra config dirs."""
        search = searchpath_overrides(self.extra_conf_dirs)
        with initialize_config_dir(config_dir=str(self.conf_dir.resolve()), version_base=None):
            loader = GlobalHydra.instance().config_loader()
            options = loader.get_group_options(group, config_name=CONFIG_NAME, overrides=search)
        return sorted(options)

    def config(self, group: Group, name: str) -> DictConfig:
        if name not in self.names(group):
            raise PluginError(f"no {group} named {name!r}; known: {self.names(group)}")
        try:
            cfg = load_config(
                [*self.overrides, f"+{group}={name}"],
                extra_conf_dirs=self.extra_conf_dirs,
                conf_dir=self.conf_dir,
            )
        except Exception as e:  # Hydra raises several unrelated types
            raise PluginError(f"{group} {name!r}: config failed to compose: {e}") from e
        return cast(DictConfig, cfg[group])

    def target(self, group: Group, name: str) -> str | None:
        target = self.config(group, name).get("_target_")
        return None if target is None else str(target)

    def _instantiate(self, group: Group, name: str, protocol: type[T]) -> T:
        cfg = self.config(group, name)
        if "_target_" not in cfg:
            raise PluginError(f"{group} {name!r}: config has no _target_")
        kwargs: dict[str, Any] = {} if "name" in cfg else {"name": name}
        try:
            obj = instantiate(cfg, _convert_="all", **kwargs)
        except Exception as e:
            raise PluginError(f"{group} {name!r}: cannot instantiate {cfg._target_}: {e}") from e
        if not isinstance(obj, protocol):
            raise PluginError(
                f"{group} {name!r}: {type(obj).__qualname__} does not implement "
                f"the {protocol.__name__} protocol"
            )
        return obj

    def dataset(self, name: str) -> Dataset:
        return self._instantiate("dataset", name, Dataset)

    def model(self, name: str) -> Model:
        return self._instantiate("model", name, Model)

    def calibrator(self, name: str) -> Calibrator:
        return self._instantiate("calibrator", name, Calibrator)

    def task(self, name: str) -> TaskSpec:
        raw = OmegaConf.to_container(self.config("task", name), resolve=True)
        try:
            return TaskSpec.from_dict(name, cast(dict[str, Any], raw))
        except (ValueError, KeyError, TypeError) as e:
            raise PluginError(f"task {name!r}: {e}") from e
