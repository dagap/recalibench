"""Download stage: fetch raw data, then load it and check it against its schema."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from omegaconf import DictConfig

from recalibench.core.paths import Paths
from recalibench.core.registry import Registry
from recalibench.core.schema import check_schema

log = logging.getLogger(__name__)


class SchemaError(Exception):
    """Downloaded data does not match its dataset's declared schema."""


@dataclass(frozen=True)
class DownloadResult:
    dataset: str
    rows: int
    columns: int


def datasets_for(reg: Registry, cfg: DictConfig, explicit: Sequence[str] = ()) -> list[str]:
    """`explicit` if given; otherwise the datasets the experiment's tasks use, in order."""
    if explicit:
        return list(dict.fromkeys(explicit))
    names: list[str] = []
    for task in cfg.experiment.tasks:
        ds = reg.task(str(task)).dataset
        if ds not in names:
            names.append(ds)
    return names


def download(reg: Registry, paths: Paths, names: Sequence[str]) -> list[DownloadResult]:
    results: list[DownloadResult] = []
    for name in names:
        ds = reg.dataset(name)
        raw_dir = paths.raw_dir(name)
        log.info("%s: raw data in %s (fetched only if missing or stale)", name, raw_dir)
        ds.download(raw_dir)
        df = ds.load(raw_dir)
        problems = check_schema(df, ds.schema())
        if problems:
            raise SchemaError(f"{name}: " + "; ".join(problems))
        log.info("%s: %d rows x %d columns, schema OK", name, len(df), df.shape[1])
        results.append(DownloadResult(name, len(df), df.shape[1]))
    return results
