import asyncio
from pathlib import Path

import pytest

from recalibench.core.protocols import Calibrator, Dataset, Model
from recalibench.core.registry import PluginError, Registry
from recalibench.core.types import Item
from tests.conftest import TOY_CONF
from tests.plugins.toy_dataset import ToyDataset


def test_builtin_groups_start_empty() -> None:
    reg = Registry()
    for group in ("dataset", "task", "model", "calibrator"):
        assert reg.names(group) == []


def test_extra_config_dir_adds_toys(toy_registry: Registry) -> None:
    assert toy_registry.names("dataset") == ["toy"]
    assert toy_registry.names("task") == ["toy_binary"]
    assert toy_registry.names("model") == ["toy"]
    assert toy_registry.names("calibrator") == ["toy_identity"]


def test_instantiates_with_injected_name_and_config(toy_registry: Registry) -> None:
    m = toy_registry.model("toy")
    assert isinstance(m, Model)
    assert (m.name, m.version) == ("toy", "0.1")
    assert isinstance(toy_registry.dataset("toy"), Dataset)
    assert isinstance(toy_registry.calibrator("toy_identity"), Calibrator)


def test_overrides_reach_plugin_configs() -> None:
    reg = Registry(extra_conf_dirs=[TOY_CONF], overrides=["seed=7"])
    ds = reg.dataset("toy")
    assert isinstance(ds, ToyDataset)
    assert ds.seed == 7  # ${seed} interpolation resolved from the main config


def test_task_parses(toy_registry: Registry) -> None:
    t = toy_registry.task("toy_binary")
    assert (t.name, t.dataset, t.sample_size) == ("toy_binary", "toy", 200)
    assert t.question.kind == "binary"


def test_toy_dataset_roundtrip(toy_registry: Registry, tmp_path: Path) -> None:
    ds = toy_registry.dataset("toy")
    ds.download(tmp_path)
    df = ds.load(tmp_path)
    assert len(df) == 200
    assert {c: str(t) for c, t in df.dtypes.items()} == ds.schema()


def test_toy_model_is_overconfident(toy_registry: Registry) -> None:
    m = toy_registry.model("toy")
    q = toy_registry.task("toy_binary").question
    items = [Item(id="a", state={"signal": 1.0}, label=1)]
    [pred] = asyncio.run(m.predict(items, q))
    assert pred.probs["yes"] > 1 / (1 + 2.718281828**-1.0)  # sharper than the true p
    assert pred.probs["yes"] + pred.probs["no"] == pytest.approx(1.0)


def test_unknown_plugin(toy_registry: Registry) -> None:
    with pytest.raises(PluginError, match="no model named 'nope'"):
        toy_registry.model("nope")


def _write(dirpath: Path, group: str, name: str, body: str) -> None:
    (dirpath / group).mkdir(parents=True, exist_ok=True)
    (dirpath / group / f"{name}.yaml").write_text(body)


def test_non_conforming_target_rejected(tmp_path: Path) -> None:
    _write(tmp_path, "model", "bad", "_target_: tests.unit.test_registry.NotAModel\n")
    with pytest.raises(PluginError, match="does not implement the Model protocol"):
        Registry(extra_conf_dirs=[tmp_path]).model("bad")


def test_missing_target_rejected(tmp_path: Path) -> None:
    _write(tmp_path, "calibrator", "bare", "foo: 1\n")
    with pytest.raises(PluginError, match="no _target_"):
        Registry(extra_conf_dirs=[tmp_path]).calibrator("bare")


def test_import_error_wrapped(tmp_path: Path) -> None:
    _write(tmp_path, "dataset", "ghost", "_target_: no.such.Module\n")
    with pytest.raises(PluginError, match="cannot instantiate"):
        Registry(extra_conf_dirs=[tmp_path]).dataset("ghost")


def test_bad_task_rejected(tmp_path: Path) -> None:
    _write(tmp_path, "task", "broken", "dataset: toy\nquestion: {kind: score, text: x}\n")
    with pytest.raises(PluginError, match="task 'broken'"):
        Registry(extra_conf_dirs=[tmp_path]).task("broken")


class NotAModel:
    def __init__(self, name: str) -> None:
        self.name = name
