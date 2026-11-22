"""Hosted models never see the ground truth: the API layer and its adapters don't read
Item.label. (Trainable baselines read labels in fit, on the fit split; the offline
simulator distorts a label-based score by design.)"""

import ast
from pathlib import Path

MODELS = Path(__file__).resolve().parents[2] / "src" / "recalibench" / "models"
API_LAYER = ["base_api.py", "jev.py", "mock.py", "cache.py", "budget.py", "context.py"]


def _item_label_reads(path: Path) -> list[int]:
    """Lines reading `<x>.label` where x isn't an option (`o.label` is an option's text)."""
    tree = ast.parse(path.read_text())
    return [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute)
        and n.attr == "label"
        and not (isinstance(n.value, ast.Name) and n.value.id in {"o", "opt", "option"})
    ]


def test_api_layer_never_reads_item_labels() -> None:
    offenders = {f: lines for f in API_LAYER if (lines := _item_label_reads(MODELS / f))}
    assert offenders == {}


def test_the_check_sees_label_reads() -> None:
    assert _item_label_reads(MODELS / "mock_generator.py")  # the scan isn't vacuous
