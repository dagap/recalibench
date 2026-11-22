"""pyright (strict) checks these assignments: a toy that drifts from its
protocol fails `make lint`, not just a runtime test."""

from recalibench.core.protocols import Calibrator, Dataset, Model
from tests.plugins.toy_calibrator import ToyIdentity
from tests.plugins.toy_dataset import ToyDataset
from tests.plugins.toy_model import ToyModel


def test_toys_satisfy_protocols() -> None:
    ds: Dataset = ToyDataset(name="toy")
    m: Model = ToyModel(name="toy", version="0.1")
    cal: Calibrator = ToyIdentity(name="toy_identity")
    assert isinstance(ds, Dataset)
    assert isinstance(m, Model)
    assert isinstance(cal, Calibrator)
