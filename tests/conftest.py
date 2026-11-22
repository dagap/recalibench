from pathlib import Path

import pytest

from recalibench.core.registry import Registry

TOY_CONF = Path(__file__).resolve().parent / "plugins" / "conf"


@pytest.fixture
def toy_registry() -> Registry:
    return Registry(extra_conf_dirs=[TOY_CONF])
