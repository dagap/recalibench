import socket
from pathlib import Path
from typing import Any

import pytest

from recalibench.core.registry import Registry

TOY_CONF = Path(__file__).resolve().parent / "plugins" / "conf"


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never touch the network (CLAUDE.md). respx mocks sit above sockets, so they work."""

    def refuse(self: socket.socket, address: Any) -> None:
        if self.family == socket.AF_UNIX:
            return original(self, address)
        raise RuntimeError(f"test tried to open a network connection to {address!r}")

    original = socket.socket.connect
    monkeypatch.setattr(socket.socket, "connect", refuse)


@pytest.fixture
def toy_registry() -> Registry:
    return Registry(extra_conf_dirs=[TOY_CONF])
