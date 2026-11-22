import socket

import pytest


def test_network_is_blocked_in_tests() -> None:
    with pytest.raises(RuntimeError, match="network connection"):
        socket.create_connection(("example.com", 80), timeout=1)
