"""Route contract: every route the SDK calls exists in backend/hypercompute/app.py. Reads the file as text only."""

from __future__ import annotations

import re
from pathlib import Path

import hypercompute_sdk
import pytest
from hypercompute_sdk.client import ROUTES

APP = Path(__file__).resolve().parents[3] / "backend" / "hypercompute" / "app.py"


@pytest.fixture(scope="module")
def app_text() -> str:
    return APP.read_text(encoding="utf-8")


@pytest.mark.parametrize(("method", "path"), ROUTES)
def test_route_exists_in_app(app_text: str, method: str, path: str) -> None:
    assert f'@app.{method.lower()}("{path}"' in app_text


def test_owner_header_name_matches_config() -> None:
    config = (APP.parent / "config.py").read_text(encoding="utf-8")
    assert re.search(r'OWNER_HEADER\s*=\s*"X-Hypercompute-Owner"', config)


def test_write_routes_use_owner_write(app_text: str) -> None:
    for method, path in [("post", "/api/plans"), ("post", "/api/deployments"),
                         ("delete", "/api/deployments/{dep_id}"), ("post", "/v1/chat/completions")]:
        line = next(line for line in app_text.splitlines() if f'@app.{method}("{path}"' in line)
        assert "OwnerWrite" in line


def test_chat_body_limit_matches_app(app_text: str) -> None:
    from hypercompute_sdk.client import CHAT_BODY_LIMIT

    assert "BODY_LIMIT_CHAT = 16384 * KiB" in app_text and CHAT_BODY_LIMIT == 16384 * 1024


def test_public_names() -> None:
    expected = {"Pool", "ChatChunk", "HypercomputeError", "GatewayUnreachable", "RequestError", "OwnerAuthError",
                "NotFoundError", "ConflictError", "BodyTooLarge", "EngineError", "ChatInterrupted",
                "UnavailableError", "StreamError", "DeploymentFailed", "ReadyTimeout", "__version__"}
    assert expected <= set(hypercompute_sdk.__all__)
    assert hypercompute_sdk.__version__ == "0.1.0"
    for name in hypercompute_sdk.__all__:
        assert hasattr(hypercompute_sdk, name)


def test_sockets_are_blocked() -> None:
    import socket

    assert socket.create_connection.__name__ == "_blocked"
    assert socket.socket.connect.__name__ == "_blocked"
