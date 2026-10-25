"""DroidRunner's session setup and the settings it depends on. A live Droid session needs the droid CLI and a
Factory key, so these check what is handed to the SDK, not a model's behaviour."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from droid_sdk import Autonomy, Mode

from autotune import config
from autotune.runner import DroidRunner


def test_droid_runner_refuses_to_start_without_key_or_cli(settings) -> None:  # noqa: ANN001
    with pytest.raises(RuntimeError, match="FACTORY_API_KEY"):
        DroidRunner(dataclasses.replace(settings, factory_api_key=None, droid_path="droid"))
    with pytest.raises(RuntimeError, match="droid CLI"):
        DroidRunner(dataclasses.replace(settings, factory_api_key="fk-test", droid_path=None))


def test_session_config_locks_autonomy_and_tools(settings) -> None:  # noqa: ANN001
    runner = DroidRunner(dataclasses.replace(settings, factory_api_key="fk-test", droid_path="droid"))
    cfg = runner._config(server=object())
    assert cfg.autonomy is Autonomy.OFF and cfg.mode is Mode.SPEC
    assert cfg.disable_builtin_skills is True
    assert cfg.system_prompt["type"] == "preset" and "check_layout" in cfg.system_prompt["append"]
    auto = DroidRunner(dataclasses.replace(settings, factory_api_key="fk-test", droid_path="droid",
                                           droid_mode="auto"))._config(server=object())
    assert auto.mode is Mode.AUTO and auto.autonomy is Autonomy.OFF


def test_settings_read_the_env_file_and_never_print_the_key(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("FACTORY_API_KEY=fk-secret-123\nexport AUTOTUNE_SETTLE_S=5\n# comment\n", encoding="utf-8")
    s = config.load(env={"AUTOTUNE_MAX_CHECKS": "4", "PATH": ""}, env_file=env)
    assert s.factory_api_key == "fk-secret-123" and s.settle_s == 5 and s.max_checks == 4
    assert "fk-secret" not in repr(s)


def test_the_environment_wins_over_the_env_file(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("AUTOTUNE_PORT=9000\n", encoding="utf-8")
    assert config.load(env={"AUTOTUNE_PORT": "9100"}, env_file=env).port == 9100


@pytest.mark.parametrize(("key", "value"), [("AUTOTUNE_MAX_CHECKS", "0"), ("AUTOTUNE_RUNNER", "shell"),
                                            ("AUTOTUNE_DROID_MODE", "yolo"), ("AUTOTUNE_SETTLE_S", "-1")])
def test_bad_settings_are_refused(tmp_path: Path, key: str, value: str) -> None:
    with pytest.raises(ValueError):
        config.load(env={key: value}, env_file=tmp_path / "missing.env")


def test_the_gateway_must_be_loopback() -> None:
    from autotune.gateway import GatewayClient

    with pytest.raises(ValueError, match="loopback"):
        GatewayClient("http://192.168.1.221:8000")
    assert GatewayClient("http://127.0.0.1:8000").base_url == "http://127.0.0.1:8000"
