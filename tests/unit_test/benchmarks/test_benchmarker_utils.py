# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.benchmarker import utils


def test_managed_omni_server_forwards_generation_server_args(
    monkeypatch, tmp_path: Path
) -> None:
    calls: dict[str, object] = {}

    class _Proc:
        pass

    def fake_start_server_from_cmd(cmd, log_file, port, timeout, env=None, tee=False):
        calls["cmd"] = cmd
        calls["log_file"] = log_file
        calls["port"] = port
        calls["timeout"] = timeout
        calls["env"] = env
        calls["tee"] = tee
        return _Proc()

    monkeypatch.setattr(utils, "start_server_from_cmd", fake_start_server_from_cmd)
    monkeypatch.setattr(
        utils,
        "_ensure_port_available",
        lambda host, port: calls.setdefault("port_check", (host, port)),
    )
    monkeypatch.setattr(
        utils,
        "stop_server",
        lambda proc: calls.setdefault("stopped", proc),
    )
    monkeypatch.setattr(
        utils,
        "wait_for_gpu_memory_release",
        lambda: calls.setdefault("gpu_cleanup", True),
    )

    with utils.managed_omni_server(
        model_path="model",
        port=8123,
        host="127.0.0.1",
        log_file=tmp_path / "server.log",
        max_running_requests=16,
        cuda_graph_max_bs=16,
        mem_fraction_static=0.85,
        timeout=42,
        wait_for_gpu_release=False,
    ):
        pass

    cmd = calls["cmd"]
    assert isinstance(cmd, list)
    assert cmd[cmd.index("--max-running-requests") + 1] == "16"
    assert cmd[cmd.index("--cuda-graph-max-bs") + 1] == "16"
    assert cmd[cmd.index("--mem-fraction-static") + 1] == "0.85"
    assert calls["port_check"] == ("127.0.0.1", 8123)
    assert calls["port"] == 8123
    assert calls["timeout"] == 42
    assert "stopped" in calls
    assert "gpu_cleanup" not in calls


def test_managed_omni_server_rejects_unavailable_port(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fail_start(*args, **kwargs):
        raise AssertionError("server should not start when port check fails")

    monkeypatch.setattr(
        utils,
        "_ensure_port_available",
        lambda host, port: (_ for _ in ()).throw(RuntimeError("busy")),
    )
    monkeypatch.setattr(utils, "start_server_from_cmd", fail_start)

    with pytest.raises(RuntimeError, match="busy"):
        with utils.managed_omni_server(
            model_path="model",
            port=8123,
            host="127.0.0.1",
            log_file=tmp_path / "server.log",
        ):
            pass
