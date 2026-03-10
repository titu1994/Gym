# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import subprocess
import time
from pathlib import Path
from threading import Lock
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.ns_opencode_agent.app import (
    NSOpenCodeAgent,
    NSOpenCodeAgentConfig,
    NSOpenCodeAgentRunRequest,
)


def _completed_process(command: list[str], returncode: int = 0, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(args=command, returncode=returncode, stdout=stdout, stderr=stderr)


@pytest.fixture(autouse=True)
def _disable_atexit(monkeypatch):
    monkeypatch.setattr("responses_api_agents.ns_opencode_agent.app.atexit.register", lambda *args, **kwargs: None)


def _create_config(data_root: Path, concurrency: int = 1) -> NSOpenCodeAgentConfig:
    return NSOpenCodeAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="ns_opencode_agent",
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        singularity_image_path="/tmp/ns-opencode.sif",
        singularity_instance_name="ns-opencode-worker",
        data_root=str(data_root),
        task_timeout=30,
        trace_monitor_interval=3.0,
        concurrency=concurrency,
    )


def _create_request(instance_id: str = "instance_001", sample: dict | None = None) -> NSOpenCodeAgentRunRequest:
    if sample is None:
        sample = {"uuid": "sample_001", "problem": "Solve this coding task"}

    return NSOpenCodeAgentRunRequest(
        instance_id=instance_id,
        sample=sample,
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
    )


def _create_server(tmp_path: Path, concurrency: int = 1) -> NSOpenCodeAgent:
    with patch.object(NSOpenCodeAgent, "_ensure_instance_started_sync", return_value=None):
        server = NSOpenCodeAgent(
            config=_create_config(tmp_path, concurrency=concurrency),
            server_client=MagicMock(spec=ServerClient),
        )
    server.server_client.global_config_dict = {
        "policy_model_name": "shared_vllm_model",
        "policy_api_key": "dummy_key",
    }
    return server


def _write_trace_from_command(command: list[str]) -> None:
    input_file = Path(command[5])
    workspace_dir = Path(command[6])
    sample = json.loads(input_file.read_text(encoding="utf-8").splitlines()[0])
    trace_dir = workspace_dir / "dataset_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / f"{sample['uuid']}-trace0.json").write_text("{}", encoding="utf-8")


def test_startup_starts_singularity_instance_when_missing(tmp_path: Path) -> None:
    config = _create_config(tmp_path)

    with (
        patch.object(NSOpenCodeAgent, "_is_instance_available_sync", side_effect=[False, True]),
        patch.object(NSOpenCodeAgent, "_run_subprocess") as run_subprocess,
    ):
        run_subprocess.return_value = _completed_process(["singularity"], returncode=0)
        server = NSOpenCodeAgent(config=config, server_client=MagicMock(spec=ServerClient))

    assert server._instance_started_by_agent is True
    start_command = run_subprocess.call_args.args[0]
    assert start_command[:3] == ["singularity", "instance", "start"]
    assert "--bind" in start_command
    assert f"{config.data_root}:{config.data_root}" in start_command


def test_shutdown_stops_only_once_when_started_by_agent(tmp_path: Path) -> None:
    server = _create_server(tmp_path)
    server._instance_started_by_agent = True

    with patch.object(server, "_run_subprocess") as run_subprocess:
        run_subprocess.return_value = _completed_process(["singularity", "instance", "stop"], returncode=0)
        server._stop_instance_sync()
        server._stop_instance_sync()

    assert run_subprocess.call_count == 1
    stop_command = run_subprocess.call_args.args[0]
    assert stop_command == ["singularity", "instance", "stop", server.config.singularity_instance_name]


def test_build_ns_opencode_command_forces_vllm(tmp_path: Path) -> None:
    server = _create_server(tmp_path)

    command = server._build_ns_opencode_command(
        input_file=Path("/data/jobs/job/input.jsonl"),
        workspace_dir=Path("/data/jobs/job/workspace"),
        model_name="shared_vllm_model",
        base_url="http://127.0.0.1:8000/v1",
        api_key="dummy_key",
    )

    assert command[0:5] == [
        "singularity",
        "exec",
        f"instance://{server.config.singularity_instance_name}",
        "ns_opencode",
        "/data/jobs/job/input.jsonl",
    ]
    assert "--provider" in command
    assert command[command.index("--provider") + 1] == "vllm"
    assert command[command.index("--model") + 1] == "shared_vllm_model"
    assert command[command.index("--openai-base-url") + 1] == "http://127.0.0.1:8000/v1"


@pytest.mark.asyncio
async def test_run_success_creates_dataset_and_returns_reward_one(tmp_path: Path) -> None:
    server = _create_server(tmp_path)
    request = _create_request()

    def fake_run_subprocess(command: list[str], timeout: int | None = None):
        _write_trace_from_command(command)
        return _completed_process(command, returncode=0, stdout="ok")

    with (
        patch.object(server, "_is_instance_available_sync", return_value=True),
        patch.object(server, "_run_subprocess", side_effect=fake_run_subprocess),
        patch(
            "responses_api_agents.ns_opencode_agent.app.get_first_server_config_dict",
            return_value={"host": "127.0.0.1", "port": 8000},
        ),
    ):
        response = await server.run(request)

    assert response.reward == 1.0
    assert response.metadata["trace_exists"] is True
    assert response.metadata["provider"] == "vllm"

    dataset_file = Path(response.metadata["dataset_file"])
    assert dataset_file.exists()
    row = json.loads(dataset_file.read_text(encoding="utf-8").splitlines()[0])
    assert row["uuid"] == "sample_001"
    assert row["problem"] == "Solve this coding task"

    trace_files = response.metadata["trace_files"]
    assert len(trace_files) == 1
    assert Path(trace_files[0]).exists()


@pytest.mark.asyncio
async def test_run_fails_if_instance_is_down(tmp_path: Path) -> None:
    server = _create_server(tmp_path)
    request = _create_request()

    with (
        patch.object(server, "_is_instance_available_sync", return_value=False),
        patch(
            "responses_api_agents.ns_opencode_agent.app.get_first_server_config_dict",
            return_value={"host": "127.0.0.1", "port": 8000},
        ),
    ):
        response = await server.run(request)

    assert response.reward == 0.0
    assert response.metadata["status"] == "failed"
    assert "unavailable" in response.metadata["error"].lower()


@pytest.mark.asyncio
async def test_run_timeout_returns_reward_zero(tmp_path: Path) -> None:
    server = _create_server(tmp_path)
    request = _create_request()

    with (
        patch.object(server, "_is_instance_available_sync", return_value=True),
        patch.object(server, "_run_subprocess", side_effect=subprocess.TimeoutExpired(cmd="ns_opencode", timeout=150)),
        patch(
            "responses_api_agents.ns_opencode_agent.app.get_first_server_config_dict",
            return_value={"host": "127.0.0.1", "port": 8000},
        ),
    ):
        response = await server.run(request)

    assert response.reward == 0.0
    assert response.metadata["status"] == "failed"
    assert "timed out" in response.metadata["error"].lower()


@pytest.mark.asyncio
async def test_run_nonzero_exit_without_trace_returns_reward_zero(tmp_path: Path) -> None:
    server = _create_server(tmp_path)
    request = _create_request()

    with (
        patch.object(server, "_is_instance_available_sync", return_value=True),
        patch.object(
            server,
            "_run_subprocess",
            return_value=_completed_process(["singularity", "exec", "instance://x", "ns_opencode"], returncode=9),
        ),
        patch(
            "responses_api_agents.ns_opencode_agent.app.get_first_server_config_dict",
            return_value={"host": "127.0.0.1", "port": 8000},
        ),
    ):
        response = await server.run(request)

    assert response.reward == 0.0
    assert response.metadata["status"] == "failed"
    assert response.metadata["ns_opencode_returncode"] == 9


def test_run_api_success_returns_reward_one(tmp_path: Path) -> None:
    server = _create_server(tmp_path)

    def fake_run_subprocess(command: list[str], timeout: int | None = None):
        _write_trace_from_command(command)
        return _completed_process(command, returncode=0)

    with (
        patch.object(server, "_is_instance_available_sync", return_value=True),
        patch.object(server, "_run_subprocess", side_effect=fake_run_subprocess),
        patch(
            "responses_api_agents.ns_opencode_agent.app.get_first_server_config_dict",
            return_value={"host": "127.0.0.1", "port": 8000},
        ),
    ):
        client = TestClient(server.setup_webserver())
        resp = client.post(
            "/run",
            json={
                "instance_id": "sample_100",
                "responses_create_params": {"input": []},
                "sample": {"uuid": "sample_100", "problem": "hello"},
            },
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["reward"] == 1.0
    assert payload["metadata"]["trace_exists"] is True


def test_run_api_no_trace_returns_reward_zero(tmp_path: Path) -> None:
    server = _create_server(tmp_path)

    with (
        patch.object(server, "_is_instance_available_sync", return_value=True),
        patch.object(
            server,
            "_run_subprocess",
            return_value=_completed_process(["singularity", "exec", "instance://x", "ns_opencode"], returncode=0),
        ),
        patch(
            "responses_api_agents.ns_opencode_agent.app.get_first_server_config_dict",
            return_value={"host": "127.0.0.1", "port": 8000},
        ),
    ):
        client = TestClient(server.setup_webserver())
        resp = client.post(
            "/run",
            json={
                "instance_id": "sample_101",
                "responses_create_params": {"input": []},
                "sample": {"uuid": "sample_101", "problem": "hello"},
            },
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["reward"] == 0.0
    assert payload["metadata"]["trace_exists"] is False


def test_run_api_timeout_returns_reward_zero(tmp_path: Path) -> None:
    server = _create_server(tmp_path)

    with (
        patch.object(server, "_is_instance_available_sync", return_value=True),
        patch.object(server, "_run_subprocess", side_effect=subprocess.TimeoutExpired(cmd="ns_opencode", timeout=150)),
        patch(
            "responses_api_agents.ns_opencode_agent.app.get_first_server_config_dict",
            return_value={"host": "127.0.0.1", "port": 8000},
        ),
    ):
        client = TestClient(server.setup_webserver())
        resp = client.post(
            "/run",
            json={
                "instance_id": "sample_102",
                "responses_create_params": {"input": []},
                "sample": {"uuid": "sample_102", "problem": "hello"},
            },
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["reward"] == 0.0
    assert payload["metadata"]["status"] == "failed"
    assert "timed out" in payload["metadata"]["error"].lower()


def test_run_api_nonzero_exit_returns_failed_metadata(tmp_path: Path) -> None:
    server = _create_server(tmp_path)

    with (
        patch.object(server, "_is_instance_available_sync", return_value=True),
        patch.object(
            server,
            "_run_subprocess",
            return_value=_completed_process(["singularity", "exec", "instance://x", "ns_opencode"], returncode=11),
        ),
        patch(
            "responses_api_agents.ns_opencode_agent.app.get_first_server_config_dict",
            return_value={"host": "127.0.0.1", "port": 8000},
        ),
    ):
        client = TestClient(server.setup_webserver())
        resp = client.post(
            "/run",
            json={
                "instance_id": "sample_103",
                "responses_create_params": {"input": []},
                "sample": {"uuid": "sample_103", "problem": "hello"},
            },
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["reward"] == 0.0
    assert payload["metadata"]["status"] == "failed"
    assert payload["metadata"]["ns_opencode_returncode"] == 11


@pytest.mark.asyncio
async def test_run_serializes_concurrency_with_semaphore(tmp_path: Path) -> None:
    server = _create_server(tmp_path, concurrency=1)
    request_one = _create_request(instance_id="instance_01", sample={"uuid": "sample_01", "problem": "a"})
    request_two = _create_request(instance_id="instance_02", sample={"uuid": "sample_02", "problem": "b"})

    state_lock = Lock()
    state = {"active": 0, "max_active": 0}

    def fake_run_subprocess(command: list[str], timeout: int | None = None):
        with state_lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])

        _write_trace_from_command(command)
        time.sleep(0.05)

        with state_lock:
            state["active"] -= 1

        return _completed_process(command, returncode=0)

    with (
        patch.object(server, "_is_instance_available_sync", return_value=True),
        patch.object(server, "_run_subprocess", side_effect=fake_run_subprocess),
        patch(
            "responses_api_agents.ns_opencode_agent.app.get_first_server_config_dict",
            return_value={"host": "127.0.0.1", "port": 8000},
        ),
    ):
        response_one, response_two = await asyncio.gather(server.run(request_one), server.run(request_two))

    assert state["max_active"] == 1
    assert response_one.reward == 1.0
    assert response_two.reward == 1.0
