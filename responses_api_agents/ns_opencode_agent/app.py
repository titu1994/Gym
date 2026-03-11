# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import atexit
import asyncio
import json
import subprocess
import time
from asyncio import Semaphore
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

import requests
from fastapi import FastAPI
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import get_first_server_config_dict


class NSOpenCodeAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    execution_backend: Literal["singularity", "remote_http"] = "singularity"
    singularity_image_path: Optional[str] = None
    singularity_instance_name: str = "ns_opencode_worker"
    remote_http_base_url: Optional[str] = None
    remote_http_timeout: int = 1920
    data_root: str = "/data"
    task_timeout: int = 1800
    trace_monitor_interval: float = 120.0
    concurrency: int = 1
    bind_paths: List[str] = Field(default_factory=list)
    instance_start_args: List[str] = Field(default_factory=list)
    ns_opencode_cli_args: List[str] = Field(default_factory=list)


class NSOpenCodeAgentRunRequest(BaseRunRequest):
    instance_id: str
    sample: Dict[str, Any]

    model_config = ConfigDict(extra="allow")


class NSOpenCodeAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class NSOpenCodeAgent(SimpleResponsesAPIAgent):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: NSOpenCodeAgentConfig
    sem: Semaphore = None

    _instance_started_by_agent: bool = False
    _instance_stop_attempted: bool = False
    _lifecycle_lock: Lock = None

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)
        self._lifecycle_lock = Lock()
        if self.config.execution_backend == "singularity":
            if not self.config.singularity_image_path:
                raise ValueError("singularity_image_path is required when execution_backend='singularity'")
            self._ensure_instance_started_sync()
            atexit.register(self._stop_instance_on_exit)
        elif self.config.execution_backend == "remote_http":
            if not self.config.remote_http_base_url:
                raise ValueError("remote_http_base_url is required when execution_backend='remote_http'")
        else:
            raise ValueError(f"Unsupported execution_backend: {self.config.execution_backend}")

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        app.post("/v1/responses")(self.responses)
        app.post("/run")(self.run)
        app.add_event_handler("shutdown", self._on_shutdown)
        return app

    async def _on_shutdown(self) -> None:
        await asyncio.to_thread(self._stop_instance_sync)

    def _stop_instance_on_exit(self) -> None:
        self._stop_instance_sync()

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        raise NotImplementedError

    def _run_subprocess(self, command: List[str], timeout: Optional[int] = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    def _is_instance_available_sync(self) -> bool:
        command = [
            "singularity",
            "exec",
            f"instance://{self.config.singularity_instance_name}",
            "true",
        ]
        try:
            result = self._run_subprocess(command, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0

    def _build_instance_start_command(self) -> List[str]:
        assert self.config.singularity_image_path
        command = ["singularity", "instance", "start"]
        command.extend(self.config.instance_start_args)

        bind_paths = list(self.config.bind_paths)
        data_bind = f"{self.config.data_root}:{self.config.data_root}"
        if data_bind not in bind_paths and not any(
            path == self.config.data_root or path.startswith(f"{self.config.data_root}:") for path in bind_paths
        ):
            bind_paths.append(data_bind)

        for bind_path in bind_paths:
            command.extend(["--bind", bind_path])

        command.extend([self.config.singularity_image_path, self.config.singularity_instance_name])
        return command

    def _ensure_instance_started_sync(self) -> None:
        if self._is_instance_available_sync():
            return

        command = self._build_instance_start_command()
        result = self._run_subprocess(command, timeout=120)

        if result.returncode != 0:
            raise RuntimeError(
                "Failed to start Singularity instance "
                f"{self.config.singularity_instance_name}: {result.stderr.strip()}"
            )

        if not self._is_instance_available_sync():
            raise RuntimeError(
                "Singularity instance started but is unavailable: "
                f"{self.config.singularity_instance_name}"
            )

        self._instance_started_by_agent = True

    def _stop_instance_sync(self) -> None:
        with self._lifecycle_lock:
            if self._instance_stop_attempted or not self._instance_started_by_agent:
                return

            command = ["singularity", "instance", "stop", self.config.singularity_instance_name]
            try:
                self._run_subprocess(command, timeout=60)
            finally:
                self._instance_stop_attempted = True

    def _default_response(self, model_name: str) -> Dict[str, Any]:
        now = float(int(time.time()))
        return {
            "id": f"resp_{uuid4().hex}",
            "created_at": now,
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "metadata": {},
            "model": model_name,
            "object": "response",
            "output": [],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "background": False,
            "max_output_tokens": None,
            "max_tool_calls": None,
            "previous_response_id": None,
            "prompt": None,
            "reasoning": {"effort": None, "generate_summary": None, "summary": None},
            "service_tier": "default",
            "status": "completed",
            "text": {"format": {"type": "text"}, "verbosity": "medium"},
            "top_logprobs": 0,
            "truncation": "disabled",
            "usage": {
                "input_tokens": 0,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 0,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 0,
            },
            "user": None,
            "prompt_cache_key": None,
            "safety_identifier": None,
            "store": True,
        }

    def _build_minimal_training_output_item(self, text: str) -> Dict[str, Any]:
        # NeMo RL rollout post-processing requires at least one output item with
        # generation token IDs and logprobs.
        return {
            "id": f"msg_{uuid4().hex}",
            "content": [
                {
                    "annotations": [],
                    "logprobs": None,
                    "text": text,
                    "type": "output_text",
                }
            ],
            "role": "assistant",
            "status": "completed",
            "type": "message",
            "prompt_token_ids": [0],
            "generation_token_ids": [0],
            "generation_log_probs": [0.0],
        }

    def _sanitize_identifier(self, text: str) -> str:
        return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text)

    def _find_trace_files(self, dataset_traces_dir: Path, sample_uuid: str) -> List[str]:
        if not dataset_traces_dir.exists():
            return []
        return sorted(str(path) for path in dataset_traces_dir.glob(f"{sample_uuid}-trace*.json") if path.is_file())

    def _build_ns_opencode_command(
        self,
        input_file: Path,
        workspace_dir: Path,
        model_name: str,
        base_url: str,
        api_key: str,
    ) -> List[str]:
        command = [
            "singularity",
            "exec",
            f"instance://{self.config.singularity_instance_name}",
            "ns_opencode",
            str(input_file),
            str(workspace_dir),
            "--model",
            model_name,
            "--provider",
            "vllm",
            "--openai-base-url",
            base_url,
            "--openai-api-key",
            api_key,
            "--batch-size",
            "1",
            "--task-timeout",
            str(self.config.task_timeout),
            "--trace-monitor-interval",
            str(self.config.trace_monitor_interval),
        ]
        command.extend(self.config.ns_opencode_cli_args)
        return command

    def _build_verify_response(
        self,
        body: NSOpenCodeAgentRunRequest,
        model_name: str,
        reward: float,
        metadata: Dict[str, Any],
    ) -> NSOpenCodeAgentVerifyResponse:
        status = str(metadata.get("status", "completed"))
        trace_exists = bool(metadata.get("trace_exists", False))
        output_text = "ns_opencode trace found" if trace_exists else f"ns_opencode run status: {status}"

        response_dict = self._default_response(model_name)
        response_dict["output"] = [self._build_minimal_training_output_item(output_text)]
        response_dict["usage"] = {
            "input_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 1,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 1,
        }
        response_obj = NeMoGymResponse.model_validate(response_dict)
        return NSOpenCodeAgentVerifyResponse(
            responses_create_params=body.responses_create_params,
            response=response_obj,
            reward=reward,
            instance_id=body.instance_id,
            metadata=metadata,
        )

    def _run_remote_opencode_sync(self, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
        assert self.config.remote_http_base_url
        endpoint = f"{self.config.remote_http_base_url.rstrip('/')}/run"
        response = requests.post(endpoint, json=payload, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict):
            raise ValueError(f"Remote response must be a dict, got {type(result)}")
        return result

    async def run(self, body: NSOpenCodeAgentRunRequest = Body()) -> NSOpenCodeAgentVerifyResponse:
        async with self.sem:
            global_config_dict = self.server_client.global_config_dict
            model_name = global_config_dict.get("policy_model_name", "unknown_model")
            api_key = global_config_dict.get("policy_api_key", "dummy_key")
            model_server_config = get_first_server_config_dict(global_config_dict, self.config.model_server.name)
            base_url = f"http://{model_server_config['host']}:{model_server_config['port']}/v1"

            safe_instance_id = self._sanitize_identifier(body.instance_id)
            job_id = f"{safe_instance_id}-{uuid4().hex[:8]}"
            sample = dict(body.sample)
            sample_uuid = str(sample.get("uuid") or body.instance_id)
            sample["uuid"] = sample_uuid
            if self.config.execution_backend == "remote_http":
                timeout_seconds = int(max(self.config.remote_http_timeout, self.config.task_timeout + 120))
                payload = {
                    "job_id": job_id,
                    "sample": sample,
                    "model_name": model_name,
                    "openai_base_url": base_url,
                    "openai_api_key": api_key,
                    "provider": "vllm",
                    "task_timeout": self.config.task_timeout,
                    "trace_monitor_interval": self.config.trace_monitor_interval,
                    "data_root": self.config.data_root,
                    "ns_opencode_cli_args": self.config.ns_opencode_cli_args,
                }

                try:
                    remote_metadata = await asyncio.to_thread(self._run_remote_opencode_sync, payload, timeout_seconds)
                except Exception as e:
                    return self._build_verify_response(
                        body=body,
                        model_name=model_name,
                        reward=0.0,
                        metadata={
                            "status": "failed",
                            "error": f"remote_http backend error: {type(e).__name__}: {e}",
                            "job_id": job_id,
                            "sample_uuid": sample_uuid,
                            "execution_backend": "remote_http",
                            "remote_http_base_url": self.config.remote_http_base_url,
                        },
                    )

                trace_files = remote_metadata.get("trace_files", [])
                trace_exists = bool(remote_metadata.get("trace_exists", len(trace_files) > 0))
                reward = 1.0 if trace_exists else 0.0
                metadata = dict(remote_metadata)
                metadata.setdefault("job_id", job_id)
                metadata.setdefault("sample_uuid", sample_uuid)
                metadata["trace_exists"] = trace_exists
                metadata["trace_files"] = trace_files
                metadata["execution_backend"] = "remote_http"
                metadata["remote_http_base_url"] = self.config.remote_http_base_url
                metadata["provider"] = "vllm"
                metadata["model_name"] = model_name
                metadata["base_url"] = base_url
                return self._build_verify_response(
                    body=body,
                    model_name=model_name,
                    reward=reward,
                    metadata=metadata,
                )

            if not await asyncio.to_thread(self._is_instance_available_sync):
                return self._build_verify_response(
                    body=body,
                    model_name=model_name,
                    reward=0.0,
                    metadata={
                        "status": "failed",
                        "error": "Singularity instance is unavailable",
                        "instance_name": self.config.singularity_instance_name,
                        "execution_backend": "singularity",
                    },
                )

            job_dir = Path(self.config.data_root) / "jobs" / job_id
            workspace_dir = job_dir / "workspace"
            logs_dir = job_dir / "logs"
            input_file = job_dir / "input.jsonl"
            dataset_traces_dir = workspace_dir / "dataset_traces"

            logs_dir.mkdir(parents=True, exist_ok=True)
            workspace_dir.mkdir(parents=True, exist_ok=True)
            with open(input_file, "w", encoding="utf-8") as f:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

            command = self._build_ns_opencode_command(
                input_file=input_file,
                workspace_dir=workspace_dir,
                model_name=model_name,
                base_url=base_url,
                api_key=api_key,
            )

            timeout_seconds = int(self.config.task_timeout + 120)
            try:
                run_result = await asyncio.to_thread(self._run_subprocess, command, timeout_seconds)
            except subprocess.TimeoutExpired:
                return self._build_verify_response(
                    body=body,
                    model_name=model_name,
                    reward=0.0,
                    metadata={
                        "status": "failed",
                        "error": f"ns_opencode timed out after {timeout_seconds}s",
                        "job_id": job_id,
                        "workspace_dir": str(workspace_dir),
                        "dataset_file": str(input_file),
                        "execution_backend": "singularity",
                    },
                )

            stdout_path = logs_dir / "ns_opencode.stdout.log"
            stderr_path = logs_dir / "ns_opencode.stderr.log"
            stdout_path.write_text(run_result.stdout or "", encoding="utf-8")
            stderr_path.write_text(run_result.stderr or "", encoding="utf-8")

            trace_files = self._find_trace_files(dataset_traces_dir, sample_uuid)
            trace_exists = len(trace_files) > 0
            reward = 1.0 if trace_exists else 0.0

            status = "completed" if run_result.returncode == 0 else "failed"
            error = None
            if run_result.returncode != 0:
                error = f"ns_opencode exited with return code {run_result.returncode}"
            elif not trace_exists:
                error = "No trace files found"

            return self._build_verify_response(
                body=body,
                model_name=model_name,
                reward=reward,
                metadata={
                    "status": status,
                    "error": error,
                    "job_id": job_id,
                    "workspace_dir": str(workspace_dir),
                    "dataset_file": str(input_file),
                    "dataset_traces_dir": str(dataset_traces_dir),
                    "sample_uuid": sample_uuid,
                    "trace_files": trace_files,
                    "trace_exists": trace_exists,
                    "ns_opencode_returncode": run_result.returncode,
                    "stdout_log": str(stdout_path),
                    "stderr_log": str(stderr_path),
                    "command": command,
                    "provider": "vllm",
                    "model_name": model_name,
                    "base_url": base_url,
                    "execution_backend": "singularity",
                },
            )


if __name__ == "__main__":
    NSOpenCodeAgent.run_webserver()
