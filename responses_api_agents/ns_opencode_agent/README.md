# NS OpenCode Agent

NeMo-Gym `responses_api_agent` for running `ns_opencode` rollouts in a long-lived Singularity instance.

## Behavior (v1)
- Starts the Singularity instance on agent startup if it is not already running.
- Stops the Singularity instance on process exit if this agent started it.
- Uses `ns_opencode` with provider fixed to `vllm`.
- Uses the same NeMo-Gym policy model server (`policy_model_name`) for both NeMo-Gym responses and `ns_opencode`.
- Returns reward `1.0` when `workspace/dataset_traces/<uuid>-trace*.json` exists.
- Returns reward `0.0` when no matching trace exists.
- If the instance is down during `/run`, request fails with reward `0.0` and no auto-restart.
- `/v1/responses` is intentionally not implemented for v1.

## Prerequisites
1. Singularity or Apptainer is installed and available as `singularity`.
2. The Singularity image contains `ns_opencode` on `PATH`.
3. A shared writable directory is mounted as `/data` in the host and visible in the instance.
4. A vLLM-compatible NeMo-Gym policy model server is configured.

If only `apptainer` exists, create a compatibility symlink:

```bash
ln -sf /usr/bin/apptainer /usr/bin/singularity
```

## Configuration
Edit `responses_api_agents/ns_opencode_agent/configs/ns_opencode_agent.yaml`.

```yaml
ns_opencode_simple_agent:
  responses_api_agents:
    ns_opencode_agent:
      entrypoint: app.py
      model_server:
        type: responses_api_models
        name: policy_model
      singularity_image_path: /path/to/ns_opencode_worker.sif
      singularity_instance_name: ns_opencode_worker
      data_root: /data
      concurrency: 1
      task_timeout: 1800
      trace_monitor_interval: 120.0
      bind_paths:
      - /data:/data
      instance_start_args: []
      ns_opencode_cli_args: []
```

Notes:
- `model_server.name` must reference your NeMo-Gym policy model server entry (`policy_model` by convention).
- `data_root` is where per-job files are created, for example `/data/jobs/<job_id>/...`.
- `concurrency` controls concurrent `/run` requests handled by this agent process.

## Start NeMo-Gym
Example with the vLLM responses model wrapper:

```bash
CONFIG_PATHS="responses_api_models/vllm_model/configs/vllm_model.yaml,responses_api_agents/ns_opencode_agent/configs/ns_opencode_agent.yaml"

ng_run +config_paths=[$CONFIG_PATHS] \
  +policy_model_name=<your_model_name> \
  +policy_base_url=http://<vllm-host>:<vllm-port>/v1 \
  +policy_api_key=dummy_key \
  +ns_opencode_simple_agent.responses_api_agents.ns_opencode_agent.singularity_image_path=/path/to/ns_opencode_worker.sif \
  +ns_opencode_simple_agent.responses_api_agents.ns_opencode_agent.singularity_instance_name=ns_opencode_worker \
  +ns_opencode_simple_agent.responses_api_agents.ns_opencode_agent.data_root=/data
```

At startup, the agent checks the instance and runs `singularity instance start ...` if needed.

## Run a rollout
Call the agent `/run` endpoint through a NeMo-Gym client or direct HTTP.

Example request body:

```json
{
  "instance_id": "sample_001",
  "responses_create_params": {
    "input": [],
    "temperature": 0.0,
    "top_p": 1.0
  },
  "sample": {
    "uuid": "sample_001",
    "problem": "Create a Python function that adds two numbers."
  }
}
```

The agent will:
1. Create `/data/jobs/<job_id>/input.jsonl` with one dataset row.
2. Execute `ns_opencode` in `instance://<singularity_instance_name>`.
3. Read traces from `/data/jobs/<job_id>/workspace/dataset_traces`.
4. Return a verify-style response with reward and metadata.

You can also use the example script:

```bash
python3 responses_api_agents/ns_opencode_agent/client.py
```

## Outputs
For each request, artifacts are written under:
- `/data/jobs/<job_id>/input.jsonl`
- `/data/jobs/<job_id>/workspace/dataset_traces/`
- `/data/jobs/<job_id>/logs/ns_opencode.stdout.log`
- `/data/jobs/<job_id>/logs/ns_opencode.stderr.log`

Response metadata includes:
- `job_id`
- `workspace_dir`
- `dataset_file`
- `dataset_traces_dir`
- `trace_files`
- `trace_exists`
- `ns_opencode_returncode`
- `stdout_log` and `stderr_log`
- `error` when applicable

## Troubleshooting
- `Singularity instance is unavailable` during `/run`: check `singularity instance list`, verify instance name matches config, and verify bind paths include `/data`.
- No traces found: inspect `stderr_log` and `stdout_log`, verify `sample.uuid` naming, and verify container write access to `/data/jobs/.../workspace`.
