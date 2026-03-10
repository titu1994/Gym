# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from asyncio import run

from nemo_gym.server_utils import ServerClient


server_client = ServerClient.load_from_global_config()

task = server_client.post(
    server_name="ns_opencode_simple_agent",
    url_path="/run",
    json={
        "instance_id": "sample_001",
        "responses_create_params": {
            "temperature": 0.0,
            "top_p": 1.0,
            "input": [],
        },
        "sample": {
            "uuid": "sample_001",
            "problem": "Create a simple python function that adds two numbers.",
        },
    },
)

result = run(task)
print(json.dumps(result.json(), indent=4))
