# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Optional

from verl.base_config import BaseConfig
from verl.workers.config import CustomAsyncServerConfig

__all__ = ["AgentFlowConfig"]




@dataclass
class AgentFlowConfig(BaseConfig):
    """Configure agent flows using ParaAgent fields and verl AgentLoopConfig aliases."""

    num_workers: int = 8

    default_agent_flow: str = "agent_env_loop"
    agent_flow_config_path: Optional[str] = None
    max_steps: int = 10
    skip_special_tokens: bool = True

    env_kwargs: Optional[dict] = None

    max_concurrent_trajectories_per_worker: Optional[int] = None

    max_concurrent_judge_per_worker: Optional[int] = None

    default_agent_loop: Optional[str] = None
    agent_loop_config_path: Optional[str] = None

    custom_async_server: CustomAsyncServerConfig = field(default_factory=CustomAsyncServerConfig)
