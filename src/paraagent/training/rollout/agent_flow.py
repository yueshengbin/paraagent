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
import asyncio
import json
import logging
import math
import os
import pickle
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional
from uuid import uuid4

import hydra
import numpy as np
import ray
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from tensordict import TensorDict
from transformers import AutoProcessor, AutoTokenizer

from paraagent.training.reward_loop import RewardLoopWorker
from verl.experimental.agent_loop.agent_loop import (
    AsyncLLMServerManager,
    DictConfigWrap,
    GlobalRequestLoadBalancer,
)
from verl.experimental.agent_loop.prometheus_utils import update_prometheus_config
from verl.experimental.agent_loop.utils import resolve_config_path
from verl.protocol import DataProto
from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.single_controller.ray.base import RayResourcePool, RayWorkerGroup
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.chat_template import apply_chat_template as apply_chat_template_compat
from verl.utils.chat_template import initialize_system_prompt
from verl.utils.tokenizer import normalize_token_ids
from verl.utils.dataset.rl_dataset import RLHFDataset, get_dataset_class
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils.profiler import simple_timer
from verl.utils.ray_utils import get_event_loop
from verl.utils.rollout_trace import (
    RolloutTraceConfig,
    rollout_trace_attr,
)

_REWARD_EXECUTOR: Optional[ThreadPoolExecutor] = None
_REWARD_EXECUTOR_LOCK = threading.Lock()

_GDPO_REWARD_COMPONENT_KEYS = ("R_traj", "R_phase", "R_format_score")

def _harmonize_reward_extra_keys(outputs: list[DataProto]) -> list[str]:
    """Align optional reward metadata across workers for DataProto concatenation.

    Union diagnostic keys and fill absent non-tensor columns with None.
    """
    all_keys = sorted(
        {
            key
            for output in outputs
            for key in output.meta_info.get("reward_extra_keys", [])
        }
    )
    for output in outputs:
        existing = set(output.non_tensor_batch)
        missing = [key for key in all_keys if key not in existing]
        for key in missing:
            values = np.empty(len(output), dtype=object)
            values.fill(None)
            output.non_tensor_batch[key] = values
        output.meta_info["reward_extra_keys"] = all_keys
    return all_keys

def _finite_reward_scalar(value: Any, field: str) -> float:
    """Convert a reward field to a finite scalar or reject the sample."""
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Reward field {field!r} must be a finite scalar, got {value!r}") from exc
    if not math.isfinite(scalar):
        raise ValueError(f"Reward field {field!r} must be finite, got {scalar!r}")
    return scalar

def _normalize_reward_extra_info(
    reward_result: dict[str, Any] | float | None,
    *,
    score_fallback: Any = None,
) -> dict[str, Any]:
    """Validate finite reward scalars and provide neutral defaults for missing GDPO components."""
    if isinstance(reward_result, dict):
        normalized = dict(reward_result)
    elif reward_result is None:
        normalized = {}
    else:
        normalized = {"score": _finite_reward_scalar(reward_result, "score")}

    if "score" in normalized:
        score = _finite_reward_scalar(normalized["score"], "score")
    elif score_fallback is not None:
        score = _finite_reward_scalar(score_fallback, "score")
    else:
        score = 0.0
    normalized["score"] = score

    if "R_traj" in normalized:
        normalized["R_traj"] = _finite_reward_scalar(normalized["R_traj"], "R_traj")
    else:
        normalized["R_traj"] = score

    if "R_outcome" in normalized:
        normalized["R_outcome"] = _finite_reward_scalar(
            normalized["R_outcome"], "R_outcome"
        )
    else:
        normalized["R_outcome"] = 0.0

    if "R_phase" in normalized:
        normalized["R_phase"] = _finite_reward_scalar(normalized["R_phase"], "R_phase")
    else:
        normalized["R_phase"] = 0.0

    if "R_format_score" in normalized:
        format_score = _finite_reward_scalar(normalized["R_format_score"], "R_format_score")
        normalized["R_format_score"] = format_score
    elif "format_reward" in normalized:
        format_score = _finite_reward_scalar(normalized["format_reward"], "format_reward")
        normalized["R_format_score"] = min(1.0, max(0.0, format_score))
    elif "R_format" in normalized:
        format_score = 1.0 + _finite_reward_scalar(normalized["R_format"], "R_format")
        normalized["R_format_score"] = min(1.0, max(0.0, format_score))
    else:
        normalized["R_format_score"] = 0.0

    for key in _GDPO_REWARD_COMPONENT_KEYS:
        normalized[key] = _finite_reward_scalar(normalized[key], key)
    return normalized

def _get_reward_executor() -> ThreadPoolExecutor:
    global _REWARD_EXECUTOR
    if _REWARD_EXECUTOR is None:
        with _REWARD_EXECUTOR_LOCK:
            if _REWARD_EXECUTOR is None:
                workers = int(os.environ.get("TOOL_REWARD_EXECUTOR_WORKERS", "32"))
                _REWARD_EXECUTOR = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="reward-judge")
    return _REWARD_EXECUTOR

def _reward_executor_kind() -> str:
    """Select thread or process execution for synchronous rewards.
    TOOL_REWARD_JUDGE_PROFILE requires the thread executor for instrumentation.
    """
    if os.environ.get("TOOL_REWARD_JUDGE_PROFILE") == "1":
        return "thread"
    kind = (os.environ.get("TOOL_REWARD_EXECUTOR") or "thread").strip().lower()
    return "process" if kind == "process" else "thread"

def _log_reward_profile(queue_wait: float, fn_dur: float, cpu: float = 0.0) -> None:
    """Record executor wait time, reward wall time, and thread CPU time.
    Called when TOOL_REWARD_JUDGE_PROFILE=1; logging failures are ignored.
    """
    try:
        d = os.environ.get("TOOL_REWARD_PROFILE_DIR", "/tmp")
        os.makedirs(d, exist_ok=True)
        with open(f"{d}/judge_profile_{os.getpid()}.jsonl", "a") as fh:
            fh.write(json.dumps({"q": round(queue_wait, 3), "fn": round(fn_dur, 3), "cpu": round(cpu, 3)}) + "\n")
    except Exception:
        pass

_REWARD_PROFILE_SAMPLE_COUNTER = [0]

def _dump_cprofile(pr) -> None:
    """Record the top 25 reward functions by cumulative time.
    Called when TOOL_REWARD_JUDGE_PROFILE=1; logging failures are ignored.
    """
    try:
        import io as _io
        import pstats

        s = _io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(25)
        d = os.environ.get("TOOL_REWARD_PROFILE_DIR", "/tmp")
        os.makedirs(d, exist_ok=True)
        with open(f"{d}/cprofile_{os.getpid()}.txt", "a") as fh:
            fh.write("\n===== reward fn cProfile (cumulative top 25) =====\n")
            fh.write(s.getvalue())
    except Exception:
        pass

try:
    from verl.utils.transferqueue_utils import create_transferqueue_client as build_transferqueue_client
    from verl.utils.transferqueue_utils import tqbridge
except Exception:  
    def tqbridge():
        def decorator(func):
            return func

        return decorator

    def build_transferqueue_client(*args, **kwargs):
        return None

from verl.workers.rollout.replica import get_rollout_replica_class

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class AgentFlowMetrics(BaseModel):
    """Agent flow performance metrics."""

    generate_sequences: float = 0.0
    """Total LLM generation time for the whole trajectory."""
    tool_calls: float = 0.0
    """Total environment/tool time for the whole trajectory."""
    reward_judge: float = 0.0
    """Total reward judge time for the whole trajectory."""
    step_metrics: list[dict[str, float]] = Field(default_factory=list)
    """Per-agent-step timing entries."""

class AgentFlowStep(BaseModel):
    """Agent flow step."""

    prompt_ids: list[int]
    """Prompt token ids."""
    response_ids: list[int]
    """Response token ids including LLM generated token, tool response token."""
    input_ids: Optional[list[int]] = None
    """Input token ids (prompt_ids + response_ids)."""
    position_ids: Optional[list[int]] = None
    """Position ids."""
    attention_mask: Optional[list[int]] = None
    """Attention mask."""
    response_mask: Optional[list[int]] = None
    """Response mask, 1 for LLM generated token, 0 for tool response token."""
    response_logprobs: Optional[list[float]] = None
    """Log probabilities for the response tokens."""
    routed_experts: Optional[Any] = None
    """Routed experts for the total tokens."""
    multi_modal_data: Optional[dict[str, Any]] = None
    """Multi-modal data for multi-modal tools."""
    reward_score: Optional[float] = None
    """Reward score for the step."""
    num_turns: int = 2
    """Number of chat turns, including user, assistant, tool."""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""

class _InternalAgentFlowStep(AgentFlowStep):
    """Internal agent flow step with padded sequences and processed multi-modal data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: torch.Tensor
    """Padded prompt token ids."""
    response_ids: torch.Tensor
    """Padded response token ids."""
    input_ids: torch.Tensor
    """Padded input ids(prompt_ids + response_ids)."""
    position_ids: torch.Tensor
    """Padded position ids."""
    response_mask: torch.Tensor
    """Padded response mask."""
    attention_mask: torch.Tensor
    """Padded attention mask."""
    response_logprobs: Optional[torch.Tensor] = None
    """Padded log probabilities for the response tokens."""
    routed_experts: Optional[torch.Tensor] = None
    """Padded routed experts for the total tokens."""
    multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None
    """Multi-modal inputs for processors (e.g., pixel_values, image_grid_thw)."""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""

class AgentFlowOutput(BaseModel):
    """Agent flow output."""

    steps: list[_InternalAgentFlowStep]
    """List of agent flow steps."""
    metrics: AgentFlowMetrics
    """Auxiliary performance metrics"""

    cum_token_ids: Optional[list[int]] = None
    """Full trajectory token ids: [initial_prompt | agent₀ | tool₀ | … | agent_N]."""
    cum_response_mask: Optional[list[int]] = None
    """Same length as cum_token_ids: 1 for agent tokens, 0 for prompt/tool tokens."""
    init_prompt_len: Optional[int] = None
    """Length of the initial prompt prefix in cum_token_ids (boundary between
    prompt portion and response portion of the packed training sample)."""

    pending_trajectory_reward: Optional[dict[str, Any]] = None
    """When set, the trajectory-level judge has been DEFERRED rather than run
    inline. The manager runs it via ``finalize_trajectory_reward`` OUTSIDE the
    generation concurrency limit (the judge is a remote call and must not hold a
    GPU rollout slot), then attaches the score to ``steps[-1]``. Carries exactly
    the kwargs ``_compute_custom_reward_direct`` needs. None once finalized, or
    when the flow/trajectory has no trajectory-level reward."""

class AgentFlowBase(ABC):
    """Interact with a model server and environment for one sample."""

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        reward_loop_worker: RewardLoopWorker,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        dataset_cls: type[RLHFDataset],
        dataset_config: DictConfig,
        **kwargs,
    ):
        """Create one model/environment loop instance for a sample."""
        self.config = trainer_config.config
        self.server_manager = server_manager
        self.reward_loop_worker = reward_loop_worker
        self.tokenizer = tokenizer
        self.processor = processor
        self.dataset_cls = dataset_cls
        self.dataset_config = dataset_config
        self.apply_chat_template_kwargs = dataset_config.get("apply_chat_template_kwargs", {})
        self.system_prompt = initialize_system_prompt(self.tokenizer, **self.apply_chat_template_kwargs)
        self.loop = get_event_loop()
        self.custom_reward_fn = get_custom_reward_fn(self.config)

        self._reward_fn_subproc_spec = None
        if self.custom_reward_fn is not None and not asyncio.iscoroutinefunction(self.custom_reward_fn):
            rf_cfg = self.config.reward.get("custom_reward_function") or {}
            rf_path, rf_name = rf_cfg.get("path"), rf_cfg.get("name")
            if rf_path and rf_name:
                rf_kwargs = rf_cfg.get("reward_kwargs", {}) or {}
                if OmegaConf.is_config(rf_kwargs):
                    rf_kwargs = OmegaConf.to_container(rf_kwargs, resolve=True)
                self._reward_fn_subproc_spec = (str(rf_path), str(rf_name), dict(rf_kwargs))

    async def process_vision_info(self, messages: list[dict]) -> dict:
        """Extract images and videos from messages.

        Args:
            messages (list[dict]): Input messages.

        Returns:
            dict: Multi-modal data with keys "images" and "videos".
        """
        multi_modal_data = {}
        if self.processor is not None:
            images, videos = await self.dataset_cls.process_vision_info(
                messages, image_patch_size=self.processor.image_processor.patch_size, config=self.dataset_config
            )
            if images is not None:
                multi_modal_data["images"] = images
            if videos is not None:
                multi_modal_data["videos"] = videos

        return multi_modal_data

    async def apply_chat_template(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        images: list[Image.Image] = None,
        videos: list[tuple[torch.Tensor, dict]] = None,
        remove_system_prompt: bool = False,
    ):
        """Return prompt token IDs for messages with optional tools, images, and videos."""
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: apply_chat_template_compat(
                    self.processor,
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )

            if videos is not None:
                videos, video_metadatas = zip(*videos, strict=False)
                videos, video_metadatas = list(videos), list(video_metadatas)
            else:
                video_metadatas = None

            model_inputs = self.processor(
                text=[raw_prompt],
                images=images,
                videos=videos,
                video_metadata=video_metadatas,
                return_tensors="pt",
                do_sample_frames=False,
            )
            prompt_ids = normalize_token_ids(model_inputs.pop("input_ids"))
        else:
            tokenized_prompt = await self.loop.run_in_executor(
                None,
                lambda: apply_chat_template_compat(
                    self.tokenizer,
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
            prompt_ids = normalize_token_ids(tokenized_prompt)

        if remove_system_prompt:
            prompt_ids = prompt_ids[len(self.system_prompt) :]

        return prompt_ids

    @abstractmethod
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentFlowOutput:
        """Run agent loop to interact with LLM server and environment.

        Args:
            sampling_params (Dict[str, Any]): LLM sampling params.
            **kwargs: dataset fields from `verl.utils.dataset.RLHFDataset`.

        Returns:
            AgentFlowOutput: Agent flow output.
        """
        raise NotImplementedError

    async def _postprocess(self, step: AgentFlowStep, **kwargs) -> _InternalAgentFlowStep:
        step.extra_fields["raw_prompt"] = kwargs["raw_prompt"]

        self.tokenizer.padding_side = "left"
        prompt_output = self.tokenizer.pad(
            {"input_ids": step.prompt_ids},
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if prompt_output["input_ids"].dim() == 1:
            prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
            prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

        self.tokenizer.padding_side = "right"
        response_output = self.tokenizer.pad(
            {"input_ids": step.response_ids},
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if response_output["input_ids"].dim() == 1:
            response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
            response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

        response_mask_ids = step.response_mask if step.response_mask is not None else [1] * len(step.response_ids)
        response_mask_output = self.tokenizer.pad(
            {"input_ids": response_mask_ids},
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=False,
        )
        if response_mask_output["input_ids"].dim() == 1:
            response_mask_output["input_ids"] = response_mask_output["input_ids"].unsqueeze(0)

        response_logprobs = None
        if step.response_logprobs is not None:
            pad_size = self.config.actor_rollout_ref.rollout.response_length - len(step.response_logprobs)
            response_logprobs = torch.tensor(step.response_logprobs + [0.0] * pad_size).unsqueeze(0)

        response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
        attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
        input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)

        routed_experts = None
        if step.routed_experts is not None:
            total_length = input_ids.shape[1]
            length, layer_num, topk_num = step.routed_experts.shape
            experts_tensor = torch.from_numpy(step.routed_experts)
            routed_experts = torch.zeros(1, total_length, layer_num, topk_num, dtype=experts_tensor.dtype)

            start_pos = prompt_output["input_ids"].shape[1] - len(step.prompt_ids)
            end_pos = min(start_pos + length, total_length)

            if start_pos < 0 or end_pos > total_length:
                raise ValueError(
                    f"Invalid position range: start_pos={start_pos}, end_pos={end_pos}, total_length={total_length}"
                )

            routed_experts[:, start_pos:end_pos] = experts_tensor.unsqueeze(0)

        multi_modal_inputs = self._compute_multi_modal_inputs(step, input_ids)
        position_ids = self._compute_position_ids(input_ids, attention_mask, multi_modal_inputs)
        await self._compute_score(
            step,
            prompts=prompt_output["input_ids"],
            responses=response_output["input_ids"],
            attention_mask=attention_mask,
            input_ids=input_ids,
            position_ids=position_ids,
            kwargs=kwargs,
        )

        assert step.reward_score is not None, "Reward score is required for agent flow"

        return _InternalAgentFlowStep(
            prompt_ids=prompt_output["input_ids"],
            response_ids=response_output["input_ids"],
            response_logprobs=response_logprobs,
            response_mask=response_mask,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            routed_experts=routed_experts,
            multi_modal_inputs=multi_modal_inputs,
            multi_modal_data=step.multi_modal_data,
            reward_score=step.reward_score,
            num_turns=step.num_turns,
            extra_fields=step.extra_fields,
        )

    def _compute_multi_modal_inputs(self, output, input_ids) -> dict[str, torch.Tensor]:
        """Compute multi-modal inputs with image and video."""
        multi_modal_inputs = {}
        if self.processor is None:
            return multi_modal_inputs

        images = output.multi_modal_data.get("images")
        videos = output.multi_modal_data.get("videos")
        
        if videos is not None:
            videos, video_metadatas = zip(*videos, strict=False)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None
        current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
        multi_modal_inputs = self.processor(
            text=[current_text],
            images=images,
            videos=videos,
            video_metadatas=video_metadatas,
            return_tensors="pt",
            do_sample_frames=False,
        )
        multi_modal_inputs.pop("input_ids", None)
        multi_modal_inputs.pop("attention_mask", None)

        multi_modal_inputs = dict(multi_modal_inputs.convert_to_tensors("pt"))
        return multi_modal_inputs

    def _compute_position_ids(self, input_ids, attention_mask, multi_modal_inputs) -> torch.Tensor:
        """Compute position ids for multi-modal inputs."""
        if self.processor is None:
            return compute_position_id_with_mask(attention_mask)  

        image_grid_thw = multi_modal_inputs.get("image_grid_thw")
        video_grid_thw = multi_modal_inputs.get("video_grid_thw")

        vision_position_ids, _ = self.processor.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
        vision_position_ids = vision_position_ids.transpose(0, 1)  

        valid_mask = attention_mask[0].bool()
        text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
        text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
        text_position_ids = text_position_ids.unsqueeze(0)
        position_ids = torch.cat((text_position_ids, vision_position_ids), dim=1)  
        return position_ids

    async def _compute_score(self, output, prompts, responses, attention_mask, input_ids, position_ids, kwargs):
        """Compute reward score for single sample."""
        if output.reward_score is None and self.reward_loop_worker is None:
            return
        if output.reward_score is None:
            batch = TensorDict(
                {
                    "prompts": prompts,  
                    "responses": responses,  
                    "attention_mask": attention_mask,  
                    "input_ids": input_ids,  
                    "position_ids": position_ids,
                },
                batch_size=1,
            )
            non_tensor_batch = {
                **{k: np.array([v]) for k, v in kwargs.items()},
                "__num_turns__": np.array([output.num_turns]),
                "tool_extra_fields": np.array([output.extra_fields], dtype=object),
            }

            data = DataProto(
                batch=batch,
                non_tensor_batch=non_tensor_batch,
            )
            result = await self.reward_loop_worker.compute_score.remote(data)
            output.reward_score = _finite_reward_scalar(result["reward_score"], "score")
            output.extra_fields["reward_extra_info"] = _normalize_reward_extra_info(
                result.get("reward_extra_info"),
                score_fallback=output.reward_score,
            )

        if output.reward_score is not None:
            output.reward_score = _finite_reward_scalar(output.reward_score, "score")
            output.extra_fields["reward_extra_info"] = _normalize_reward_extra_info(
                output.extra_fields.get("reward_extra_info"),
                score_fallback=output.reward_score,
            )

    async def _compute_custom_reward_direct(
        self,
        *,
        data_source: Any,
        solution_str: str,
        ground_truth: Any,
        extra_info: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any] | float | None:
        """Score the completed trajectory with the configured reward function.
        Offload synchronous functions to an executor to keep the event loop responsive.
        """
        if self.custom_reward_fn is None:
            return None

        fn = self.custom_reward_fn
        kwargs = dict(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info or {},
        )

        if asyncio.iscoroutinefunction(fn):
            return await fn(**kwargs)

        subproc_spec = getattr(self, "_reward_fn_subproc_spec", None)
        if _reward_executor_kind() == "process" and subproc_spec is not None:
            from paraagent.rewards.reward_subproc import run_in_reward_subprocess

            module_path, fn_name, reward_kwargs = subproc_spec
            try:
                return await run_in_reward_subprocess(
                    self.loop,
                    module_path=module_path,
                    fn_name=fn_name,
                    reward_kwargs=reward_kwargs,
                    call_kwargs=kwargs,
                )
            except (TypeError, AttributeError, pickle.PicklingError) as exc:
                logger.warning(
                    "reward subprocess dispatch failed (%s); falling back to thread executor", exc
                )

        executor = _get_reward_executor()
        if os.environ.get("TOOL_REWARD_JUDGE_PROFILE") != "1":
            return await self.loop.run_in_executor(executor, lambda: fn(**kwargs))

        submit_t = time.perf_counter()
        marks: dict[str, float] = {}

        def _run_with_marks():
            marks["start"] = time.perf_counter()
            marks["cpu0"] = time.thread_time()  
            pr = None
            if os.environ.get("TOOL_REWARD_JUDGE_PROFILE") == "1":
                _REWARD_PROFILE_SAMPLE_COUNTER[0] += 1
                if _REWARD_PROFILE_SAMPLE_COUNTER[0] % 30 == 1:  
                    import cProfile

                    pr = cProfile.Profile()
                    pr.enable()
            try:
                return fn(**kwargs)
            finally:
                marks["end"] = time.perf_counter()
                marks["cpu1"] = time.thread_time()
                if pr is not None:
                    pr.disable()
                    _dump_cprofile(pr)

        result = await self.loop.run_in_executor(executor, _run_with_marks)
        start = marks.get("start", submit_t)

        cpu = marks.get("cpu1", 0.0) - marks.get("cpu0", 0.0)
        _log_reward_profile(start - submit_t, marks.get("end", start) - start, cpu)
        return result

    async def finalize_trajectory_reward(self, output: "AgentFlowOutput") -> "AgentFlowOutput":
        """Resolve a pending trajectory reward outside the generation concurrency limit.

        The payload captures all required environment data. The manager awaits this
        before postprocessing reads scores and metrics; absent payloads are a no-op.
        """
        pending = getattr(output, "pending_trajectory_reward", None)
        if not pending or not output.steps:
            output.pending_trajectory_reward = None
            return output

        reward_timer: dict[str, float] = {}
        with simple_timer("reward_judge", reward_timer):
            reward_result = await self._compute_custom_reward_direct(**pending)
        output.metrics.reward_judge += float(reward_timer.get("reward_judge", 0.0))

        reward_extra_info = _normalize_reward_extra_info(reward_result)
        total_reward = reward_extra_info["score"]

        for step in output.steps[:-1]:
            step.reward_score = 0.0
        output.steps[-1].reward_score = total_reward
        output.steps[-1].extra_fields["reward_extra_info"] = reward_extra_info
        output.pending_trajectory_reward = None
        return output

"""Agent flow registry: key is agent_name, value is a dict of agent flow config
used by hydra.utils.instantiate to initialize agent flow instance.

https://hydra.cc/docs/advanced/instantiate_objects/overview/
"""
_agent_flow_registry: dict[str, dict] = {}

def register(agent_name: str):
    """Register agent flow class."""

    def decorator(subclass: type[AgentFlowBase]) -> type[AgentFlowBase]:
        fqdn = f"{subclass.__module__}.{subclass.__qualname__}"
        _agent_flow_registry[agent_name] = {"_target_": fqdn}
        return subclass

    return decorator

class AgentFlowWorkerBase:
    """Agent flow worker takes a batch of messages and run each message in an agent flow."""

    def __init__(
        self,
        config: DictConfig,
        servers: list[tuple[str, ray.actor.ActorHandle]],
        load_balancer_handle: ray.actor.ActorHandle,
        reward_router_address: str = None,
    ):
        """Initialize agent flow manager.

        Args:
            config (DictConfig): YAML config.
            servers (List[Tuple[str, ray.actor.ActorHandle]]): OpenAI compatible LLM server address/handle pairs.
        """
        self.config = config

        if not hasattr(self, "server_manager"):
            self.server_manager = AsyncLLMServerManager(config, servers, load_balancer_handle)

        self.dataset_cls = get_dataset_class(config.data)
        self.reward_router_address = reward_router_address

        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.processor = hf_processor(local_path, trust_remote_code=True)

        agent_flow_config_path = config.actor_rollout_ref.rollout.agent.agent_flow_config_path
        if agent_flow_config_path:
            resolved_path = resolve_config_path(agent_flow_config_path)
            agent_flow_configs = OmegaConf.load(resolved_path)
            for agent_flow_config in agent_flow_configs:
                _agent_flow_registry[agent_flow_config.name] = agent_flow_config
        if self.config.actor_rollout_ref.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.actor_rollout_ref.model.custom_chat_template
            self.tokenizer.chat_template = self.config.actor_rollout_ref.model.custom_chat_template

        self.reward_loop_worker = None
        needs_reward_loop_worker = self.config.reward_model.enable or self.config.reward.custom_reward_function.path is None
        if needs_reward_loop_worker:
            self.reward_loop_worker = RewardLoopWorker.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                ),
            ).remote(self.config, self.reward_router_address)

        trace_config = self.config.actor_rollout_ref.rollout.get("trace", {})
        RolloutTraceConfig.init(
            self.config.trainer.project_name,
            self.config.trainer.experiment_name,
            trace_config.get("backend"),
            trace_config.get("token2text", False),
            trace_config.get("max_samples_per_step_per_worker", None),
        )

    @tqbridge()
    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate model and observation turns with a mask over model tokens."""
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=config.get("repetition_penalty", 1.0),
            logprobs=config.calculate_log_probs,
        )

        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature

        if "agent_name" not in batch.non_tensor_batch:
            default_agent_flow = config.agent.default_agent_flow
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_flow] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        max_samples_per_worker = RolloutTraceConfig.get_instance().max_samples_per_step_per_worker

        if max_samples_per_worker is not None:
            unique_sample_indices = np.unique(index)
            if max_samples_per_worker < len(unique_sample_indices):
                selected_samples = set(
                    np.random.choice(unique_sample_indices, max_samples_per_worker, replace=False).tolist()
                )
                traced_indices = set(i for i in range(len(batch)) if index[i] in selected_samples)
            else:
                traced_indices = set(range(len(batch)))
        else:
            traced_indices = set(range(len(batch)))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
        )

        max_concurrent = config.agent.get("max_concurrent_trajectories_per_worker", None)
        max_concurrent = int(max_concurrent) if max_concurrent is not None else None
        semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent and max_concurrent > 0 else None

        max_concurrent_judge = config.agent.get("max_concurrent_judge_per_worker", None)
        max_concurrent_judge = int(max_concurrent_judge) if max_concurrent_judge is not None else max_concurrent
        judge_semaphore = (
            asyncio.Semaphore(max_concurrent_judge) if max_concurrent_judge and max_concurrent_judge > 0 else None
        )

        async def run_one_agent_flow(
            *,
            sampling_params: dict[str, Any],
            trajectory: dict[str, Any],
            trace: bool,
            kwargs: dict[str, Any],
        ) -> AgentFlowOutput:

            decouple_judge = _reward_executor_kind() == "process"

            async def _finalize(agent_flow: "AgentFlowBase", output: "AgentFlowOutput") -> "AgentFlowOutput":
                
                if getattr(output, "pending_trajectory_reward", None) is not None:
                    await agent_flow.finalize_trajectory_reward(output)
                return output

            async def _run_and_judge() -> "AgentFlowOutput":
                agent_flow, output = await self._run_agent_flow(
                    sampling_params, trajectory, trace=trace, **kwargs
                )
                return await _finalize(agent_flow, output)

            if semaphore is None:
                return await _run_and_judge()
            if not decouple_judge:
                async with semaphore:
                    return await _run_and_judge()
            async with semaphore:
                agent_flow, output = await self._run_agent_flow(
                    sampling_params, trajectory, trace=trace, **kwargs
                )
            if judge_semaphore is None:
                return await _finalize(agent_flow, output)
            async with judge_semaphore:
                return await _finalize(agent_flow, output)

        tasks = []
        for i in range(len(batch)):
            trace_this_sample = i in traced_indices
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            tasks.append(
                asyncio.create_task(
                    run_one_agent_flow(
                        sampling_params=sampling_params,
                        trajectory=trajectory_info[i],
                        trace=trace_this_sample,
                        kwargs=kwargs,
                    )
                )
            )
        outputs = await asyncio.gather(*tasks)

        output = self._postprocess(outputs)
        return output

    async def _run_agent_flow(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        trace: bool = True,
        **kwargs,
    ) -> tuple["AgentFlowBase", AgentFlowOutput]:
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_flow",
            trace=trace,
        ):
            assert agent_name in _agent_flow_registry, (
                f"Agent flow {agent_name} not registered, registered agent flows: {_agent_flow_registry.keys()}"
            )

            agent_flow_config = _agent_flow_registry[agent_name]
            agent_flow = hydra.utils.instantiate(
                config=agent_flow_config,
                trainer_config=DictConfigWrap(config=self.config),
                server_manager=self.server_manager,
                reward_loop_worker=self.reward_loop_worker,
                tokenizer=self.tokenizer,
                processor=self.processor,
                dataset_cls=self.dataset_cls,
                dataset_config=self.config.data,
            )
            kwargs["_agent_validate"] = bool(trajectory.get("validate", False))
            output: AgentFlowOutput = await agent_flow.run(sampling_params, **kwargs)

            return agent_flow, output

    def _postprocess(self, inputs: list[AgentFlowOutput]) -> DataProto:
        """Process the padded outputs from _run_agent_flow and combine them into a batch."""
        rollout_cfg = self.config.actor_rollout_ref.rollout
        prompt_length = int(rollout_cfg.prompt_length)
        response_length = int(
            getattr(getattr(self.config, "data", None), "max_response_length", 0)
            or getattr(rollout_cfg, "max_model_len", 0)
            or rollout_cfg.response_length
        )
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        num_steps = []
        trajectory_num_steps = []
        trajectory_uids = []
        step_indices = []
        prompt_ids = []
        response_ids = []
        response_mask = []
        attention_mask = []
        input_ids = []
        position_ids = []
        multi_modal_data = []
        multi_modal_inputs = []
        num_turns = []
        reward_tensors = []
        response_logprobs_list = []
        routed_experts_list = []
        reward_extra_infos = []

        def _append_packed_trajectory(input_item: AgentFlowOutput, trajectory_uid: str) -> None:
            if input_item.cum_token_ids is None or input_item.cum_response_mask is None:
                raise ValueError("Multi-step AgentEnvLoop output requires cumulative token tracking for loss masking.")
            if input_item.init_prompt_len is None:
                raise ValueError("Cumulative token tracking requires init_prompt_len.")
            if len(input_item.cum_token_ids) != len(input_item.cum_response_mask):
                raise ValueError(
                    "Cumulative buffer length mismatch: "
                    f"tokens={len(input_item.cum_token_ids)} mask={len(input_item.cum_response_mask)}"
                )
            if input_item.init_prompt_len < 0 or input_item.init_prompt_len > len(input_item.cum_token_ids):
                raise ValueError(
                    f"Invalid init_prompt_len={input_item.init_prompt_len} "
                    f"for cumulative length {len(input_item.cum_token_ids)}"
                )

            raw_prompt_ids = list(input_item.cum_token_ids[: input_item.init_prompt_len])
            raw_response_ids = list(input_item.cum_token_ids[input_item.init_prompt_len :])
            raw_response_mask = [int(v) for v in input_item.cum_response_mask[input_item.init_prompt_len :]]
            if any(v not in (0, 1) for v in raw_response_mask):
                raise ValueError("cum_response_mask must contain only 0/1 values.")
            if len(raw_prompt_ids) > prompt_length:
                raise ValueError(
                    f"Packed prompt length {len(raw_prompt_ids)} exceeds prompt_length={prompt_length}."
                )
            if len(raw_response_ids) > response_length:
                raise ValueError(
                    f"Packed response length {len(raw_response_ids)} exceeds max_response_length={response_length}."
                )

            prompt_pad = [pad_token_id] * (prompt_length - len(raw_prompt_ids)) + raw_prompt_ids
            prompt_attention = [0] * (prompt_length - len(raw_prompt_ids)) + [1] * len(raw_prompt_ids)
            response_pad = raw_response_ids + [pad_token_id] * (response_length - len(raw_response_ids))
            response_attention = [1] * len(raw_response_ids) + [0] * (response_length - len(raw_response_ids))
            response_mask_pad = raw_response_mask + [0] * (response_length - len(raw_response_mask))

            prompt_tensor = torch.tensor([prompt_pad], dtype=torch.long)
            response_tensor = torch.tensor([response_pad], dtype=torch.long)
            response_mask_tensor = torch.tensor([response_mask_pad], dtype=torch.long)
            attention_tensor = torch.tensor([prompt_attention + response_attention], dtype=torch.long)
            input_tensor = torch.cat([prompt_tensor, response_tensor], dim=1)
            multi_modal_input = {}
            position_tensor = compute_position_id_with_mask(attention_tensor)

            prompt_ids.append(prompt_tensor)
            response_ids.append(response_tensor)
            response_mask.append(response_mask_tensor)
            attention_mask.append(attention_tensor)
            input_ids.append(input_tensor)
            position_ids.append(position_tensor)
            multi_modal_data.append(None)
            multi_modal_inputs.append(None)
            num_turns.append(input_item.steps[-1].num_turns if input_item.steps else 1)

            packed_logprobs = None
            if input_item.steps and all(step.response_logprobs is not None for step in input_item.steps):
                action_logprobs = []
                for step in input_item.steps:
                    step_mask = step.response_mask[0].to(dtype=torch.bool)
                    action_logprobs.append(step.response_logprobs[0][step_mask])
                if action_logprobs:
                    action_logprobs = torch.cat(action_logprobs, dim=0)
                    action_positions = response_mask_tensor[0].to(dtype=torch.bool)
                    if int(action_positions.sum().item()) == int(action_logprobs.numel()):
                        packed_logprobs = torch.zeros((1, response_length), dtype=action_logprobs.dtype)
                        packed_logprobs[0, action_positions] = action_logprobs
            response_logprobs_list.append(packed_logprobs)
            routed_experts_list.append(None)

            last_step = input_item.steps[-1] if input_item.steps else None
            if last_step is not None and last_step.reward_score is not None:
                last_step.extra_fields["reward_extra_info"] = _normalize_reward_extra_info(
                    last_step.extra_fields.get("reward_extra_info"),
                    score_fallback=last_step.reward_score,
                )
                reward_tensor = torch.zeros_like(response_mask_tensor, dtype=torch.float32)
                action_positions = torch.nonzero(response_mask_tensor[0], as_tuple=False).flatten()
                if action_positions.numel() > 0:
                    reward_tensor[0, int(action_positions[-1].item())] = float(last_step.reward_score)
                reward_tensors.append(reward_tensor)
            else:
                reward_tensors.append(None)
            reward_extra_infos.append(last_step.extra_fields.get("reward_extra_info", {}) if last_step else {})

        def _append_step(step: _InternalAgentFlowStep, trajectory_uid: str, step_index: int) -> None:
            prompt_ids.append(step.prompt_ids)
            response_ids.append(step.response_ids)
            response_mask.append(step.response_mask)
            attention_mask.append(step.attention_mask)
            input_ids.append(step.input_ids)
            position_ids.append(step.position_ids)
            multi_modal_data.append(step.multi_modal_data)
            multi_modal_inputs.append(step.multi_modal_inputs)
            num_turns.append(step.num_turns)
            response_logprobs_list.append(step.response_logprobs)
            routed_experts_list.append(step.routed_experts)
            if step.reward_score is not None:
                step.extra_fields["reward_extra_info"] = _normalize_reward_extra_info(
                    step.extra_fields.get("reward_extra_info"),
                    score_fallback=step.reward_score,
                )
                reward_tensor = torch.zeros_like(step.response_mask, dtype=torch.float32)
                action_positions = torch.nonzero(step.response_mask[0], as_tuple=False).flatten()
                if action_positions.numel() > 0:
                    reward_tensor[0, int(action_positions[-1].item())] = float(step.reward_score)
                reward_tensors.append(reward_tensor)
            else:
                reward_tensors.append(None)
            reward_extra_infos.append(step.extra_fields.get("reward_extra_info", {}))

        for input in inputs:
            traj_steps = len(input.steps)
            trajectory_uid = uuid4().hex
            if input.cum_token_ids is not None or traj_steps > 1:
                _append_packed_trajectory(input, trajectory_uid)
                num_steps.append(1)
                trajectory_num_steps.append(traj_steps)
                trajectory_uids.append(trajectory_uid)
                step_indices.append(0)
            else:
                num_steps.append(traj_steps)
                trajectory_num_steps.append(traj_steps)
                for step_index, step in enumerate(input.steps):
                    _append_step(step, trajectory_uid, step_index)
                    trajectory_uids.append(trajectory_uid)
                    step_indices.append(step_index)

        prompt_ids = torch.cat(prompt_ids, dim=0)
        response_ids = torch.cat(response_ids, dim=0)
        response_mask = torch.cat(response_mask, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)
        input_ids = torch.cat(input_ids, dim=0)
        position_ids = torch.cat(position_ids, dim=0)

        optional_outputs = {}
        if all(logprobs is not None for logprobs in response_logprobs_list):
            optional_outputs["rollout_log_probs"] = torch.cat(response_logprobs_list, dim=0)
        if all(routed_experts is not None for routed_experts in routed_experts_list):
            optional_outputs["routed_experts"] = torch.cat(routed_experts_list, dim=0)

        batch = TensorDict(
            {
                "prompts": prompt_ids,
                "responses": response_ids,
                "response_mask": response_mask,
                "attention_mask": attention_mask,
                "input_ids": input_ids,
                "position_ids": position_ids,
                **optional_outputs,
            },
            batch_size=prompt_ids.size(0),
        )

        if all(reward_tensor is not None for reward_tensor in reward_tensors):
            reward_tensor = torch.cat(reward_tensors, dim=0)
            batch["rm_scores"] = reward_tensor

        non_tensor_batch = {
            "trajectory_uids": np.array(trajectory_uids, dtype=object),
            "step_indices": np.array(step_indices, dtype=np.int32),
            "trajectory_num_steps": np.array(trajectory_num_steps, dtype=np.int32),
            "__num_turns__": np.array(num_turns, dtype=np.int32),
        }

        all_reward_keys = set()
        for info in reward_extra_infos:
            all_reward_keys.update(info.keys())
        reward_extra_keys = sorted(all_reward_keys)
        for key in reward_extra_keys:

            vals = [info.get(key) for info in reward_extra_infos]
            arr = np.empty(len(vals), dtype=object)
            for _i, _v in enumerate(vals):
                arr[_i] = _v
            non_tensor_batch[key] = arr

        if any(mmi is not None for mmi in multi_modal_inputs):
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs, dtype=object)

        metrics = [input.metrics.model_dump() for input in inputs]

        for i, metric in enumerate(metrics):
            metric["num_steps"] = num_steps[i]
            metric["trajectory_num_steps"] = trajectory_num_steps[i]

        extra_fields = {}
        all_keys = set(
            key
            for input_item in inputs
            if input_item.steps
            for key in input_item.steps[-1].extra_fields
            if key != "reward_extra_info"  
        )
        for key in all_keys:
            temp_list = []
            for input_item in inputs:
                if input_item.steps:
                    temp_list.append(input_item.steps[-1].extra_fields.get(key))
            arr = np.empty(len(temp_list), dtype=object)
            for _i, _v in enumerate(temp_list):
                arr[_i] = _v
            extra_fields[key] = arr

        non_tensor_batch.update(extra_fields)
        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info={"metrics": metrics, "reward_extra_keys": reward_extra_keys},
        )

    def create_transferqueue_client(
        self,
    ):
        """Create a client for data system (TransferQueue)."""
        from verl.single_controller.ray.base import get_random_string

        client_name = get_random_string(length=6)

        self.tq_client = build_transferqueue_client(
            client_id=f"AgentLoopWorker_{client_name}",
            config=self.config.transfer_queue,
        )

@ray.remote
class AgentFlowWorker(AgentFlowWorkerBase):
    """Agent flow worker takes a batch of messages and run each message in an agent flow."""

    def __init__(
        self,
        config: DictConfig,
        servers: list[tuple[str, ray.actor.ActorHandle]],
        load_balancer_handle: ray.actor.ActorHandle,
        reward_router_address: str = None,
    ):
        """Initialize agent flow manager.
        Args:
            config (DictConfig): YAML config.
            servers (List[Tuple[str, ray.actor.ActorHandle]]): OpenAI compatible LLM server address/handle pairs.
            reward_router_address (str): reward router address.
        """
        super().__init__(config, servers, load_balancer_handle, reward_router_address)

async def get_trajectory_info(step, index, validate):
    """Build trajectory metadata from trainer step, dataset indices, and validation mode."""
    trajectory_info = []
    rollout_n = 0
    for i in range(len(index)):
        if i > 0 and index[i - 1] == index[i]:
            rollout_n += 1
        else:
            rollout_n = 0
        trajectory_info.append({"step": step, "sample_index": index[i], "rollout_n": rollout_n, "validate": validate})
    return trajectory_info

class AgentFlowManager:
    """Agent flow manager that manages a group of agent flow workers."""

    def __init__(
        self, config: DictConfig, worker_group: RayWorkerGroup = None, rm_resource_pool: RayResourcePool = None
    ):
        """Initialize agent flow manager.

        Args:
            config (DictConfig): trainer config.
            worker_group (RayWorkerGroup): ActorRolloutRef worker group for hybrid mode; None for standalone mode.
            rm_resource_pool (RayResourcePool): Resource pool for reward model (Standalone mode).
        """
        self.config = config
        self.worker_group = worker_group
        self.reward_model_manager = None
        self.reward_router_address = None
        if self.config.reward_model.enable:
            from verl.experimental.reward_loop import RewardModelManager

            self.reward_model_manager = RewardModelManager(config.reward_model, rm_resource_pool)
            self.reward_router_address = self.reward_model_manager.get_router_address()

        if not hasattr(self, "rollout_replica_class"):
            self.rollout_replica_class = get_rollout_replica_class(self.config.actor_rollout_ref.rollout.name)
        if not hasattr(self, "agent_flow_workers_class"):
            self.agent_flow_workers_class = AgentFlowWorker

        try:
            self._initialize_llm_servers()
            self._init_agent_flow_workers()

            if self._should_manage_rollout_cache():
                self.sleep()
        except BaseException:

            from paraagent.training.shutdown import close_rollout_manager

            try:
                close_rollout_manager(self)
            except Exception:
                logging.getLogger(__name__).exception("Cleanup after rollout initialization failed")
            raise

    def _should_manage_rollout_cache(self) -> bool:
        """Manage rollout sleep/wake only when the backend supports explicit calls.
        Hybrid engines wake through weight updates.
        """
        return self.config.actor_rollout_ref.rollout.free_cache_engine and not self.config.actor_rollout_ref.hybrid_engine

    def _initialize_llm_servers(self):
        rollout_world_size = (
            self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
            * self.config.actor_rollout_ref.rollout.data_parallel_size
            * self.config.actor_rollout_ref.rollout.pipeline_model_parallel_size
        )
        world_size = (
            self.worker_group.world_size
            if self.worker_group
            else self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        )
        num_replicas = world_size // rollout_world_size

        rollout_config = self.config.actor_rollout_ref.rollout
        model_config = self.config.actor_rollout_ref.model
        self.rollout_replicas = [
            self.rollout_replica_class(
                replica_rank=replica_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=self.config.trainer.n_gpus_per_node,
            )
            for replica_rank in range(num_replicas)
        ]
        if self.worker_group:
            self._run_all([server.init_hybrid(self.worker_group) for server in self.rollout_replicas])
        else:
            self._run_all([server.init_standalone() for server in self.rollout_replicas])
        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]
        self.global_load_balancer = GlobalRequestLoadBalancer.remote(server_actor_ids=self.server_addresses)

        print(f"AgentFlowManager: {self.server_addresses}")

        if rollout_config.prometheus.enable:
            if rollout_config.disable_log_stats:
                raise ValueError("PROMETHEUS needs disable_log_stats==False, but it is currently True.")
            update_prometheus_config(rollout_config.prometheus, self.server_addresses)

    def _init_agent_flow_workers(self):
        self.agent_flow_workers = []
        num_workers = self.config.actor_rollout_ref.rollout.agent.num_workers
        load_balancer_handle = self.global_load_balancer
        servers = list(zip(self.server_addresses, self.server_handles, strict=True))

        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        for i in range(num_workers):
            
            node_id = node_ids[i % len(node_ids)]
            self.agent_flow_workers.append(
                self.agent_flow_workers_class.options(
                    name=f"agent_flow_worker_{i}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(self.config, servers, load_balancer_handle, self.reward_router_address)
            )

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """

        if self._should_manage_rollout_cache():
            self.wake_up()
        if self.reward_model_manager:
            self.reward_model_manager.wake_up()

        num_workers = min(len(self.agent_flow_workers), len(prompts))
        split_size = (len(prompts) - 1) // num_workers + 1
        chunks = prompts.split(split_size)
        outputs = ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_flow_workers, chunks)
            ]
        )
        _harmonize_reward_extra_keys(outputs)
        output = DataProto.concat(outputs)
        if self._should_manage_rollout_cache():
            self.sleep()
        if self.reward_model_manager:
            self.reward_model_manager.sleep()

        metrics = [output.meta_info.pop("metrics") for output in outputs]  

        num_steps = [metric["num_steps"] for chunk in metrics for metric in chunk]
        timing = self._performance_metrics(metrics, num_steps, output)

        output.meta_info = {"timing": timing, "num_steps": num_steps, **outputs[0].meta_info}
        return output

    def _performance_metrics(
        self, metrics: list[list[dict[str, str]]], num_steps: list[int], output: DataProto
    ) -> dict[str, float]:
        timing = {}

        flat_metrics = [metric for chunk in metrics for metric in chunk]
        step_metrics = []
        for metric in flat_metrics:
            entries = metric.get("step_metrics") or []
            if entries:
                step_metrics.extend(entries)
            else:
                step_metrics.append(metric)

        t_generate_sequences = np.array([m.get("generate_sequences", 0.0) for m in step_metrics], dtype=float)
        t_tool_calls = np.array([m.get("tool_calls", 0.0) for m in step_metrics], dtype=float)

        timing["agent_flow/step/generate_sequences/min"] = t_generate_sequences.min()
        timing["agent_flow/step/generate_sequences/max"] = t_generate_sequences.max()
        timing["agent_flow/step/generate_sequences/mean"] = t_generate_sequences.mean()
        timing["agent_flow/step/tool_calls/min"] = t_tool_calls.min()
        timing["agent_flow/step/tool_calls/max"] = t_tool_calls.max()
        timing["agent_flow/step/tool_calls/mean"] = t_tool_calls.mean()

        trajectory_generate_times = np.array([m.get("generate_sequences", 0.0) for m in flat_metrics], dtype=float)
        trajectory_tool_times = np.array([m.get("tool_calls", 0.0) for m in flat_metrics], dtype=float)
        trajectory_reward_times = np.array([m.get("reward_judge", 0.0) for m in flat_metrics], dtype=float)
        trajectory_total_times = trajectory_generate_times + trajectory_tool_times + trajectory_reward_times
        trajectory_real_steps = [
            int(m.get("trajectory_num_steps", m.get("num_steps", n)))
            for m, n in zip(flat_metrics, num_steps, strict=False)
        ]

        extra_step_keys = sorted(
            {
                key
                for entry in step_metrics
                for key in entry
                if key not in {"generate_sequences", "tool_calls", "reward_judge"}
            }
        )
        for key in extra_step_keys:
            vals = np.array([entry.get(key, 0.0) for entry in step_metrics], dtype=float)
            timing[f"agent_flow/step/{key}/mean"] = vals.mean()
        extra_trajectory_keys = sorted(
            {
                key
                for metric in flat_metrics
                for step_entry in (metric.get("step_metrics") or [])
                for key in step_entry
                if key not in {"generate_sequences", "tool_calls", "reward_judge"}
            }
        )
        for key in extra_trajectory_keys:
            vals = []
            for metric in flat_metrics:
                vals.append(sum(float(step.get(key, 0.0)) for step in (metric.get("step_metrics") or [])))
            timing[f"agent_flow/trajectory/{key}/mean"] = float(np.mean(vals)) if vals else 0.0

        timing["agent_flow/trajectory/generate_sequences/min"] = trajectory_generate_times.min()
        timing["agent_flow/trajectory/generate_sequences/max"] = trajectory_generate_times.max()
        timing["agent_flow/trajectory/generate_sequences/mean"] = trajectory_generate_times.mean()
        timing["agent_flow/trajectory/tool_calls/min"] = trajectory_tool_times.min()
        timing["agent_flow/trajectory/tool_calls/max"] = trajectory_tool_times.max()
        timing["agent_flow/trajectory/tool_calls/mean"] = trajectory_tool_times.mean()
        timing["agent_flow/trajectory/reward_judge/min"] = trajectory_reward_times.min()
        timing["agent_flow/trajectory/reward_judge/max"] = trajectory_reward_times.max()
        timing["agent_flow/trajectory/reward_judge/mean"] = trajectory_reward_times.mean()
        timing["agent_flow/trajectory/total/min"] = trajectory_total_times.min()
        timing["agent_flow/trajectory/total/max"] = trajectory_total_times.max()
        timing["agent_flow/trajectory/total/mean"] = trajectory_total_times.mean()
        timing["agent_flow/trajectory/num_steps/mean"] = float(np.mean(trajectory_real_steps))

        slowest_traj_idx = np.argmax(trajectory_total_times)
        
        slowest_step_start_idx = sum(num_steps[:slowest_traj_idx])
        slowest_step_end_idx = slowest_step_start_idx + num_steps[slowest_traj_idx]

        prompt_length = output.batch["prompts"].shape[1]
        total_prompt_length = 0
        total_response_length = 0
        for step_idx in range(slowest_step_start_idx, slowest_step_end_idx):
            attention_mask = output.batch["attention_mask"][step_idx]
            total_prompt_length += attention_mask[:prompt_length].sum().item()
            total_response_length += attention_mask[prompt_length:].sum().item()

        timing["agent_flow/slowest/num_steps"] = trajectory_real_steps[slowest_traj_idx]
        timing["agent_flow/slowest/total_prompt_length"] = total_prompt_length
        timing["agent_flow/slowest/total_response_length"] = total_response_length

        return timing

    def wake_up(self):
        """Wake up all rollout replica instances."""
        self._run_all([replica.wake_up() for replica in self.rollout_replicas])

    def sleep(self):
        """Sleep all rollout replica instances."""
        self._run_all([replica.sleep() for replica in self.rollout_replicas])

    def clear_kv_cache(self):
        """Clear all rollout kv cache, but don`t sleep."""
        self._run_all([replica.clear_kv_cache() for replica in self.rollout_replicas])

    def _run_all(self, tasks: list[asyncio.Task]):
        async def run_all():
            await asyncio.gather(*tasks)

        asyncio.run(run_all())
