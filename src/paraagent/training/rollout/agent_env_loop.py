import json
import logging
import os
import threading
from typing import Any
from uuid import uuid4

from paraagent.training.rollout.agent_flow import (
    AgentFlowBase,
    AgentFlowOutput,
    AgentFlowStep,
    _normalize_reward_extra_info,
    register,
)
from paraagent.toolenv.env import AgentEnv
from paraagent.toolenv.env.base import Action, Observation
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_TURN_BREAK_CACHE: dict[tuple, tuple[list[int], list[int]]] = {}
_TURN_BREAK_CACHE_LOCK = threading.Lock()


@register("agent_env_loop")
class AgentEnvLoop(AgentFlowBase):
    """Run agent/environment turns as independent AgentFlowStep objects.

    Merge global environment defaults with per-sample env_kwargs (sample wins),
    then select the registered environment using env_type.
    """

    def __init__(self, *args, env_kwargs: dict[str, Any] | None = None, **kwargs):
        base_kwarg_names = {
            "trainer_config",
            "server_manager",
            "reward_loop_worker",
            "tokenizer",
            "processor",
            "dataset_cls",
            "dataset_config",
        }
        constructor_env_kwargs = {k: v for k, v in kwargs.items() if k not in base_kwarg_names}
        if env_kwargs is not None:
            if isinstance(env_kwargs, str):
                env_kwargs = json.loads(env_kwargs)
            if hasattr(env_kwargs, "items"):
                constructor_env_kwargs.update(dict(env_kwargs))
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.max_model_len = self.config.actor_rollout_ref.rollout.max_model_len
        self.max_trajectory_response_len = int(
            getattr(getattr(self.config, "data", None), "max_response_length", 0)
            or getattr(self.config.actor_rollout_ref.rollout, "max_model_len", 0)
            or self.response_length
        )
        self.max_steps: int = self.config.actor_rollout_ref.rollout.agent.get("max_steps", 10)
        self.skip_special_tokens: bool = self.config.actor_rollout_ref.rollout.agent.get(
            "skip_special_tokens", True
        )
        config_env_kwargs = self.config.actor_rollout_ref.rollout.agent.get("env_kwargs") or {}
        if hasattr(config_env_kwargs, "items"):
            config_env_kwargs = dict(config_env_kwargs)
        self.env_kwargs: dict[str, Any] = {**config_env_kwargs, **constructor_env_kwargs}

        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is None or im_end_id == self.tokenizer.unk_token_id:
            raise ValueError(
                "AgentEnvLoop requires a Qwen-style chat template that uses the "
                "<|im_end|> special token. Detected tokenizer does not recognise it."
            )
        self._im_end_id = int(im_end_id)

    def _ensure_turn_break_template(self) -> tuple[list[int], list[int]]:
        """Compute the token-exact wrapper between assistant generations.
        Use the generation prompt plus raw response tokens as the diff baseline.
        """
        kwargs = dict(self.apply_chat_template_kwargs)
        cache_key = (id(self.tokenizer), tuple(sorted(kwargs.items())))
        cached = _TURN_BREAK_CACHE.get(cache_key)
        if cached is not None:
            return cached

        with _TURN_BREAK_CACHE_LOCK:
            cached = _TURN_BREAK_CACHE.get(cache_key)
            if cached is not None:
                return cached

            placeholder = "☃PLACEHOLDER_TOOL_RESPONSE_TEXT☃"
            placeholder_ids = self.tokenizer.encode(placeholder, add_special_tokens=False)
            if not placeholder_ids:
                raise RuntimeError("Tokenizer produced empty ids for the wrapper placeholder.")

            user_msg = {"role": "user", "content": "u"}
            assistant_text = "a"
            ids_before = list(
                self.tokenizer.apply_chat_template(
                    [user_msg],
                    add_generation_prompt=True,
                    tokenize=True,
                    **kwargs,
                )
            ) + self.tokenizer.encode(assistant_text, add_special_tokens=False)
            ids_after = list(
                self.tokenizer.apply_chat_template(
                    [
                        user_msg,
                        {"role": "assistant", "content": assistant_text},
                        {"role": "user", "content": placeholder},
                    ],
                    add_generation_prompt=True,
                    tokenize=True,
                    **kwargs,
                )
            )
            if len(ids_after) <= len(ids_before) or ids_after[: len(ids_before)] != ids_before:
                raise RuntimeError(
                    "Chat-template output for [user, generated assistant, user] is not a strict "
                    "extension of [user generation prompt + generated assistant tokens]; "
                    "cannot derive a token-exact tool-turn wrapper."
                )

            delta = ids_after[len(ids_before) :]
            n = len(placeholder_ids)
            split_at = -1
            for i in range(len(delta) - n + 1):
                if delta[i : i + n] == placeholder_ids:
                    split_at = i
                    break
            if split_at < 0:
                raise RuntimeError(
                    "Could not locate placeholder ids inside the chat-template delta; "
                    "tokenizer must be merging across the placeholder boundary. "
                    "Try a different placeholder string."
                )

            prefix_ids = list(delta[:split_at])
            suffix_ids = list(delta[split_at + n :])
            _TURN_BREAK_CACHE[cache_key] = (prefix_ids, suffix_ids)
            return prefix_ids, suffix_ids

    def _build_tool_turn_ids(self, tool_text: str, prev_response_ids: list[int]) -> list[int]:
        """Wrap tool observations using the detected chat-template turn boundary.

        Avoid duplicating im_end when the previous response already contains it.
        """
        prefix_ids, suffix_ids = self._ensure_turn_break_template()

        if (
            prev_response_ids
            and prev_response_ids[-1] == self._im_end_id
            and prefix_ids
            and prefix_ids[0] == self._im_end_id
        ):
            prefix_ids = prefix_ids[1:]

        tool_ids = self.tokenizer.encode(tool_text, add_special_tokens=False)
        return list(prefix_ids) + list(tool_ids) + list(suffix_ids)

    @staticmethod
    def _extract_new_tool_text(next_obs: Observation) -> str | None:
        """Return the trailing user message appended by env.step, or empty text otherwise."""
        messages = getattr(next_obs, "messages", None)
        if not messages:
            return None
        last = messages[-1]
        if not isinstance(last, dict) or last.get("role") != "user":
            return None
        content = last.get("content")
        return content if isinstance(content, str) else None

    def _create_env(self, **kwargs) -> AgentEnv:
        """Create a trajectory environment from merged global and per-sample env_kwargs."""

        env_kwargs = kwargs.get("env_kwargs") or {}
        if isinstance(env_kwargs, str):
            env_kwargs = json.loads(env_kwargs)
        if not hasattr(env_kwargs, "items"):
            raise TypeError(f"env_kwargs must be a mapping or JSON object, got {type(env_kwargs).__name__}")

        merged = {**self.env_kwargs, **dict(env_kwargs)}

        env_type = merged.pop("env_type", None)
        if not isinstance(env_type, str) or not env_type:
            raise ValueError(
                "Missing required env_kwargs.env_type in both the agent-flow "
                "configuration and the dataset row"
            )
        return AgentEnv.from_config(env_type, **merged)

    def _is_max_model_len_reached(self, prompt_ids: list[int], response_ids: list[int]) -> bool:
        return len(prompt_ids) + len(response_ids) >= self.max_model_len

    def _should_use_trajectory_reward(self, env: AgentEnv) -> bool:
        return env.use_trajectory_reward and self.custom_reward_fn is not None

    def _extract_ground_truth(self, **kwargs) -> Any:
        reward_model = kwargs.get("reward_model", {})
        if isinstance(reward_model, str):
            try:
                reward_model = json.loads(reward_model)
            except json.JSONDecodeError:
                reward_model = {}
        if isinstance(reward_model, dict):
            return reward_model.get("ground_truth")
        return None

    def _build_trajectory_reward_extra_info(
        self,
        steps: list[AgentFlowStep],
        *,
        env: AgentEnv | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        base_extra_info = kwargs.get("extra_info", {})
        if isinstance(base_extra_info, str):
            try:
                base_extra_info = json.loads(base_extra_info)
            except json.JSONDecodeError:
                base_extra_info = {"raw_extra_info": base_extra_info}
        if not isinstance(base_extra_info, dict):
            base_extra_info = {"raw_extra_info": base_extra_info}
        if kwargs.get("_agent_validate") and base_extra_info.get("split") in (None, ""):
            base_extra_info["split"] = "val"

        trajectory_steps = []
        for step in steps:
            trajectory_steps.append(
                {
                    "step_index": step.extra_fields.get("step_index"),
                    "response_text": step.extra_fields.get("response_text", ""),
                    "done": step.extra_fields.get("done", False),
                    "env_info": step.extra_fields.get("trajectory_step_info", {}),
                }
            )

        final_env_info = trajectory_steps[-1]["env_info"] if trajectory_steps else {}
        reward_extra_info = {
            **base_extra_info,
            "question": base_extra_info.get("question") or kwargs.get("prompt"),
            "trajectory_steps": trajectory_steps,
            "trajectory_tool_context": final_env_info.get(
                "visible_trajectory_tool_context",
                final_env_info.get("trajectory_tool_context", ""),
            ),
            "num_steps": len(trajectory_steps),
        }

        getter = getattr(env, "get_trajectory_reward_extra_info", None)
        if callable(getter):
            try:
                env_reward_extra_info = getter()
            except Exception as exc:
                logger.exception("Failed to capture environment reward state")
                env_reward_extra_info = {
                    "tau_state_capture_ok": False,
                    "tau_gt_replay_ok": False,
                    "tau_state_capture_error": f"{type(exc).__name__}: {exc}",
                }
            if isinstance(env_reward_extra_info, dict):
                reward_extra_info.update(env_reward_extra_info)
        return reward_extra_info

    def _should_build_trajectory_solution_str(self) -> bool:
        """Build concatenated solution_str for custom rewards that need it.
        The ToolEnv reward consumes trajectory_steps directly.
        """
        fn = self.custom_reward_fn
        if fn is None:
            return False
        return not (
            getattr(fn, "__module__", "") == "paraagent.rewards.tool_reward"
            and getattr(fn, "__name__", "") == "compute_score"
        )

    def _summarize_step_info(self, info: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(info, dict):
            return {"raw_info_type": type(info).__name__}

        summary = {
            "env_action_type": info.get("env_action_type"),
            "plan_update_valid": info.get("plan_update_valid"),
            "search_phase_adherence": info.get("search_phase_adherence"),
            "tool_phase_adherence": info.get("tool_phase_adherence"),
            "num_successful_searches": info.get("num_successful_searches"),
            "num_successful_tool_calls": info.get("num_successful_tool_calls"),
        }
        for key in ("retrieval_success", "retrieval_nonempty", "tool_success", "tool_error_types"):
            if key in info:
                summary[key] = info.get(key)
        return summary

    async def _obs_to_prompt(
        self, obs: Observation, tools: list[dict] | None = None
    ) -> tuple[list[int], dict]:
        """Convert observation token_ids, messages, or text into prompt token IDs.

        Messages use the chat template with optional tool schemas; text uses
        the raw tokenizer.
        """
        if obs.token_ids is not None:
            return obs.token_ids
        if obs.messages is not None:
            prompt_ids = await self.apply_chat_template(
                obs.messages,
            )
            return prompt_ids
        if obs.text is not None:
            prompt_ids = self.tokenizer.encode(obs.text)
            return prompt_ids
        raise ValueError("Observation must have at least one field set")

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentFlowOutput:
        """Run the agent-environment interaction loop.

        Args:
            sampling_params (dict[str, Any]): LLM sampling parameters.
            **kwargs: Dataset fields from ``verl.utils.dataset.RLHFDataset``.

        Returns:
            AgentFlowOutput: Output containing one ``AgentFlowStep`` per turn.
        """
        env = self._create_env(**kwargs)
        try:
            return await self._run_inner(env, sampling_params, **kwargs)
        finally:
            env.cleanup()

    async def _run_inner(self, env: AgentEnv, sampling_params: dict[str, Any], **kwargs) -> AgentFlowOutput:
        obs = await env.reset(**kwargs)
        tools = getattr(env, "tool_schemas", None)
        use_trajectory_reward = self._should_use_trajectory_reward(env)

        trajectory_request_id = uuid4().hex

        if logger.isEnabledFor(logging.DEBUG):
            if isinstance(kwargs.get("prompt"), list):
                logger.debug("AgentEnvLoop raw dataset prompt: %s", kwargs.get("prompt"))
            if getattr(obs, "messages", None) is not None:
                logger.debug("AgentEnvLoop reset obs.messages: %s", obs.messages)

        steps: list = []
        metrics = {"generate_sequences": 0.0, "tool_calls": 0.0, "reward_judge": 0.0, "step_metrics": []}

        initial_prompt_ids = await self._obs_to_prompt(obs, tools=tools)
        cum_token_ids: list[int] = list(initial_prompt_ids)
        cum_response_mask: list[int] = [0] * len(cum_token_ids)
        init_prompt_len = len(cum_token_ids)
        visible_trajectory_tool_context = ""
        stop_reason = "max_steps"
        skip_loop = len(initial_prompt_ids) > self.prompt_length
        if skip_loop:
            logger.warning(
                "Initial prompt length (%d) exceeds configured prompt_length (%d). "
                "Stopping rollout before any generation; dummy step will be emitted.",
                len(initial_prompt_ids),
                self.prompt_length,
            )
            stop_reason = "initial_prompt_too_long"

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "AgentEnvLoop initial prompt len=%d head=%s tail=%s",
                init_prompt_len,
                cum_token_ids[:64],
                cum_token_ids[-64:] if init_prompt_len > 64 else cum_token_ids,
            )

        for step_idx in range(0 if skip_loop else self.max_steps):
            step_metrics = {}

            trajectory_response_len = len(cum_token_ids) - init_prompt_len
            if trajectory_response_len >= self.max_trajectory_response_len:
                logger.warning(
                    "Cumulative trajectory response length (%d) reaches max_response_length (%d) at step %d. "
                    "Stopping rollout early because training packs the full untruncated trajectory.",
                    trajectory_response_len,
                    self.max_trajectory_response_len,
                    step_idx,
                )
                stop_reason = "response_overflow"
                break

            if len(cum_token_ids) > self.prompt_length:
                logger.warning(
                    "Prompt length (%d) exceeds configured prompt_length (%d) at step %d. "
                    "Stopping rollout at the configured per-step prompt boundary.",
                    len(cum_token_ids),
                    self.prompt_length,
                    step_idx,
                )
                stop_reason = "prompt_overflow"
                break

            prompt_ids = list(cum_token_ids)

            if len(prompt_ids) >= self.max_model_len:
                logger.warning(
                    "Prompt length (%d) reaches configured max_model_len (%d) at step %d. "
                    "Stopping rollout early because there is no room for response tokens.",
                    len(prompt_ids),
                    self.max_model_len,
                    step_idx,
                )
                stop_reason = "max_model_len"
                break

            step_max_tokens = min(
                self.response_length,
                self.max_model_len - len(prompt_ids),
                self.max_trajectory_response_len - trajectory_response_len,
            )
            if step_max_tokens <= 0:
                logger.warning(
                    "No response budget left at step %d: prompt_len=%d max_model_len=%d "
                    "trajectory_response_len=%d max_response_length=%d.",
                    step_idx,
                    len(prompt_ids),
                    self.max_model_len,
                    trajectory_response_len,
                    self.max_trajectory_response_len,
                )
                stop_reason = "response_overflow"
                break

            with simple_timer("generate_sequences", step_metrics):
                output = await self.server_manager.generate(
                    request_id=trajectory_request_id,
                    prompt_ids=prompt_ids,
                    sampling_params={**sampling_params, "max_tokens": step_max_tokens},
                )

            response_ids = output.token_ids[: self.response_length]
            if not response_ids:
                logger.warning(
                    "AgentEnvLoop step %d: vLLM returned empty response_ids; "
                    "step will produce no agent tokens.",
                    step_idx,
                )

            response_text = await self.loop.run_in_executor(
                None,
                lambda _ids=response_ids: self.tokenizer.decode(
                    _ids, skip_special_tokens=self.skip_special_tokens
                ),
            )

            if step_idx <= 5 and logger.isEnabledFor(logging.DEBUG):
                logger.debug("AgentEnvLoop step%d response_text: %s", step_idx, response_text)

            action = Action(text=response_text, token_ids=response_ids)

            with simple_timer("tool_calls", step_metrics):
                next_obs, reward, done, info = await env.step(action)
            if isinstance(info, dict):
                env_timing_ms = info.get("env_timing_ms")
                if isinstance(env_timing_ms, dict):
                    for timing_name, timing_value in env_timing_ms.items():
                        try:
                            step_metrics[f"env/{timing_name}"] = float(timing_value) / 1000.0
                        except (TypeError, ValueError):
                            continue
                for count_name in ("env_num_search_queries", "env_num_tool_calls"):
                    try:
                        step_metrics[count_name] = float(info.get(count_name, 0.0))
                    except (TypeError, ValueError):
                        continue
            metrics["generate_sequences"] += float(step_metrics.get("generate_sequences", 0.0))
            metrics["tool_calls"] += float(step_metrics.get("tool_calls", 0.0))
            metrics["step_metrics"].append(dict(step_metrics))

            max_model_len_reached = self._is_max_model_len_reached(prompt_ids, response_ids)
            if max_model_len_reached:
                if not isinstance(info, dict):
                    info = {"raw_info": info}
                info.update(
                    {
                        "max_model_len_reached": True,
                        "rollout_stop_reason": "max_model_len",
                        "model_len": len(prompt_ids) + len(response_ids),
                        "max_model_len": self.max_model_len,
                    }
                )
                done = True
                stop_reason = "max_model_len"
            elif done:
                stop_reason = "natural_done"

            tool_text = None
            will_append_observation = False
            if not done:
                tool_text = self._extract_new_tool_text(next_obs)
                will_append_observation = tool_text is not None
            if isinstance(info, dict):
                info["observation_visible_to_model"] = bool(will_append_observation)
                if info.get("env_action_type") == "tool_call":
                    info["tool_observation_visible_to_model"] = bool(will_append_observation)
                info["visible_trajectory_tool_context"] = visible_trajectory_tool_context

            if step_idx <= 5 and logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "AgentEnvLoop step%d outcome: done=%s reward=%s info=%s",
                    step_idx,
                    done,
                    reward,
                    self._summarize_step_info(info),
                )

            step = AgentFlowStep(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_logprobs=(output.log_probs[: len(response_ids)] if output.log_probs else None),
                routed_experts=(
                    output.routed_experts[: len(response_ids)] if output.routed_experts is not None else None
                ),
                reward_score=0.0 if use_trajectory_reward and reward is None else reward,
                num_turns=2 * step_idx + 3,
                extra_fields={
                    "trajectory_step_info": info,
                    "response_text": response_text,
                    "step_index": step_idx,
                    "done": done,
                },
            )
            step = await self._postprocess(step, **kwargs)
            steps.append(step)

            cum_token_ids.extend(response_ids)
            cum_response_mask.extend([1] * len(response_ids))

            if done:
                break

            if tool_text is not None:
                tool_turn_ids = await self.loop.run_in_executor(
                    None,
                    lambda _txt=tool_text, _resp=response_ids: self._build_tool_turn_ids(_txt, _resp),
                )
                next_response_len = len(cum_token_ids) + len(tool_turn_ids) - init_prompt_len
                next_model_len = len(cum_token_ids) + len(tool_turn_ids)
                if (
                    next_response_len > self.max_trajectory_response_len
                    or next_model_len > self.prompt_length
                    or next_model_len >= self.max_model_len
                ):
                    if isinstance(step.extra_fields.get("trajectory_step_info"), dict):
                        step.extra_fields["trajectory_step_info"]["observation_visible_to_model"] = False
                        if step.extra_fields["trajectory_step_info"].get("env_action_type") == "tool_call":
                            step.extra_fields["trajectory_step_info"]["tool_observation_visible_to_model"] = (
                                False
                            )
                    if next_response_len > self.max_trajectory_response_len:
                        stop_reason = "response_overflow"
                    elif next_model_len > self.prompt_length:
                        stop_reason = "prompt_overflow"
                    else:
                        stop_reason = "max_model_len"
                    logger.warning(
                        "Stopping before appending tool observation at step %d: next_response_len=%d "
                        "max_response_length=%d next_model_len=%d prompt_length=%d max_model_len=%d.",
                        step_idx,
                        next_response_len,
                        self.max_trajectory_response_len,
                        next_model_len,
                        self.prompt_length,
                        self.max_model_len,
                    )
                    break
                cum_token_ids.extend(tool_turn_ids)
                cum_response_mask.extend([0] * len(tool_turn_ids))
                if isinstance(step.extra_fields.get("trajectory_step_info"), dict):
                    visible_trajectory_tool_context = step.extra_fields["trajectory_step_info"].get(
                        "trajectory_tool_context",
                        visible_trajectory_tool_context,
                    )
                    step.extra_fields["trajectory_step_info"]["visible_trajectory_tool_context"] = (
                        visible_trajectory_tool_context
                    )
            else:
                logger.error(
                    "AgentEnvLoop step %d: env returned done=False but appended no user "
                    "message; ending trajectory early to keep token buffer consistent. "
                    "Check env.step() implementation.",
                    step_idx,
                )
                stop_reason = "tool_text_missing"
                break

            obs = next_obs

        if steps:
            steps[-1].extra_fields["rollout_stop_reason"] = stop_reason

        dummy_step_added = False
        if not steps:
            logger.error(
                "Agent flow produced zero steps. This typically means the initial "
                "prompt exceeded prompt_length. Generating a dummy step."
            )
            prompt_ids = initial_prompt_ids[: self.prompt_length]
            eos_id = self.tokenizer.eos_token_id or 0
            dummy_step = AgentFlowStep(
                prompt_ids=prompt_ids,
                response_ids=[eos_id],
                reward_score=-1.0,
                num_turns=1,
                extra_fields={
                    "response_text": "",
                    "step_index": 0,
                    "done": True,
                    "rollout_stop_reason": stop_reason or "initial_prompt_too_long",
                    "reward_extra_info": _normalize_reward_extra_info(-1.0),
                },
            )
            dummy_step = await self._postprocess(dummy_step, **kwargs)
            steps.append(dummy_step)
            dummy_step_added = True

            cum_token_ids = list(prompt_ids) + [eos_id]
            cum_response_mask = [0] * len(prompt_ids) + [1]
            init_prompt_len = len(prompt_ids)

        pending_trajectory_reward = None
        if use_trajectory_reward and steps and not dummy_step_added:
            solution_str = (
                env.build_trajectory_solution_str(steps)
                if self._should_build_trajectory_solution_str()
                else ""
            )
            pending_trajectory_reward = {
                "data_source": kwargs.get("data_source"),
                "solution_str": solution_str,
                "ground_truth": self._extract_ground_truth(**kwargs),
                "extra_info": self._build_trajectory_reward_extra_info(
                    steps,
                    env=env,
                    **kwargs,
                ),
            }

            for step in steps[:-1]:
                step.reward_score = 0.0
            steps[-1].reward_score = 0.0

        return AgentFlowOutput(
            steps=steps,
            metrics=metrics,
            cum_token_ids=cum_token_ids,
            cum_response_mask=cum_response_mask,
            init_prompt_len=init_prompt_len,
            pending_trajectory_reward=pending_trajectory_reward,
        )
