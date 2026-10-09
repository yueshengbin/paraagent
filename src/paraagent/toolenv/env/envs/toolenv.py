import json
import logging
import os
import re
import time
import uuid
from copy import deepcopy
from collections import Counter
from typing import Any

from paraagent.paraact.protocol import (
    MAX_ACCEPTED_PLANS_PER_PHASE,
    PLAN_CONTROLLER_LIFECYCLE_VERSION,
    action_calls_follow_flow,
    action_controller_progress,
    action_plan_dependencies_valid,
    action_plan_dependency_edges,
    action_plan_format_valid,
    action_plan_frontier,
    action_plan_occurrence_tools,
    action_plan_stage_specs,
    classify_plan_requirement,
    classify_plan_refresh,
    has_explicit_recovery_trigger,
    make_call_key,
    match_action_calls_to_occurrences,
    max_refreshes_for,
    serialize_call_keys,
    search_plan_format_valid,
)
from paraagent.toolenv.runtime.tools.toolenv_simulator import ToolEnvSimulatorTool
from paraagent.toolenv.runtime.tools.tool_retrieval import (
    AUTONOMOUS_RETAIL_TOOL_DESCRIPTIONS,
    ToolSearchTool,
)
from paraagent.toolenv.outcome import (
    BASIS_LOCAL_VALIDATION,
    CATEGORY_INVALID_REQUEST,
    CATEGORY_TOOL_NOT_FOUND,
    RETRY_NEVER,
    STAGE_DISPATCH,
    STAGE_REQUEST,
    make_tool_outcome,
)
from paraagent.toolenv.runtime.utils import (
    get_tool_display_name,
    is_business_level_tool_failure,
    is_effective_tool_observation,
    normalize_tool_name,
)

from ..base import Action, AgentEnv, Observation

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_TOOL_DESC_MAX_CHARS = 160

_DESC_WRAPPER_RE = re.compile(r'The description of this function is:\s*"(.*)"\s*$', re.DOTALL)
_FIRST_SENTENCE_RE = re.compile(r"[.!?。！？]")


@AgentEnv.register("toolenv")
class ToolEnv(AgentEnv):
    """Run tool retrieval and execution through the AgentEnv API.

    The loop is:
    1. model emits one or more ``<search_tool>...</search_tool>``
    2. env returns ``<tools>...</tools>``
    3. model emits one or more ``<tool_call>{...}</tool_call>``
    4. env executes Simia tools natively or uses the ToolEnv cache/simulator,
       then returns ``<tool_response>...</tool_response>``
    """

    use_trajectory_reward = True

    _TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    _SEARCH_TOOL_RE = re.compile(r"<search_tool>(.*?)</search_tool>", re.DOTALL)
    _PLAN_RE = re.compile(r"<plan>(.*?)</plan>", re.DOTALL)
    _ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

    def __init__(
        self,
        **kwargs,
    ):
        from paraagent.toolenv.runtime.modes import require_current_native_modes

        self._tau_feedback_mode = require_current_native_modes()["feedback_mode"]

        self.retriever = ToolSearchTool()

        self.simulator = None

        self._tau_executor = None
        self._tau_init_failed = False
        try:
            from paraagent.toolenv.runtime.tools.tau_native_executor import get_tau_executor

            self._tau_executor = get_tau_executor()
        except Exception as exc:
            logger.warning("TauNativeExecutor unavailable: %s", exc)
        self._tau_sample_id: str | None = None
        self._tau_session_id: str | None = None
        self._messages: list[dict[str, Any]] = []
        self._active_search_plan: dict[str, Any] | None = None
        self._active_action_plan: dict[str, Any] | None = None
        self._plan_prev_action_type: str | None = None
        self._plan_refresh_counts: dict[str, int] = {
            "search_tool": 0,
            "tool_call": 0,
        }
        self._plan_previous_recovery_trigger: dict[str, bool] = {
            "search_tool": False,
            "tool_call": False,
        }
        self._feedback_retrieved_tool_names: set[str] = set()
        self._feedback_retrieved_tool_meta: dict[str, dict[str, Any]] = {}
        self._credit_retrieved_tool_names: set[str] = set()
        self._executed_tool_names: list[str] = []
        self._action_controller_issued_calls: set[tuple[str, str]] = set()
        self._action_controller_succeeded_calls: set[tuple[str, str]] = set()
        self._action_controller_failed_calls: set[tuple[str, str]] = set()
        self._action_controller_issued_occurrences: set[str] = set()
        self._action_controller_succeeded_occurrences: set[str] = set()
        self._action_controller_failed_occurrences: set[str] = set()
        self._action_controller_stale_succeeded_calls: set[tuple[str, str]] = set()
        self._action_controller_revisitable_tools: set[str] = set()
        self._tool_contexts: list[str] = []
        self._successful_searches = 0
        self._successful_tool_calls = 0
        self.max_tool_context_chars = kwargs.get("max_tool_context_chars")
        self.max_tool_context_item_chars = kwargs.get("max_tool_context_item_chars")

        self._judge_tool_desc_enabled = os.environ.get("TOOLENV_JUDGE_TOOL_DESC", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
        }

    async def reset(self, **kwargs) -> Observation:
        messages = kwargs.get("prompt")
        if messages is None:
            messages = kwargs.get("raw_prompt", [])
        self._messages = list(messages)

        from paraagent.toolenv.context import extract_tau_prompt_time

        self._tau_prompt_time = extract_tau_prompt_time(self._messages)
        self._active_search_plan = None
        self._active_action_plan = None
        self._plan_prev_action_type = None
        self._plan_refresh_counts = {"search_tool": 0, "tool_call": 0}
        self._plan_previous_recovery_trigger = {
            "search_tool": False,
            "tool_call": False,
        }
        self._feedback_retrieved_tool_names = set()
        self._feedback_retrieved_tool_meta = {}
        self._credit_retrieved_tool_names = set()
        self._executed_tool_names = []
        self._reset_action_controller_progress()
        self._tool_contexts = []
        self._successful_searches = 0
        self._successful_tool_calls = 0

        if self._tau_executor is not None and self._tau_session_id is not None:
            self._tau_executor.cleanup(self._tau_session_id)
        self._tau_sample_id = None
        self._tau_session_id = None
        self._tau_init_failed = False

        extra = kwargs.get("extra_info") or {}
        if isinstance(extra, str):
            try:
                extra = json.loads(extra)
            except (ValueError, TypeError):
                extra = {}
        sid = extra.get("sample_id") if isinstance(extra, dict) else None
        requested_tau = (
            (isinstance(sid, str) and sid.startswith(("simia_", "tau_")))
            or str(kwargs.get("data_source")).lower() in {"tau", "simia"}
            or (
                isinstance(extra, dict)
                and (
                    str(extra.get("data_source")).lower() in {"tau", "simia"}
                    or extra.get("subset_name") in {"Airline", "Retail"}
                )
            )
        )
        if requested_tau:
            self._tau_init_failed = True
            if (
                self._tau_executor is None
                or getattr(self._tau_executor, "_feedback_mode", None) != "guard_v1"
                or not isinstance(sid, str)
                or not self._tau_executor.has_state(sid)
            ):
                raise RuntimeError(
                    "TAU environment initialization failed: matching executor and initial state are required; simulator fallback is disabled."
                )
        if self._tau_executor is not None:
            extra_info = kwargs.get("extra_info") or {}
            if isinstance(extra_info, str):
                try:
                    extra_info = json.loads(extra_info)
                except Exception:
                    extra_info = {}
            if isinstance(extra_info, dict):
                sid = extra_info.get("sample_id")
                if isinstance(sid, str) and self._tau_executor.has_state(sid):
                    gt_write_calls = extra_info.get("gt_write_calls")
                    if isinstance(gt_write_calls, str):
                        try:
                            gt_write_calls = json.loads(gt_write_calls)
                        except Exception:
                            gt_write_calls = None
                    if not isinstance(gt_write_calls, list):
                        gt_write_calls_json = extra_info.get("gt_write_calls_json")
                        if isinstance(gt_write_calls_json, str):
                            try:
                                parsed_gt_write_calls = json.loads(gt_write_calls_json)
                            except Exception:
                                parsed_gt_write_calls = None
                            if isinstance(parsed_gt_write_calls, list):
                                gt_write_calls = parsed_gt_write_calls
                    self._tau_sample_id = sid
                    self._tau_session_id = uuid.uuid4().hex
                    initialized = self._tau_executor.reset(
                        self._tau_session_id,
                        sid,
                        gt_write_calls=gt_write_calls if isinstance(gt_write_calls, list) else None,
                    )
                    if requested_tau:
                        if not initialized:
                            raise RuntimeError(
                                "TAU environment initialization failed: reset did not create a session."
                            )
                        self._tau_init_failed = False
        return Observation(messages=list(self._messages))

    def _tau_feedback_enabled(self):
        return self._tau_sample_id is not None

    def _tau_request_error(self, call, message):
        name = call.get("name") if isinstance(call.get("name"), str) else ""
        error = {"type": "InvalidRequestError", "msg": message}
        return json.dumps({"name": name, "result": {"error": error}}, ensure_ascii=False), {
            "success": False,
            "error": error,
            "error_type": error["type"],
            "injected_error": False,
            "tool_outcome": make_tool_outcome(
                False,
                stage=STAGE_REQUEST,
                category=CATEGORY_INVALID_REQUEST,
                code="INVALID_TOOL_CALL",
                basis=BASIS_LOCAL_VALIDATION,
                retry_hint=RETRY_NEVER,
                causal_inputs=("tool_call",),
                raw_type=error["type"],
                raw_message=message,
                source="tau_env",
            ),
        }

    def _tau_retrieval_result(self, response, extra):
        """Keep retrieval rank/membership, replacing only mounted TAU schemas."""
        lines = []
        text = response.text
        for line in text.splitlines():
            try:
                item = json.loads(line)
            except ValueError:
                return response, extra
            if not isinstance(item, dict):
                return response, extra
            name = item.get("function", {}).get("name")
            schema = self._tau_executor.tool_schema(self._tau_sample_id, name)
            lines.append(json.dumps(schema if schema is not None else item, ensure_ascii=False))
        response = deepcopy(response)
        response.text = "\n".join(lines)
        extra = deepcopy(extra)
        candidates = []
        for candidate in extra.get("tool_candidates", []):
            name = (
                candidate.get("function", {}).get("name")
                if candidate.get("type") == "function"
                else candidate.get("name")
            )
            schema = self._tau_executor.tool_schema(self._tau_sample_id, name)
            candidates.append(schema if schema is not None else candidate)
        extra["tool_candidates"] = candidates
        return response, extra

    def get_trajectory_reward_extra_info(self) -> dict[str, Any]:
        """Return the TAU final-state outcome payload before session cleanup."""
        if self._tau_executor is None or self._tau_sample_id is None or self._tau_session_id is None:
            return {}
        getter = getattr(self._tau_executor, "get_session_state_summary", None)
        if not callable(getter):
            return {
                "tau_state_capture_ok": False,
                "tau_gt_replay_ok": False,
                "tau_state_capture_error": "executor_state_summary_unavailable",
                "tau_gt_replay_error": "executor_state_summary_unavailable",
            }
        return {
            **getter(self._tau_session_id),
            "tau_simulation_time_context": deepcopy(getattr(self, "_tau_prompt_time", None)),
        }

    def cleanup(self) -> None:
        """Release this trajectory's session from the shared native executor."""
        if self._tau_executor is not None and self._tau_session_id is not None:
            self._tau_executor.cleanup(self._tau_session_id)
        self._tau_session_id = None
        self._tau_sample_id = None

    def _clear_grounding_pool(self) -> None:
        self._feedback_retrieved_tool_names = set()
        self._feedback_retrieved_tool_meta = {}
        self._credit_retrieved_tool_names = set()

    def _reset_action_controller_progress(self, *, keep_history: bool = False) -> None:
        """Reset the execution cursor; keep_history preserves strict call identities for repeat detection."""
        self._action_controller_issued_calls = set()
        self._action_controller_succeeded_calls = set()
        self._action_controller_failed_calls = set()
        self._action_controller_issued_occurrences = set()
        self._action_controller_succeeded_occurrences = set()
        self._action_controller_failed_occurrences = set()
        self._action_controller_revisitable_tools = set()
        if not keep_history:
            self._action_controller_stale_succeeded_calls = set()

    def _current_action_controller_progress(self):
        return action_controller_progress(
            self._active_action_plan,
            issued_tools=self._action_controller_issued_calls,
            succeeded_tools=self._action_controller_succeeded_calls,
            failed_tools=self._action_controller_failed_calls,
            stale_succeeded_calls=self._action_controller_stale_succeeded_calls,
            revisitable_tools=self._action_controller_revisitable_tools,
            issued_occurrences=self._action_controller_issued_occurrences,
            succeeded_occurrences=self._action_controller_succeeded_occurrences,
            failed_occurrences=self._action_controller_failed_occurrences,
        )

    def _activate_action_controller(
        self,
        plan_payload: dict[str, Any],
        transition_kind: str,
    ) -> None:
        """Install a controller and retain valid DAG progress for an updated plan.
        Initial plans and restarts reset the cursor; call history supports repeat detection.
        """
        if transition_kind == "revise":
            old_occurrences = action_plan_occurrence_tools(self._active_action_plan)
            new_occurrences = action_plan_occurrence_tools(plan_payload)
            carried_refs = {ref for ref, tool in new_occurrences.items() if old_occurrences.get(ref) == tool}
            self._action_controller_issued_occurrences &= carried_refs
            self._action_controller_succeeded_occurrences &= carried_refs
            self._action_controller_failed_occurrences &= (
                carried_refs - self._action_controller_succeeded_occurrences
            )
            scheduled = action_plan_frontier(plan_payload)
            self._action_controller_issued_calls = {
                key for key in self._action_controller_issued_calls if key[0] in scheduled
            }
            self._action_controller_succeeded_calls = {
                key for key in self._action_controller_succeeded_calls if key[0] in scheduled
            }
            carried = {new_occurrences[ref] for ref in self._action_controller_succeeded_occurrences}
            self._action_controller_failed_calls = {
                key
                for key in self._action_controller_failed_calls
                if key[0] in scheduled and key[0] not in carried
            }
            self._action_controller_revisitable_tools = carried
        else:
            if transition_kind == "restart":
                self._action_controller_stale_succeeded_calls |= self._action_controller_succeeded_calls
            self._reset_action_controller_progress(keep_history=transition_kind == "restart")
        self._active_action_plan = plan_payload

    def _record_action_controller_outcomes(
        self,
        step_info: dict[str, Any],
        tool_calls: list[dict[str, Any]],
        success_flags: list[bool],
        injected_error_flags: list[bool],
    ) -> None:
        """Track structurally valid successes for rollout diagnostics.
        Grounding and reward replay are evaluated separately.
        """
        duplicate_flags = [False] * len(tool_calls)
        if self._active_action_plan is not None:
            scheduled = action_plan_frontier(self._active_action_plan)
            progress_before = self._current_action_controller_progress()
            keys = [
                make_call_key(tool_call.get("name"), tool_call.get("arguments")) for tool_call in tool_calls
            ]
            duplicate_flags = [
                key in progress_before.succeeded_calls or key in progress_before.stale_succeeded_calls
                for key in keys
            ]
            named_calls = [(tool_call.get("name"), tool_call.get("arguments")) for tool_call in tool_calls]
            matched_occurrences = match_action_calls_to_occurrences(progress_before, named_calls)
            for key in keys:
                if key[0] in scheduled:
                    self._action_controller_issued_calls.add(key)

                    self._action_controller_revisitable_tools.discard(key[0])
            grouped_indices: dict[str, list[int]] = {}
            for index, (key, occurrence_ref) in enumerate(zip(keys, matched_occurrences)):
                if occurrence_ref is None:
                    continue
                if occurrence_ref:
                    self._action_controller_issued_occurrences.add(occurrence_ref)
                    grouped_indices.setdefault(occurrence_ref, []).append(index)
                succeeded = bool(success_flags[index]) if index < len(success_flags) else False
                injected = bool(injected_error_flags[index]) if index < len(injected_error_flags) else False
                if succeeded and not injected:
                    self._action_controller_succeeded_calls.add(key)
                else:
                    self._action_controller_failed_calls.add(key)

            for occurrence_ref, indices in grouped_indices.items():
                group_succeeded = all(
                    index < len(success_flags)
                    and bool(success_flags[index])
                    and not (
                        bool(injected_error_flags[index]) if index < len(injected_error_flags) else False
                    )
                    for index in indices
                )
                if group_succeeded:
                    self._action_controller_succeeded_occurrences.add(occurrence_ref)
                    self._action_controller_failed_occurrences.discard(occurrence_ref)
                else:
                    self._action_controller_succeeded_occurrences.discard(occurrence_ref)
                    self._action_controller_failed_occurrences.add(occurrence_ref)

        progress = self._current_action_controller_progress()
        step_info.update(
            {
                "plan_controller_issued_tools_after_step": sorted(progress.issued_tools),
                "plan_controller_succeeded_tools_after_step": sorted(progress.succeeded_tools),
                "plan_controller_failed_tools_after_step": sorted(progress.failed_tools),
                "plan_controller_remaining_tools_after_step": sorted(progress.remaining_tools),
                "plan_controller_issued_occurrences_after_step": sorted(progress.issued_occurrences),
                "plan_controller_succeeded_occurrences_after_step": sorted(progress.succeeded_occurrences),
                "plan_controller_failed_occurrences_after_step": sorted(progress.failed_occurrences),
                "plan_controller_remaining_occurrences_after_step": sorted(progress.remaining_occurrences),
                "plan_controller_ready_occurrences_after_step": sorted(progress.ready_occurrences),
                "plan_controller_issued_calls_after_step": serialize_call_keys(progress.issued_calls),
                "plan_controller_succeeded_calls_after_step": serialize_call_keys(progress.succeeded_calls),
                "plan_controller_failed_calls_after_step": serialize_call_keys(progress.failed_calls),
                "plan_controller_stale_succeeded_calls_after_step": (
                    serialize_call_keys(progress.stale_succeeded_calls)
                ),
                "plan_controller_revisitable_tools_after_step": sorted(progress.revisitable_tools),
                "plan_controller_duplicate_calls": duplicate_flags,
                "plan_controller_duplicate_call_count": sum(duplicate_flags),
                "plan_controller_exhausted_after_step": progress.exhausted,
                "plan_controller_failed_after_step": (progress.has_unresolved_failure),
            }
        )

    async def step(self, action: Action) -> tuple[Observation, float | None, bool, dict[str, Any]]:
        if getattr(self, "_tau_init_failed", False):
            raise RuntimeError("TAU environment is not initialized; simulator fallback is disabled.")
        if not isinstance(action, Action) or action.text is None:
            raise TypeError("ToolEnv only accepts Action with text")

        step_start = time.perf_counter()
        timing_ms: dict[str, float] = {}

        self._messages.append({"role": "assistant", "content": action.text})
        parse_start = time.perf_counter()
        parsed_step = self._parse_step_content(action.text)
        step_info = self._build_step_info(parsed_step)
        timing_ms["parse"] = (time.perf_counter() - parse_start) * 1000.0

        tool_queries = parsed_step["tool_queries"]
        if tool_queries:
            retrieval_start = time.perf_counter()
            if hasattr(self.retriever, "execute_many"):
                raw_retrieval_results = await self.retriever.execute_many(
                    [{"query": query} for query in tool_queries]
                )
            else:
                raw_retrieval_results = [await self.retriever.run({"query": query}) for query in tool_queries]
            timing_ms["retrieval_wall"] = (time.perf_counter() - retrieval_start) * 1000.0

            tool_lists = []
            success_flags = []
            retrieval_nonempty = []
            retrieval_result_tool_names = []
            retrieval_errors = []
            retrieval_error_types = []
            retrieval_elapsed_ms = []
            retrieval_status_codes = []
            retrieval_attempts = []
            for tool_response, _, extra_info in raw_retrieval_results:
                if self._tau_feedback_enabled():
                    tool_response, extra_info = self._tau_retrieval_result(tool_response, extra_info)
                tool_text = tool_response.text
                tool_names = self._extract_retrieved_tool_names(tool_text)
                tool_lists.append(tool_text)
                success = bool(extra_info.get("success"))
                success_flags.append(success)
                retrieval_errors.append(extra_info.get("error"))
                retrieval_error_types.append(extra_info.get("error_type"))
                retrieval_elapsed_ms.append(extra_info.get("elapsed_ms"))
                retrieval_status_codes.append(extra_info.get("status_code"))
                retrieval_attempts.append(extra_info.get("attempts"))
                self._cache_feedback_retrieved_tool_meta(extra_info.get("tool_candidates", []))
                retrieval_result_tool_names.append(tool_names)
                retrieval_nonempty.append(bool(tool_names))
                self._feedback_retrieved_tool_names.update(
                    self._normalize_name(name) for name in tool_names if name
                )

            if self._should_commit_retrieval_credit(step_info):
                for ok, tool_names in zip(success_flags, retrieval_result_tool_names, strict=False):
                    if ok:
                        self._credit_retrieved_tool_names.update(
                            self._normalize_name(name) for name in tool_names if name
                        )

            format_start = time.perf_counter()
            observation = self.format_retrieval_response(tool_lists)
            timing_ms["format_observation"] = (time.perf_counter() - format_start) * 1000.0
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "ToolEnv.step retrieval queries=%s success=%s nonempty=%s tool_names=%s observation=%s",
                    tool_queries,
                    success_flags,
                    retrieval_nonempty,
                    retrieval_result_tool_names,
                    observation,
                )
            self._messages.append({"role": "user", "content": observation})
            self._successful_searches += sum(1 for ok in success_flags if ok)
            render_start = time.perf_counter()
            trajectory_tool_context = self._render_tool_context()
            timing_ms["render_tool_context"] = (time.perf_counter() - render_start) * 1000.0
            timing_ms["total"] = (time.perf_counter() - step_start) * 1000.0
            step_info.update(
                {
                    "env_timing_ms": timing_ms,
                    "env_num_search_queries": len(tool_queries),
                    "env_num_tool_calls": 0,
                    "retrieval_success": success_flags,
                    "retrieval_nonempty": retrieval_nonempty,
                    "retrieval_result_tool_names": retrieval_result_tool_names,
                    "retrieval_errors": retrieval_errors,
                    "retrieval_error_types": retrieval_error_types,
                    "retrieval_elapsed_ms": retrieval_elapsed_ms,
                    "retrieval_status_codes": retrieval_status_codes,
                    "retrieval_attempts": retrieval_attempts,
                    "num_successful_searches": self._successful_searches,
                    "num_successful_tool_calls": self._successful_tool_calls,
                    "trajectory_tool_context": trajectory_tool_context,
                }
            )
            self._plan_previous_recovery_trigger["search_tool"] = has_explicit_recovery_trigger(
                "search_tool", step_info
            )
            return Observation(messages=list(self._messages)), None, False, step_info

        tool_calls = parsed_step["tool_calls"]
        invalid_tool_call_count = parsed_step.get("invalid_tool_call_count", 0)
        if parsed_step.get("tool_call_block_count", 0) > 0:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("ToolEnv.step entering tool_call branch with tool_calls=%s", tool_calls)
            tool_responses = []
            success_flags = []
            error_types = []
            injected_error_flags = []
            is_tau_sample = self._tau_executor is not None and self._tau_sample_id is not None

            tau_tool_outcomes: list[dict[str, Any]] | None = [] if is_tau_sample else None
            tau_outcomes_complete = is_tau_sample

            if tool_calls:
                simulator_start = time.perf_counter()

                tau_idx_results: dict[int, tuple[str, dict[str, Any]]] = {}
                simulator_pending: list[tuple[int, dict[str, Any]]] = []
                if is_tau_sample:
                    has_state = self._tau_executor.has_state(self._tau_sample_id)
                    for i, tc in enumerate(tool_calls):
                        if self._tau_feedback_enabled() and tc.get("_tau_parse_error"):
                            tau_idx_results[i] = self._tau_request_error(tc, tc["_tau_parse_error"])
                        elif has_state and self._is_tau_tool_call(tc):
                            tau_idx_results[i] = self._invoke_tau_tool(tc)
                        else:
                            tau_idx_results[i] = self._tau_tool_unavailable_result(tc)
                else:
                    simulator_pending = list(enumerate(tool_calls))

                if simulator_pending:
                    if self.simulator is None:
                        self.simulator = ToolEnvSimulatorTool()
                    pending_calls = [tc for _, tc in simulator_pending]
                    sim_results = await self.simulator.execute_many(
                        pending_calls,
                        retrieved_tool_meta=self._feedback_retrieved_tool_meta,
                    )
                else:
                    sim_results = []

                simulator_results: list[tuple[str, dict[str, Any]]] = [None] * len(tool_calls)
                for (i, _tc), out in zip(simulator_pending, sim_results, strict=False):
                    simulator_results[i] = out
                for i, out in tau_idx_results.items():
                    simulator_results[i] = out

                timing_ms["simulator_wall"] = (time.perf_counter() - simulator_start) * 1000.0
                for text, extra_info in simulator_results:
                    tool_responses.append(text)
                    success = bool(extra_info.get("success"))
                    error_type = self._extract_error_type(text, extra_info)
                    injected_error = bool(extra_info.get("injected_error", False))
                    success_flags.append(success)
                    error_types.append(error_type)
                    injected_error_flags.append(injected_error)
                    if tau_tool_outcomes is not None:
                        outcome = extra_info.get("tool_outcome")
                        if isinstance(outcome, dict):
                            tau_tool_outcomes.append(dict(outcome))
                        else:
                            tau_outcomes_complete = False
                    if success and not injected_error:
                        self._tool_contexts.append(text)

            for _ in range(0 if self._tau_feedback_enabled() else invalid_tool_call_count):
                tool_responses.append(
                    json.dumps(
                        {
                            "name": "",
                            "result": {
                                "error": {
                                    "type": "InvalidRequestError",
                                    "msg": "Invalid tool call JSON",
                                }
                            },
                        },
                        ensure_ascii=False,
                    )
                )
                success_flags.append(False)
                error_types.append("InvalidRequestError")
                injected_error_flags.append(False)
                if tau_tool_outcomes is not None:
                    tau_tool_outcomes.append(
                        make_tool_outcome(
                            False,
                            stage=STAGE_REQUEST,
                            category=CATEGORY_INVALID_REQUEST,
                            code="invalid_tool_call_json",
                            basis=BASIS_LOCAL_VALIDATION,
                            retry_hint=RETRY_NEVER,
                            causal_inputs=("tool_call",),
                            raw_type="InvalidRequestError",
                            raw_message="Invalid tool call JSON",
                            source="tau_env",
                        )
                    )

            format_start = time.perf_counter()
            observation = self.format_tool_response(tool_responses)
            timing_ms["format_observation"] = (time.perf_counter() - format_start) * 1000.0
            self._messages.append({"role": "user", "content": observation})
            self._successful_tool_calls += sum(1 for ok in success_flags if ok)
            self._executed_tool_names.extend(
                self._normalize_name(tool_call.get("name"))
                for tool_call in tool_calls
                if tool_call.get("name")
            )
            render_start = time.perf_counter()
            trajectory_tool_context = self._render_tool_context()
            timing_ms["render_tool_context"] = (time.perf_counter() - render_start) * 1000.0
            timing_ms["total"] = (time.perf_counter() - step_start) * 1000.0
            tool_step_info = {
                "env_timing_ms": timing_ms,
                "env_num_search_queries": 0,
                "env_num_tool_calls": len(tool_calls),
                "tool_success": success_flags,
                "tool_error_types": error_types,
                "tool_injected_error_flags": injected_error_flags,
                "tool_response_payloads": list(tool_responses),
                "num_successful_searches": self._successful_searches,
                "num_successful_tool_calls": self._successful_tool_calls,
                "trajectory_tool_context": trajectory_tool_context,
            }
            if (
                tau_tool_outcomes is not None
                and tau_outcomes_complete
                and len(tau_tool_outcomes) == len(success_flags)
            ):
                tool_step_info["tool_outcomes_v1"] = tau_tool_outcomes
            step_info.update(tool_step_info)
            self._record_action_controller_outcomes(
                step_info,
                tool_calls,
                success_flags,
                injected_error_flags,
            )
            self._plan_previous_recovery_trigger["tool_call"] = has_explicit_recovery_trigger(
                "tool_call", step_info
            )
            return Observation(messages=list(self._messages)), None, False, step_info

        render_start = time.perf_counter()
        trajectory_tool_context = self._render_tool_context()
        timing_ms["render_tool_context"] = (time.perf_counter() - render_start) * 1000.0
        timing_ms["total"] = (time.perf_counter() - step_start) * 1000.0
        step_info.update(
            {
                "env_timing_ms": timing_ms,
                "env_num_search_queries": 0,
                "env_num_tool_calls": 0,
                "num_successful_searches": self._successful_searches,
                "num_successful_tool_calls": self._successful_tool_calls,
                "trajectory_tool_context": trajectory_tool_context,
            }
        )
        return Observation(messages=list(self._messages)), None, True, step_info

    def _is_tau_tool_call(self, tool_call: dict[str, Any]) -> bool:
        """True when this call should be routed to TauNativeExecutor."""
        if self._tau_executor is None or self._tau_sample_id is None:
            return False
        from paraagent.toolenv.runtime.tools.tau_native_executor import (
            TAU_TOOL_NAMES_AIRLINE,
            TAU_TOOL_NAMES_RETAIL,
            domain_for_sample_id,
        )

        tn = tool_call.get("name")
        domain = domain_for_sample_id(self._tau_sample_id)
        if domain == "airline":
            return tn in TAU_TOOL_NAMES_AIRLINE
        if domain == "retail":
            return tn in TAU_TOOL_NAMES_RETAIL
        return False

    def _invoke_tau_tool(self, tool_call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Run a single tau-native tool call, return simulator-shaped (text, extra_info)."""
        tool_name = tool_call.get("name")
        arguments = tool_call.get("arguments", {})
        if not self._tau_feedback_enabled():
            arguments = arguments or {}
        result = self._tau_executor.execute(
            session_id=self._tau_session_id,
            sample_id=self._tau_sample_id,
            tool_name=tool_name,
            arguments=arguments,
        )
        content = result.get("content") or {}
        if isinstance(content, dict) and content.get("error") is None:
            result_payload = content.get("response")
        else:
            result_payload = {"error": content.get("error") if isinstance(content, dict) else None}
        payload = json.dumps({"name": tool_name, "result": result_payload}, ensure_ascii=False)
        error_obj = content.get("error") if isinstance(content, dict) else None
        extra_info = {
            "success": bool(result.get("success")),
            "tool_name": tool_name,
            "error": error_obj,
            "error_type": error_obj.get("type") if isinstance(error_obj, dict) else None,
            "error_msg": result.get("error_msg"),
            "injected_error": False,
            "tau_native": True,
        }
        if isinstance(result.get("tool_outcome"), dict):
            extra_info["tool_outcome"] = result["tool_outcome"]
        return payload, extra_info

    def _tau_tool_unavailable_result(self, tool_call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Expose ToolUnavailableError for tools outside this session.
        Record a non-terminal dispatch failure with zero call and step credit.
        """
        tool_name = tool_call.get("name")
        error_obj = {
            "type": "ToolUnavailableError",
            "msg": (
                f"The tool '{tool_name}' is currently unavailable. Do not retry it, select a different tool."
            ),
        }
        result_payload = {"error": error_obj}
        payload = json.dumps({"name": tool_name, "result": result_payload}, ensure_ascii=False)
        extra_info = {
            "success": False,
            "tool_name": tool_name,
            "error": error_obj,
            "error_type": "tool_not_found",
            "error_msg": error_obj["msg"],
            "injected_error": False,
            "tau_native": True,
            "tool_outcome": make_tool_outcome(
                False,
                stage=STAGE_DISPATCH,
                category=CATEGORY_TOOL_NOT_FOUND,
                code="tool_not_found",
                basis=BASIS_LOCAL_VALIDATION,
                retry_hint=RETRY_NEVER,
                raw_type="tool_not_found",
                raw_message=error_obj["msg"],
                source="tau_env",
            ),
        }
        return payload, extra_info

    def extract_tool_calls(self, raw_response: str) -> list[dict[str, Any]]:

        tool_calls: list[dict[str, Any]] = []
        for tool_call_text in self._TOOL_CALL_RE.findall(raw_response):
            try:
                tool_call = json.loads(tool_call_text)
            except json.JSONDecodeError:
                continue
            if not isinstance(tool_call, dict):
                continue

            arguments = tool_call.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            tool_calls.append({"name": tool_call.get("name"), "arguments": arguments})
        if "<tool_call>" in raw_response and logger.isEnabledFor(logging.DEBUG):
            logger.debug("ToolEnv.extract_tool_calls parsed=%s from_response=%s", tool_calls, raw_response)
        return tool_calls

    def extract_retrieval_calls(self, raw_response: str) -> list[str]:
        return [item.strip() for item in self._SEARCH_TOOL_RE.findall(raw_response) if item.strip()]

    def _parse_step_content(self, raw_response: str) -> dict[str, Any]:
        plan_payload, plan_meta = self._parse_plan(raw_response)
        tool_queries = self.extract_retrieval_calls(raw_response)
        tool_calls = [] if self._tau_feedback_enabled() else self.extract_tool_calls(raw_response)
        tool_call_block_count = len(self._TOOL_CALL_RE.findall(raw_response))
        invalid_tool_call_count = max(0, tool_call_block_count - len(tool_calls))
        has_answer = bool(self._ANSWER_RE.findall(raw_response))
        if self._tau_feedback_enabled():
            from paraagent.toolenv.runtime.tau_feedback import parse_calls

            tool_calls = parse_calls(raw_response)
            tool_call_block_count = len(tool_calls)
            if tool_calls and (tool_queries or has_answer):
                for call in tool_calls:
                    call["_tau_parse_error"] = (
                        "Mixed action types: send tool calls separately from search_tool and answer blocks. No tools were executed."
                    )
                tool_queries = []
            invalid_tool_call_count = sum("_tau_parse_error" in call for call in tool_calls)

        action_types = []
        if tool_queries:
            action_types.append("search_tool")

        if tool_call_block_count:
            action_types.append("tool_call")
        if has_answer:
            action_types.append("answer")

        if len(action_types) == 1:
            action_type = action_types[0]
        elif action_types:
            action_type = "mixed"
        else:
            action_type = "invalid"

        structure_info = self._parse_step_structure(raw_response, action_type)
        return {
            "raw_response": raw_response,
            "tool_queries": tool_queries,
            "tool_calls": tool_calls,
            "tool_call_block_count": tool_call_block_count,
            "invalid_tool_call_count": invalid_tool_call_count,
            "plan_payload": plan_payload,
            "plan_meta": plan_meta,
            "has_answer": has_answer,
            "action_type": action_type,
            "structure_info": structure_info,
        }

    def _build_step_info(self, raw_response_or_parsed: str | dict[str, Any]) -> dict[str, Any]:
        parsed_step = (
            raw_response_or_parsed
            if isinstance(raw_response_or_parsed, dict)
            else self._parse_step_content(raw_response_or_parsed)
        )
        plan_payload = parsed_step["plan_payload"]
        plan_meta = parsed_step["plan_meta"]
        tool_queries = parsed_step["tool_queries"]
        tool_calls = parsed_step["tool_calls"]
        tool_call_block_count = parsed_step["tool_call_block_count"]
        invalid_tool_call_count = parsed_step["invalid_tool_call_count"]
        action_type = parsed_step["action_type"]
        structure_info = parsed_step["structure_info"]

        if action_type in {"search_tool", "tool_call"}:
            phase_boundary = (
                self._plan_prev_action_type is not None and action_type != self._plan_prev_action_type
            )
            if phase_boundary:
                if action_type == "search_tool":
                    self._active_search_plan = None
                else:
                    self._active_action_plan = None
                    self._reset_action_controller_progress()
                self._plan_refresh_counts[action_type] = 0
                self._plan_previous_recovery_trigger[action_type] = False
        else:
            phase_boundary = False

        action_progress_before = self._current_action_controller_progress()
        active_plan = (
            self._active_search_plan
            if action_type == "search_tool"
            else self._active_action_plan
            if action_type == "tool_call"
            else None
        )
        has_live_controller = isinstance(active_plan, dict)
        controller_exhausted = bool(
            action_type == "tool_call" and has_live_controller and action_progress_before.exhausted
        )

        plan_requirement = (
            classify_plan_requirement(
                has_live_controller=has_live_controller,
                controller_exhausted=controller_exhausted,
                previous_recovery_trigger=self._plan_previous_recovery_trigger[action_type],
                refresh_count=int(self._plan_refresh_counts.get(action_type, 0)),
                max_refreshes=max_refreshes_for(action_type),
                action_type=action_type,
            )
            if action_type in {"search_tool", "tool_call"}
            else "forbidden"
        )

        plan_format_valid = False
        plan_update_valid = False
        plan_phase_mismatch = False
        plan_redundant = False
        plan_refresh_accepted = False
        plan_refresh_kind = "none"
        plan_refresh_reason = "no_plan"
        plan_refresh_count_before = int(self._plan_refresh_counts.get(action_type, 0))
        if plan_meta["has_plan"] and not plan_meta["plan_json_valid"]:
            plan_refresh_kind = "invalid"
            plan_refresh_reason = (
                "multiple_plan_blocks" if plan_meta["plan_count"] != 1 else "invalid_plan_json"
            )
        if plan_payload is not None and plan_meta["plan_json_valid"]:
            if action_type == "search_tool":
                plan_format_valid = (
                    self._is_valid_search_plan_format(plan_payload)
                    and structure_info["plan_activation_valid"]
                )
                plan_update_valid = plan_format_valid and self._is_valid_search_plan(plan_payload)

                decision = classify_plan_refresh(
                    action_type,
                    plan_payload,
                    self._active_search_plan,
                    plan_valid=plan_update_valid,
                    plan_requirement=plan_requirement,
                    controller_unfinished=None,
                    refresh_count=plan_refresh_count_before,
                )
                plan_redundant = decision.same_controller
                plan_refresh_accepted = decision.accepted
                plan_refresh_kind = decision.kind
                plan_refresh_reason = decision.reason
                if decision.accepted:
                    if decision.kind != "initial":
                        self._plan_refresh_counts[action_type] += 1
                    self._active_search_plan = plan_payload
            elif action_type == "tool_call":
                plan_format_valid = (
                    self._is_valid_action_plan_format(plan_payload)
                    and structure_info["plan_activation_valid"]
                )
                plan_update_valid = plan_format_valid and self._is_valid_action_plan(plan_payload)
                decision = classify_plan_refresh(
                    action_type,
                    plan_payload,
                    self._active_action_plan,
                    plan_valid=plan_update_valid,
                    plan_requirement=plan_requirement,
                    controller_unfinished=(
                        set(action_progress_before.remaining_occurrences)
                        if self._active_action_plan is not None
                        else None
                    ),
                    refresh_count=plan_refresh_count_before,
                )
                plan_redundant = decision.same_controller
                plan_refresh_accepted = decision.accepted
                plan_refresh_kind = decision.kind
                plan_refresh_reason = decision.reason
                if decision.accepted:
                    if decision.kind != "initial":
                        self._plan_refresh_counts[action_type] += 1
                    self._activate_action_controller(plan_payload, decision.kind)
            else:
                plan_phase_mismatch = True
                plan_refresh_kind = "invalid"
                plan_refresh_reason = "plan_phase_mismatch"

        if action_type in {"search_tool", "tool_call"}:
            self._plan_prev_action_type = action_type

        step_info = {
            "env_action_type": action_type,
            "has_think": structure_info["has_think"],
            "think_before_actions": structure_info["think_before_actions"],
            "outer_text_valid": structure_info["outer_text_valid"],
            "has_plan": plan_meta["has_plan"],
            "plan_count": plan_meta["plan_count"],
            "plan_json_valid": plan_meta["plan_json_valid"],
            "plan_type": plan_meta["plan_type"],
            "plan_position_valid": structure_info["plan_position_valid"],
            "plan_activation_valid": structure_info["plan_activation_valid"],
            "plan_format_valid": plan_format_valid,
            "plan_update_valid": plan_update_valid,
            "plan_requirement": plan_requirement,
            "plan_phase_mismatch": plan_phase_mismatch,
            "plan_redundant": plan_redundant,
            "plan_phase_boundary": phase_boundary,
            "plan_refresh_accepted": plan_refresh_accepted,
            "plan_refresh_kind": plan_refresh_kind,
            "plan_refresh_reason": plan_refresh_reason,
            "plan_refresh_count_before": plan_refresh_count_before,
            "plan_refresh_count_after": int(self._plan_refresh_counts.get(action_type, 0)),
            "plan_refresh_limit": max_refreshes_for(action_type),
            "plan_limit_per_phase": MAX_ACCEPTED_PLANS_PER_PHASE,
            "plan_controller_lifecycle_version": (PLAN_CONTROLLER_LIFECYCLE_VERSION),
            "plan_controller_authoritative": False,
            "plan_controller_unfinished_tools_before_step": sorted(action_progress_before.remaining_tools),
            "plan_controller_unfinished_occurrences_before_step": sorted(
                action_progress_before.remaining_occurrences
            ),
            "plan_controller_ready_occurrences_before_step": sorted(action_progress_before.ready_occurrences),
            "plan_controller_issued_occurrences_before_step": sorted(
                action_progress_before.issued_occurrences
            ),
            "plan_controller_succeeded_occurrences_before_step": sorted(
                action_progress_before.succeeded_occurrences
            ),
            "plan_controller_failed_occurrences_before_step": sorted(
                action_progress_before.failed_occurrences
            ),
            "plan_controller_issued_tools_before_step": sorted(action_progress_before.issued_tools),
            "plan_controller_succeeded_tools_before_step": sorted(action_progress_before.succeeded_tools),
            "plan_controller_failed_tools_before_step": sorted(action_progress_before.failed_tools),
            "plan_controller_remaining_tools_before_step": sorted(action_progress_before.remaining_tools),
            "plan_controller_exhausted_before_step": (
                action_type == "tool_call" and action_progress_before.exhausted
            ),
            "plan_controller_failed_before_step": (
                action_type == "tool_call" and action_progress_before.has_unresolved_failure
            ),
            "tool_call_block_count": tool_call_block_count,
            "invalid_tool_call_count": invalid_tool_call_count,
            "tool_call_json_valid": invalid_tool_call_count == 0,
            "has_active_search_plan": self._active_search_plan is not None,
            "has_active_action_plan": self._active_action_plan is not None,
            "search_phase_adherence": None,
            "tool_phase_adherence": None,
            "search_slot_matches": [],
            "search_slot_overflow": None,
            "tool_names_in_available_tools": [],
            "tool_names_in_retrieved_tools": [],
            "tool_order_valid": None,
            "tool_dependencies_valid": None,
            "tool_called_without_retrieval": [],
            "tool_called_outside_available_tools": [],
            "tool_called_out_of_order": False,
            "tool_called_with_missing_dependencies": False,
            "executed_tools_before": list(self._executed_tool_names),
        }

        if action_type == "search_tool":
            step_info.update(
                self._compute_search_phase_info(
                    tool_queries,
                    own_plan=plan_payload,
                    own_plan_rejected=bool(plan_meta["has_plan"] and not plan_refresh_accepted),
                    own_plan_unparseable=bool(plan_meta["has_plan"] and not plan_meta["plan_json_valid"]),
                )
            )
        elif action_type == "tool_call":
            step_info.update(self._compute_tool_phase_info(tool_calls))

        return step_info

    def _tag_spans(self, raw_response: str, tag: str) -> list[tuple[int, int]]:
        pattern = re.compile(rf"<{tag}>.*?</{tag}>", re.DOTALL)
        return [match.span() for match in pattern.finditer(raw_response)]

    def _has_only_whitespace_outside_spans(self, raw_response: str, spans: list[tuple[int, int]]) -> bool:
        if not raw_response:
            return True
        merged_spans: list[tuple[int, int]] = []
        for start, end in sorted(spans):
            if start < 0 or end < start:
                return False
            if merged_spans and start <= merged_spans[-1][1]:
                merged_spans[-1] = (merged_spans[-1][0], max(merged_spans[-1][1], end))
            else:
                merged_spans.append((start, end))

        cursor = 0
        for start, end in merged_spans:
            if raw_response[cursor:start].strip():
                return False
            cursor = end
        return not raw_response[cursor:].strip()

    def _parse_step_structure(self, raw_response: str, action_type: str) -> dict[str, Any]:
        think_spans = self._tag_spans(raw_response, "think")
        plan_spans = self._tag_spans(raw_response, "plan")
        search_spans = self._tag_spans(raw_response, "search_tool")
        tool_spans = self._tag_spans(raw_response, "tool_call")
        answer_spans = self._tag_spans(raw_response, "answer")
        tag_spans = think_spans + plan_spans + search_spans + tool_spans + answer_spans
        outer_text_valid = self._has_only_whitespace_outside_spans(raw_response, tag_spans)

        action_positions = search_spans + tool_spans + answer_spans
        first_action_start = min((start for start, _ in action_positions), default=None)
        last_think_end = max((end for _, end in think_spans), default=None)

        has_think = bool(think_spans)
        think_before_actions = bool(has_think and first_action_start is not None)
        if think_before_actions:
            think_before_actions = all(end <= first_action_start for _, end in think_spans)

        plan_position_valid = False
        if not plan_spans:
            plan_position_valid = True
        elif len(plan_spans) == 1 and first_action_start is not None and last_think_end is not None:
            plan_start, plan_end = plan_spans[0]
            has_think_before_plan = any(end <= plan_start for _, end in think_spans)
            plan_position_valid = has_think_before_plan and plan_end <= first_action_start

        plan_activation_valid = (
            len(plan_spans) == 1
            and has_think
            and think_before_actions
            and plan_position_valid
            and outer_text_valid
            and action_type in {"search_tool", "tool_call"}
        )

        return {
            "has_think": has_think,
            "think_before_actions": think_before_actions,
            "outer_text_valid": outer_text_valid,
            "plan_position_valid": plan_position_valid,
            "plan_activation_valid": plan_activation_valid,
        }

    def _match_queries_against_slots(
        self,
        tool_queries: list[str],
        plan: dict[str, Any] | None,
    ) -> tuple[bool | None, list[bool], bool | None]:
        """Check queries against one plan's declared capacity slots."""
        if plan is None:
            return None, [], None
        slots = plan.get("capacity_slots", [])
        remaining_slots = Counter(
            self._normalize_name(slot)
            for slot in slots
            if isinstance(slot, str) and self._normalize_name(slot)
        )
        query_names = [self._extract_search_name(query) for query in tool_queries]
        slot_matches = []
        for name in query_names:
            matched = bool(name and remaining_slots[name] > 0)
            slot_matches.append(matched)
            if matched:
                remaining_slots[name] -= 1
        slot_overflow = len(query_names) > len(slots)
        phase_ok = bool(query_names) and all(slot_matches) and not slot_overflow
        return phase_ok, slot_matches, slot_overflow

    def _compute_search_phase_info(
        self,
        tool_queries: list[str],
        *,
        own_plan: dict[str, Any] | None = None,
        own_plan_rejected: bool = False,
        own_plan_unparseable: bool = False,
    ) -> dict[str, Any]:
        """Score controller adherence against the current step's declared slots.
        Unparseable plans leave retrieval credit unchanged and are scored by the protocol checker.
        """
        phase_ok, slot_matches, slot_overflow = self._match_queries_against_slots(
            tool_queries, self._active_search_plan
        )
        credit_ok = phase_ok is True
        if not credit_ok and own_plan_rejected:
            if own_plan_unparseable:
                credit_ok = True
            else:
                credit_ok = self._match_queries_against_slots(tool_queries, own_plan)[0] is True
        if self._active_search_plan is None and not own_plan_rejected:
            return {"search_phase_adherence": None}
        return {
            "search_phase_adherence": phase_ok,
            "search_slot_matches": slot_matches,
            "search_slot_overflow": slot_overflow,
            "search_retrieval_credit_ok": credit_ok,
        }

    def _compute_tool_phase_info(self, tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
        named_calls = [
            (self._normalize_name(tool_call.get("name")), tool_call.get("arguments"))
            for tool_call in tool_calls
            if tool_call.get("name")
        ]
        tool_names = [name for name, _ in named_calls]
        in_retrieved_tools = [name in self._credit_retrieved_tool_names for name in tool_names]
        if self._active_action_plan is None:
            return {
                "tool_phase_adherence": None,
                "tool_names_in_retrieved_tools": in_retrieved_tools,
                "tool_called_without_retrieval": [not present for present in in_retrieved_tools],
            }

        available_tools = {
            self._normalize_name(name)
            for name in self._active_action_plan.get("available_tools", [])
            if isinstance(name, str)
        }
        controller_progress = self._current_action_controller_progress()
        in_available_tools = [name in available_tools for name in tool_names]
        order_valid = self._check_execution_flow(named_calls)
        dependencies_valid = self._check_dependencies(named_calls)
        phase_ok = (
            bool(tool_names)
            and not controller_progress.exhausted
            and all(in_available_tools)
            and all(in_retrieved_tools)
            and (order_valid is not False)
            and (dependencies_valid is not False)
        )
        return {
            "tool_phase_adherence": phase_ok,
            "tool_names_in_available_tools": in_available_tools,
            "tool_names_in_retrieved_tools": in_retrieved_tools,
            "tool_order_valid": order_valid,
            "tool_dependencies_valid": dependencies_valid,
            "tool_called_without_retrieval": [not present for present in in_retrieved_tools],
            "tool_called_outside_available_tools": [not present for present in in_available_tools],
            "tool_called_out_of_order": order_valid is False,
            "tool_called_with_missing_dependencies": dependencies_valid is False,
        }

    def _parse_plan(self, raw_response: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        matches = self._PLAN_RE.findall(raw_response)
        meta = {
            "has_plan": bool(matches),
            "plan_count": len(matches),
            "plan_json_valid": False,
            "plan_type": None,
        }
        if len(matches) != 1:
            return None, meta

        try:
            payload = json.loads(matches[0])
        except json.JSONDecodeError:
            return None, meta

        meta["plan_json_valid"] = isinstance(payload, dict)
        meta["plan_type"] = self._infer_plan_type(payload)
        return payload if isinstance(payload, dict) else None, meta

    def _infer_plan_type(self, payload: dict[str, Any] | None) -> str | None:
        if not isinstance(payload, dict):
            return None
        if isinstance(payload.get("capacity_slots"), list):
            return "search"
        if isinstance(payload.get("available_tools"), list) and isinstance(
            payload.get("execution_flow"), list
        ):
            return "action"
        return "unknown"

    def _is_valid_search_plan_format(self, payload: dict[str, Any]) -> bool:
        return search_plan_format_valid(payload)

    def _is_valid_search_plan(self, payload: dict[str, Any]) -> bool:

        return self._is_valid_search_plan_format(payload)

    def _is_valid_action_plan_format(self, payload: dict[str, Any]) -> bool:
        """Validate only the public action-plan scheme.

        Membership, dependency order, grounding, lifecycle transitions, and
        refresh allowance are controller semantics and intentionally excluded.
        """
        return action_plan_format_valid(payload)

    def _is_valid_action_plan(self, payload: dict[str, Any]) -> bool:
        if not self._is_valid_action_plan_format(payload):
            return False
        if not self._validate_action_plan_membership(payload):
            return False
        if not self._validate_action_plan_dependency_order(payload):
            return False
        return True

    def _is_valid_subgoals(self, payload: dict[str, Any]) -> bool:
        if "subgoals" not in payload:
            return True
        subgoals = payload.get("subgoals")
        if subgoals is None:
            return True
        if not isinstance(subgoals, list) or not subgoals:
            return False
        for item in subgoals:
            if not isinstance(item, dict):
                return False
            if set(item) != {"id", "subgoal"}:
                return False
            if not isinstance(item["id"], str) or not item["id"].strip():
                return False
            if not isinstance(item["subgoal"], str) or not item["subgoal"].strip():
                return False
        return True

    def _validate_action_plan_membership(self, payload: dict[str, Any]) -> bool:
        available_tools = {
            self._normalize_name(tool)
            for tool in payload.get("available_tools", [])
            if isinstance(tool, str) and tool.strip()
        }
        if not available_tools:
            return False

        seen_refs = set()
        for stage in action_plan_stage_specs(payload):
            for node in stage:
                normalized = node["tool"]
                ref = node["ref"]
                if not normalized or normalized not in available_tools or ref in seen_refs:
                    return False
                seen_refs.add(ref)

        dependencies = payload.get("dependencies", []) or []
        for dep in dependencies:
            if not isinstance(dep, dict):
                return False
            target = self._normalize_name(dep.get("to"))
            sources = [self._normalize_name(item) for item in dep.get("from", [])]
            if not target or not sources:
                return False
        return action_plan_dependencies_valid(payload)

    def _validate_action_plan_dependency_order(self, payload: dict[str, Any]) -> bool:
        stages = action_plan_stage_specs(payload)
        if not stages or not action_plan_dependencies_valid(payload):
            return False
        layer_map = {node["ref"]: layer_idx for layer_idx, stage in enumerate(stages) for node in stage}
        edges = action_plan_dependency_edges(payload)
        adjacency: dict[str, set[str]] = {ref: set() for ref in layer_map}
        for source, target in edges:
            if source not in layer_map or target not in layer_map:
                return False
            if layer_map[source] >= layer_map[target]:
                return False
            adjacency[source].add(target)

        occurrences = action_plan_occurrence_tools(payload)
        refs_by_tool: dict[str, list[str]] = {}
        for ref, tool in occurrences.items():
            refs_by_tool.setdefault(tool, []).append(ref)

        def _reachable(source: str, target: str) -> bool:
            frontier = [source]
            seen: set[str] = set()
            while frontier:
                current = frontier.pop()
                if current == target:
                    return True
                if current in seen:
                    continue
                seen.add(current)
                frontier.extend(adjacency[current] - seen)
            return False

        for refs in refs_by_tool.values():
            for index, left in enumerate(refs):
                for right in refs[index + 1 :]:
                    left_layer = layer_map[left]
                    right_layer = layer_map[right]
                    if left_layer == right_layer:
                        continue
                    earlier, later = (left, right) if left_layer < right_layer else (right, left)
                    if not _reachable(earlier, later):
                        return False
        return True

    def _build_execution_layer_map(self, execution_flow: list[dict[str, Any]]) -> dict[str, int] | None:
        if not isinstance(execution_flow, list) or not execution_flow:
            return None

        layer_map: dict[str, int] = {}
        for layer_idx, step in enumerate(execution_flow):
            if not isinstance(step, dict):
                return None
            parallel = step.get("parallel", [])
            if not isinstance(parallel, list) or not parallel:
                return None
            for tool in parallel:
                normalized = self._normalize_name(tool)
                if not normalized or normalized in layer_map:
                    return None
                layer_map[normalized] = layer_idx
        return layer_map

    def _is_redundant_plan(self, new_plan: dict[str, Any], active_plan: dict[str, Any] | None) -> bool:
        if active_plan is None:
            return False
        return json.dumps(
            self._normalize_plan_for_repeat(new_plan), sort_keys=True, ensure_ascii=False
        ) == json.dumps(self._normalize_plan_for_repeat(active_plan), sort_keys=True, ensure_ascii=False)

    def _normalize_plan_for_repeat(self, plan: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(plan)
        if isinstance(normalized.get("capacity_slots"), list):
            normalized["capacity_slots"] = sorted(
                [
                    slot.strip()
                    for slot in normalized["capacity_slots"]
                    if isinstance(slot, str) and slot.strip()
                ],
                key=self._normalize_name,
            )
        if isinstance(normalized.get("available_tools"), list):
            normalized["available_tools"] = sorted(
                {
                    self._normalize_name(tool)
                    for tool in normalized["available_tools"]
                    if self._normalize_name(tool)
                }
            )
        if isinstance(normalized.get("execution_flow"), list):
            normalized["execution_flow"] = [
                {
                    **step,
                    "parallel": sorted(
                        [
                            tool.strip()
                            for tool in step.get("parallel", [])
                            if isinstance(tool, str) and tool.strip()
                        ],
                        key=self._normalize_name,
                    ),
                }
                if isinstance(step, dict)
                else step
                for step in normalized["execution_flow"]
            ]
        if isinstance(normalized.get("dependencies"), list):
            dependencies = []
            for dep in normalized["dependencies"]:
                if not isinstance(dep, dict):
                    dependencies.append(dep)
                    continue
                sources = sorted(
                    [
                        source.strip()
                        for source in dep.get("from", [])
                        if isinstance(source, str) and source.strip()
                    ],
                    key=self._normalize_name,
                )
                target = dep.get("to").strip() if isinstance(dep.get("to"), str) else dep.get("to")
                dependencies.append({"from": sources, "to": target})
            normalized["dependencies"] = sorted(
                dependencies,
                key=lambda dep: (
                    self._normalize_name(dep.get("to")) if isinstance(dep, dict) else "",
                    tuple(self._normalize_name(source) for source in dep.get("from", []))
                    if isinstance(dep, dict)
                    else (),
                ),
            )
        if isinstance(normalized.get("subgoals"), list):
            normalized["subgoals"] = sorted(
                normalized["subgoals"],
                key=lambda item: (
                    str(item.get("id", "")) if isinstance(item, dict) else "",
                    json.dumps(item, sort_keys=True, ensure_ascii=False),
                ),
            )
        return normalized

    def _check_execution_flow(self, current_calls: list[tuple[str, Any]]) -> bool | None:
        """Check calls against the earliest unfinished stage, allowing retries and split steps.

        Later stages require current-stage successes. The shared classifier also
        allows argument-changing revisits of successes retained from a prior controller.
        """
        if self._active_action_plan is None:
            return None
        return action_calls_follow_flow(self._current_action_controller_progress(), current_calls)

    def _check_dependencies(self, current_calls: list[tuple[str, Any]]) -> bool | None:
        if self._active_action_plan is None:
            return None
        dependencies = self._active_action_plan.get("dependencies", [])
        if not dependencies:
            return None
        return action_calls_follow_flow(self._current_action_controller_progress(), current_calls)

    def _extract_retrieved_tool_names(self, tool_blob_text: str) -> list[str]:
        if not tool_blob_text:
            return []
        payload = self._parse_retrieved_tools_payload(tool_blob_text)
        if not isinstance(payload, list):
            return []

        names = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            function = item.get("function", {})
            if isinstance(function, dict):
                name = function.get("name")
                if isinstance(name, str) and name.strip():
                    names.append(name.strip())
        return names

    def _extract_error_type(self, text: str, extra_info: dict[str, Any]) -> str | None:
        error_type = extra_info.get("error_type")
        if isinstance(error_type, str) and error_type:
            return error_type

        error_msg = extra_info.get("error_msg")
        if isinstance(error_msg, str) and error_msg:
            return error_msg.split(":", 1)[0].strip()

        try:
            payload = json.loads(text)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        result = payload.get("result")
        if not isinstance(result, dict):
            return None
        error = result.get("error")
        if isinstance(error, dict):
            err_type = error.get("type")
            if isinstance(err_type, str) and err_type.strip():
                return err_type.strip()
        return None

    def _is_consumable_tool_observation(
        self, success: bool, error_type: Any, injected_error: bool = False
    ) -> bool:
        return is_effective_tool_observation(success, error_type, injected_error=injected_error)

    def _is_business_level_failure(self, error_type: Any) -> bool:
        return is_business_level_tool_failure(error_type)

    def _short_tool_description(self, name: str) -> str:
        """Return a short purpose for judge context, using mounted schemas for Simia retail writes."""
        if not name:
            return ""
        normalized_name = normalize_tool_name(name)
        desc = AUTONOMOUS_RETAIL_TOOL_DESCRIPTIONS.get(normalized_name, "")
        if not desc:
            meta = self._feedback_retrieved_tool_meta.get(name) or self._feedback_retrieved_tool_meta.get(
                normalized_name
            )
            if not isinstance(meta, dict):
                return ""
            blob = meta.get("tool_meta")
            if not isinstance(blob, dict):
                return ""

            fn = blob.get("function")
            if isinstance(fn, dict):
                desc = str(fn.get("description"))
            if not desc:
                desc = str(blob.get("description") or blob.get("api_description"))
        desc = desc.strip()
        if not desc:
            return ""
        wrapped = _DESC_WRAPPER_RE.search(desc)
        if wrapped:
            desc = wrapped.group(1).strip()
        desc = " ".join(desc.split())
        sentence = _FIRST_SENTENCE_RE.search(desc)
        if sentence:
            desc = desc[: sentence.end()]
        if len(desc) > _TOOL_DESC_MAX_CHARS:
            desc = desc[: _TOOL_DESC_MAX_CHARS - 3].rstrip() + "..."
        return desc

    def _render_tool_context(self) -> str:
        if not self._tool_contexts:
            return ""

        def _trim_text(text: str, max_chars: Any) -> str:
            if max_chars is None:
                return text
            try:
                max_chars = int(max_chars)
            except (TypeError, ValueError):
                return text
            if max_chars <= 0:
                return ""
            if len(text) <= max_chars:
                return text
            if max_chars <= 3:
                return "." * max_chars
            return text[: max_chars - 3] + "..."

        lines = []
        for seq, raw_text in enumerate(self._tool_contexts, start=1):
            try:
                parsed = json.loads(raw_text)
                name = str(parsed.get("name", "")).strip()
                result = parsed.get("result", "")
                result_str = json.dumps(result, ensure_ascii=False) if not isinstance(result, str) else result
            except Exception:
                name = ""
                result_str = raw_text.strip() if isinstance(raw_text, str) else ""
            if name:
                desc = self._short_tool_description(name) if self._judge_tool_desc_enabled else ""
                prefix = f"[{seq}] {name} ({desc}): " if desc else f"[{seq}] {name}: "
            else:
                prefix = f"[{seq}] "
            lines.append(_trim_text(prefix + result_str, self.max_tool_context_item_chars))

        max_context_chars = self.max_tool_context_chars
        if max_context_chars is None:
            return "\n".join(lines)
        try:
            max_context_chars = int(max_context_chars)
        except (TypeError, ValueError):
            return "\n".join(lines)
        if max_context_chars <= 0:
            return ""

        packed_reversed = []
        total_len = 0
        for line in reversed(lines):
            line_len = len(line)
            sep_len = 1 if packed_reversed else 0
            if total_len + sep_len + line_len > max_context_chars:
                if not packed_reversed:
                    packed_reversed.append(_trim_text(line, max_context_chars))
                break
            packed_reversed.append(line)
            total_len += sep_len + line_len

        return "\n".join(reversed(packed_reversed))

    def _should_commit_retrieval_credit(self, step_info: dict[str, Any]) -> bool:

        del step_info
        return True

    def _cache_feedback_retrieved_tool_meta(self, tool_candidates: list[dict[str, Any]]) -> None:
        for tool_candidate in tool_candidates:
            raw_name = get_tool_display_name(tool_candidate)
            normalized_name = normalize_tool_name(raw_name)
            if not normalized_name:
                continue
            self._feedback_retrieved_tool_meta[normalized_name] = {
                "raw_name": raw_name,
                "tool_meta": tool_candidate,
            }

    def _extract_search_name(self, query: str) -> str:
        head, _, _ = query.partition(":")
        return self._normalize_name(head or query)

    def _normalize_name(self, value: Any) -> str:
        if not isinstance(value, str):
            return ""
        return value.strip().lower()

    def format_tool_response(self, tool_responses: list[str]) -> str:
        rendered = []
        for tool_response in tool_responses:
            rendered.append(f"<tool_response>\n{tool_response}\n</tool_response>")
        return "\n".join(rendered)

    def format_retrieval_response(self, tool_lists: list[str]) -> str:
        rendered_blocks = []
        for tool_list in tool_lists:
            payload = self._parse_retrieved_tools_payload(tool_list)
            if isinstance(payload, list) and payload:
                tool_content = "\n".join(
                    json.dumps(item, ensure_ascii=False) for item in payload if isinstance(item, dict)
                )
            else:
                tool_content = tool_list.strip()
            if tool_content:
                rendered_blocks.append(f"<tools>\n{tool_content}\n</tools>")
        return "\n".join(rendered_blocks)

    def _parse_retrieved_tools_payload(self, tool_blob_text: str) -> list[dict[str, Any]] | None:
        text = tool_blob_text.strip()
        if not text:
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None

        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            return [payload]

        parsed_items: list[dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                parsed_items.append(item)
        return parsed_items

    def build_trajectory_solution_str(self, steps: list) -> str:
        """Build Qwen-style assistant blocks for trajectory-level reward."""
        blocks = []
        for step in steps:
            response_text = step.extra_fields.get("response_text", "")
            blocks.append(f"<|im_start|>assistant\n{response_text}\n<|im_end|>")
        return "\n".join(blocks)
