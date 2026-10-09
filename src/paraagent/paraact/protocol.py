from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

MAX_ACCEPTED_PLANS_PER_PHASE = 2
MAX_PLAN_REFRESHES_PER_PHASE = MAX_ACCEPTED_PLANS_PER_PHASE - 1


def max_refreshes_for(action_type: str) -> int:
    return MAX_PLAN_REFRESHES_PER_PHASE


PLAN_CONTROLLER_LIFECYCLE_VERSION = 11

PLAN_REQUIREMENTS = frozenset({"required", "optional", "forbidden"})

_SEARCH_TOOL_BLOCK_RE = re.compile(r"<search_tool>(.*?)</search_tool>", re.DOTALL)
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_ANSWER_BLOCK_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _public_subgoals_format_valid(payload: dict[str, Any]) -> bool:

    if "subgoals" not in payload:
        return True
    subgoals = payload.get("subgoals")
    if not isinstance(subgoals, list) or not subgoals:
        return False
    for item in subgoals:
        if not isinstance(item, dict) or set(item) != {"id", "subgoal"}:
            return False
        if not isinstance(item["id"], str) or not item["id"].strip():
            return False
        if not isinstance(item["subgoal"], str) or not item["subgoal"].strip():
            return False
    return True


def search_plan_format_valid(payload: dict[str, Any] | None) -> bool:

    if not isinstance(payload, dict):
        return False
    if "capacity_slots" not in payload or not set(payload).issubset({"capacity_slots", "subgoals"}):
        return False
    slots = payload.get("capacity_slots")
    return bool(
        isinstance(slots, list)
        and slots
        and all(isinstance(slot, str) and slot.strip() for slot in slots)
        and _public_subgoals_format_valid(payload)
    )


def action_plan_format_valid(payload: dict[str, Any] | None) -> bool:

    if not isinstance(payload, dict):
        return False
    required = {"available_tools", "execution_flow", "dependencies"}
    allowed = required | {"subgoals"}
    if not required.issubset(payload) or not set(payload).issubset(allowed):
        return False

    available_tools = payload.get("available_tools")
    if not (
        isinstance(available_tools, list)
        and available_tools
        and all(isinstance(tool, str) and tool.strip() for tool in available_tools)
    ):
        return False

    execution_flow = payload.get("execution_flow")
    if not isinstance(execution_flow, list) or not execution_flow:
        return False
    for expected_step, stage in enumerate(execution_flow, start=1):
        if not isinstance(stage, dict) or set(stage) != {"step", "parallel"}:
            return False
        step_number = stage.get("step")
        if not isinstance(step_number, int) or isinstance(step_number, bool) or step_number != expected_step:
            return False
        parallel = stage.get("parallel")
        if not (
            isinstance(parallel, list)
            and parallel
            and all(isinstance(tool, str) and tool.strip() for tool in parallel)
        ):
            return False

    dependencies = payload.get("dependencies")
    if not isinstance(dependencies, list):
        return False
    for dependency in dependencies:
        if not isinstance(dependency, dict) or set(dependency) != {"from", "to"}:
            return False
        sources = dependency.get("from")
        target = dependency.get("to")
        if not (
            isinstance(sources, list)
            and sources
            and all(isinstance(source, str) and source.strip() for source in sources)
        ):
            return False
        if not isinstance(target, str) or not target.strip():
            return False
    return _public_subgoals_format_valid(payload)


def parse_executable_tool_calls(
    raw_response: str,
) -> tuple[list[dict[str, Any]], int, int]:

    blocks = _TOOL_CALL_BLOCK_RE.findall(raw_response or "")
    calls: list[dict[str, Any]] = []
    for block in blocks:
        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        arguments = payload.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                continue
        if not isinstance(arguments, dict):
            continue
        calls.append({"name": payload.get("name"), "arguments": arguments})
    return calls, len(blocks), len(blocks) - len(calls)


def parse_response_action(raw_response: str) -> dict[str, Any]:

    text = raw_response or ""
    search_queries = [block.strip() for block in _SEARCH_TOOL_BLOCK_RE.findall(text) if block.strip()]
    tool_calls, tool_call_block_count, invalid_tool_call_count = parse_executable_tool_calls(text)
    answer_count = len(_ANSWER_BLOCK_RE.findall(text))

    action_families: list[str] = []
    if search_queries:
        action_families.append("search_tool")
    if tool_call_block_count:
        action_families.append("tool_call")
    if answer_count:
        action_families.append("answer")

    if len(action_families) == 1:
        action_type = action_families[0]
    elif action_families:
        action_type = "mixed"
    else:
        action_type = "invalid"

    return {
        "action_type": action_type,
        "search_queries": search_queries,
        "tool_calls": tool_calls,
        "tool_call_block_count": tool_call_block_count,
        "invalid_tool_call_count": invalid_tool_call_count,
        "answer_count": answer_count,
    }


def classify_plan_requirement(
    *,
    has_live_controller: bool,
    controller_exhausted: bool,
    previous_recovery_trigger: bool,
    refresh_count: int,
    max_refreshes: int,
    action_type: str = "tool_call",
) -> str:

    refresh_budget_open = max_refreshes < 0 or refresh_count < max_refreshes
    if not has_live_controller:
        return "required"
    if action_type == "search_tool":
        return "optional" if refresh_budget_open else "forbidden"
    if controller_exhausted:
        return "required" if refresh_budget_open else "forbidden"
    if previous_recovery_trigger and refresh_budget_open:
        return "optional"
    return "forbidden"


def score_plan_requirement(
    requirement: str,
    *,
    has_plan: bool,
    plan_format_valid: bool,
) -> float:

    if requirement not in PLAN_REQUIREMENTS:
        raise ValueError(f"unknown plan requirement: {requirement!r}")
    valid_plan = bool(has_plan and plan_format_valid)
    if requirement == "required":
        return 1.0 if valid_plan else 0.0
    if requirement == "optional":
        return 1.0 if not has_plan or valid_plan else 0.0
    return 1.0 if not has_plan else 0.0


def has_explicit_recovery_trigger(
    action_type: str,
    env_info: dict[str, Any] | None,
) -> bool:

    info = env_info if isinstance(env_info, dict) else {}
    observation_visible = info.get(
        "tool_observation_visible_to_model",
        info.get("observation_visible_to_model"),
    )
    if observation_visible is False:
        return False
    if action_type == "tool_call":
        success_flags = info.get("tool_success") if isinstance(info.get("tool_success"), list) else []
        error_types = info.get("tool_error_types") if isinstance(info.get("tool_error_types"), list) else []
        injected_flags = (
            info.get("tool_injected_error_flags")
            if isinstance(info.get("tool_injected_error_flags"), list)
            else []
        )
        outside_available = (
            info.get("tool_called_outside_available_tools")
            if isinstance(info.get("tool_called_outside_available_tools"), list)
            else []
        )
        return any(
            (
                any(not bool(flag) for flag in success_flags),
                any(error_type is not None for error_type in error_types),
                any(bool(flag) for flag in injected_flags),
                any(bool(flag) for flag in outside_available),
                bool(info.get("tool_called_with_missing_dependencies")),
                bool(info.get("tool_called_out_of_order")),
            )
        )
    if action_type == "search_tool":
        retrieval_success = (
            info.get("retrieval_success") if isinstance(info.get("retrieval_success"), list) else []
        )
        retrieval_nonempty = (
            info.get("retrieval_nonempty") if isinstance(info.get("retrieval_nonempty"), list) else []
        )
        retrieval_errors = (
            info.get("retrieval_error_types") if isinstance(info.get("retrieval_error_types"), list) else []
        )
        return any(
            (
                any(flag is False for flag in retrieval_success),
                any(flag is False for flag in retrieval_nonempty),
                any(error_type is not None for error_type in retrieval_errors),
            )
        )
    return False


ANY_ARGUMENTS = "\x00any"


def _normalize_name(value: Any) -> str:
    return value.strip().lower() if isinstance(value, str) else ""


def _canonicalize_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _canonicalize_jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonicalize_jsonable(item) for item in value]
    return value


def canonical_call_arguments(arguments: Any) -> str:

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (TypeError, ValueError):
            return arguments
    try:
        return json.dumps(_canonicalize_jsonable(arguments), ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return repr(arguments)


def make_call_key(name: Any, arguments: Any = None) -> tuple[str, str]:
    return (_normalize_name(name), canonical_call_arguments(arguments))


def normalize_call_keys(values: Any) -> set[tuple[str, str]]:

    if not isinstance(values, (list, tuple, set, frozenset)):
        return set()
    keys: set[tuple[str, str]] = set()
    for value in values:
        if isinstance(value, str):
            name = _normalize_name(value)
            if name:
                keys.add((name, ANY_ARGUMENTS))
            continue
        if isinstance(value, (list, tuple)) and len(value) == 2:
            name = _normalize_name(value[0])
            if not name:
                continue
            raw = value[1]
            keys.add(
                (name, raw) if isinstance(raw, str) and raw == ANY_ARGUMENTS else make_call_key(name, raw)
            )
    return keys


def call_key_names(keys: Any) -> set[str]:
    return {name for name, _ in normalize_call_keys(keys)}


def serialize_call_keys(keys: Any) -> list[list[str]]:

    return [list(key) for key in sorted(normalize_call_keys(keys))]


def _normalize_name_set(values: Any) -> set[str]:
    if not isinstance(values, list):
        return set()
    return {name for name in (_normalize_name(value) for value in values) if name}


def _normalize_name_counter(values: Any) -> Counter[str]:
    if not isinstance(values, list):
        return Counter()
    return Counter(name for name in (_normalize_name(value) for value in values) if name)


def _normalize_names(values: Any) -> set[str]:
    if not isinstance(values, (list, tuple, set, frozenset)):
        return set()
    return {name for name in (_normalize_name(value) for value in values) if name}


def normalize_action_plan_ref(value: Any) -> str:

    return _normalize_name(value)


def normalize_action_plan_node(value: Any) -> dict[str, str] | None:

    if not isinstance(value, str):
        return None
    tool = _normalize_name(value)
    return {"ref": tool, "tool": tool, "legacy": "1"} if tool else None


def action_plan_stage_specs(
    plan_payload: dict[str, Any] | None,
) -> list[list[dict[str, str]]]:
    if not isinstance(plan_payload, dict):
        return []
    raw_stages: list[list[dict[str, str]]] = []
    tool_counts: Counter[str] = Counter()
    for stage in plan_payload.get("execution_flow", []) or []:
        if not isinstance(stage, dict):
            return []
        parallel = stage.get("parallel")
        if not isinstance(parallel, list) or not parallel:
            return []
        nodes = [normalize_action_plan_node(item) for item in parallel]
        if any(node is None for node in nodes):
            return []
        normalized = [node for node in nodes if node is not None]
        raw_stages.append(normalized)
        tool_counts.update(node["tool"] for node in normalized if node.get("legacy") == "1")

    ordinals: Counter[str] = Counter()
    stages: list[list[dict[str, str]]] = []
    for raw_stage in raw_stages:
        stage_nodes: list[dict[str, str]] = []
        for raw_node in raw_stage:
            node = dict(raw_node)
            if node.get("legacy") == "1" and tool_counts[node["tool"]] > 1:
                ordinals[node["tool"]] += 1
                node["ref"] = f"{node['tool']}#{ordinals[node['tool']]}"
            stage_nodes.append(node)
        stages.append(stage_nodes)
    return stages


def _declared_dependency_edges(
    plan_payload: dict[str, Any] | None,
    available_refs: set[str] | None = None,
) -> set[tuple[str, str]]:
    if not isinstance(plan_payload, dict):
        return set()
    dependencies = plan_payload.get("dependencies")
    if dependencies is None:
        dependencies = []
    if not isinstance(dependencies, list):
        return set()

    stage_specs = action_plan_stage_specs(plan_payload)
    flat_nodes = [node for stage in stage_specs for node in stage]
    if available_refs is None:
        available_refs = {node["ref"] for node in flat_nodes}

    refs_by_tool: dict[str, list[str]] = {}
    for node in flat_nodes:
        refs_by_tool.setdefault(node["tool"], []).append(node["ref"])

    edges, _valid = _resolve_explicit_dependency_edges(
        plan_payload,
        available_refs=available_refs,
        flat_nodes=flat_nodes,
    )

    for refs in refs_by_tool.values():
        refs_by_layer: list[list[str]] = []
        for stage in stage_specs:
            layer_refs = [node["ref"] for node in stage if node["ref"] in refs and node.get("legacy") == "1"]
            if layer_refs:
                refs_by_layer.append(layer_refs)
        for previous, current in zip(refs_by_layer, refs_by_layer[1:]):
            edges.update((source, target) for source in previous for target in current)
    return edges


def _resolve_explicit_dependency_edges(
    plan_payload: dict[str, Any] | None,
    *,
    available_refs: set[str] | None = None,
    flat_nodes: list[dict[str, str]] | None = None,
) -> tuple[set[tuple[str, str]], bool]:

    if not isinstance(plan_payload, dict):
        return set(), False
    dependencies = plan_payload.get("dependencies")
    if dependencies is None:
        dependencies = []
    if not isinstance(dependencies, list):
        return set(), False
    if flat_nodes is None:
        flat_nodes = [node for stage in action_plan_stage_specs(plan_payload) for node in stage]
    if available_refs is None:
        available_refs = {node["ref"] for node in flat_nodes}
    layer_by_ref = {
        node["ref"]: layer_index
        for layer_index, stage in enumerate(action_plan_stage_specs(plan_payload))
        for node in stage
    }
    refs_by_tool: dict[str, list[str]] = {}
    for node in flat_nodes:
        refs_by_tool.setdefault(node["tool"], []).append(node["ref"])

    def _candidates(value: Any) -> list[str]:
        normalized = normalize_action_plan_ref(value)
        if normalized in available_refs:
            return [normalized]
        return list(refs_by_tool.get(normalized, []))

    edges: set[tuple[str, str]] = set()
    for dependency in dependencies:
        if not isinstance(dependency, dict):
            return set(), False
        targets = _candidates(dependency.get("to"))
        source_values = dependency.get("from", [])
        if not targets or not isinstance(source_values, list) or not source_values:
            return set(), False
        source_candidates = [_candidates(value) for value in source_values]
        if any(not candidates for candidates in source_candidates):
            return set(), False

        feasible: list[tuple[str, list[str]]] = []
        for target in targets:
            chosen_sources: list[str] = []
            for candidates in source_candidates:
                earlier = [
                    ref
                    for ref in candidates
                    if ref != target and layer_by_ref.get(ref, -1) < layer_by_ref.get(target, -1)
                ]
                if not earlier:
                    break
                latest_layer = max(layer_by_ref[ref] for ref in earlier)
                latest = [ref for ref in earlier if layer_by_ref[ref] == latest_layer]
                if len(latest) != 1:
                    break
                chosen_sources.append(latest[0])
            if len(chosen_sources) == len(source_candidates):
                feasible.append((target, chosen_sources))

        if not feasible:
            return set(), False
        earliest_layer = min(layer_by_ref[target] for target, _ in feasible)
        earliest = [item for item in feasible if layer_by_ref[item[0]] == earliest_layer]
        if len(earliest) != 1:
            return set(), False
        target, chosen_sources = earliest[0]
        edges.update((source, target) for source in chosen_sources)
    return edges, True


def action_plan_dependencies_valid(
    plan_payload: dict[str, Any] | None,
) -> bool:
    occurrences = action_plan_occurrence_tools(plan_payload)
    if not occurrences:
        return False
    flat_nodes = [node for stage in action_plan_stage_specs(plan_payload) for node in stage]
    _edges, valid = _resolve_explicit_dependency_edges(
        plan_payload,
        available_refs=set(occurrences),
        flat_nodes=flat_nodes,
    )
    return valid


def search_plan_frontier(plan_payload: dict[str, Any] | None) -> set[str]:

    if not isinstance(plan_payload, dict):
        return set()
    return set(search_plan_capacity(plan_payload))


def search_plan_capacity(
    plan_payload: dict[str, Any] | None,
) -> Counter[str]:

    if not isinstance(plan_payload, dict):
        return Counter()
    return _normalize_name_counter(plan_payload.get("capacity_slots"))


def action_plan_available_tools(plan_payload: dict[str, Any] | None) -> set[str]:
    if not isinstance(plan_payload, dict):
        return set()
    return _normalize_name_set(plan_payload.get("available_tools"))


def action_plan_frontier(plan_payload: dict[str, Any] | None) -> set[str]:

    return {node["tool"] for stage in action_plan_stage_specs(plan_payload) for node in stage}


def action_plan_occurrence_tools(
    plan_payload: dict[str, Any] | None,
) -> dict[str, str]:

    stages = action_plan_stage_specs(plan_payload)
    if not stages:
        return {}
    result: dict[str, str] = {}
    for stage in stages:
        for node in stage:
            if node["ref"] in result:
                return {}
            result[node["ref"]] = node["tool"]
    return result


def action_plan_occurrence_order(
    plan_payload: dict[str, Any] | None,
) -> tuple[str, ...]:
    return tuple(node["ref"] for stage in action_plan_stage_specs(plan_payload) for node in stage)


def action_plan_dependency_edges(
    plan_payload: dict[str, Any] | None,
) -> set[tuple[str, str]]:

    occurrences = action_plan_occurrence_tools(plan_payload)
    if not occurrences:
        return set()
    return _declared_dependency_edges(plan_payload, set(occurrences))


def action_plan_active_stage(
    plan_payload: dict[str, Any] | None,
    succeeded_tools: Any,
) -> set[str]:

    progress = action_controller_progress(plan_payload, succeeded_tools=succeeded_tools)
    return set(progress.active_stage_tools)


@dataclass(frozen=True)
class ActionControllerProgress:
    scheduled_tools: frozenset[str]
    issued_tools: frozenset[str]
    succeeded_tools: frozenset[str]
    failed_tools: frozenset[str]
    remaining_tools: frozenset[str]
    active_stage_tools: frozenset[str]
    exhausted: bool
    has_unresolved_failure: bool

    issued_calls: frozenset[tuple[str, str]] = frozenset()
    succeeded_calls: frozenset[tuple[str, str]] = frozenset()
    failed_calls: frozenset[tuple[str, str]] = frozenset()

    stale_succeeded_calls: frozenset[tuple[str, str]] = frozenset()

    revisitable_tools: frozenset[str] = frozenset()

    scheduled_occurrences: frozenset[str] = frozenset()
    issued_occurrences: frozenset[str] = frozenset()
    succeeded_occurrences: frozenset[str] = frozenset()
    failed_occurrences: frozenset[str] = frozenset()
    remaining_occurrences: frozenset[str] = frozenset()
    ready_occurrences: frozenset[str] = frozenset()
    occurrence_tools: tuple[tuple[str, str], ...] = ()
    dependency_edges: frozenset[tuple[str, str]] = frozenset()

    def is_duplicate_call(self, name: Any, arguments: Any = None) -> bool:

        key = make_call_key(name, arguments)
        return key in self.succeeded_calls or key in self.stale_succeeded_calls


def action_controller_progress(
    plan_payload: dict[str, Any] | None,
    *,
    issued_tools: Any = (),
    succeeded_tools: Any = (),
    failed_tools: Any = (),
    stale_succeeded_calls: Any = (),
    revisitable_tools: Any = (),
    issued_occurrences: Any = None,
    succeeded_occurrences: Any = None,
    failed_occurrences: Any = None,
) -> ActionControllerProgress:

    occurrence_tools = action_plan_occurrence_tools(plan_payload)
    occurrence_order = action_plan_occurrence_order(plan_payload)
    scheduled_occurrences = set(occurrence_tools)
    scheduled = set(occurrence_tools.values())
    issued_calls = {key for key in normalize_call_keys(issued_tools) if key[0] in scheduled}
    succeeded_calls = {key for key in normalize_call_keys(succeeded_tools) if key[0] in scheduled}
    failed_calls = {key for key in normalize_call_keys(failed_tools) if key[0] in scheduled}

    def _explicit_refs(values: Any) -> set[str]:
        if not isinstance(values, (list, tuple, set, frozenset)):
            return set()
        return {
            ref
            for ref in (normalize_action_plan_ref(value) for value in values)
            if ref in scheduled_occurrences
        }

    def _infer_refs(call_keys: set[tuple[str, str]]) -> set[str]:

        counts = Counter(name for name, _ in call_keys)
        inferred: set[str] = set()
        for ref in occurrence_order:
            tool = occurrence_tools.get(ref, "")
            if counts[tool] > 0:
                inferred.add(ref)
                counts[tool] -= 1
        return inferred

    issued_refs = (
        _infer_refs(issued_calls) if issued_occurrences is None else _explicit_refs(issued_occurrences)
    )
    succeeded_refs = (
        _infer_refs(succeeded_calls)
        if succeeded_occurrences is None
        else _explicit_refs(succeeded_occurrences)
    )
    failed_refs = (
        _infer_refs(failed_calls) if failed_occurrences is None else _explicit_refs(failed_occurrences)
    ) - succeeded_refs
    succeeded = {occurrence_tools[ref] for ref in succeeded_refs}
    failed_calls = {key for key in failed_calls if key[0] not in succeeded}
    issued = {occurrence_tools[ref] for ref in issued_refs}
    failed = {occurrence_tools[ref] for ref in failed_refs}
    remaining_refs = scheduled_occurrences - succeeded_refs
    remaining = {occurrence_tools[ref] for ref in remaining_refs}
    dependency_edges = _declared_dependency_edges(plan_payload, scheduled_occurrences)
    predecessors: dict[str, set[str]] = {ref: set() for ref in scheduled_occurrences}
    for source, target in dependency_edges:
        predecessors[target].add(source)
    ready_refs = {ref for ref in remaining_refs if predecessors.get(ref, set()).issubset(succeeded_refs)}
    active_stage = {occurrence_tools[ref] for ref in ready_refs}
    return ActionControllerProgress(
        scheduled_tools=frozenset(scheduled),
        issued_tools=frozenset(issued),
        succeeded_tools=frozenset(succeeded),
        failed_tools=frozenset(failed),
        remaining_tools=frozenset(remaining),
        active_stage_tools=frozenset(active_stage),
        exhausted=bool(scheduled_occurrences and not remaining_refs),
        has_unresolved_failure=bool(failed),
        issued_calls=frozenset(issued_calls),
        succeeded_calls=frozenset(succeeded_calls),
        failed_calls=frozenset(failed_calls),
        stale_succeeded_calls=frozenset(normalize_call_keys(stale_succeeded_calls)),
        revisitable_tools=frozenset(_normalize_names(revisitable_tools) & succeeded),
        scheduled_occurrences=frozenset(scheduled_occurrences),
        issued_occurrences=frozenset(issued_refs),
        succeeded_occurrences=frozenset(succeeded_refs),
        failed_occurrences=frozenset(failed_refs),
        remaining_occurrences=frozenset(remaining_refs),
        ready_occurrences=frozenset(ready_refs),
        occurrence_tools=tuple((ref, occurrence_tools[ref]) for ref in occurrence_order),
        dependency_edges=frozenset(dependency_edges),
    )


def match_action_calls_to_occurrences(
    progress: ActionControllerProgress,
    calls: Any,
) -> list[str | None]:

    available_by_tool: dict[str, list[str]] = {}
    for ref, tool in progress.occurrence_tools:
        if ref in progress.ready_occurrences:
            available_by_tool.setdefault(tool, []).append(ref)

    matched: list[str | None] = []
    last_match_by_tool: dict[str, str] = {}
    for entry in calls or ():
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            name, arguments = _normalize_name(entry[0]), entry[1]
        else:
            name, arguments = _normalize_name(entry), None
        candidates = available_by_tool.get(name, [])
        if candidates:
            ref = candidates.pop(0)
            last_match_by_tool[name] = ref
            matched.append(ref)
            continue

        if name in last_match_by_tool:
            matched.append(last_match_by_tool[name])
            continue
        if (
            name in progress.revisitable_tools
            and make_call_key(name, arguments) not in progress.succeeded_calls
        ):
            matched.append("")
            continue
        matched.append(None)
    return matched


def action_calls_follow_flow(
    progress: ActionControllerProgress,
    calls: Any,
) -> bool | None:

    if progress.exhausted or not progress.ready_occurrences:
        return None
    matches = match_action_calls_to_occurrences(progress, calls)
    return all(match is not None for match in matches)


def plan_controller_signature(
    action_type: str,
    plan_payload: dict[str, Any] | None,
) -> Any:

    if not isinstance(plan_payload, dict):
        return None
    if action_type == "search_tool":
        return tuple(sorted(search_plan_capacity(plan_payload).elements()))
    if action_type != "tool_call":
        return None

    available_set = action_plan_available_tools(plan_payload)
    available = tuple(sorted(available_set))
    flow = tuple(
        tuple(sorted((node["ref"], node["tool"]) for node in stage))
        for stage in action_plan_stage_specs(plan_payload)
    )
    refs = set(action_plan_occurrence_tools(plan_payload))
    dependencies = tuple(sorted(_declared_dependency_edges(plan_payload, refs)))
    return available, flow, dependencies


def action_plan_execution_signature(
    plan_payload: dict[str, Any] | None,
) -> Any:

    if not isinstance(plan_payload, dict):
        return None
    occurrence_tools = action_plan_occurrence_tools(plan_payload)
    if not occurrence_tools:
        return None
    flow = tuple(
        tuple(sorted((node["ref"], node["tool"]) for node in stage))
        for stage in action_plan_stage_specs(plan_payload)
    )
    dependencies = tuple(sorted(_declared_dependency_edges(plan_payload, set(occurrence_tools))))
    return flow, dependencies


def plan_new_capacity(
    action_type: str,
    current_plan: dict[str, Any] | None,
    live_plan: dict[str, Any] | None,
) -> set[str]:

    if action_type == "tool_call":
        return action_plan_frontier(current_plan) - action_plan_frontier(live_plan)
    if action_type == "search_tool":
        current = search_plan_capacity(current_plan)
        live = search_plan_capacity(live_plan)
        return {name for name in current if current[name] > live[name]}
    return set()


def plan_abandons_unfinished_work(
    current_plan: dict[str, Any] | None,
    live_plan: dict[str, Any] | None,
    unfinished_tools: Any,
) -> bool:

    unfinished_refs = _normalize_names(unfinished_tools)
    if not unfinished_refs:
        return False
    live_occurrences = action_plan_occurrence_tools(live_plan)
    current_occurrences = action_plan_occurrence_tools(current_plan)
    exact_kept = {
        ref
        for ref in unfinished_refs
        if ref in live_occurrences and current_occurrences.get(ref) == live_occurrences.get(ref)
    }
    unmatched_refs = {ref for ref in unfinished_refs if ref in live_occurrences} - exact_kept

    new_ref_capacity = Counter(
        tool for ref, tool in current_occurrences.items() if ref not in live_occurrences
    )
    unmatched_by_tool: dict[str, list[str]] = {}
    for ref in unmatched_refs:
        unmatched_by_tool.setdefault(live_occurrences[ref], []).append(ref)
    same_tool_kept: set[str] = set()
    for tool, refs in unmatched_by_tool.items():
        keep_count = min(len(refs), new_ref_capacity.get(tool, 0))
        same_tool_kept.update(sorted(refs)[:keep_count])
        new_ref_capacity[tool] -= keep_count

    dropped_refs = unmatched_refs - same_tool_kept

    live_tools = set(live_occurrences.values())
    current_tools = set(current_occurrences.values())
    unknown_legacy_tools = {
        value
        for value in unfinished_refs
        if value not in live_occurrences and value in live_tools and value not in current_tools
    }
    dropped_count = len(dropped_refs) + len(unknown_legacy_tools)
    if dropped_count <= 0:
        return False

    genuinely_new_tools = current_tools - live_tools
    return len(genuinely_new_tools) < dropped_count


@dataclass(frozen=True)
class PlanRefreshDecision:
    accepted: bool
    kind: str
    reason: str
    same_controller: bool


def classify_plan_refresh(
    action_type: str,
    current_plan: dict[str, Any] | None,
    live_plan: dict[str, Any] | None,
    *,
    plan_valid: bool,
    plan_requirement: str,
    controller_unfinished: set[str] | None = None,
    refresh_count: int = 0,
    max_refreshes: int | None = None,
) -> PlanRefreshDecision:

    if plan_requirement not in PLAN_REQUIREMENTS:
        raise ValueError(f"unknown plan requirement: {plan_requirement!r}")
    if max_refreshes is None:
        max_refreshes = max_refreshes_for(action_type)
    current_signature = plan_controller_signature(action_type, current_plan)
    live_signature = plan_controller_signature(action_type, live_plan)
    same_controller = bool(
        current_signature is not None and live_signature is not None and current_signature == live_signature
    )
    if action_type == "tool_call" and not same_controller:
        current_execution_signature = action_plan_execution_signature(current_plan)
        live_execution_signature = action_plan_execution_signature(live_plan)
        same_controller = bool(
            current_execution_signature is not None
            and current_execution_signature == live_execution_signature
        )

    if not plan_valid or current_signature is None:
        return PlanRefreshDecision(False, "invalid", "plan_invalid", same_controller)

    def _spend(kind: str, reason: str) -> PlanRefreshDecision:

        if 0 <= max_refreshes <= refresh_count:
            return PlanRefreshDecision(False, "limited", "refresh_limit_reached", False)
        return PlanRefreshDecision(True, kind, reason, False)

    if plan_requirement == "forbidden":
        return PlanRefreshDecision(False, "forbidden", "plan_forbidden_by_protocol", same_controller)

    if same_controller:
        if (
            action_type == "tool_call"
            and plan_requirement == "required"
            and controller_unfinished is not None
            and not controller_unfinished
        ):
            return _spend("restart", "controller_exhausted")
        return PlanRefreshDecision(False, "redundant", "same_controller", True)

    if live_plan is None or live_signature is None:
        return PlanRefreshDecision(True, "initial", "controller_missing", False)

    if controller_unfinished is None:
        return _spend("revise", "controller_revised")
    if not controller_unfinished:
        return _spend("restart", "controller_exhausted")
    if plan_abandons_unfinished_work(current_plan, live_plan, controller_unfinished):
        return PlanRefreshDecision(False, "unsupported", "abandons_unfinished_work", False)
    return _spend("revise", "controller_revised")
