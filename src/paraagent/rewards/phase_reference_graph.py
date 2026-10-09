"""Build phase dependencies from the task graph, retaining occurrence identity."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from paraagent.paraact.protocol import action_plan_occurrence_tools, action_plan_stage_specs

def _name(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower()

def _names(values: Iterable[Any] | None) -> set[str]:
    if values is None:
        return set()
    return {item for raw in values if (item := _name(raw))}

def _pair_set(values: Iterable[Iterable[Any]] | None) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    if values is None:
        return result
    for raw in values:
        try:
            source, target = raw
        except (TypeError, ValueError):
            continue
        source_name, target_name = _name(source), _name(target)
        if source_name and target_name and source_name != target_name:
            result.add((source_name, target_name))
    return result

def resolve_phase_declared_edges(
    plan_payload: Mapping[str, Any] | None,
) -> set[tuple[str, str]]:
    """Resolve edges before judging layers; keep ambiguous bare names as tool-level edges."""
    if not isinstance(plan_payload, Mapping):
        return set()
    occurrence_tools = action_plan_occurrence_tools(dict(plan_payload))
    if not occurrence_tools:
        return set()
    refs_by_tool: dict[str, list[str]] = defaultdict(list)
    for ref, tool in occurrence_tools.items():
        refs_by_tool[_name(tool)].append(ref)
    tool_names = set(refs_by_tool)

    def resolve(value: Any) -> str:
        token = _name(value)
        if not token:
            return ""
        if token in occurrence_tools:
            return _name(occurrence_tools[token])
        if token in tool_names:
            return token

        available = {
            _name(item) for item in (plan_payload.get("available_tools") or [])
        }
        return token if token in available else ""

    dependencies = plan_payload.get("dependencies")
    if not isinstance(dependencies, list):
        return set()
    edges: set[tuple[str, str]] = set()
    for dependency in dependencies:
        if not isinstance(dependency, Mapping):
            continue
        target = resolve(dependency.get("to"))
        sources = dependency.get("from")
        if not target or not isinstance(sources, list):
            continue
        for source_value in sources:
            source = resolve(source_value)
            if source and source != target:
                edges.add((source, target))
    return edges

@dataclass(frozen=True)
class PhaseReferenceGraph:
    """Canonical phase dependencies and readiness.
    Levels are zero-based; carried/cyclic nodes use -1. Any successful occurrence
    satisfies a tool prerequisite; explicit occurrence chains require their predecessor.
    """

    occurrence_tools: dict[str, str]
    occurrence_edges: frozenset[tuple[str, str]]
    required_edges: frozenset[tuple[str, str]]
    precompleted_tools: frozenset[str]
    carried_refs: frozenset[str]
    non_precompleted_dependency_edges: frozenset[tuple[str, str]]
    levels: dict[str, int]
    unreachable_refs: frozenset[str] = frozenset()
    synthetic_refs: frozenset[str] = frozenset()
    upstream_tools: dict[str, frozenset[str]] = field(default_factory=dict)
    _occurrence_predecessors: dict[str, frozenset[str]] = field(default_factory=dict)

    @property
    def nodes(self) -> frozenset[str]:
        return frozenset(self.occurrence_tools)

    @property
    def layers(self) -> dict[int, tuple[str, ...]]:
        grouped: dict[int, list[str]] = defaultdict(list)
        for ref, level in self.levels.items():
            if level >= 0 and ref not in self.carried_refs:
                grouped[level].append(ref)
        return {
            level: tuple(sorted(refs))
            for level, refs in sorted(grouped.items())
        }

    def ready_occurrences(self, succeeded_refs: Iterable[str] | None = None) -> set[str]:
        """Return canonical occurrences ready at the start of a step."""
        succeeded = {
            ref.strip().lower()
            for ref in (succeeded_refs or ())
            if isinstance(ref, str) and ref.strip().lower() in self.occurrence_tools
        }
        satisfied_refs = succeeded | set(self.carried_refs)
        succeeded_tools = {
            self.occurrence_tools[ref]
            for ref in satisfied_refs
            if ref in self.occurrence_tools
        }
        current_succeeded_tools = {
            self.occurrence_tools[ref]
            for ref in succeeded
            if ref in self.occurrence_tools
        }
        ready: set[str] = set()
        for ref, tool in self.occurrence_tools.items():
            if ref in satisfied_refs or ref in self.unreachable_refs:
                continue
            if not self._occurrence_predecessors.get(ref, frozenset()).issubset(
                satisfied_refs
            ):
                continue
            blocked = False
            for source in self.upstream_tools.get(ref, frozenset()):
                edge = (source, tool)
                if edge in self.non_precompleted_dependency_edges:
                    if source not in current_succeeded_tools:
                        blocked = True
                        break
                elif source not in self.precompleted_tools and source not in succeeded_tools:
                    blocked = True
                    break
            if not blocked:
                ready.add(ref)
        return ready

    def is_ready(self, ref: str, succeeded_refs: Iterable[str] | None = None) -> bool:
        return ref.strip().lower() in self.ready_occurrences(succeeded_refs)

def _synthetic_ref(tool: str, occupied: set[str]) -> str:
    base = f"__phase__:{tool}"
    candidate = base
    suffix = 2
    while candidate in occupied:
        candidate = f"{base}#{suffix}"
        suffix += 1
    return candidate

def _cycle_nodes(nodes: set[str], edges: set[tuple[str, str]]) -> set[str]:
    indegree = {node: 0 for node in nodes}
    outgoing: dict[str, set[str]] = {node: set() for node in nodes}
    for source, target in edges:
        if source not in nodes or target not in nodes:
            continue
        if target not in outgoing[source]:
            outgoing[source].add(target)
            indegree[target] += 1
    queue = deque(node for node, degree in indegree.items() if degree == 0)
    visited: set[str] = set()
    while queue:
        current = queue.popleft()
        visited.add(current)
        for target in outgoing[current]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    return nodes - visited

def build_phase_reference_graph(
    available_tools: set[str],
    occurrence_tools: dict[str, str],
    occurrence_edges: set[tuple[str, str]],
    required_edges: set[tuple[str, str]],
    precompleted_tools: set[str],
    precompleted_occurrences: set[str],
    non_precompleted_dependency_edges: set[tuple[str, str]],
) -> PhaseReferenceGraph:
    """Build a canonical graph independent of model-declared layer numbers."""
    available = _names(available_tools)
    precompleted = _names(precompleted_tools) & available
    raw_occurrences: dict[str, str] = {}
    for raw_ref, raw_tool in (occurrence_tools or {}).items():
        ref = _name(raw_ref)
        tool = _name(raw_tool)
        if ref and tool and tool in available and ref not in raw_occurrences:
            raw_occurrences[ref] = tool

    synthetic: set[str] = set()
    occupied = set(raw_occurrences)
    for tool in sorted(available - set(raw_occurrences.values())):
        ref = _synthetic_ref(tool, occupied)
        occupied.add(ref)
        raw_occurrences[ref] = tool
        synthetic.add(ref)

    nodes = set(raw_occurrences)
    carried = {
        _name(ref)
        for ref in (precompleted_occurrences or set())
        if _name(ref) in nodes
    }
    required = {
        (source, target)
        for source, target in _pair_set(required_edges)
        if source in available and target in available
    }
    non_precompleted = _pair_set(non_precompleted_dependency_edges) & required
    same_tool_edges = {
        (source, target)
        for source, target in _pair_set(occurrence_edges)
        if source in nodes
        and target in nodes
        and raw_occurrences[source] == raw_occurrences[target]
    }

    upstream: dict[str, frozenset[str]] = {
        ref: frozenset(
            source
            for source, target in required
            if target == tool
        )
        for ref, tool in raw_occurrences.items()
    }
    predecessors: dict[str, frozenset[str]] = {
        ref: frozenset(source for source, target in same_tool_edges if target == ref)
        for ref in nodes
    }

    tool_nodes = available - precompleted
    tool_edges = {
        (source, target)
        for source, target in required
        if source in tool_nodes and target in tool_nodes
    }
    cyclic_tools = _cycle_nodes(tool_nodes, tool_edges)
    tool_upstream: dict[str, set[str]] = {tool: set() for tool in tool_nodes}
    for source, target in tool_edges:
        tool_upstream[target].add(source)

    tool_levels: dict[str, int] = {tool: -1 for tool in cyclic_tools}

    def level(tool: str, visiting: set[str]) -> int:
        if tool in tool_levels:
            return tool_levels[tool]
        if tool in cyclic_tools or tool in visiting:
            return -1
        visiting.add(tool)
        parent_levels = [level(parent, visiting) for parent in tool_upstream[tool]]
        visiting.remove(tool)
        if any(parent_level < 0 for parent_level in parent_levels):
            tool_levels[tool] = -1
        else:
            tool_levels[tool] = 0 if not parent_levels else 1 + max(parent_levels)
        return tool_levels[tool]

    for tool in sorted(tool_nodes):
        level(tool, set())

    levels: dict[str, int] = {}
    for ref, tool in raw_occurrences.items():
        levels[ref] = -1 if ref in carried else tool_levels.get(tool, 0)

    cyclic_refs = _cycle_nodes(nodes - carried, same_tool_edges)
    changed = True
    while changed:
        changed = False
        for source, target in sorted(same_tool_edges):
            if source in carried or target in carried or source in cyclic_refs:
                continue
            if levels.get(source, -1) < 0:
                continue
            candidate = levels[target]
            if candidate >= 0 and candidate < levels[source] + 1:
                levels[target] = levels[source] + 1
                changed = True
    unreachable = {
        ref
        for ref, tool in raw_occurrences.items()
        if ref in cyclic_refs or levels.get(ref, -1) < 0
    }
    for ref in unreachable:
        levels[ref] = -1

    return PhaseReferenceGraph(
        occurrence_tools=dict(raw_occurrences),
        occurrence_edges=frozenset(same_tool_edges),
        required_edges=frozenset(required),
        precompleted_tools=frozenset(precompleted),
        carried_refs=frozenset(carried),
        non_precompleted_dependency_edges=frozenset(non_precompleted),
        levels=levels,
        unreachable_refs=frozenset(unreachable),
        synthetic_refs=frozenset(synthetic),
        upstream_tools={ref: frozenset(sources) for ref, sources in upstream.items()},
        _occurrence_predecessors={
            ref: frozenset(sources) for ref, sources in predecessors.items()
        },
    )
