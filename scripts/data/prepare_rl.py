#!/usr/bin/env python3
"""Convert released ParaAgent RL tasks to the training Parquet schema."""

import argparse
from collections import Counter
import json
import re
from pathlib import Path
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCES = {"toolbench", "toucan", "simia"}
TASK_FIELDS = {
    "task_id", "dataset", "question", "context", "target_tool_names",
    "expected",
}
EXPECTED_FIELDS = {"task_type", "actions", "outputs", "dependencies"}


def training_schema():
    return pa.schema([
        ("data_source", pa.large_string()),
        ("agent_name", pa.large_string()),
        ("prompt", pa.list_(pa.struct([("content", pa.string()), ("role", pa.string())]))),
        ("reward_model", pa.struct([("ground_truth", pa.string()), ("style", pa.string())])),
        ("extra_info", pa.struct([
            ("index", pa.int64()), ("sample_id", pa.string()), ("question", pa.string()),
            ("split", pa.string()), ("target_tool_names", pa.list_(pa.string())),
            ("is_tau_task", pa.bool_()), ("task_type", pa.string()),
            ("gt_write_calls_json", pa.string()), ("gt_outputs_json", pa.string()),
            ("tool_dependency_pairs_graph", pa.string()),
            ("tau_require_complete_writes", pa.bool_()),
            ("tau_require_exact_write_args", pa.bool_()),
        ])),
        ("env_kwargs", pa.struct([("env_type", pa.string())])),
    ])


def convert_task(task, *, split, index, system, data_dir):
    if not isinstance(task, dict) or set(task) != TASK_FIELDS:
        raise ValueError("Task fields must be: " + ", ".join(sorted(TASK_FIELDS)))
    sample_id = task["task_id"]
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError("task_id must be a nonempty string")
    source = task["dataset"]
    if source not in SOURCES:
        raise ValueError(f"{sample_id}: unsupported dataset {source!r}")
    question, context = task["question"], task["context"]
    if not isinstance(question, str) or not question.strip() or not isinstance(context, str):
        raise ValueError(f"{sample_id}: question and context must be strings")
    targets = task["target_tool_names"]
    if not isinstance(targets, list) or not targets or any(
        not isinstance(name, str) or not name.strip() for name in targets
    ):
        raise ValueError(f"{sample_id}: target_tool_names must be nonempty tool names")
    expected = task["expected"]
    if not isinstance(expected, dict) or set(expected) != EXPECTED_FIELDS:
        raise ValueError(f"{sample_id}: expected must contain task_type, actions, outputs, and dependencies")
    extra = {
        "index": index, "sample_id": sample_id, "question": question,
        "split": split, "target_tool_names": targets,
        "tool_dependency_pairs_graph": expected["dependencies"],
    }
    if expected["dependencies"] is not None:
        if not isinstance(expected["dependencies"], str):
            raise ValueError(f"{sample_id}: expected.dependencies must be JSON text")
        graph = json.loads(expected["dependencies"])
        if not isinstance(graph, dict) or not isinstance(graph.get("edges"), list):
            raise ValueError(f"{sample_id}: dependency graph requires an edges list")
        if any(
            not isinstance(edge, list) or len(edge) not in {2, 3}
            or any(not isinstance(value, str) or not value.strip() for value in edge)
            for edge in graph["edges"]
        ):
            raise ValueError(f"{sample_id}: dependency edges must name two tools and an optional relation")
    if source == "simia":
        match = re.fullmatch(r"simia_(airline|retail)_[0-9]{6}", sample_id)
        if match is None:
            raise ValueError(f"{sample_id}: expected simia_airline_NNNNNN or simia_retail_NNNNNN")
        domain = match.group(1)
        if expected["task_type"] not in {"write", "inquiry"}:
            raise ValueError(f"{sample_id}: unsupported expected.task_type")
        decoded = {}
        for key, expected_type in (
            ("actions", list), ("outputs", list), ("dependencies", dict),
        ):
            if not isinstance(expected[key], str):
                raise ValueError(f"{sample_id}: expected.{key} must be JSON text")
            decoded[key] = json.loads(expected[key])
            if not isinstance(decoded[key], expected_type):
                raise ValueError(f"{sample_id}: invalid expected.{key}")
        is_write = expected["task_type"] == "write"
        if bool(decoded["actions"]) != is_write:
            raise ValueError(f"{sample_id}: write calls disagree with task_type")
        for action in decoded["actions"]:
            if (not isinstance(action, dict) or not isinstance(action.get("tool"), str)
                    or action["tool"] not in targets or not isinstance(action.get("args"), dict)):
                raise ValueError(f"{sample_id}: each expected action requires a target tool and argument object")
        obligation_ids = set()
        for output in decoded["outputs"]:
            obligation_id = output.get("obligation_id") if isinstance(output, dict) else None
            if (not isinstance(obligation_id, str) or not obligation_id.startswith(sample_id + ":")
                    or len(obligation_id) <= len(sample_id) + 1):
                raise ValueError(f"{sample_id}: each expected output requires an obligation_id linked to this task")
            if obligation_id in obligation_ids:
                raise ValueError(f"{sample_id}: duplicate obligation_id {obligation_id}")
            obligation_ids.add(obligation_id)
        state_path = data_dir / "state" / f"{sample_id}.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(state, dict) or state.get("__shared_db__") != domain:
            raise ValueError(f"{sample_id}: state database does not match its domain")
        collections = ("flights", "reservations", "users") if domain == "airline" else (
            "orders", "products", "users"
        )
        for collection in collections:
            required = data_dir / "full_db" / domain / "data" / f"{collection}.json"
            if not required.is_file():
                raise FileNotFoundError(required)
        extra.update({
            "task_type": expected["task_type"],
            "gt_write_calls_json": expected["actions"],
            "gt_outputs_json": expected["outputs"],
            "is_tau_task": True,
            "tau_require_complete_writes": is_write,
            "tau_require_exact_write_args": is_write,
        })
    elif any(expected[key] is not None for key in ("task_type", "actions", "outputs")):
        raise ValueError(f"{sample_id}: execution annotations must be null for {source}")
    return {
        "data_source": "Simia" if source == "simia" else source,
        "agent_name": "agent_env_loop",
        "prompt": [
            {"role": "system", "content": system},
            {"role": "user", "content": context + question},
        ],
        "reward_model": {"ground_truth": question, "style": "rule"},
        "extra_info": extra,
        "env_kwargs": {"env_type": "toolenv"},
    }


def prepare(data_dir, output_dir, system_prompt):
    system = system_prompt.read_text(encoding="utf-8")
    if not system.strip():
        raise ValueError("The system prompt is empty")
    seen = set()
    tables, report = {}, {}
    for split in ("train", "validation"):
        records = []
        path = data_dir / f"{split}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    task = json.loads(line)
                    row = convert_task(
                        task, split=split, index=len(seen) + 1, system=system, data_dir=data_dir
                    )
                    sample_id = row["extra_info"]["sample_id"]
                    if sample_id in seen:
                        raise ValueError(f"Duplicate or cross-split sample ID: {sample_id}")
                    seen.add(sample_id)
                    records.append(row)
                except (ValueError, TypeError, KeyError, OSError) as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
        if not records:
            raise ValueError(f"{path}: split is empty")
        tables[split] = pa.Table.from_pylist(records, schema=training_schema())
        report[split] = {
            "rows": len(records),
            "sources": dict(Counter(row["data_source"] for row in records)),
        }
    # Validate both splits before replacing either existing training file.
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-rl-", dir=output_dir) as temporary:
        for split, table in tables.items():
            pq.write_table(table, Path(temporary) / f"{split}.parquet", compression="zstd")
        for split in tables:
            (Path(temporary) / f"{split}.parquet").replace(output_dir / f"{split}.parquet")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/rl/paraagent-rl")
    parser.add_argument("--output-dir", type=Path,
                        help="Write Parquet here; defaults to --data-dir. Resources are not copied.")
    parser.add_argument("--system-prompt", type=Path,
                        default=PROJECT_ROOT / "configs/prompts/paraagent.txt")
    args = parser.parse_args()
    report = prepare(args.data_dir, args.output_dir or args.data_dir, args.system_prompt)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
