
import argparse
import json
from pathlib import Path
import statistics

from paraagent.evaluation.run import benchmark_tasks
from paraagent.evaluation.api_config import load_api_config, public_api_config


def score_runs(root, benchmark, predictions, runs, scoring_config=None):
    tasks = benchmark_tasks(root, benchmark)
    if benchmark == "toolbench" and scoring_config is None:
        raise ValueError("ToolBench scoring requires --scoring-config")
    toolbench_sources = set()
    if benchmark == "toolbench":
        for run in runs:
            for split, qid, _ in tasks:
                path = Path(predictions) / run / split / f"{qid}_Agent@1.json"
                if not path.is_file():
                    raise FileNotFoundError(f"Incomplete run; missing {path}")
                prediction = json.loads(path.read_text())
                toolbench_sources.add((prediction.get("toolbench_backend", "unknown_legacy"),
                                       prediction.get("toolbench_service_url"),
                                       prediction.get("toolbench_cache_policy", "unknown_legacy")))
        if len(toolbench_sources) != 1:
            raise ValueError("ToolBench predictions mix observation backends or service endpoints")
    if benchmark == "toolbench":
        from paraagent.evaluation.toolbench.judge import load_evaluator
        from paraagent.evaluation.toolbench.metrics import compute_tool_hit_metrics, extract_called_tools, load_toolbench_name_api_set
        from paraagent.evaluation.toolbench.trace import convert_result
        judge = load_evaluator(profile_name="completeness",
            profiles_root=str(Path(__file__).parent / "toolbench/judge"), api_config=scoring_config)
        names = load_toolbench_name_api_set(str(Path(root) / "data/benchmarks/toolbench/name_tool.tsv"))
    else:
        from paraagent.evaluation.apibank.score import evaluate_level3_sample
    all_scores = []
    for run in runs:
        result = {}
        for split, qid, query in tasks:
            path = Path(predictions) / run / split / f"{qid}_Agent@1.json"
            if not path.is_file():
                raise FileNotFoundError(f"Incomplete run; missing {path}")
            raw_result = json.loads(path.read_text())
            answer = raw_result["answer_generation"]
            if benchmark == "apibank":
                entry = evaluate_level3_sample(query, answer)
            else:
                converted = convert_result(raw_result)
                answer = converted["answer"]
                final = answer.get("final_answer", "")
                if isinstance(final, dict):
                    final = json.dumps(final, ensure_ascii=False)
                called_tools = extract_called_tools(answer["answer_details"])
                if not called_tools or called_tools[-1] != "Finish" or not final.strip():
                    entry = {"success": 0.0, "judge_credit": 0.0, "judge_status": "Unsolved",
                             "reason": "Final tool call must be Finish with a nonempty answer"}
                else:
                    try:
                        judge_answer = {**answer, "final_answer": final}
                        status, reason = judge.check_is_solved({"query": query["query"], "available_tools": converted["available_tools"]}, judge_answer, return_reason=True)
                        entry = {"success": float(status.name == "Solved"),
                                 "judge_credit": {"Solved": 1.0, "Unsure": 0.5, "Unsolved": 0.0}[status.name],
                                 "judge_status": status.name, "reason": reason}
                    except Exception as exc:
                        if "content_filter" not in str(exc):
                            raise
                        entry = {"success": 0.0, "judge_credit": 0.5, "judge_status": "Unsure",
                                 "reason": "Evaluator content_filter (Unsure)"}
                entry.update(compute_tool_hit_metrics({"answer": answer}, query.get("relevant APIs", []), names))
            result[f"{split}/{qid}"] = entry
        all_scores.append(result)
        (Path(predictions) / run / "scores.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    keys = list(all_scores[0])
    per_run = [statistics.mean(row[k]["success"] for k in keys) for row in all_scores]
    summary = {"benchmark": benchmark, "tasks": len(keys), "runs": runs,
        "success_per_run": per_run, "pass_at_k": statistics.mean(any(r[k]["success"] for r in all_scores) for k in keys),
        "k": len(runs)}
    if benchmark == "toolbench":
        summary["judge_pass_rate_per_run"] = [statistics.mean(r[k]["judge_credit"] for k in keys) for r in all_scores]
        split_keys = {}
        for split, qid, _ in tasks:
            split_keys.setdefault(split, []).append(f"{split}/{qid}")
        summary["judge_pass_rate_by_split_per_run"] = [
            {split: statistics.mean(row[key]["judge_credit"] for key in subset)
             for split, subset in split_keys.items()} for row in all_scores]
        summary["scoring_api"] = public_api_config(scoring_config)
        backend, service_url, cache_policy = next(iter(toolbench_sources))
        summary["toolbench_backend"] = backend
        summary["toolbench_cache_policy"] = cache_policy
        if service_url is not None:
            summary["toolbench_service_url"] = service_url
        summary["tool_path_per_run"] = [statistics.mean(r[k]["tool_hit_ratio"] for k in keys) for r in all_scores]
    else:
        summary["api_accuracy_per_run"] = [sum(x["correct_api_calls"] for x in r.values()) / sum(x["gt_api_calls"] for x in r.values()) for r in all_scores]
    return summary


def main():
    parser = argparse.ArgumentParser(description="Score complete ToolBench or API-Bank evaluation runs.")
    parser.add_argument("--benchmark", choices=["toolbench", "apibank"], required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", default=["run-1"])
    parser.add_argument("--scoring-config", type=Path, help="ToolBench judge JSON with model, base_url and api_key_env.")
    args = parser.parse_args()
    if len(set(args.runs)) != len(args.runs):
        parser.error("Each run must be distinct")
    if args.benchmark == "toolbench" and not args.scoring_config:
        parser.error("--scoring-config is required for ToolBench")
    if args.benchmark != "toolbench" and args.scoring_config:
        parser.error("--scoring-config only applies to ToolBench")
    try:
        scoring_config = load_api_config(args.scoring_config) if args.scoring_config else None
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    result = score_runs(args.root, args.benchmark, args.predictions, args.runs, scoring_config)
    (args.predictions / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
