
import argparse
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

from paraagent.evaluation.api_config import load_api_config, public_api_config
from paraagent.evaluation.experiment import fingerprint, prediction_identity
from paraagent.security import validate_public_url


def benchmark_tasks(root, benchmark):
    base = Path(root) / "data/benchmarks" / benchmark
    if benchmark == "apibank":
        return [("level-3", i, q) for i, q in enumerate(json.loads((base / "level-3.json").read_text()))]
    tasks = []
    for path in sorted((base / "test_instruction").glob("G*.json")):
        for index, q in enumerate(json.loads(path.read_text())):
            tasks.append((path.stem, q.get("query_id", index), q))
    if not tasks:
        raise FileNotFoundError(f"No benchmark tasks in {base}")
    return tasks


def make_environment(root, benchmark, query, paradigm="EaE", *, toolbench_backend="simulator",
                     toolbench_service_url=None, toolbench_key=None):
    args = SimpleNamespace(
        paradigm=paradigm, f_top_k=int(os.environ.get("TOOL_SEARCH_DEFAULT_TOP_K", "3")), max_observation_length=8000,
        observ_compress_method="truncate",
        tool_root_dir=str(Path(root) / "data/benchmarks/toolbench/name_tool.tsv"),
        apibank_apis_dir=str(Path(root) / "data/benchmarks/apibank/lv3_apis"),
        apibank_database_dir=str(Path(root) / "data/benchmarks/apibank/init_database"),
        toolbench_backend=toolbench_backend,
        toolbench_service_url=toolbench_service_url,
        toolbench_key=toolbench_key,
    )
    if benchmark == "apibank":
        from paraagent.evaluation.apibank.environment import ApiBankEnvironment
        return ApiBankEnvironment(query, args, paradigm=paradigm)
    from paraagent.evaluation.toolbench.environment import ToolBenchEnvironment
    return ToolBenchEnvironment(query, [], None, args)


def run_episode(model, environment, max_depth=60, protocol="paraagent"):
    from paraagent.paraact.loop import ParaAct

    chain = ParaAct(model, environment, protocol=protocol)
    start = time.monotonic()
    chain.start(max_depth=max_depth, pass_at=1, answer=1)
    result = chain.to_json(answer=True, process=True)
    answer = result["answer_generation"]
    answer["inference_time"] = time.monotonic() - start
    node = next((n for n in chain.terminal_node if not n.pruned), None)
    if node is None and chain.terminal_node:
        node = chain.terminal_node[-1]
    state = node.io_state if node is not None else None
    if state is not None:
        if hasattr(state, "called_apis"):
            answer["called_apis"] = state.called_apis
        if hasattr(state, "final_answer"):
            answer["final_answer"] = state.final_answer
    if isinstance(answer["final_answer"], dict):
        answer["final_answer"] = json.dumps(answer["final_answer"], ensure_ascii=False)

    details = []


    messages = node.messages if node is not None else []
    answer["messages"] = messages
    for message in messages:
        for call in message.get("tool_calls", []) or []:
            function = call.get("function", call)
            details.append({"role": "tool", "message": {"name": function.get("name", "")}, "next": []})
    answer["answer_details"] = details
    return result


def _validate_existing_prediction(target, experiment, query):
    previous = json.loads(target.read_text())
    if previous.get("inference_api") != experiment["inference_api"] or previous.get("agent") != experiment["agent"]:
        raise ValueError(f"Inference API or agent differs from existing prediction: {target}")
    if experiment["benchmark"] == "toolbench":
        if any(previous.get(field) != experiment[field] for field in (
            "toolbench_backend", "toolbench_service_url", "toolbench_cache_policy",
        )):
            raise ValueError(f"ToolBench backend differs from existing prediction: {target}")
    previous_identity = prediction_identity(previous, experiment["benchmark"], query)
    if previous_identity != experiment:
        raise ValueError(f"Experiment settings differ from existing prediction; use a new output directory: {target}")


def main():
    parser = argparse.ArgumentParser(description="Run ParaAgent or GPT-ReAct on ToolBench or API-Bank.")
    parser.add_argument("--benchmark", choices=["toolbench", "apibank"], required=True)
    parser.add_argument("--agent", choices=["paraagent", "react"], default="paraagent")
    parser.add_argument("--paradigm", choices=["ETE", "EaE"], help="Required for the GPT-ReAct baseline.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--inference-config", type=Path, help="JSON with model, base_url and api_key_env.")
    parser.add_argument("--retriever-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-depth", type=int, default=60, help="Trace depth (three nodes per typical reasoning/action round).")
    parser.add_argument("--top-k", type=int, default=os.environ.get("TOOL_SEARCH_DEFAULT_TOP_K", "3"), help="Tools per retrieval (default: 3).")
    parser.add_argument("--toolbench-backend", choices=["simulator", "virtual", "mirrorapi", "live"], default="simulator",
                        help="ToolBench observation source; service modes use TOOLBENCH_SERVICE_URL, live also requires TOOLBENCH_KEY.")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.max_depth < 1:
        parser.error("--max-depth must be positive")
    if args.runs < 1 or (args.limit is not None and args.limit < 1):
        parser.error("runs and limit must be positive")
    if args.agent == "react" and args.paradigm is None:
        parser.error("--paradigm ETE or --paradigm EaE is required with --agent react")
    if args.agent == "paraagent" and args.paradigm is not None:
        parser.error("--paradigm is only used with --agent react")
    if args.benchmark != "toolbench" and args.toolbench_backend != "simulator":
        parser.error("--toolbench-backend only applies to ToolBench")
    if args.inference_config and (args.model or args.base_url):
        parser.error("--inference-config cannot be combined with --model or --base-url")
    if args.agent == "react" and not args.inference_config:
        parser.error("--inference-config is required with --agent react")
    try:
        inference_config = load_api_config(args.inference_config) if args.inference_config else None
        validate_public_url(args.retriever_url, "--retriever-url")
        if args.base_url:
            validate_public_url(args.base_url, "--base-url")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    service_url = os.environ.get("TOOLBENCH_SERVICE_URL") if args.toolbench_backend in {"live", "virtual", "mirrorapi"} else None
    toolbench_key = os.environ.get("TOOLBENCH_KEY") if args.toolbench_backend in {"live", "virtual", "mirrorapi"} else None
    cache_policy = {"live": "service-managed", "virtual": "service-managed-virtual",
                    "mirrorapi": "service-managed-mirrorapi",
                    "simulator": "simulator-cache-then-model"}[args.toolbench_backend]
    service_timeout = None
    if args.toolbench_backend in {"live", "virtual", "mirrorapi"}:
        from paraagent.evaluation.toolbench.service import ToolBenchServiceClient
        try:
            service_timeout = ToolBenchServiceClient(service_url, toolbench_key, backend=args.toolbench_backend).timeout
        except ValueError as exc:
            parser.error(str(exc))

    root = args.root.resolve()
    os.environ["TOOL_SEARCH_API_URL"] = args.retriever_url
    os.environ["TOOL_SEARCH_DEFAULT_TOP_K"] = str(args.top_k)
    os.environ.setdefault("AGENT_MAX_TOKENS", "2048")
    os.environ.setdefault("MIRRORAPI_CONFIG", str(root / "configs/eval/toolbench-simulator.yaml"))
    tasks = benchmark_tasks(root, args.benchmark)
    tasks_fingerprint = fingerprint(tasks)
    if args.limit:
        tasks = tasks[:args.limit]
    if args.agent == "paraagent":
        from paraagent.paraact.model import AgentVLLMFunction
        model_class = AgentVLLMFunction
        model_name = inference_config["model"] if inference_config else (args.model or "paraagent-rl")
        base_url = inference_config["base_url"] if inference_config else (args.base_url or "http://127.0.0.1:8000/v1")
        api_key = inference_config["api_key"] if inference_config else os.environ.get("POLICY_API_KEY", "EMPTY")
        inference_identity = public_api_config(inference_config) if inference_config else {
            "model": model_name, "base_url": base_url, "api_key_env": "POLICY_API_KEY"}
        protocol = "paraagent"
        environment_paradigm = "EaE"
    else:
        from paraagent.paraact.react_model import ReActChatModel
        model_class = ReActChatModel
        model_name = inference_config["model"]
        base_url = inference_config["base_url"]
        api_key = inference_config["api_key"]
        inference_identity = public_api_config(inference_config)
        protocol = "react"
        environment_paradigm = args.paradigm
    experiment = {
        "schema_version": 1, "benchmark": args.benchmark, "agent": args.agent,
        "paradigm": environment_paradigm, "inference_api": inference_identity,
        "retriever_url": args.retriever_url, "retrieval_top_k": args.top_k,
        "max_depth": args.max_depth, "agent_max_tokens": int(os.environ["AGENT_MAX_TOKENS"]),
        "benchmark_tasks_fingerprint": tasks_fingerprint,
    }
    if args.benchmark == "toolbench":
        experiment.update(toolbench_backend=args.toolbench_backend, toolbench_service_url=service_url,
                          toolbench_cache_policy=cache_policy,
                          toolbench_timeout=list(service_timeout) if service_timeout else None)
        if args.toolbench_backend == "simulator":
            config_path = Path(os.environ["MIRRORAPI_CONFIG"])
            experiment["simulator_config_fingerprint"] = (
                fingerprint(config_path.read_text()) if config_path.is_file() else None
            )
            simulator_url = os.environ.get("TOOLENV_BASE_URL")
            if simulator_url is not None:
                validate_public_url(simulator_url, "TOOLENV_BASE_URL")
            experiment["simulator_base_url_override"] = simulator_url
    # Check every existing prediction before executing any missing episodes.
    for run in range(1, args.runs + 1):
        for split, qid, query in tasks:
            target = args.output / f"run-{run}" / split / f"{qid}_Agent@1.json"
            if target.exists():
                _validate_existing_prediction(target, experiment, query)
    for run in range(1, args.runs + 1):
        for split, qid, query in tasks:
            target = args.output / f"run-{run}" / split / f"{qid}_Agent@1.json"
            if target.exists():
                _validate_existing_prediction(target, experiment, query)
                continue
            if args.agent == "paraagent":
                model = model_class(model=model_name, base_url=base_url, openai_key=api_key)
            else:
                model = model_class(model=model_name, base_url=base_url, api_key=api_key)
            try:
                environment = make_environment(
                    root, args.benchmark, query, environment_paradigm,
                    toolbench_backend=args.toolbench_backend,
                    toolbench_service_url=service_url,
                    toolbench_key=toolbench_key,
                )
                result = run_episode(model, environment, args.max_depth, protocol=protocol)
            finally:
                model.client.close()
            result["query_id"] = qid
            result["benchmark"] = args.benchmark
            result["split"] = split
            result["agent"] = args.agent
            result["inference_api"] = inference_identity
            result["retrieval_top_k"] = args.top_k
            result["experiment"] = experiment
            result["experiment_fingerprint"] = fingerprint(experiment)
            result["query_fingerprint"] = fingerprint(query)
            if args.benchmark == "toolbench":
                result["toolbench_backend"] = args.toolbench_backend
                result["toolbench_cache_policy"] = cache_policy
                if service_url is not None:
                    result["toolbench_service_url"] = service_url
            if args.agent == "react":
                result["paradigm"] = args.paradigm
            result["answer_generation"]["query"] = query.get("query", query.get("requirement", ""))
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            temporary.replace(target)
            print(f"Saved {target}")


if __name__ == "__main__":
    main()
