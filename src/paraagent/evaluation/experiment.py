"""Identity checks for evaluation predictions, resumption and aggregation."""

import hashlib
import json

from paraagent.security import validate_public_url


def fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def prediction_identity(prediction, benchmark, query, *, allow_legacy_results=False):
    """Validate recorded identity, returning only comparable experiment settings."""
    experiment = prediction.get("experiment")
    if experiment is None:
        if not allow_legacy_results:
            raise ValueError(
                "Prediction lacks experiment metadata; regenerate in a new output directory, "
                "or explicitly use --allow-legacy-results for scoring with partial identity checks"
            )
        if prediction.get("benchmark", benchmark) != benchmark:
            raise ValueError("Prediction benchmark differs from requested benchmark")
        recorded_query = prediction.get("answer_generation", {}).get("query")
        if recorded_query is not None and recorded_query != query.get("query", query.get("requirement", "")):
            raise ValueError("Prediction query differs from the benchmark task")
        api = prediction.get("inference_api")
        if api is not None:
            validate_public_url(api["base_url"])
        fields = (
            "benchmark", "agent", "paradigm", "inference_api", "retrieval_top_k",
            "retriever_url", "max_depth", "agent_max_tokens", "toolbench_backend",
            "toolbench_service_url", "toolbench_cache_policy", "toolbench_timeout",
        )
        return {"legacy": True, **{field: prediction.get(field) for field in fields}}
    required = {
        "schema_version", "benchmark", "agent", "paradigm", "inference_api",
        "retriever_url", "retrieval_top_k", "max_depth", "agent_max_tokens", "benchmark_tasks_fingerprint",
    }
    if not isinstance(experiment, dict) or not required.issubset(experiment) or experiment["schema_version"] != 1:
        raise ValueError("Prediction has incomplete or unsupported experiment metadata")
    if prediction.get("experiment_fingerprint") != fingerprint(experiment):
        raise ValueError("Prediction experiment fingerprint is inconsistent")
    if experiment["benchmark"] != benchmark:
        raise ValueError("Prediction benchmark differs from requested benchmark")
    if prediction.get("query_fingerprint") != fingerprint(query):
        raise ValueError("Prediction query differs from the benchmark task")
    api = experiment["inference_api"]
    if not isinstance(api, dict) or set(api) != {"model", "base_url", "api_key_env"}:
        raise ValueError("Prediction inference identity must contain only model, base_url and api_key_env")
    validate_public_url(api["base_url"])
    validate_public_url(experiment["retriever_url"], "retriever_url")
    for field in ("benchmark", "agent", "paradigm", "inference_api", "retrieval_top_k",
                  "toolbench_backend", "toolbench_service_url", "toolbench_cache_policy"):
        if field in prediction and prediction[field] != experiment.get(field):
            raise ValueError(f"Prediction {field} differs from its experiment metadata")
    return experiment
