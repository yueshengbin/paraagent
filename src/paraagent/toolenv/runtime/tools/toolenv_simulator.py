import atexit
import asyncio
import fcntl
import json
import logging
import os
import pickle
import random
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import requests
import yaml
from jsonschema import SchemaError, ValidationError, validate
from openai import APITimeoutError, OpenAI
from referencing.exceptions import Unresolvable

from paraagent.toolenv.runtime.utils import normalize_tool_name, standardize

from ..base import BaseTool
from ..schema import ToolResponse

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _parse_bool_env(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


_SIMULATOR_PROFILE_ENABLED = _parse_bool_env(os.environ.get("TOOLENV_SIMULATOR_PROFILE", "0"))
_SIMULATOR_PROFILE_LOG_EVERY = max(1, int(os.environ.get("TOOLENV_SIMULATOR_PROFILE_LOG_EVERY", "50")))
_SIMULATOR_PROFILE_PENDING_WINDOW = max(
    1, int(os.environ.get("TOOLENV_SIMULATOR_PROFILE_PENDING_WINDOW", "512"))
)
_SIMULATOR_PROFILE_LATENCY_WINDOW = max(
    1, int(os.environ.get("TOOLENV_SIMULATOR_PROFILE_LATENCY_WINDOW", "4096"))
)

_CACHE_LOG_PATH = os.environ.get("TOOLENV_CACHE_LOG_PATH", "").strip() or None
_CACHE_LOG_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _tau_bench_tool_names() -> set[str]:
    try:
        from paraagent.toolenv.runtime.tools.tau_native_executor import TAU_TOOL_NAMES_ALL
    except Exception:
        return set()
    return {str(name).lower().replace("-", "_") for name in TAU_TOOL_NAMES_ALL}


def _is_tau_bench_tool_name(name: str | None) -> bool:
    key = str(name or "").strip().lower().replace(" ", "_").replace("-", "_")
    return bool(key) and key in _tau_bench_tool_names()


def _persist_cache_miss(tool_call: dict, payload: dict | None) -> None:
    """Append simulator responses to JSONL using the cache_flat lookup keys.
    Store the (tool_for_category, api) key, sorted JSON arguments, and parsed payload.
    """
    if not _CACHE_LOG_PATH or not isinstance(payload, dict):
        return
    try:
        tool_name = tool_call.get("tool_name", "") or ""
        api_name = tool_call.get("api_name", "") or ""
        category = tool_call.get("category_name", "") or ""
        arguments = tool_call.get("arguments", {}) or {}
        if not api_name:
            return
        api_key = api_name.lower().replace(" ", "_").replace("-", "_")
        if _is_tau_bench_tool_name(api_key) and not tool_name:
            return
        if tool_name:
            std_tool = standardize(tool_name) + "_for_" + category
        else:
            std_tool = ""
        record = {
            "tool": std_tool,
            "api": api_key,
            "args_key": json.dumps(arguments, ensure_ascii=False, sort_keys=True),
            "payload": payload,
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"

        with _CACHE_LOG_LOCK:
            with open(_CACHE_LOG_PATH, "a", encoding="utf-8") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(line)
                    f.flush()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception as exc:
        logger.debug("persist_cache_miss failed: %s", exc)


_SIMULATOR_PROFILE_LOCK = threading.Lock()
_SIMULATOR_PROFILE_STATE = {
    "batch_count": 0,
    "pending_counts": deque(maxlen=_SIMULATOR_PROFILE_PENDING_WINDOW),
    "total_counts": deque(maxlen=_SIMULATOR_PROFILE_PENDING_WINDOW),
    "tool_exec_count": 0,
    "tool_retry_count": 0,
    "api_call_latencies_ms": deque(maxlen=_SIMULATOR_PROFILE_LATENCY_WINDOW),
}


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * percentile
    lower_idx = int(rank)
    upper_idx = min(lower_idx + 1, len(values) - 1)
    lower = values[lower_idx]
    upper = values[upper_idx]
    if upper_idx == lower_idx:
        return lower
    weight = rank - lower_idx
    return lower + (upper - lower) * weight


def _record_simulator_pending_batch(pending_count: int, total_count: int) -> None:
    if not _SIMULATOR_PROFILE_ENABLED:
        return

    summary: dict[str, float | int] | None = None
    with _SIMULATOR_PROFILE_LOCK:
        state = _SIMULATOR_PROFILE_STATE
        state["batch_count"] += 1
        state["pending_counts"].append(int(pending_count))
        state["total_counts"].append(int(total_count))

        if state["batch_count"] % _SIMULATOR_PROFILE_LOG_EVERY == 0:
            pending_values = list(state["pending_counts"])
            total_values = list(state["total_counts"])
            latency_values = sorted(float(v) for v in state["api_call_latencies_ms"])
            tool_exec_count = int(state["tool_exec_count"])
            tool_retry_count = int(state["tool_retry_count"])
            retry_rate = (tool_retry_count / tool_exec_count) if tool_exec_count else 0.0
            total_sum = sum(total_values)
            pending_sum = sum(pending_values)
            hit_rate = 1.0 - (pending_sum / total_sum) if total_sum > 0 else 0.0
            summary = {
                "batch_count": state["batch_count"],
                "pending_mean": (pending_sum / len(pending_values)) if pending_values else 0.0,
                "pending_max": max(pending_values) if pending_values else 0,
                "hit_rate": hit_rate,
                "retry_rate": retry_rate,
                "api_p50_ms": _percentile(latency_values, 0.50),
                "api_p95_ms": _percentile(latency_values, 0.95),
                "api_p99_ms": _percentile(latency_values, 0.99),
                "api_samples": len(latency_values),
                "tool_exec_samples": tool_exec_count,
            }

    if summary is not None:
        print(
            "ToolEnvSimulator profile batches=%d cache_hit_rate=%.4f llm_pending/mean=%.2f llm_pending/max=%d "
            "retry_rate=%.4f mirror_api_latency_ms/p50=%.2f p95=%.2f p99=%.2f "
            "api_samples=%d tool_exec_samples=%d"
            % (
                summary["batch_count"],
                summary["hit_rate"],
                summary["pending_mean"],
                summary["pending_max"],
                summary["retry_rate"],
                summary["api_p50_ms"],
                summary["api_p95_ms"],
                summary["api_p99_ms"],
                summary["api_samples"],
                summary["tool_exec_samples"],
            ),
            flush=True,
        )


def _record_simulator_retry(retried: bool) -> None:
    if not _SIMULATOR_PROFILE_ENABLED:
        return
    with _SIMULATOR_PROFILE_LOCK:
        _SIMULATOR_PROFILE_STATE["tool_exec_count"] += 1
        if retried:
            _SIMULATOR_PROFILE_STATE["tool_retry_count"] += 1


def _record_simulator_api_latency(latency_ms: float) -> None:
    if not _SIMULATOR_PROFILE_ENABLED:
        return
    with _SIMULATOR_PROFILE_LOCK:
        _SIMULATOR_PROFILE_STATE["api_call_latencies_ms"].append(float(latency_ms))


_SFT_SYSTEM = (
    "You are an API simulator acting as a backend server. Your task is to handle API requests "
    "and return realistic, logically consistent responses that strictly follow the API documentation "
    "and provided input parameters.\n\n"
    "### RESPONSE RULES\n\n"
    "1. **Output Format**\n"
    "   - Only return valid, well-formed JSON (no markdown, no explanations, no comments, no extra text).\n"
    "   - Response schema:\n"
    "     {\n"
    '       "error": "none" | { "type": "<error type>", "msg": "<error message>" },\n'
    '       "response": <object | string | number | array | "none">\n'
    "     }\n\n"
    "2. **Error Handling**\n"
    '   - Use "error": "none" for successful executions.\n'
    "   - For failures, return:\n"
    "     {\n"
    '       "error": { "type": "<error type>", "msg": "<brief message>" },\n'
    '       "response": "none"\n'
    "     }\n"
    "   - Possible error types include (but are not limited to): InvalidRequestError, NetworkError, "
    "NotFoundError, PermissionError, ToolExecutionError.\n\n"
    "3. **Data Generation**\n"
    "   - Generate realistic, type-correct, domain-appropriate data fully aligned with the API "
    "documentation and parameters (e.g., timestamps, valid URLs, unique numeric IDs, correct currency codes, usable email formats).\n\n"
    "4. **Logical Consistency**\n"
    "   - Maintain meaningful and coherent relationships between fields. Avoid contradictions or obviously artificial data.\n\n"
    "5. **Quality Requirements**\n"
    "   - Do not use placeholders, meaningless filler data, or repetitive patterns.\n"
    "   - Ensure outputs appear production-grade and believable.\n\n"
    "6. **Final Output Restriction**\n"
    "   - Return only the JSON object — no additional formatting, commentary, explanation, or wrapping."
)

_USER_TEMPLATE = "## API Documentation:\n{api_doc}\n\n## Input Parameters:\n{request}\n"
_REALISTIC_ERRORS = [
    ("RateLimitError", "Too many requests. Please retry after a moment.", 0.15),
    ("AuthenticationError", "Invalid or expired API key.", 0.10),
    ("NetworkError", "Connection timeout while reaching the endpoint.", 0.10),
    ("NotFoundError", "The requested resource was not found.", 0.1),
    ("ServiceUnavailable", "The service is temporarily unavailable.", 0.05),
]
_ERROR_TYPES = [e[0] for e in _REALISTIC_ERRORS]
_ERROR_MSGS = [e[1] for e in _REALISTIC_ERRORS]
_ERROR_WEIGHTS = [e[2] for e in _REALISTIC_ERRORS]
_BAD_RESPONSE_VALUES = {"none", "null", "", None}

_REFUSAL_ACTIONS = (
    r"provide|generate|fulfill|assist|help|access|retrieve|fetch|browse|create|"
    r"simulate|answer|comply|produce|summarize|explain|verify"
)
_REFUSAL_PREFIX_PATTERNS = (
    re.compile(
        rf"^(?:(?:i(?:'m|\s+am)\s+sorry|i\s+apologize|sorry|apologies)"
        rf"[\s,;:!.\-]+)?(?:but\s+)?(?:"
        rf"i\s+can(?:not|'t|\s+not)|"
        rf"i(?:'m|\s+am|\s+was)\s+(?:unable|not\s+able)"
        rf")\s+(?:to\s+)?(?:(?:directly|currently)\s+)?"
        rf"(?:{_REFUSAL_ACTIONS})\b"
    ),
    re.compile(
        r"^(?:(?:i(?:'m|\s+am)\s+sorry|i\s+apologize|sorry|apologies)"
        r"[\s,;:!.\-]+)?(?:but\s+)?i\s+"
        r"(?:do\s+not|don't)\s+have\s+"
        r"(?:access|the\s+(?:ability|capability)|enough\s+information|"
        r"specific\s+data|the\s+requested\s+(?:data|information))\b"
    ),
    re.compile(
        r"^as\s+an?\s+(?:ai|language\s+model)\b.{0,180}\b"
        r"(?:cannot|can't|unable|not\s+able|do\s+not\s+have|don't\s+have)\b"
    ),
    re.compile(r"^(?:unable\s+to|cannot|can't)\s+simulate\b"),
    re.compile(
        r"^(?:unfortunately[\s,;:!\-]+)?(?:the\s+)?(?:api\s+)?"
        r"input(?:\s+provided)?\s+(?:does\s+not|doesn't)\s+"
        r"(?:contain|include|provide)\b"
    ),
)


def _is_refusal_text(text: str) -> bool:
    """Detect refusal or apology text that must be treated as a tool failure."""
    if not isinstance(text, str):
        return False

    head = " ".join(text.lstrip()[:500].lower().replace("’", "'").replace("‘", "'").split())
    return any(pattern.search(head) for pattern in _REFUSAL_PREFIX_PATTERNS)


def _is_bad_response(response: Any) -> bool:
    if isinstance(response, (dict, list)):
        return not response
    if isinstance(response, str):
        stripped = response.strip()
        if stripped.lower() in _BAD_RESPONSE_VALUES:
            return True

        return _is_refusal_text(stripped)
    if response is None:
        return True
    return False


def _is_unusable_success_payload(payload: dict[str, Any]) -> bool:
    """Whether a nominally successful simulator envelope has no usable result."""
    return payload.get("error") is None and _is_bad_response(payload.get("response"))


def _looks_like_json_prefix(text: str) -> bool:
    """Return true for an object/array prefix that may be cut off by max_tokens."""
    stripped = text.lstrip()
    if not stripped:
        return False
    opener = stripped[0]
    if opener not in "{[":
        return False

    try:
        _, decoded_end = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError:
        pass
    else:
        if stripped[decoded_end:].strip():
            return False
    remainder = stripped[1:].lstrip()
    if not remainder:
        return True
    if opener == "{":
        return remainder[0] in {'"', "}"}
    return remainder[0] in '{["-0123456789tfn]'


def _cache_argument_key_candidates(arguments: dict[str, Any]) -> list[str]:
    return list(dict.fromkeys((json.dumps(arguments, ensure_ascii=False, sort_keys=True), str(arguments))))


@lru_cache(maxsize=8)
def _load_api_name_reflect_cached(tsv_path: str) -> dict[str, dict[str, Any]]:
    if not tsv_path:
        return {}
    path = Path(tsv_path)
    if not path.exists():
        return {}

    dataset_df = pd.read_csv(path, sep="\t")
    api_name_reflect = {}
    for item in dataset_df.itertuples():
        tool_meta = ToolEnvSimulatorTool.normalize_runtime_tool_meta(
            item.name, json.loads(item.document_content)
        )
        api_name_reflect[normalize_tool_name(item.name)] = tool_meta
    return api_name_reflect


class _SimEndpoint:
    __slots__ = ("client", "base_url", "inflight", "fail_streak", "disabled_until", "use_proxy")

    def __init__(self, client, base_url, use_proxy):
        self.client = client
        self.base_url = base_url
        self.use_proxy = use_proxy
        self.inflight = 0
        self.fail_streak = 0
        self.disabled_until = 0.0


class _SimEndpointPool:
    """Thread-safe least-inflight routing with failure cooldowns."""

    def __init__(self, endpoints, fail_threshold: int, cooldown_s: float):
        self._eps = endpoints
        self._lock = threading.Lock()
        self._fail_threshold = max(1, fail_threshold)
        self._cooldown_s = max(0.0, cooldown_s)

    @property
    def endpoints(self):
        return self._eps

    def pick(self):
        now = time.monotonic()
        with self._lock:
            healthy = [e for e in self._eps if e.disabled_until <= now]
            if not healthy:
                healthy = self._eps
            ep = min(healthy, key=lambda e: e.inflight)
            ep.inflight += 1
            return ep

    def release(self, ep, ok: bool):
        with self._lock:
            ep.inflight = ep.inflight - 1 if ep.inflight > 0 else 0
            if ok:
                ep.fail_streak = 0
            elif len(self._eps) > 1:
                ep.fail_streak += 1
                if ep.fail_streak >= self._fail_threshold:
                    ep.disabled_until = time.monotonic() + self._cooldown_s
                    ep.fail_streak = 0

    def stats(self) -> list[dict]:
        now = time.monotonic()
        with self._lock:
            return [
                {
                    "base_url": e.base_url,
                    "inflight": e.inflight,
                    "disabled": e.disabled_until > now,
                    "fail_streak": e.fail_streak,
                }
                for e in self._eps
            ]


def _sim_url_needs_proxy(base_url: str) -> bool:
    """Use the proxy for remote endpoints and direct connections for local endpoints.
    MIRRORAPI_FORCE_PROXY=1/0 overrides automatic selection.
    """
    force = os.environ.get("MIRRORAPI_FORCE_PROXY", "").strip().lower()
    if force in ("1", "true", "yes"):
        return True
    if force in ("0", "false", "no"):
        return False
    b = base_url.lower()
    return not any(h in b for h in ("127.0.0.1", "localhost", "0.0.0.0", "[::1]"))


def _make_sim_endpoint(base_url: str, api_key: str, call_timeout, *, use_proxy: bool) -> "_SimEndpoint":

    http_client = httpx.Client(trust_env=use_proxy, timeout=call_timeout)
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        http_client=http_client,
        max_retries=0,
        timeout=call_timeout,
    )
    return _SimEndpoint(client, base_url, use_proxy)


def _build_sim_endpoint_pool(config: dict, call_timeout) -> "_SimEndpointPool":
    api_key = os.environ.get("TOOLENV_API_KEY", config.get("api_key", "EMPTY"))
    local_base = os.environ.get("TOOLENV_BASE_URL", config.get("api_base", "http://127.0.0.1:12348/v1"))

    eps = [_make_sim_endpoint(local_base, api_key, call_timeout, use_proxy=_sim_url_needs_proxy(local_base))]

    extra = os.environ.get("MIRRORAPI_EXTRA_ENDPOINTS", "").strip()
    for base in (b.strip() for b in extra.split(",")):
        if base:
            eps.append(_make_sim_endpoint(base, api_key, call_timeout, use_proxy=_sim_url_needs_proxy(base)))
    fail_threshold = int(os.environ.get("MIRRORAPI_ENDPOINT_FAIL_THRESHOLD", "5"))
    cooldown_s = float(os.environ.get("MIRRORAPI_ENDPOINT_COOLDOWN_S", "30"))
    return _SimEndpointPool(eps, fail_threshold, cooldown_s)


class MirrorApiClient:
    _shared_executor: ThreadPoolExecutor | None = None
    _shared_executor_lock = threading.Lock()
    _flat_cache: dict | None = None
    _flat_cache_lock = threading.Lock()
    _shared_openai_client: "OpenAI | None" = None
    _shared_openai_client_lock = threading.Lock()

    _endpoint_pool: "_SimEndpointPool | None" = None
    _endpoint_pool_lock = threading.Lock()

    _no_category_index: dict | None = None

    _variants_cache: dict | None = None
    _variants_cache_lock = threading.Lock()

    def __init__(self, config_path: str | None = None):
        path = Path(config_path) if config_path else None
        config = {}
        if path and path.exists():
            config = yaml.safe_load(path.read_text()) or {}

        self.config = config
        self.model = os.environ.get("MIRRORAPI_MODEL", config.get("model", "MirrorAPI"))
        self.temperature = float(os.environ.get("MIRRORAPI_TEMPERATURE", config.get("temperature", 0.0)))
        self.max_tokens = int(os.environ.get("MIRRORAPI_MAX_TOKENS", config.get("max_tokens", 1024)))
        self.max_workers = int(os.environ.get("MIRRORAPI_MAX_WORKERS", "128"))
        self.max_attempts = max(1, int(os.environ.get("MIRRORAPI_MAX_ATTEMPTS", "2")))

        call_timeout = float(os.environ.get("MIRRORAPI_CALL_TIMEOUT", "0")) or None
        with MirrorApiClient._endpoint_pool_lock:
            if MirrorApiClient._endpoint_pool is None:
                MirrorApiClient._endpoint_pool = _build_sim_endpoint_pool(config, call_timeout)
        self.pool = MirrorApiClient._endpoint_pool

        self.client = self.pool.endpoints[0].client
        self.error_injection_rate = float(os.environ.get("OWPTA_ERROR_RATE", "0.01"))
        with MirrorApiClient._shared_executor_lock:
            if MirrorApiClient._shared_executor is None:
                MirrorApiClient._shared_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._executor = MirrorApiClient._shared_executor

    @classmethod
    def shutdown_shared_executor(cls, wait: bool = False, cancel_futures: bool = True) -> None:
        """Shut down the shared simulator executor so the process can exit cleanly."""
        with cls._shared_executor_lock:
            executor = cls._shared_executor
            cls._shared_executor = None
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=cancel_futures)

    def close(self) -> None:
        """No-op: both the executor and OpenAI client are process-wide singletons.

        Lifecycle is managed separately via ``shutdown_shared_executor``.
        """

    @classmethod
    def _load_flat_cache(cls) -> dict:

        if cls._flat_cache is not None:
            return cls._flat_cache
        with cls._flat_cache_lock:
            if cls._flat_cache is not None:
                return cls._flat_cache
            pkl_path = os.environ.get("MIRRORAPI_CACHE_PKL", "")
            if not pkl_path or not os.path.exists(pkl_path):
                cls._flat_cache = {}
                if pkl_path:
                    logger.warning("MIRRORAPI_CACHE_PKL not found: %s", pkl_path)
                return cls._flat_cache
            try:
                with open(pkl_path, "rb") as f:
                    cls._flat_cache = pickle.load(f)
                logger.warning("Loaded flat cache from %s: %d keys", pkl_path, len(cls._flat_cache))
                cls._no_category_index = cls._build_no_category_index(cls._flat_cache)
                logger.warning("Built no-category index: %d entries", len(cls._no_category_index))
            except Exception as exc:
                logger.warning("Failed to load flat cache from %s: %s", pkl_path, exc)
                cls._flat_cache = {}
                cls._no_category_index = {}
            return cls._flat_cache

    @classmethod
    def _load_variants_cache(cls) -> dict:
        """Lazy-load variants pool. (tool, api) -> [payload,...].

        Used only when empty-args call hits an entry — randomly pick one.
        Disabled if MIRRORAPI_VARIANTS_PKL is unset / missing.
        """
        if cls._variants_cache is not None:
            return cls._variants_cache
        with cls._variants_cache_lock:
            if cls._variants_cache is not None:
                return cls._variants_cache
            pkl_path = os.environ.get("MIRRORAPI_VARIANTS_PKL", "")
            if not pkl_path or not os.path.exists(pkl_path):
                cls._variants_cache = {}
                if pkl_path:
                    logger.warning("MIRRORAPI_VARIANTS_PKL not found: %s", pkl_path)
                return cls._variants_cache
            try:
                with open(pkl_path, "rb") as f:
                    cls._variants_cache = pickle.load(f)
                n_apis = len(cls._variants_cache)
                n_payloads = sum(len(v) for v in cls._variants_cache.values() if isinstance(v, list))
                logger.warning(
                    "Loaded variants from %s: %d APIs, %d payloads",
                    pkl_path,
                    n_apis,
                    n_payloads,
                )
            except Exception as exc:
                logger.warning("Failed to load variants from %s: %s", pkl_path, exc)
                cls._variants_cache = {}
            return cls._variants_cache

    @classmethod
    def _build_no_category_index(cls, flat_cache: dict) -> dict:
        """Index cache entries by (tool prefix, API), merging categories for lookup."""
        index: dict[tuple[str, str], dict] = {}
        for (tool_key, api_key), entries in flat_cache.items():
            if not isinstance(entries, dict):
                continue

            if "_for_" in tool_key:
                tool_prefix = tool_key.rsplit("_for_", 1)[0]
            else:
                tool_prefix = tool_key
            idx_key = (tool_prefix, api_key)
            if idx_key not in index:
                index[idx_key] = {}
            index[idx_key].update(entries)
        return index

    def _load_from_cache(self, tool_call: dict[str, Any]) -> dict[str, Any] | None:

        if os.environ.get("TOOLENV_DISABLE_CACHE", "").lower() in ("1", "true", "yes"):
            return None

        tool_name = tool_call.get("tool_name", "")
        api_name = tool_call.get("api_name", "")
        category = tool_call.get("category_name", "")
        arguments = tool_call.get("arguments", {})

        if not api_name:
            return None

        api_key = api_name.lower().replace(" ", "_").replace("-", "_")
        if _is_tau_bench_tool_name(api_key) and not tool_name:
            return None
        argument_key_candidates = _cache_argument_key_candidates(
            arguments,
        )

        cache_key_candidates: list[tuple[str, str]] = []
        if tool_name:
            std_tool = standardize(tool_name) + "_for_" + category
            simple_tool = re.sub(r"[^\w]", "_", tool_name.lower()) + "_for_" + category
            cache_key_candidates.append((std_tool, api_key))
            if simple_tool != std_tool:
                cache_key_candidates.append((simple_tool, api_key))
        cache_key_candidates.append(("", api_key))

        if not arguments or arguments == {}:
            variants_pool = self._load_variants_cache()
            if variants_pool:
                for ck in cache_key_candidates:
                    vlist = variants_pool.get(ck)
                    if not vlist:
                        continue
                    payload = random.choice(vlist)
                    cached_result = self._cached_result(payload)
                    if cached_result is not None:
                        return cached_result

        flat_cache = self._load_flat_cache()
        if not flat_cache:
            return None

        seen: set[tuple[str, str]] = set()
        for ck in cache_key_candidates:
            if ck in seen:
                continue
            seen.add(ck)
            entries = flat_cache.get(ck)
            if not entries:
                continue
            for ak in argument_key_candidates:
                val = entries.get(ak)
                if val is not None:
                    if isinstance(val, list) and val:
                        val = random.choice(val)
                    cached_result = self._cached_result(val)
                    if cached_result is not None:
                        return cached_result

        no_cat_index = self.__class__._no_category_index
        if no_cat_index and tool_name:
            std_prefix = standardize(tool_name)
            simple_prefix = re.sub(r"[^\w]", "_", tool_name.lower())
            for prefix in dict.fromkeys([std_prefix, simple_prefix]):
                idx_key = (prefix, api_key)
                entries = no_cat_index.get(idx_key)
                if not entries:
                    continue
                for ak in argument_key_candidates:
                    val = entries.get(ak)
                    if val is not None:
                        if isinstance(val, list) and val:
                            val = random.choice(val)
                        cached_result = self._cached_result(val)
                        if cached_result is not None:
                            return cached_result
        return None

    def _cached_result(self, payload: Any) -> dict[str, Any] | None:
        """Normalize a cached payload; reject refusals and empty successes.
        Preserve explicit API errors as tool outcomes.
        """
        content = self._normalize_payload(payload)
        if content is None or _is_unusable_success_payload(content):
            return None
        return {
            "success": content.get("error") is None,
            "content": content,
            "error_msg": self._format_error_message(content.get("error")),
        }

    def _normalize_error(self, error: Any) -> Any:
        if error is None:
            return None
        if isinstance(error, str) and error.strip().lower() in _BAD_RESPONSE_VALUES:
            return None
        return error

    def _normalize_payload(self, payload: Any) -> dict[str, Any] | None:
        if isinstance(payload, list):
            return {"error": None, "response": payload}
        if not isinstance(payload, dict):
            return None
        if "error" not in payload and "response" not in payload:
            return {"error": None, "response": payload}
        error = self._normalize_error(payload.get("error"))
        response = payload.get("response")
        return {"error": error, "response": response}

    def _format_error_message(self, error: Any) -> str | None:
        if error is None:
            return None
        if isinstance(error, dict):
            err_type = error.get("type") or "ToolExecutionError"
            err_msg = error.get("msg") or ""
            return f"{err_type}: {err_msg}" if err_msg else str(err_type)
        return str(error)

    def _parse_payload(
        self,
        output: str,
        *,
        allow_partial_json: bool = False,
    ) -> dict[str, Any] | None:
        if not output:
            return None
        output = output.strip()

        try:
            return self._normalize_payload(json.loads(output))
        except Exception:
            pass

        repaired = self._try_repair_json(output)
        if repaired is not None:
            content = self._normalize_payload(repaired)
            if content is not None:
                resp = content.get("response")
                if isinstance(resp, str):
                    content["response"] = resp + " ......"
                elif isinstance(resp, list) and resp:
                    resp.append("......")
                elif isinstance(resp, dict) and resp:
                    resp["__truncated__"] = "......"

                if not _is_unusable_success_payload(content):
                    return content

        if _is_refusal_text(output):
            return None

        if _looks_like_json_prefix(output):
            if allow_partial_json:
                return {"error": None, "response": output + " ......"}
            return None
        if len(output) >= 100:
            return {"error": None, "response": output + " ......"}
        return None

    def _try_repair_json(self, text: str) -> dict[str, Any] | list[Any] | None:
        """Balance unclosed { [ and strings in truncated JSON."""
        if not text or text[0] not in "{[":
            return None
        stack: list[str] = []
        in_string = False
        escape = False
        for c in text:
            if escape:
                escape = False
                continue
            if in_string:
                if c == "\\":
                    escape = True
                elif c == '"':
                    in_string = False
                continue
            if c == '"':
                in_string = True
            elif c in "{[":
                stack.append(c)
            elif c in "}]":
                if stack:
                    stack.pop()
        if not stack and not in_string:
            return None
        closers = "".join("}" if op == "{" else "]" for op in reversed(stack))

        base = text + ('"' if in_string else "")
        candidates = [
            base.rstrip(", \n\t") + closers,
            re.sub(r',\s*"[^"]*"\s*:\s*[^,{}\[\]]*$', "", base).rstrip(", \n\t") + closers,
            re.sub(r',\s*"[^"]*"\s*$', "", base).rstrip(", \n\t") + closers,
            base.rstrip(",: \n\t") + closers,
        ]
        for attempt in candidates:
            try:
                return json.loads(attempt)
            except Exception:
                continue
        return None

    def _call_api(
        self,
        messages: list[dict[str, str]],
        temperature: float,
    ) -> tuple[Any, str | None]:
        """Return (payload, transport_error_type) for one simulator attempt.
        Invalid or refused payloads return (None, None); client timeouts return
        (None, "TimeoutError"). Keep attempt state local to each calling thread.
        """
        start_time = time.perf_counter()
        ep = self.pool.pick()
        ok = False
        try:
            resp = ep.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=self.max_tokens,
                seed=42,
                response_format={"type": "json_object"},
            )
            ok = True
            choice = resp.choices[0]
            text = choice.message.content or ""
            finish_reason = str(getattr(choice, "finish_reason", "") or "").lower()
            return self._parse_payload(
                text,
                allow_partial_json=finish_reason == "length",
            ), None
        except (APITimeoutError, httpx.TimeoutException, TimeoutError):
            return None, "TimeoutError"
        except Exception:
            return None, None
        finally:
            self.pool.release(ep, ok)
            _record_simulator_api_latency((time.perf_counter() - start_time) * 1000.0)

    def _execute_one(self, tool_call: dict[str, Any], messages: list[dict[str, str]]) -> dict[str, Any]:
        if random.random() < self.error_injection_rate:
            err_type = random.choices(_ERROR_TYPES, weights=_ERROR_WEIGHTS, k=1)[0]
            err_msg = _ERROR_MSGS[_ERROR_TYPES.index(err_type)]
            return {
                "success": False,
                "content": {"error": {"type": err_type, "msg": err_msg}, "response": "none"},
                "error_msg": f"{err_type}: {err_msg}",
                "injected_error": True,
            }

        retried = False
        payload, final_transport_error_type = self._call_api(
            messages,
            temperature=self.temperature,
        )
        if self.max_attempts > 1 and (payload is None or _is_unusable_success_payload(payload)):
            retried = True

            payload, final_transport_error_type = self._call_api(
                messages,
                temperature=0.0,
            )

        _record_simulator_retry(retried)

        if payload is None or _is_unusable_success_payload(payload):
            error_type = final_transport_error_type or "ToolExecutionError"
            error_message = (
                "MirrorAPI request timed out." if error_type == "TimeoutError" else "Tool execution failed."
            )
            return {
                "success": False,
                "content": {
                    "error": {"type": error_type, "msg": error_message},
                    "response": "none",
                },
                "error_msg": f"{error_type}: {error_message}",
            }

        error = payload.get("error")

        _persist_cache_miss(tool_call, payload)
        return {
            "success": error is None,
            "content": payload,
            "error_msg": self._format_error_message(error),
        }

    def _build_messages(self, tool_call: dict[str, Any]) -> list[dict[str, str]]:

        api_name_full = tool_call.get("display_name") or tool_call.get("api_name", "")
        parameters = tool_call.get("parameters")
        if parameters is None:
            parameters = tool_call.get("required_parameters", []) + tool_call.get("optional_parameters", [])
        platform_overview = tool_call.get("platform_overview", "")

        api_doc_dict = {
            "- Category": tool_call.get("category_name", ""),
            "- API Name": api_name_full,
            "- API Description": tool_call.get("api_description", ""),
            "- Parameters": parameters,
        }
        if platform_overview:
            api_doc_dict["- Platform Overview"] = platform_overview

        api_doc_str = "\n".join(f"{k}: {v}" for k, v in api_doc_dict.items())
        arguments = tool_call.get("arguments", {})
        request_str = (
            json.dumps(arguments, ensure_ascii=False) if isinstance(arguments, dict) else str(arguments)
        )
        instruction = _USER_TEMPLATE.format(api_doc=api_doc_str, request=request_str)
        return [
            {"role": "system", "content": _SFT_SYSTEM},
            {"role": "user", "content": instruction},
        ]

    def execute_batch(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any] | None] = [None] * len(tool_calls)
        llm_pending: list[tuple[int, dict[str, Any], list[dict[str, str]]]] = []

        for idx, tool_call in enumerate(tool_calls):
            cached = self._load_from_cache(tool_call)
            if cached is not None:
                results[idx] = cached
            else:
                llm_pending.append((idx, tool_call, self._build_messages(tool_call)))

        _record_simulator_pending_batch(len(llm_pending), len(tool_calls))

        if not llm_pending:
            return [
                result
                or {
                    "success": False,
                    "content": {
                        "error": {"type": "ToolExecutionError", "msg": "unknown"},
                        "response": "none",
                    },
                }
                for result in results
            ]

        future_to_idx = {
            self._executor.submit(self._execute_one, tc, messages): idx for idx, tc, messages in llm_pending
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                results[idx] = {
                    "success": False,
                    "content": {"error": {"type": "ToolExecutionError", "msg": str(exc)}, "response": "none"},
                    "error_msg": str(exc),
                }

        return [
            result
            or {
                "success": False,
                "content": {"error": {"type": "ToolExecutionError", "msg": "unknown"}, "response": "none"},
            }
            for result in results
        ]

    async def execute_batch_async(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any] | None] = [None] * len(tool_calls)
        llm_pending: list[tuple[int, dict[str, Any], list[dict[str, str]]]] = []

        for idx, tool_call in enumerate(tool_calls):
            cached = self._load_from_cache(tool_call)
            if cached is not None:
                results[idx] = cached
            else:
                llm_pending.append((idx, tool_call, self._build_messages(tool_call)))

        _record_simulator_pending_batch(len(llm_pending), len(tool_calls))

        if not llm_pending:
            return [
                result
                or {
                    "success": False,
                    "content": {
                        "error": {"type": "ToolExecutionError", "msg": "unknown"},
                        "response": "none",
                    },
                }
                for result in results
            ]

        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(self._executor, self._execute_one, tc, messages)
            for _, tc, messages in llm_pending
        ]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for batch_result, (idx, _, _) in zip(batch_results, llm_pending):
            if isinstance(batch_result, Exception):
                results[idx] = {
                    "success": False,
                    "content": {
                        "error": {"type": "ToolExecutionError", "msg": str(batch_result)},
                        "response": "none",
                    },
                    "error_msg": str(batch_result),
                }
            else:
                results[idx] = batch_result

        return [
            result
            or {
                "success": False,
                "content": {"error": {"type": "ToolExecutionError", "msg": "unknown"}, "response": "none"},
            }
            for result in results
        ]


def _shutdown_mirrorapi_shared_executor_at_exit() -> None:
    """Best-effort process-exit cleanup for the shared simulator executor."""
    MirrorApiClient.shutdown_shared_executor(wait=False, cancel_futures=True)


atexit.register(_shutdown_mirrorapi_shared_executor_at_exit)


@BaseTool.register("toolenv_simulator")
class ToolEnvSimulatorTool(BaseTool):
    name = "toolenv_simulator"
    description = "Execute a retrieved API schema against the local MirrorAPI simulator."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Retrieved tool name"},
            "arguments": {"type": "object", "description": "Arguments to pass to the retrieved tool"},
        },
        "required": ["name", "arguments"],
    }

    def __init__(self):
        super().__init__()
        default_config = Path(__file__).resolve().parent / "config_mirrorapi.yml"
        self.config_path = os.environ.get("MIRRORAPI_CONFIG_PATH", str(default_config))
        self.tool_name_api_tsv = os.environ.get("TOOL_NAME_API_TSV")
        self.timeout = float(os.environ.get("TOOLENV_SIMULATOR_TIMEOUT", "30"))
        self.mirror_api = MirrorApiClient(self.config_path)
        resolved_tsv_path = str(Path(self.tool_name_api_tsv).resolve()) if self.tool_name_api_tsv else ""
        self.api_name_reflect = _load_api_name_reflect_cached(resolved_tsv_path)

    def close(self) -> None:
        mirror_api = getattr(self, "mirror_api", None)
        if mirror_api is not None and hasattr(mirror_api, "close"):
            try:
                mirror_api.close()
            except Exception:
                pass

    def _summarize_args_list(self, args_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "name": args.get("name"),
                "arguments": args.get("arguments"),
            }
            for args in args_list
        ]

    def _summarize_results(self, results: list[tuple[str, dict[str, Any]] | None]) -> list[dict[str, Any]]:
        summary = []
        for result in results:
            if result is None:
                summary.append({"result": None})
                continue
            text, extra_info = result
            summary.append(
                {
                    "success": extra_info.get("success"),
                    "tool_name": extra_info.get("tool_name"),
                    "error": extra_info.get("error"),
                    "error_msg": extra_info.get("error_msg"),
                    "text_head": text[:300],
                }
            )
        return summary

    def _format_error_content(self, error_type: str, message: str) -> dict[str, Any]:
        return {
            "error": {"type": error_type, "msg": message},
            "response": "",
        }

    def _format_eval_payload(
        self,
        tool_name: str,
        batch_result: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        content = batch_result.get("content")
        result_payload: Any = content
        if isinstance(content, dict) and "error" in content and "response" in content:
            error = content.get("error")
            if error is None:
                result_payload = content.get("response")
            else:
                result_payload = {"error": error}
        payload = json.dumps({"name": tool_name, "result": result_payload}, ensure_ascii=False)

        response_truncated = False
        response_original_len: int | None = None
        max_chars = int(os.environ.get("TOOLENV_TOOL_RESPONSE_MAX_CHARS", "8000"))
        if max_chars > 0 and len(payload) > max_chars:
            response_original_len = len(payload)
            marker = f"... [truncated, original_len={response_original_len}]"
            payload = payload[: max(0, max_chars - len(marker))] + marker
            response_truncated = True

        return (
            payload,
            {
                "success": bool(batch_result.get("success")),
                "tool_name": tool_name,
                "error": content.get("error") if isinstance(content, dict) else None,
                "error_msg": batch_result.get("error_msg"),
                "injected_error": bool(batch_result.get("injected_error", False)),
                "response_truncated": response_truncated,
                "response_original_len": response_original_len,
            },
        )

    async def execute(self, args: dict[str, Any], **kwargs) -> tuple[ToolResponse, float | None, dict]:
        results = await self.execute_many([args])
        text, extra_info = results[0]
        return ToolResponse(text=text), None, extra_info

    async def execute_many(
        self,
        args_list: list[dict[str, Any]],
        retrieved_tool_meta: dict[str, dict[str, Any]] | None = None,
    ) -> list[tuple[str, dict[str, Any]]]:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "ToolEnvSimulatorTool.execute_many args_list=%s", self._summarize_args_list(args_list)
            )
        prepared_payloads = []
        prepared_indices = []
        results: list[tuple[str, dict[str, Any]] | None] = [None] * len(args_list)

        for idx, args in enumerate(args_list):
            tool_name = args.get("name")
            tool_args = args.get("arguments", {})
            if not isinstance(tool_name, str) or not tool_name:
                results[idx] = self._format_eval_payload(
                    "",
                    {
                        "success": False,
                        "content": self._format_error_content("InvalidRequestError", "No tool name"),
                        "error_msg": "InvalidRequestError: No tool name",
                    },
                )
                continue
            if not isinstance(tool_args, dict):
                results[idx] = self._format_eval_payload(
                    tool_name,
                    {
                        "success": False,
                        "content": self._format_error_content(
                            "InvalidRequestError", "Invalid tool arguments"
                        ),
                        "error_msg": "InvalidRequestError: Invalid tool arguments",
                    },
                )
                continue

            tool_meta = self._resolve_tool_meta(tool_name, retrieved_tool_meta)
            if tool_meta is None:
                results[idx] = self._format_eval_payload(
                    tool_name,
                    {
                        "success": False,
                        "content": self._format_error_content(
                            "InvalidRequestError", f"No such tool : {tool_name}"
                        ),
                        "error_msg": f"InvalidRequestError: No such tool : {tool_name}",
                    },
                )
                continue
            try:
                schema_ok = self.validate_schema(tool_args, tool_meta["parameters"])
            except (SchemaError, Unresolvable) as exc:
                logger.warning(
                    "ToolEnvSimulatorTool.execute_many tool=%s has an invalid parameter schema: %s",
                    tool_name,
                    exc,
                )
                schema_ok = False
            if not schema_ok:
                keys_str = ",".join(tool_args.keys()) if isinstance(tool_args, dict) else "Invalid Input"
                results[idx] = self._format_eval_payload(
                    tool_name,
                    {
                        "success": False,
                        "content": self._format_error_content(
                            "InvalidRequestError",
                            f"Invalid tool parameters: {keys_str}",
                        ),
                        "error_msg": f"InvalidRequestError: Invalid tool parameters: {keys_str}",
                    },
                )
                continue

            tn = tool_meta.get("tool_name", "")
            an = tool_meta["api_name"]
            display_name = f"{tn} : {an}" if tn else an
            prepared_payloads.append(
                {
                    "tool_name": tool_meta.get("tool_name", tool_name),
                    "api_name": an,
                    "display_name": display_name,
                    "category_name": tool_meta["category_name"],
                    "api_description": tool_meta.get("api_description", ""),
                    "parameters": tool_meta.get("parameters", {}),
                    "required_parameters": tool_meta.get("required_parameters", []),
                    "optional_parameters": tool_meta.get("optional_parameters", []),
                    "platform_overview": tool_meta.get("tool_description", ""),
                    "arguments": tool_args,
                }
            )
            prepared_indices.append((idx, tool_name))

        if prepared_payloads:
            if hasattr(self.mirror_api, "execute_batch_async"):
                batch_results = await self.mirror_api.execute_batch_async(prepared_payloads)
            else:
                loop = asyncio.get_running_loop()
                batch_results = await loop.run_in_executor(
                    None, self.mirror_api.execute_batch, prepared_payloads
                )
            for batch_result, (idx, tool_name) in zip(batch_results, prepared_indices):
                results[idx] = self._format_eval_payload(tool_name, batch_result)

        finalized_results = [
            result
            or self._format_eval_payload(
                "",
                {
                    "success": False,
                    "content": self._format_error_content("ServiceUnavailable", "API failed error"),
                    "error_msg": "ServiceUnavailable: API failed error",
                },
            )
            for result in results
        ]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "ToolEnvSimulatorTool.execute_many results=%s",
                self._summarize_results(finalized_results),
            )
        return finalized_results

    def _resolve_tool_meta(
        self,
        tool_name: str,
        retrieved_tool_meta: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        cached = (retrieved_tool_meta or {}).get(tool_name)
        if isinstance(cached, dict):
            raw_name = str(cached.get("raw_name", "")).strip() or tool_name
            tool_meta = cached.get("tool_meta")
            if isinstance(tool_meta, dict):
                normalized = self.normalize_runtime_tool_meta(raw_name, tool_meta)
                return normalized
        return self.api_name_reflect.get(tool_name)

    @staticmethod
    def normalize_runtime_tool_meta(raw_name: str, tool_meta: dict[str, Any]) -> dict[str, Any]:
        if "api_name" in tool_meta and "category_name" in tool_meta:
            return tool_meta

        if tool_meta.get("type") == "function" and isinstance(tool_meta.get("function"), dict):
            tool_meta = tool_meta["function"]

        tool_name = ""
        api_name = raw_name
        if " : " in raw_name:
            tool_name, api_name = raw_name.split(" : ", 1)

        parameters = tool_meta.get("parameters") or {"type": "object", "properties": {}, "required": []}
        properties = parameters.get("properties") or {}
        required_names = set(parameters.get("required") or [])

        def infer_param_type(schema_type: Any) -> str:
            if schema_type in {"integer", "number"}:
                return "NUMBER"
            if schema_type == "boolean":
                return "BOOLEAN"
            if schema_type == "array":
                return "ARRAY"
            if schema_type == "object":
                return "OBJECT"
            return "STRING"

        required_parameters = []
        optional_parameters = []
        for param_name, param_schema in properties.items():
            param_info = {
                "name": param_name,
                "description": (param_schema or {}).get("description", ""),
                "type": infer_param_type((param_schema or {}).get("type")),
                "default": (
                    (param_schema or {})["example_value"]
                    if "example_value" in (param_schema or {})
                    else (param_schema or {}).get("default", "")
                ),
            }
            if param_name in required_names:
                required_parameters.append(param_info)
            else:
                optional_parameters.append(param_info)

        return {
            "tool_name": tool_name,
            "api_name": api_name,
            "category_name": tool_meta.get("category", ""),
            "api_description": tool_meta.get("description", ""),
            "tool_description": tool_meta.get("platform_overview", ""),
            "platform_overview": tool_meta.get("platform_overview", ""),
            "parameters": parameters,
            "required_parameters": required_parameters,
            "optional_parameters": optional_parameters,
        }

    def validate_schema(self, input_data: dict[str, Any], schema: dict[str, Any]) -> bool:
        """Validate call arguments; return False for ValidationError.
        Malformed schemas and unresolved references propagate as metadata errors.
        """

        if isinstance(schema, dict) and isinstance(schema.get("required"), list):
            schema = {**schema, "required": list(dict.fromkeys(schema["required"]))}
        try:
            validate(instance=input_data, schema=schema)
            return True
        except ValidationError:
            return False

    def healthcheck(self) -> tuple[bool, str]:
        if not self.tool_name_api_tsv:
            return False, "TOOL_NAME_API_TSV is not set"
        if not self.api_name_reflect:
            return False, f"Failed to load tool metadata from {self.tool_name_api_tsv}"
        try:
            requests.get(
                os.environ.get(
                    "TOOLENV_BASE_URL", self.mirror_api.config.get("api_base", "http://127.0.0.1:12348/v1")
                ),
                timeout=self.timeout,
            )
        except Exception as exc:
            return False, f"MirrorAPI endpoint is not reachable: {exc}"
        return True, "ok"
