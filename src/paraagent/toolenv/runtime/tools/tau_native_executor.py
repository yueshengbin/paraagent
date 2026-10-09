"""Run native tau-bench retail/airline tools against per-trajectory state.

Initial states come from TAU_INITIAL_STATE_DIRS (default:
data/rl/paraagent-rl/state). Reset deep-copies state for each session.
Tool errors remain visible in the success/content response envelope.
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
import logging
import os
import re
import sys
import threading
from collections.abc import Iterable
from copy import deepcopy
from functools import wraps
from paraagent.toolenv.runtime.modes import require_current_native_modes
from pathlib import Path
from typing import Any

from paraagent.toolenv.outcome import (
    BASIS_LOCAL_VALIDATION,
    BASIS_NATIVE_MESSAGE_REGISTRY,
    BASIS_NATIVE_RETURN_SITE,
    BASIS_UNKNOWN,
    CATEGORY_CAPACITY_UNAVAILABLE,
    CATEGORY_EXECUTOR_FAILURE,
    CATEGORY_INVALID_REQUEST,
    CATEGORY_RESOURCE_NOT_FOUND,
    CATEGORY_STATE_CONFLICT,
    CATEGORY_SUCCESS,
    CATEGORY_TOOL_NOT_FOUND,
    CATEGORY_UNKNOWN_FAILURE,
    RETRY_NEVER,
    RETRY_NOT_APPLICABLE,
    RETRY_UNKNOWN,
    STAGE_COMPLETED,
    STAGE_DISPATCH,
    STAGE_DOMAIN,
    STAGE_EXECUTOR,
    STAGE_REQUEST,
    make_tool_outcome,
)

logger = logging.getLogger(__name__)

TAU_STATE_HASH_VERSION = "tau_state_sha256_json_v1"
_AIRLINE_TEMPORAL_MODULE_NAME = "_tau_airline_temporal"
_AIRLINE_TEMPORAL_LOCK = threading.Lock()
_AIRLINE_BUSINESS_MODULE_NAME = "_tau_airline_business"
_AIRLINE_BUSINESS_LOCK = threading.Lock()
_AIRLINE_BOOKING_MODULE_NAME = "_tau_airline_booking"
_AIRLINE_BOOKING_LOCK = threading.Lock()
_AIRLINE_ELIGIBILITY_MODULE_NAME = "_tau_airline_eligibility"
_AIRLINE_ELIGIBILITY_LOCK = threading.Lock()


def _airline_temporal_module():
    """Load the pure guard without importing the optional training tool package."""
    with _AIRLINE_TEMPORAL_LOCK:
        module = sys.modules.get(_AIRLINE_TEMPORAL_MODULE_NAME)
        if module is None:
            path = Path(__file__).resolve().parent.parent / "airline_temporal.py"
            spec = importlib.util.spec_from_file_location(_AIRLINE_TEMPORAL_MODULE_NAME, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[_AIRLINE_TEMPORAL_MODULE_NAME] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(_AIRLINE_TEMPORAL_MODULE_NAME, None)
                raise
        return module


def _airline_business_module():
    """Load airline business rules without importing training packages."""
    with _AIRLINE_BUSINESS_LOCK:
        module = sys.modules.get(_AIRLINE_BUSINESS_MODULE_NAME)
        if module is None:
            path = Path(__file__).resolve().parent.parent / "airline_business.py"
            spec = importlib.util.spec_from_file_location(_AIRLINE_BUSINESS_MODULE_NAME, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[_AIRLINE_BUSINESS_MODULE_NAME] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(_AIRLINE_BUSINESS_MODULE_NAME, None)
                raise
        return module


def _airline_booking_module():
    """Load booking preflight checks without optional training dependencies."""
    with _AIRLINE_BOOKING_LOCK:
        module = sys.modules.get(_AIRLINE_BOOKING_MODULE_NAME)
        if module is None:
            path = Path(__file__).resolve().parent.parent / "airline_booking.py"
            spec = importlib.util.spec_from_file_location(_AIRLINE_BOOKING_MODULE_NAME, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[_AIRLINE_BOOKING_MODULE_NAME] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(_AIRLINE_BOOKING_MODULE_NAME, None)
                raise
        return module


def _airline_eligibility_module():
    with _AIRLINE_ELIGIBILITY_LOCK:
        module = sys.modules.get(_AIRLINE_ELIGIBILITY_MODULE_NAME)
        if module is None:
            path = Path(__file__).resolve().parent.parent / "airline_eligibility.py"
            spec = importlib.util.spec_from_file_location(_AIRLINE_ELIGIBILITY_MODULE_NAME, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[_AIRLINE_ELIGIBILITY_MODULE_NAME] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(_AIRLINE_ELIGIBILITY_MODULE_NAME, None)
                raise
        return module


def _serialized_session_operation(method):
    """Serialize lifecycle, lookup, and hashing operations with the transaction lock."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._transaction_lock:
            return method(self, *args, **kwargs)
        return method(self, *args, **kwargs)

    return wrapped


def canonical_tau_state_hash(state: Any) -> str:
    """Hash the complete JSON database state for deterministic outcome comparison.
    The SHA-256 digest is stable across processes and Python versions.
    """
    payload = json.dumps(
        state,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ensure_tau_module_registered() -> Any:
    """Register the bundled Tool class under tau_bench.envs.tool for native imports."""
    if "tau_bench.envs.tool" in sys.modules:
        return sys.modules["tau_bench.envs.tool"]

    here = Path(__file__).resolve().parent.parent
    vendored_path = here / "tau_bench" / "envs" / "tool.py"
    if not vendored_path.exists():
        raise RuntimeError(f"vendored tau_bench tool not found at {vendored_path}")

    for pkg in ("tau_bench", "tau_bench.envs"):
        if pkg not in sys.modules:
            m = type(sys)("module")
            m.__path__ = []
            sys.modules[pkg] = m

    spec = importlib.util.spec_from_file_location("tau_bench.envs.tool", vendored_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tau_bench.envs.tool"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_domain_tools(domain: str) -> dict[str, type]:
    """Load tau-bench {retail,airline} tool classes by name."""
    _ensure_tau_module_registered()
    tau_tool_mod = sys.modules["tau_bench.envs.tool"]

    here = Path(__file__).resolve().parent.parent
    tools_dir = here / "tau_bench" / "envs" / domain / "tools"
    if not tools_dir.is_dir():
        raise RuntimeError(f"tau_bench {domain} tools dir not found: {tools_dir}")

    out: dict[str, type] = {}
    for fn in sorted(os.listdir(tools_dir)):
        if fn.startswith("_") or not fn.endswith(".py"):
            continue
        path = tools_dir / fn
        mod_name = f"_tau_native_{domain}_{fn[:-3]}"
        s = importlib.util.spec_from_file_location(mod_name, path)
        m = importlib.util.module_from_spec(s)
        sys.modules[mod_name] = m
        try:
            s.loader.exec_module(m)
        except Exception as e:
            logger.warning("tau_bench %s tool %s failed to load: %s", domain, fn, e)
            continue
        for attr in dir(m):
            v = getattr(m, attr)
            if isinstance(v, type) and issubclass(v, tau_tool_mod.Tool) and v is not tau_tool_mod.Tool:
                try:
                    nm = v.get_info()["function"]["name"]
                    out[nm] = v
                except Exception:
                    pass
    return out


_STATE_CACHES: dict[tuple[str, ...], dict[str, dict]] = {}
_STATE_CACHE_LOCK = threading.Lock()

_DEFAULT_STATE_DIR = Path.cwd() / "data" / "rl" / "paraagent-rl" / "state"
_STATE_DIRS_ENV = "TAU_INITIAL_STATE_DIRS"

_SHARED_DB_KEY = "__shared_db__"
_TAU_DATA_ROOT = Path(__file__).resolve().parent.parent / "tau_bench" / "envs"
_TAU_DATA_ROOT_ENV = "TAU_FULL_DB_ROOT"
_FULL_DB_CACHE: dict[tuple[str, str], dict | None] = {}
_FULL_DB_CACHE_LOCK = threading.Lock()
_EXPECTED_DB_COLLECTIONS = {
    "retail": frozenset({"orders", "products", "users"}),
    "airline": frozenset({"flights", "reservations", "users"}),
}


def _resolve_state_dirs(
    state_dir: str | Path | Iterable[str | Path] | None = None,
) -> tuple[Path, ...]:
    """Use explicit state_dir, then TAU_INITIAL_STATE_DIRS, then the default state directory.
    Environment paths are ordered and separated by os.pathsep.
    """
    if state_dir is None:
        configured = os.environ.get(_STATE_DIRS_ENV, "").strip()
        raw_dirs: list[str | Path] = (
            [item for item in configured.split(os.pathsep) if item.strip()]
            if configured
            else [_DEFAULT_STATE_DIR]
        )
    elif isinstance(state_dir, (str, Path)):
        raw_dirs = [state_dir]
    else:
        raw_dirs = list(state_dir)

    resolved: list[Path] = []
    seen: set[str] = set()
    for raw in raw_dirs:
        path = Path(raw).expanduser().resolve(strict=False)
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        resolved.append(path)
    if not resolved:
        resolved.append(_DEFAULT_STATE_DIR.expanduser().resolve(strict=False))
    return tuple(resolved)


def _state_dirs_cache_key(state_dirs: Iterable[Path]) -> tuple[str, ...]:
    return tuple(str(path) for path in state_dirs)


def _load_full_domain_db(domain: str, full_db_root: Path | None = None) -> dict | None:
    """Load one vendored full tau-bench DB, cached by data-root and domain."""
    expected = _EXPECTED_DB_COLLECTIONS.get(domain)
    if expected is None:
        logger.warning("unknown tau shared-DB domain: %r", domain)
        return None

    configured_root = os.environ.get(_TAU_DATA_ROOT_ENV, "").strip()
    data_root = Path(full_db_root or configured_root or _TAU_DATA_ROOT).expanduser().resolve(strict=False)
    cache_key = (str(data_root), domain)
    cached = _FULL_DB_CACHE.get(cache_key)
    if cache_key in _FULL_DB_CACHE:
        return cached

    with _FULL_DB_CACHE_LOCK:
        if cache_key in _FULL_DB_CACHE:
            return _FULL_DB_CACHE[cache_key]
        data_dir = data_root / domain / "data"
        db: dict | None = None
        if not data_dir.is_dir():
            logger.warning("tau full-DB data dir not found: %s", data_dir)
        else:
            loaded: dict[str, Any] = {}
            for path in sorted(data_dir.glob("*.json")):
                try:
                    with open(path, "r", encoding="utf-8") as file:
                        loaded[path.stem] = json.load(file)
                except Exception as exc:
                    logger.warning("failed to load tau full-DB %s: %s", path, exc)
            missing = sorted(expected - loaded.keys())
            if missing:
                logger.warning(
                    "tau full-DB %s is incomplete; missing collections: %s",
                    domain,
                    missing,
                )
            else:
                db = loaded
                logger.warning(
                    "TauNativeExecutor: loaded full %s DB (%d collections) from %s",
                    domain,
                    len(db),
                    data_dir,
                )
        _FULL_DB_CACHE[cache_key] = db
        return db


def _load_all_initial_states(state_dir: Path, full_db_root: Path | None = None) -> dict[str, dict]:
    """Load every ``{sample_id}.json`` from one directory.

    A ``{"__shared_db__": "retail|airline"}`` file resolves to the cached
    vendored full-domain DB. Sessions still receive a deep copy in ``reset``.
    """
    out: dict[str, dict] = {}
    if not state_dir.is_dir():
        logger.warning("tau initial-state dir not found: %s — TauNativeExecutor disabled", state_dir)
        return out
    n_loaded = 0
    n_shared = 0
    for p in state_dir.iterdir():
        if not p.suffix == ".json" or p.name.startswith("_"):
            continue
        sid = p.stem
        raw = None
        try:
            with open(p, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict) and _SHARED_DB_KEY in raw:
                domain = raw.get(_SHARED_DB_KEY)
                full_db = _load_full_domain_db(domain, full_db_root) if isinstance(domain, str) else None
                if full_db is None:
                    logger.warning(
                        "shared-DB state %s references unavailable domain %r",
                        sid,
                        domain,
                    )
                    continue
                if "__flight_time_overrides__" in raw:
                    if domain != "airline":
                        raise ValueError("Flight time overrides require Airline")
                    from paraagent.toolenv.runtime.airline_connections import apply_time_overrides

                    out[sid] = apply_time_overrides(full_db, raw["__flight_time_overrides__"])
                else:
                    out[sid] = full_db
                n_shared += 1
            else:
                out[sid] = raw
            n_loaded += 1
        except Exception as e:
            if isinstance(locals().get("raw"), dict) and "__flight_time_overrides__" in raw:
                raise ValueError(f"Invalid per-task flight time override: {sid}") from e
            logger.warning("failed to load tau state %s: %s", p.name, e)

    logger.warning(
        "TauNativeExecutor: loaded %d initial states from %s (%d shared-DB pointers)",
        n_loaded,
        state_dir,
        n_shared,
    )
    return out


def _load_initial_state_dirs(
    state_dirs: tuple[Path, ...], full_db_root: Path | None = None
) -> dict[str, dict]:
    """Union directories in order; a conflicting later sample never overwrites."""
    merged: dict[str, dict] = {}
    sources: dict[str, Path] = {}
    for state_dir in state_dirs:
        for sample_id, state in _load_all_initial_states(state_dir, full_db_root).items():
            if sample_id not in merged:
                merged[sample_id] = state
                sources[sample_id] = state_dir
                continue
            if merged[sample_id] != state:
                logger.warning(
                    "duplicate tau state %s differs between %s and %s; keeping first",
                    sample_id,
                    sources[sample_id],
                    state_dir,
                )
    logger.warning(
        "TauNativeExecutor: unioned %d initial states from %d directories",
        len(merged),
        len(state_dirs),
    )
    return merged


def ensure_state_cache(
    state_dir: str | Path | Iterable[str | Path] | None = None,
    *,
    full_db_root: Path | None = None,
) -> dict[str, dict]:
    state_dirs = _resolve_state_dirs(state_dir)
    cache_key = _state_dirs_cache_key(state_dirs)
    if full_db_root is not None:
        cache_key += ("full_db_root", str(full_db_root.resolve()))
    cached = _STATE_CACHES.get(cache_key)
    if cached is not None:
        return cached
    with _STATE_CACHE_LOCK:
        cached = _STATE_CACHES.get(cache_key)
        if cached is None:
            cached = _load_initial_state_dirs(state_dirs, full_db_root)
            _STATE_CACHES[cache_key] = cached
    return cached


TAU_TOOL_NAMES_RETAIL = frozenset(
    {
        "calculate",
        "cancel_pending_order",
        "exchange_delivered_order_items",
        "find_user_id_by_email",
        "find_user_id_by_name_zip",
        "get_order_details",
        "get_product_details",
        "get_user_details",
        "list_all_product_types",
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "modify_user_address",
        "return_delivered_order_items",
        "think",
        "transfer_to_human_agents",
    }
)
TAU_TOOL_NAMES_AIRLINE = frozenset(
    {
        "book_reservation",
        "calculate",
        "cancel_reservation",
        "get_reservation_details",
        "get_user_details",
        "list_all_airports",
        "search_direct_flight",
        "search_onestop_flight",
        "send_certificate",
        "think",
        "transfer_to_human_agents",
        "update_reservation_baggages",
        "update_reservation_flights",
        "update_reservation_passengers",
    }
)
TAU_TOOL_NAMES_ALL = TAU_TOOL_NAMES_RETAIL | TAU_TOOL_NAMES_AIRLINE


def domain_for_sample_id(sample_id: str) -> str | None:
    """Resolve the domain from a released Simia task ID."""
    if not isinstance(sample_id, str):
        return None
    match = re.fullmatch(r"simia_(airline|retail)_[0-9]{6}", sample_id)
    return match.group(1) if match else None


class TauNativeExecutor:
    """Per-process registry; produce per-trajectory sessions via `reset(sample_id)`.

    Thread/process-safe: each session keeps its own deep-copied state dict.
    """

    _tools_airline: dict[str, type] | None = None
    _tools_retail: dict[str, type] | None = None
    _tools_lock = threading.Lock()

    _transaction_lock = threading.RLock()

    @classmethod
    def _ensure_tools_loaded(cls):
        if cls._tools_airline is not None and cls._tools_retail is not None:
            return
        with cls._tools_lock:
            if cls._tools_airline is None:
                cls._tools_airline = _load_domain_tools("airline")
            if cls._tools_retail is None:
                cls._tools_retail = _load_domain_tools("retail")

    def __init__(
        self,
        state_dir: str | Path | Iterable[str | Path] | None = None,
        *,
        write_schema_mode: str | None = None,
        temporal_mode: str | None = None,
        business_mode: str | None = None,
        transaction_mode: str | None = None,
        inventory_mode: str | None = None,
        isolation_mode: str | None = None,
        feedback_mode: str | None = None,
        connection_mode: str | None = None,
    ):
        modes = require_current_native_modes(
            write_schema_mode=write_schema_mode,
            temporal_mode=temporal_mode,
            business_mode=business_mode,
            transaction_mode=transaction_mode,
            inventory_mode=inventory_mode,
            isolation_mode=isolation_mode,
            feedback_mode=feedback_mode,
            connection_mode=connection_mode,
        )
        for name, value in modes.items():
            setattr(self, "_" + name, value)
        self._transaction_lock = threading.RLock()
        self._ensure_tools_loaded()
        self._state_dirs = _resolve_state_dirs(state_dir)
        self._full_db_root = Path(os.environ.get(_TAU_DATA_ROOT_ENV) or _TAU_DATA_ROOT).expanduser().resolve()

        self._initial_states = deepcopy(ensure_state_cache(self._state_dirs, full_db_root=self._full_db_root))
        self._session_owners: dict[str, str] = {}
        self._sessions: dict[str, dict] = {}
        self._session_meta: dict[str, dict[str, Any]] = {}

        self._initial_hash_cache: dict[int, str] = {}
        self._initial_hash_lock = threading.Lock()

        self._expected_summary_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._expected_summary_lock = threading.Lock()

    def has_state(self, sample_id: str) -> bool:
        cache = self._initial_cache()
        return sample_id in cache

    def tool_schema(self, sample_id: str, tool_name: str):
        """Return a detached schema for a tool actually mounted in this domain."""
        domain = domain_for_sample_id(sample_id)
        tools = (
            self._tools_airline if domain == "airline" else self._tools_retail if domain == "retail" else {}
        )
        if not isinstance(tool_name, str) or tool_name not in (tools or {}):
            return None
        info = tools[tool_name].get_info()
        if domain == "airline":
            info = deepcopy(info)
            if tool_name in {"book_reservation", "update_reservation_flights", "search_onestop_flight"}:
                info["function"]["description"] += (
                    " Use dated estimated arrival/departure times when present. "
                    "Adjacent active flights must connect at the same airport with at least 60 minutes between arrival and next departure; "
                    "the 60-minute floor is this simulator's connection rule."
                )
        from paraagent.toolenv.runtime.tau_feedback import mounted_schema

        return mounted_schema(info)

    def _initial_cache(self):
        return self._initial_states

    def _initial_state_hash(self, initial: dict) -> str:
        cache_key = id(initial)
        cached = self._initial_hash_cache.get(cache_key)
        if cached is not None:
            return cached
        digest = canonical_tau_state_hash(initial)
        with self._initial_hash_lock:
            return self._initial_hash_cache.setdefault(cache_key, digest)

    @staticmethod
    def _normalize_gt_call(call: Any) -> tuple[str, dict[str, Any]]:
        if not isinstance(call, dict):
            raise ValueError("GT write call is not an object")
        tool_name = call.get("tool") or call.get("name")
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValueError("GT write call has no valid tool/name")
        if "args" in call:
            arguments = call.get("args")
        elif "kwargs" in call:
            arguments = call.get("kwargs")
        else:
            arguments = call.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ValueError(f"GT write call {tool_name!r} arguments are not an object")
        return tool_name.strip(), arguments

    def _expected_state_summary(
        self,
        sample_id: str,
        initial: dict,
        initial_hash: str,
        gt_write_calls: list[Any],
    ) -> dict[str, Any]:
        gt_json = json.dumps(
            gt_write_calls,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        gt_digest = hashlib.sha256(gt_json.encode("utf-8")).hexdigest()
        cache_key = (sample_id, gt_digest)
        cache_key += (initial_hash, "seats_v1")
        cached = self._expected_summary_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        summary: dict[str, Any] = {
            "tau_gt_replay_ok": False,
            "tau_expected_final_state_hash": "",
            "tau_expected_state_changed": False,
            "tau_gt_replay_error": "",
        }
        expected = deepcopy(initial)
        for index, raw_call in enumerate(gt_write_calls):
            try:
                tool_name, arguments = self._normalize_gt_call(raw_call)
            except Exception as exc:
                summary["tau_gt_replay_error"] = f"gt_call[{index}]: {type(exc).__name__}: {exc}"
                break
            result = self._invoke_on_state(
                state=expected,
                sample_id=sample_id,
                tool_name=tool_name,
                arguments=arguments,
            )
            if not bool(result.get("success")):
                content = result.get("content") or {}
                error = content.get("error") if isinstance(content, dict) else None
                summary["tau_gt_replay_error"] = (
                    f"gt_call[{index}] {tool_name}: {json.dumps(error, ensure_ascii=False, sort_keys=True)}"
                )
                break
        else:
            try:
                expected_hash = canonical_tau_state_hash(expected)
            except Exception as exc:
                summary["tau_gt_replay_error"] = f"expected_state_hash: {type(exc).__name__}: {exc}"
            else:
                summary.update(
                    {
                        "tau_gt_replay_ok": True,
                        "tau_expected_final_state_hash": expected_hash,
                        "tau_expected_state_changed": expected_hash != initial_hash,
                    }
                )

        with self._expected_summary_lock:
            stored = self._expected_summary_cache.setdefault(cache_key, dict(summary))
        return dict(stored)

    @_serialized_session_operation
    def reset(
        self,
        session_id: str,
        sample_id: str,
        gt_write_calls: list[Any] | None = None,
    ) -> bool:
        """Initialize a fresh session for a trajectory.

        Returns True if state was loaded; False if sample_id unknown.
        """
        cache = self._initial_cache()
        if session_id in self._sessions:
            raise ValueError("Active rollout session ID already exists; cleanup before reusing it")
        initial = cache.get(sample_id)
        if initial is None:
            return False
        self._sessions[session_id] = deepcopy(initial)
        self._session_owners[session_id] = sample_id
        meta: dict[str, Any] = {
            "tau_state_hash_version": TAU_STATE_HASH_VERSION,
            "tau_state_capture_ok": False,
            "tau_initial_state_hash": "",
            "tau_expected_final_state_hash": "",
            "tau_agent_final_state_hash": "",
            "tau_expected_state_changed": False,
            "tau_agent_state_changed": False,
            "tau_gt_replay_ok": False,
            "tau_gt_replay_error": "",
            "tau_state_capture_error": "",
        }
        try:
            initial_hash = self._initial_state_hash(initial)
        except Exception as exc:
            meta["tau_state_capture_error"] = f"initial_state_hash: {type(exc).__name__}: {exc}"
        else:
            meta["tau_initial_state_hash"] = initial_hash
            if not isinstance(gt_write_calls, list):
                meta["tau_gt_replay_error"] = "missing_or_invalid_gt_write_calls"
            else:
                try:
                    meta.update(
                        self._expected_state_summary(
                            sample_id,
                            initial,
                            initial_hash,
                            gt_write_calls,
                        )
                    )
                except Exception as exc:
                    meta["tau_gt_replay_error"] = f"gt_replay_setup: {type(exc).__name__}: {exc}"
        self._session_meta[session_id] = meta
        return True

    @_serialized_session_operation
    def cleanup(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._session_meta.pop(session_id, None)
        self._session_owners.pop(session_id, None)

    @_serialized_session_operation
    def has_session(self, session_id: str) -> bool:
        return session_id in self._sessions

    @_serialized_session_operation
    def get_session_state_summary(self, session_id: str) -> dict[str, Any]:
        """Capture the final session state without exposing the full database."""
        state = self._sessions.get(session_id)
        meta = self._session_meta.get(session_id)
        if state is None or meta is None:
            return {
                "tau_state_hash_version": TAU_STATE_HASH_VERSION,
                "tau_state_capture_ok": False,
                "tau_gt_replay_ok": False,
                "tau_initial_state_hash": "",
                "tau_expected_final_state_hash": "",
                "tau_agent_final_state_hash": "",
                "tau_expected_state_changed": False,
                "tau_agent_state_changed": False,
                "tau_gt_replay_error": "missing_tau_session",
                "tau_state_capture_error": "missing_tau_session",
            }

        summary = dict(meta)
        try:
            agent_hash = canonical_tau_state_hash(state)
        except Exception as exc:
            summary["tau_state_capture_ok"] = False
            summary["tau_state_capture_error"] = f"agent_final_state_hash: {type(exc).__name__}: {exc}"
            return summary

        initial_hash = summary.get("tau_initial_state_hash", "")
        summary["tau_agent_final_state_hash"] = agent_hash
        summary["tau_agent_state_changed"] = bool(initial_hash and agent_hash != initial_hash)
        summary["tau_state_capture_ok"] = bool(initial_hash)
        return summary

    @_serialized_session_operation
    def execute(
        self,
        session_id: str,
        sample_id: str,
        tool_name: str,
        arguments: Any,
    ) -> dict:
        """Invoke a tau-native tool against session state.

        Returns simulator-shaped payload:
            {"success": bool, "content": {"error": dict|None, "response": str|dict}, "error_msg": str|None}
        """
        state = self._sessions.get(session_id)
        if state is not None and self._session_owners.get(session_id) != sample_id:
            return _err_payload(
                "InvalidRequestError", "Rollout session does not belong to this sample_id", tool_name
            )
        if state is None:
            return _err_payload(
                "ToolExecutionError",
                f"tau session {session_id!r} not initialized for sample_id "
                f"{sample_id!r} (ToolEnv.reset must run before tool calls)",
                tool_name,
                outcome=_tau_guard_outcome(
                    category=CATEGORY_EXECUTOR_FAILURE,
                    stage=STAGE_EXECUTOR,
                    code="TAU_SESSION_NOT_INITIALIZED",
                    retry_hint=RETRY_UNKNOWN,
                    raw_type="ToolExecutionError",
                    causal_inputs=(),
                ),
            )
        return self._invoke_on_state(
            state=state,
            sample_id=sample_id,
            tool_name=tool_name,
            arguments=arguments,
        )

    def _invoke_on_state(
        self,
        *,
        state: dict,
        sample_id: str,
        tool_name: str,
        arguments: Any,
    ) -> dict:
        """Invoke a tool through the shared agent/GT replay path.

        With transactions enabled, commit in-memory writes only on success.
        Copy arguments as well as state because native tools may mutate nested values.
        """
        mode = "all_atomic_v1"
        if not isinstance(tool_name, str) or not tool_name.strip():
            return _err_payload("InvalidRequestError", "Invalid tool name: expected a non-empty string.", "")
        domain = domain_for_sample_id(sample_id)
        atomic = domain in {"airline", "retail"}
        if not atomic:
            return self._invoke_inplace(
                state=state,
                sample_id=sample_id,
                tool_name=tool_name,
                arguments=arguments,
            )

        with self._transaction_lock:
            if tool_name not in _MUTATING_TOOLS:
                try:
                    result = self._invoke_inplace(
                        state=state,
                        sample_id=sample_id,
                        tool_name=tool_name,
                        arguments=arguments,
                    )
                except Exception as exc:
                    return _err_payload(
                        "ToolExecutionError",
                        f"Tool execution failed ({type(exc).__name__}).",
                        tool_name,
                        outcome=_tau_guard_outcome(
                            category=CATEGORY_EXECUTOR_FAILURE,
                            stage=STAGE_EXECUTOR,
                            code="TAU_READ_EXECUTION_FAILURE",
                            retry_hint=RETRY_UNKNOWN,
                            raw_type="ToolExecutionError",
                            causal_inputs=(),
                        ),
                    )
                return deepcopy(result)
            try:
                staged = deepcopy(state)
                staged_arguments = deepcopy(arguments)
                result = self._invoke_inplace(
                    state=staged,
                    sample_id=sample_id,
                    tool_name=tool_name,
                    arguments=staged_arguments,
                )
            except Exception as exc:
                return _err_payload(
                    "ToolExecutionError",
                    f"transaction aborted: {type(exc).__name__}: {exc}",
                    tool_name,
                    outcome=_tau_guard_outcome(
                        category=CATEGORY_EXECUTOR_FAILURE,
                        stage=STAGE_EXECUTOR,
                        code="TAU_TRANSACTION_ABORTED",
                        retry_hint=RETRY_UNKNOWN,
                        raw_type="ToolExecutionError",
                        causal_inputs=(),
                    ),
                )
            if result.get("success") is True:
                state.clear()
                state.update(staged)
            return deepcopy(result)

    def _invoke_inplace(
        self,
        *,
        state: dict,
        sample_id: str,
        tool_name: str,
        arguments: Any,
    ) -> dict:
        """Shared native adapter; state may be a private transaction snapshot."""
        feedback = True
        if feedback and (not isinstance(tool_name, str) or not tool_name.strip()):
            return _err_payload("InvalidRequestError", "Invalid tool name: expected a non-empty string.", "")
        domain = domain_for_sample_id(sample_id)
        if domain == "airline":
            tools = type(self)._tools_airline or {}
        elif domain == "retail":
            tools = type(self)._tools_retail or {}
        else:
            return _err_payload(
                "InvalidRequestError",
                f"sample_id {sample_id!r} is not a tau task",
                tool_name,
                outcome=_tau_guard_outcome(
                    category=CATEGORY_INVALID_REQUEST,
                    stage=STAGE_REQUEST,
                    code="INVALID_SAMPLE_ID",
                    retry_hint=RETRY_NEVER,
                    raw_type="InvalidRequestError",
                    causal_inputs=("sample_id",),
                ),
            )

        if tool_name not in tools:
            return _err_payload(
                "InvalidRequestError",
                f"tau {domain} has no tool named {tool_name!r}",
                tool_name,
                outcome=_tau_guard_outcome(
                    category=CATEGORY_TOOL_NOT_FOUND,
                    stage=STAGE_DISPATCH,
                    code="TOOL_NOT_FOUND",
                    retry_hint=RETRY_NEVER,
                    raw_type="InvalidRequestError",
                    causal_inputs=("tool_name",),
                    raw_message=f"tau {domain} has no tool named {tool_name!r}",
                ),
            )

        if not isinstance(arguments, dict):
            return _err_payload(
                "InvalidRequestError",
                f"tool arguments must be a JSON object, got {type(arguments).__name__}",
                tool_name,
                outcome=_tau_guard_outcome(
                    category=CATEGORY_INVALID_REQUEST,
                    stage=STAGE_REQUEST,
                    code="ARGUMENTS_NOT_OBJECT",
                    retry_hint=RETRY_NEVER,
                    raw_type="InvalidRequestError",
                    causal_inputs=("arguments",),
                ),
            )

        if feedback:
            from paraagent.toolenv.runtime.tau_feedback import argument_errors

            schema = self.tool_schema(sample_id, tool_name)["function"]["parameters"]
            errors = argument_errors(arguments, schema)
            if errors:
                return _err_payload(
                    "InvalidRequestError",
                    "Invalid tool parameters: " + "; ".join(errors[:8]),
                    tool_name,
                    outcome=_tau_guard_outcome(
                        category=CATEGORY_INVALID_REQUEST,
                        stage=STAGE_REQUEST,
                        code="INVALID_ARGUMENT_STRUCTURE",
                        retry_hint=RETRY_NEVER,
                        raw_type="InvalidRequestError",
                        causal_inputs=tuple(errors[:8]),
                    ),
                )

        parameter_schema: dict[str, Any] = {}
        try:
            info = tools[tool_name].get_info()
            parameter_schema = info["function"]["parameters"]
            props = parameter_schema.get("properties", {})
            filtered_args = {k: v for k, v in arguments.items() if k in props}
        except Exception:
            filtered_args = dict(arguments)
        invalid_argument_inputs = _schema_invalid_causal_inputs(
            parameter_schema,
            filtered_args,
        )
        if (
            not feedback
            and tool_name in _MUTATING_TOOLS
            and (invalid_argument_inputs or not parameter_schema)
        ):
            return _err_payload(
                "InvalidRequestError",
                "write arguments violate native JSON schema: "
                + ", ".join(invalid_argument_inputs or ("schema unavailable",)),
                tool_name,
                outcome=_tau_guard_outcome(
                    category=CATEGORY_INVALID_REQUEST,
                    stage=STAGE_REQUEST,
                    code="INVALID_ARGUMENT_STRUCTURE",
                    retry_hint=RETRY_NEVER,
                    raw_type="InvalidRequestError",
                    causal_inputs=invalid_argument_inputs or ("arguments",),
                ),
            )

        _oid = filtered_args.get("order_id")
        if isinstance(_oid, str) and re.match(r"^[Ww]\d", _oid):
            filtered_args["order_id"] = "#" + _oid.lstrip("#")

        try:
            if feedback and domain == "airline" and tool_name == "update_reservation_flights":
                from paraagent.toolenv.runtime.tau_feedback import retained_flight_error

                error = retained_flight_error(state, filtered_args)
                if error:
                    return _err_payload(
                        "InvalidRequestError",
                        error,
                        tool_name,
                        outcome=_tau_guard_outcome(
                            category=CATEGORY_INVALID_REQUEST,
                            stage=STAGE_REQUEST,
                            code="INVALID_RETAINED_FLIGHT_METADATA",
                            retry_hint=RETRY_NEVER,
                            raw_type="InvalidRequestError",
                            causal_inputs=("flights",),
                        ),
                    )
            if feedback and domain == "airline" and tool_name == "send_certificate":
                from paraagent.toolenv.runtime.tau_feedback import certificate_error

                error = certificate_error(state, filtered_args)
                if error:
                    return _err_payload(
                        "BusinessRuleError",
                        error,
                        tool_name,
                        outcome=_tau_guard_outcome(
                            category=CATEGORY_STATE_CONFLICT,
                            stage=STAGE_DOMAIN,
                            code="AIRLINE_CERTIFICATE_ID_SPACE_EXHAUSTED",
                            retry_hint=RETRY_NEVER,
                            raw_type="BusinessRuleError",
                            causal_inputs=("user_id",),
                        ),
                    )
            if domain == "airline" and tool_name == "book_reservation":
                booking_error = _airline_booking_module().check_airline_booking(
                    state,
                    filtered_args,
                )
                if booking_error is not None:
                    return _err_payload(
                        "BusinessRuleError",
                        booking_error["message"],
                        tool_name,
                        outcome=_tau_guard_outcome(
                            category=CATEGORY_STATE_CONFLICT,
                            stage=STAGE_DOMAIN,
                            code=booking_error["code"],
                            retry_hint=RETRY_NEVER,
                            raw_type="BusinessRuleError",
                            causal_inputs=booking_error["causal_inputs"],
                        ),
                    )
            if domain == "airline" and tool_name in {"cancel_reservation", "update_reservation_baggages"}:
                business_error = _airline_business_module().check_airline_business(
                    state,
                    tool_name,
                    filtered_args,
                )
                if business_error is not None:
                    return _err_payload(
                        "BusinessRuleError",
                        business_error["message"],
                        tool_name,
                        outcome=_tau_guard_outcome(
                            category=CATEGORY_STATE_CONFLICT,
                            stage=STAGE_DOMAIN,
                            code=business_error["code"],
                            retry_hint=RETRY_NEVER,
                            raw_type="BusinessRuleError",
                            causal_inputs=business_error["causal_inputs"],
                        ),
                    )
            if domain == "airline" and tool_name in {
                "update_reservation_baggages",
                "update_reservation_flights",
                "update_reservation_passengers",
            }:
                temporal_error = _airline_temporal_module().check_airline_temporal(
                    state,
                    tool_name,
                    filtered_args,
                )
                if temporal_error is not None:
                    return _err_payload(
                        "BusinessRuleError",
                        temporal_error["message"],
                        tool_name,
                        outcome=_tau_guard_outcome(
                            category=CATEGORY_STATE_CONFLICT,
                            stage=STAGE_DOMAIN,
                            code=temporal_error["code"],
                            retry_hint=RETRY_NEVER,
                            raw_type="BusinessRuleError",
                            causal_inputs=temporal_error["causal_inputs"],
                        ),
                    )
            if domain == "airline" and tool_name in {"cancel_reservation", "update_reservation_flights"}:
                eligibility_error = _airline_eligibility_module().check_airline_eligibility(
                    state,
                    tool_name,
                    filtered_args,
                )
                if eligibility_error is not None:
                    return _err_payload(
                        "BusinessRuleError",
                        eligibility_error["message"],
                        tool_name,
                        outcome=_tau_guard_outcome(
                            category=CATEGORY_STATE_CONFLICT,
                            stage=STAGE_DOMAIN,
                            code=eligibility_error["code"],
                            retry_hint=RETRY_NEVER,
                            raw_type="BusinessRuleError",
                            causal_inputs=eligibility_error["causal_inputs"],
                        ),
                    )
            if domain == "airline":
                from paraagent.toolenv.runtime.airline_connections import check_airline_connections

                connection_error = check_airline_connections(state, tool_name, filtered_args)
                if connection_error is not None:
                    return _err_payload(
                        "BusinessRuleError",
                        connection_error["message"],
                        tool_name,
                        outcome=_tau_guard_outcome(
                            category=CATEGORY_STATE_CONFLICT,
                            stage=STAGE_DOMAIN,
                            code=connection_error["code"],
                            retry_hint=RETRY_NEVER,
                            raw_type="BusinessRuleError",
                            causal_inputs=connection_error["causal_inputs"],
                        ),
                    )
            inventory_changes = []
            if domain == "airline":
                from paraagent.toolenv.runtime import airline_inventory

                try:
                    inventory_changes = airline_inventory.plan(state, tool_name, filtered_args)
                except airline_inventory.InventoryError as exc:
                    return _err_payload(
                        "BusinessRuleError",
                        str(exc),
                        tool_name,
                        outcome=_tau_guard_outcome(
                            category=CATEGORY_CAPACITY_UNAVAILABLE
                            if exc.code == "AIRLINE_INVENTORY_INSUFFICIENT"
                            else CATEGORY_STATE_CONFLICT,
                            stage=STAGE_DOMAIN,
                            code=exc.code,
                            retry_hint=RETRY_NEVER,
                            raw_type="BusinessRuleError",
                            causal_inputs=("flights",),
                        ),
                    )
            if domain == "airline" and tool_name == "search_onestop_flight":
                from paraagent.toolenv.runtime.airline_connections import search_onestop

                result = search_onestop(state, **filtered_args)
            else:
                result = tools[tool_name].invoke(data=state, **filtered_args)
            if feedback and result is None:
                return _err_payload(
                    "ToolExecutionError",
                    "Tool returned no result; the operation was not committed.",
                    tool_name,
                    outcome=_tau_guard_outcome(
                        category=CATEGORY_EXECUTOR_FAILURE,
                        stage=STAGE_EXECUTOR,
                        code="TAU_MISSING_TOOL_RESULT",
                        retry_hint=RETRY_UNKNOWN,
                        raw_type="ToolExecutionError",
                        causal_inputs=(),
                    ),
                )
            if inventory_changes and not (isinstance(result, str) and result.startswith("Error:")):
                airline_inventory.commit(state, inventory_changes)
        except KeyError as e:
            if invalid_argument_inputs:
                outcome = _tau_guard_outcome(
                    category=CATEGORY_INVALID_REQUEST,
                    stage=STAGE_REQUEST,
                    code="INVALID_ARGUMENT_STRUCTURE",
                    retry_hint=RETRY_NEVER,
                    raw_type="ToolExecutionError",
                    causal_inputs=invalid_argument_inputs,
                )
            else:
                outcome = _tau_guard_outcome(
                    category=CATEGORY_EXECUTOR_FAILURE,
                    stage=STAGE_EXECUTOR,
                    code="TAU_STATE_FIELD_MISSING",
                    retry_hint=RETRY_UNKNOWN,
                    raw_type="ToolExecutionError",
                    causal_inputs=(),
                )
            return _err_payload(
                "ToolExecutionError",
                f"missing field: {e}",
                tool_name,
                outcome=outcome,
            )
        except TypeError as e:
            return _err_payload(
                "InvalidRequestError",
                f"invalid arguments: {e}",
                tool_name,
                outcome=_tau_guard_outcome(
                    category=CATEGORY_INVALID_REQUEST,
                    stage=STAGE_REQUEST,
                    code="INVALID_ARGUMENTS",
                    retry_hint=RETRY_NEVER,
                    raw_type="InvalidRequestError",
                    causal_inputs=(invalid_argument_inputs or _type_error_causal_inputs(e)),
                ),
            )
        except Exception as e:
            if invalid_argument_inputs:
                outcome = _tau_guard_outcome(
                    category=CATEGORY_INVALID_REQUEST,
                    stage=STAGE_REQUEST,
                    code="INVALID_ARGUMENT_STRUCTURE",
                    retry_hint=RETRY_NEVER,
                    raw_type="ToolExecutionError",
                    causal_inputs=invalid_argument_inputs,
                )
            else:
                outcome = _tau_guard_outcome(
                    category=CATEGORY_EXECUTOR_FAILURE,
                    stage=STAGE_EXECUTOR,
                    code="TAU_TOOL_INTERNAL_ERROR",
                    retry_hint=RETRY_UNKNOWN,
                    raw_type="ToolExecutionError",
                    causal_inputs=(),
                )
            return _err_payload(
                "ToolExecutionError",
                f"{type(e).__name__}: {e}",
                tool_name,
                outcome=outcome,
            )

        if isinstance(result, str) and result.startswith("Error:"):
            raw = result.replace("Error:", "", 1).strip()
            raw_type = _classify_tau_error_type(raw)
            return _err_payload(
                raw_type,
                raw,
                tool_name,
                outcome=_classify_tau_tool_outcome(
                    tool_name=tool_name,
                    raw_message=raw,
                    raw_type=raw_type,
                ),
            )

        parsed: Any = result
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
            except Exception:
                parsed = result

        if feedback:
            try:
                if parsed is None:
                    raise ValueError("null result")
                json.dumps(parsed, allow_nan=False)
            except (ValueError, TypeError, OverflowError, RecursionError):
                return _err_payload(
                    "ToolExecutionError",
                    "Tool returned an invalid result; the operation was not committed.",
                    tool_name,
                    outcome=_tau_guard_outcome(
                        category=CATEGORY_EXECUTOR_FAILURE,
                        stage=STAGE_EXECUTOR,
                        code="TAU_INVALID_TOOL_RESULT",
                        retry_hint=RETRY_UNKNOWN,
                        raw_type="ToolExecutionError",
                        causal_inputs=(),
                    ),
                )

        return {
            "success": True,
            "content": {"error": None, "response": parsed},
            "error_msg": None,
            "tool_outcome": make_tool_outcome(
                True,
                stage=STAGE_COMPLETED,
                category=CATEGORY_SUCCESS,
                code="SUCCESS",
                basis=BASIS_NATIVE_RETURN_SITE,
                retry_hint=RETRY_NOT_APPLICABLE,
                causal_inputs=(),
                raw_type=None,
                raw_message=None,
                source="tau_native",
            ),
        }


def _classify_tau_error_type(raw_msg: str) -> str:
    """Classify native rejections as NotFoundError or BusinessRuleError.

    Keep business markers free of non-business substrings for reward matching.
    Argument/schema failures are handled separately as InvalidRequestError.
    """
    low = raw_msg.lower()
    if "not found" in low or "not exist" in low:
        return "NotFoundError"
    return "BusinessRuleError"


def _tau_guard_outcome(
    *,
    category: str,
    stage: str,
    code: str,
    retry_hint: str,
    raw_type: str,
    causal_inputs: Iterable[str],
    raw_message: Any = None,
) -> dict[str, Any]:
    """Build a reward-only outcome for an executor-owned validation branch."""
    return make_tool_outcome(
        False,
        stage=stage,
        category=category,
        code=code,
        basis=BASIS_LOCAL_VALIDATION,
        retry_hint=retry_hint,
        causal_inputs=causal_inputs,
        raw_type=raw_type,
        raw_message=raw_message,
        source="tau_native",
    )


def _type_error_causal_inputs(error: TypeError) -> tuple[str, ...]:
    """Extract only an explicitly named bad/missing parameter, else fail closed."""
    message = str(error)
    names: list[str] = []
    for pattern in (
        r"missing \d+ required positional argument(?:s)?: (.+)$",
        r"unexpected keyword argument ['\"]([^'\"]+)['\"]",
    ):
        match = re.search(pattern, message)
        if match is None:
            continue
        quoted = re.findall(r"['\"]([^'\"]+)['\"]", match.group(1))
        names.extend(quoted or [match.group(1).strip()])
        break
    return tuple(dict.fromkeys(name for name in names if name))


def _schema_invalid_causal_inputs(
    schema: Any,
    arguments: Any,
) -> tuple[str, ...]:
    """Identify top-level inputs violating the native schema.
    Strict write mode rejects them before execution; business and date checks are separate.
    """
    if not isinstance(schema, dict) or not isinstance(arguments, dict):
        return ()
    invalid: list[str] = []
    required = schema.get("required")
    if isinstance(required, list):
        invalid.extend(name for name in required if isinstance(name, str) and name not in arguments)
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return tuple(dict.fromkeys(invalid))
    for name, value in arguments.items():
        field_schema = properties.get(name)
        if not isinstance(field_schema, dict):
            continue
        if not _json_schema_value_shape_valid(value, field_schema):
            invalid.append(name)
    return tuple(dict.fromkeys(invalid))


def _json_schema_value_shape_valid(value: Any, schema: dict[str, Any]) -> bool:
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        return False
    expected = schema.get("type")
    if expected == "string":
        return isinstance(value, str)
    if expected == "object":
        if not isinstance(value, dict):
            return False
        return not _schema_invalid_causal_inputs(schema, value)
    if expected == "array":
        if not isinstance(value, list):
            return False
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            return all(_json_schema_value_shape_valid(item, item_schema) for item in value)
        return True
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    return True


_MUTATING_TOOLS = frozenset(
    {
        "modify_pending_order_items",
        "modify_pending_order_address",
        "modify_pending_order_payment",
        "cancel_pending_order",
        "return_delivered_order_items",
        "exchange_delivered_order_items",
        "modify_user_address",
        "book_reservation",
        "cancel_reservation",
        "send_certificate",
        "update_reservation_baggages",
        "update_reservation_flights",
        "update_reservation_passengers",
    }
)

_ORDER_TOOLS = frozenset(
    {
        "cancel_pending_order",
        "exchange_delivered_order_items",
        "get_order_details",
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "return_delivered_order_items",
    }
)
_RESERVATION_TOOLS = frozenset(
    {
        "cancel_reservation",
        "get_reservation_details",
        "update_reservation_baggages",
        "update_reservation_flights",
        "update_reservation_passengers",
    }
)
_PAYMENT_METHOD_ID_TOOLS = frozenset(
    {
        "exchange_delivered_order_items",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "return_delivered_order_items",
    }
)
_PAYMENT_ID_TOOLS = frozenset({"update_reservation_baggages", "update_reservation_flights"})


def _native_domain_outcome(
    *,
    category: str,
    code: str,
    causal_inputs: Iterable[str],
    raw_type: str,
    raw_message: str,
) -> dict[str, Any]:
    return make_tool_outcome(
        False,
        stage=STAGE_DOMAIN,
        category=category,
        code=code,
        basis=BASIS_NATIVE_MESSAGE_REGISTRY,
        retry_hint=RETRY_NEVER,
        causal_inputs=causal_inputs,
        raw_type=raw_type,
        raw_message=raw_message,
        source="tau_native",
    )


def _classify_tau_tool_outcome(
    *,
    tool_name: str,
    raw_message: str,
    raw_type: str,
) -> dict[str, Any]:
    """Classify rejections by native tool and exact return pattern; preserve the payload.
    Unrecognized branches remain unknown_failure.
    """
    message = raw_message.strip()
    low = message.lower()

    def native(category: str, code: str, *inputs: str) -> dict[str, Any]:
        return _native_domain_outcome(
            category=category,
            code=code,
            causal_inputs=inputs,
            raw_type=raw_type,
            raw_message=message,
        )

    if low == "order not found" and tool_name in _ORDER_TOOLS:
        return native(CATEGORY_RESOURCE_NOT_FOUND, "ORDER_NOT_FOUND", "order_id")
    if low == "reservation not found" and tool_name in _RESERVATION_TOOLS:
        return native(
            CATEGORY_RESOURCE_NOT_FOUND,
            "RESERVATION_NOT_FOUND",
            "reservation_id",
        )

    if low == "user not found" and tool_name == "get_reservation_details":
        return native(
            CATEGORY_RESOURCE_NOT_FOUND,
            "RESERVATION_NOT_FOUND",
            "reservation_id",
        )
    if low == "user not found":
        if tool_name == "find_user_id_by_email":
            return native(CATEGORY_RESOURCE_NOT_FOUND, "USER_NOT_FOUND", "email")
        if tool_name == "find_user_id_by_name_zip":
            return native(
                CATEGORY_RESOURCE_NOT_FOUND,
                "USER_NOT_FOUND",
                "first_name",
                "last_name",
                "zip",
            )
        if tool_name in {
            "book_reservation",
            "get_user_details",
            "modify_user_address",
            "send_certificate",
        }:
            return native(CATEGORY_RESOURCE_NOT_FOUND, "USER_NOT_FOUND", "user_id")
    if low == "product not found" and tool_name == "get_product_details":
        return native(CATEGORY_RESOURCE_NOT_FOUND, "PRODUCT_NOT_FOUND", "product_id")
    if low == "payment method not found":
        if tool_name in _PAYMENT_METHOD_ID_TOOLS:
            return native(
                CATEGORY_RESOURCE_NOT_FOUND,
                "PAYMENT_METHOD_NOT_FOUND",
                "payment_method_id",
            )
        if tool_name in _PAYMENT_ID_TOOLS:
            return native(
                CATEGORY_RESOURCE_NOT_FOUND,
                "PAYMENT_METHOD_NOT_FOUND",
                "payment_id",
            )
    if low == "some item not found" and tool_name == "return_delivered_order_items":
        return native(CATEGORY_RESOURCE_NOT_FOUND, "ITEM_NOT_FOUND", "item_ids")
    if (
        re.fullmatch(r"item .+ not found in order", message, flags=re.IGNORECASE)
        and tool_name == "return_delivered_order_items"
    ):
        return native(CATEGORY_RESOURCE_NOT_FOUND, "ITEM_NOT_FOUND", "item_ids")

    if low == "non-pending order cannot be cancelled" and tool_name == "cancel_pending_order":
        return native(CATEGORY_STATE_CONFLICT, "ORDER_NOT_PENDING", "order_id")
    if low == "non-pending order cannot be modified" and tool_name in {
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
    }:
        return native(CATEGORY_STATE_CONFLICT, "ORDER_NOT_PENDING", "order_id")
    if low == "non-delivered order cannot be exchanged" and tool_name == "exchange_delivered_order_items":
        return native(CATEGORY_STATE_CONFLICT, "ORDER_NOT_DELIVERED", "order_id")
    if low == "non-delivered order cannot be returned" and tool_name == "return_delivered_order_items":
        return native(CATEGORY_STATE_CONFLICT, "ORDER_NOT_DELIVERED", "order_id")
    if (
        low == "there should be exactly one payment for a pending order"
        and tool_name == "modify_pending_order_payment"
    ):
        return native(
            CATEGORY_STATE_CONFLICT,
            "ORDER_PAYMENT_HISTORY_CONFLICT",
            "order_id",
        )

    if low == "invalid reason" and tool_name == "cancel_pending_order":
        return native(CATEGORY_INVALID_REQUEST, "INVALID_CANCELLATION_REASON", "reason")
    if low == "the number of items to be exchanged should match" and tool_name in {
        "exchange_delivered_order_items",
        "modify_pending_order_items",
    }:
        return native(
            CATEGORY_INVALID_REQUEST,
            "ITEM_COUNT_MISMATCH",
            "item_ids",
            "new_item_ids",
        )
    if low == "the number of items to be modified should match" and tool_name == "modify_pending_order_items":
        return native(
            CATEGORY_INVALID_REQUEST,
            "ITEM_COUNT_MISMATCH",
            "item_ids",
            "new_item_ids",
        )
    if low == "number of passengers does not match" and tool_name == "update_reservation_passengers":
        return native(
            CATEGORY_INVALID_REQUEST,
            "PASSENGER_COUNT_MISMATCH",
            "reservation_id",
            "passengers",
        )
    if (
        low == "payment method should be either the original payment method or a gift card"
        and tool_name == "return_delivered_order_items"
    ):
        return native(
            CATEGORY_INVALID_REQUEST,
            "PAYMENT_METHOD_NOT_ALLOWED",
            "order_id",
            "payment_method_id",
        )
    if (
        low == "the new payment method should be different from the current one"
        and tool_name == "modify_pending_order_payment"
    ):
        return native(
            CATEGORY_INVALID_REQUEST,
            "PAYMENT_METHOD_UNCHANGED",
            "order_id",
            "payment_method_id",
        )
    if low == "certificate cannot be used to update reservation" and tool_name in {
        "update_reservation_baggages",
        "update_reservation_flights",
    }:
        return native(
            CATEGORY_STATE_CONFLICT,
            "CERTIFICATE_PAYMENT_NOT_ALLOWED",
            "reservation_id",
            "payment_id",
        )
    if tool_name == "calculate":
        code = (
            "CALCULATION_INVALID_CHARACTERS"
            if low == "invalid characters in expression"
            else "CALCULATION_INVALID_EXPRESSION"
        )
        return native(CATEGORY_INVALID_REQUEST, code, "expression")

    if low in {
        "gift card balance is not enough",
        "insufficient gift card balance to pay for the new item",
        "insufficient gift card balance to pay for the order",
        "insufficient gift card balance to pay for the price difference",
    }:
        if tool_name in _PAYMENT_METHOD_ID_TOOLS:
            return native(
                CATEGORY_CAPACITY_UNAVAILABLE,
                "INSUFFICIENT_PAYMENT_BALANCE",
                "payment_method_id",
            )
        if tool_name in _PAYMENT_ID_TOOLS:
            return native(
                CATEGORY_CAPACITY_UNAVAILABLE,
                "INSUFFICIENT_PAYMENT_BALANCE",
                "payment_id",
            )

    if tool_name in {"book_reservation", "update_reservation_flights"}:
        if re.fullmatch(r"flight .+ not found on date .+", message, flags=re.IGNORECASE):
            return native(
                CATEGORY_RESOURCE_NOT_FOUND,
                "FLIGHT_DATE_NOT_FOUND",
                "flights",
            )
        if re.fullmatch(r"flight .+ not found", message, flags=re.IGNORECASE):
            return native(CATEGORY_RESOURCE_NOT_FOUND, "FLIGHT_NOT_FOUND", "flights")
        if re.fullmatch(r"flight .+ not available on date .+", message, flags=re.IGNORECASE):
            return native(
                CATEGORY_CAPACITY_UNAVAILABLE,
                "FLIGHT_UNAVAILABLE",
                "flights",
            )
        if re.fullmatch(r"not enough seats on flight .+", message, flags=re.IGNORECASE):
            causal_inputs = (
                ("flights", "cabin", "passengers")
                if tool_name == "book_reservation"
                else ("reservation_id", "flights", "cabin")
            )
            return native(
                CATEGORY_CAPACITY_UNAVAILABLE,
                "INSUFFICIENT_SEATS",
                *causal_inputs,
            )
    if tool_name == "book_reservation":
        if re.fullmatch(r"payment method .+ not found", message, flags=re.IGNORECASE):
            return native(
                CATEGORY_RESOURCE_NOT_FOUND,
                "PAYMENT_METHOD_NOT_FOUND",
                "payment_methods",
            )
        if re.fullmatch(r"not enough balance in payment method .+", message, flags=re.IGNORECASE):
            return native(
                CATEGORY_CAPACITY_UNAVAILABLE,
                "INSUFFICIENT_PAYMENT_BALANCE",
                "payment_methods",
            )
        if re.fullmatch(
            r"payment amount does not add up, total price is .+, but paid .+",
            message,
            flags=re.IGNORECASE,
        ):
            return native(
                CATEGORY_INVALID_REQUEST,
                "PAYMENT_AMOUNT_MISMATCH",
                "payment_methods",
                "flights",
                "cabin",
                "passengers",
                "insurance",
                "nonfree_baggages",
            )
    if tool_name in {"exchange_delivered_order_items", "modify_pending_order_items"}:
        if re.fullmatch(r"new item .+ not found", message, flags=re.IGNORECASE):
            return native(
                CATEGORY_RESOURCE_NOT_FOUND,
                "ITEM_NOT_FOUND",
                "new_item_ids",
            )
        if re.fullmatch(r"new item .+ not available", message, flags=re.IGNORECASE):
            return native(
                CATEGORY_CAPACITY_UNAVAILABLE,
                "ITEM_UNAVAILABLE",
                "new_item_ids",
            )

        if re.fullmatch(r"new item .+ not found or available", message, flags=re.IGNORECASE):
            return make_tool_outcome(
                False,
                stage=STAGE_DOMAIN,
                category=CATEGORY_UNKNOWN_FAILURE,
                code="ITEM_NOT_FOUND_OR_UNAVAILABLE",
                basis=BASIS_NATIVE_MESSAGE_REGISTRY,
                retry_hint=RETRY_UNKNOWN,
                causal_inputs=("new_item_ids",),
                raw_type=raw_type,
                raw_message=message,
                source="tau_native",
            )
        if re.fullmatch(r".+ not found", message, flags=re.IGNORECASE):
            return native(CATEGORY_RESOURCE_NOT_FOUND, "ITEM_NOT_FOUND", "item_ids")

    return make_tool_outcome(
        False,
        stage=STAGE_DOMAIN,
        category=CATEGORY_UNKNOWN_FAILURE,
        code="TAU_UNCLASSIFIED_DOMAIN_ERROR",
        basis=BASIS_UNKNOWN,
        retry_hint=RETRY_UNKNOWN,
        causal_inputs=(),
        raw_type=raw_type,
        raw_message=message,
        source="tau_native",
    )


def _err_payload(
    err_type: str,
    msg: str,
    tool_name: str,
    *,
    outcome: dict[str, Any] | None = None,
) -> dict:
    payload = {
        "success": False,
        "content": {
            "error": {"type": err_type, "msg": msg},
            "response": "none",
        },
        "error_msg": f"{err_type}: {msg}",
    }
    if outcome is not None:
        payload["tool_outcome"] = outcome
    return payload


_EXECUTORS: dict[tuple[str, ...], TauNativeExecutor] = {}
_SINGLETON_LOCK = threading.Lock()


def get_tau_executor(state_dir: str | Path | Iterable[str | Path] | None = None) -> TauNativeExecutor:
    require_current_native_modes()
    state_dirs = _resolve_state_dirs(state_dir)
    cache_key = _state_dirs_cache_key(state_dirs) + (
        str(Path(os.environ.get(_TAU_DATA_ROOT_ENV) or _TAU_DATA_ROOT).expanduser().resolve()),
    )
    executor = _EXECUTORS.get(cache_key)
    if executor is not None:
        return executor
    with _SINGLETON_LOCK:
        executor = _EXECUTORS.get(cache_key)
        if executor is None:
            executor = TauNativeExecutor(state_dirs)
            _EXECUTORS[cache_key] = executor
    return executor
