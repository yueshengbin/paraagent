"""Dispatch rewards through a spawn-based pool; keep child imports lightweight."""

import importlib.util
import hashlib
import multiprocessing
import os
import sys
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from typing import Any, Optional

_POOL: Optional[ProcessPoolExecutor] = None
_POOL_LOCK = threading.Lock()

_CHILD_FN_CACHE: dict[tuple[str, str], Any] = {}

def _load_reward_fn(module_path: str, fn_name: str):
    canonical_path = os.path.realpath(os.path.abspath(module_path))
    key = (canonical_path, fn_name)
    fn = _CHILD_FN_CACHE.get(key)
    if fn is not None:
        return fn
    stem = os.path.basename(canonical_path).removesuffix(".py")
    path_digest = hashlib.sha256(canonical_path.encode("utf-8")).hexdigest()[:16]
    mod_name = f"reward_subproc__{stem}_{path_digest}"
    mod = sys.modules.get(mod_name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(mod_name, canonical_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load reward module from {canonical_path!r}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        try:
            spec.loader.exec_module(mod)

            preload = getattr(mod, "preload_for_subprocess", None)
            if callable(preload):
                preload()
        except BaseException:

            if sys.modules.get(mod_name) is mod:
                sys.modules.pop(mod_name, None)
            raise
    fn = getattr(mod, fn_name)
    _CHILD_FN_CACHE[key] = fn
    return fn

def call_reward_fn(module_path: str, fn_name: str, reward_kwargs: dict, call_kwargs: dict):
    """Top-level (picklable) entry point executed inside pool children."""
    fn = _load_reward_fn(module_path, fn_name)
    
    merged = {**call_kwargs, **(reward_kwargs or {})}
    return fn(**merged)

def _pool_size() -> int:
    try:
        return max(1, int(os.environ.get("TOOL_REWARD_PROC_WORKERS", "6")))
    except ValueError:
        return 6

def _get_pool() -> ProcessPoolExecutor:
    global _POOL
    if _POOL is None:
        with _POOL_LOCK:
            if _POOL is None:

                _POOL = ProcessPoolExecutor(
                    max_workers=_pool_size(),
                    mp_context=multiprocessing.get_context("spawn"),
                )
    return _POOL

def _discard_pool(broken: ProcessPoolExecutor) -> None:
    global _POOL
    with _POOL_LOCK:
        if _POOL is broken:
            _POOL = None
    broken.shutdown(wait=False, cancel_futures=True)

async def run_in_reward_subprocess(
    loop,
    *,
    module_path: str,
    fn_name: str,
    reward_kwargs: dict,
    call_kwargs: dict,
):
    """Await a reward in the process pool; rebuild and retry once if the pool breaks."""
    job = partial(call_reward_fn, module_path, fn_name, reward_kwargs, call_kwargs)
    pool = _get_pool()
    try:
        return await loop.run_in_executor(pool, job)
    except BrokenProcessPool:
        _discard_pool(pool)
        return await loop.run_in_executor(_get_pool(), job)
