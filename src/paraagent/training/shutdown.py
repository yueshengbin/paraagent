"""Release trainer-owned Ray actors at final shutdown."""
import asyncio
from contextlib import contextmanager
import gc
import logging
import sys
import time
import traceback

logger = logging.getLogger(__name__)
SHUTDOWN_TIMEOUT = 60.0

def finalize_process_resources():
    """Run resource finalizers before terminating dedicated training actors.
    Keep shared trackers alive and avoid creating trackers or pools during cleanup.
    """
    gc.collect()
    for name in ("multiprocessing.util", "multiprocess.util"):
        module = sys.modules.get(name)
        if module is not None:
            module._run_finalizers()

def _close_loaded_executors():
    
    flow = sys.modules.get("paraagent.training.rollout.agent_flow")
    if flow is not None:
        with flow._REWARD_EXECUTOR_LOCK:
            executor, flow._REWARD_EXECUTOR = flow._REWARD_EXECUTOR, None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
    reward = sys.modules.get("paraagent.rewards.reward_subproc")
    if reward is not None:
        with reward._POOL_LOCK:
            pool, reward._POOL = reward._POOL, None
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=False)
    simulator = sys.modules.get("paraagent.toolenv.runtime.tools.toolenv_simulator")
    if simulator is not None:
        simulator.MirrorApiClient.shutdown_shared_executor(wait=True, cancel_futures=False)

def _uvicorn_server(task):
    """Resolve the uvicorn.Server owner from its suspended serve task at shutdown.
    Reject unrecognized live tasks.
    """
    coro = task.get_coro()
    while coro is not None:
        frame = getattr(coro, "cr_frame", None)
        owner = frame.f_locals.get("self") if frame is not None else None
        if owner is not None and all(hasattr(owner, key) for key in ("should_exit", "servers", "shutdown")):
            return owner
        coro = getattr(coro, "cr_await", None)
    raise RuntimeError("Cannot locate uvicorn server for graceful terminal shutdown")

async def _close_vllm(actor):
    errors = []
    task = getattr(actor, "_server_task", None)
    engine = getattr(actor, "engine", None)
    http_server = None
    try:
        if task is not None and not task.done():
            http_server = _uvicorn_server(task)
            
            for listener in http_server.servers:
                listener.close()
            http_server.should_exit = True
        if engine is not None:
            drain = getattr(engine, "wait_for_requests_to_drain", None)
            if drain is not None:
                await asyncio.wait_for(drain(), timeout=SHUTDOWN_TIMEOUT / 3)
        if task is not None:
            await asyncio.wait_for(asyncio.shield(task), timeout=SHUTDOWN_TIMEOUT / 3)
    except (Exception, asyncio.CancelledError):
        errors.append(traceback.format_exc())
    finally:

        if engine is not None:
            try:
                engine.shutdown()
            except Exception:
                errors.append(traceback.format_exc())
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if http_server is not None:
                for listener in http_server.servers:
                    listener.close()
        actor.engine = None
        actor._server_task = None
    if errors:
        raise RuntimeError("vLLM terminal shutdown failed:\n" + "\n".join(errors))

def begin_actor_shutdown(actor, role):
    """Sync RPC entrypoint: schedule cleanup on async actors without blocking it."""
    state = getattr(actor, "_paraagent_shutdown_state", None)
    if state is not None:
        return dict(state)
    state = {"status": "running", "error": None}
    actor._paraagent_shutdown_state = state

    async def close():
        errors = []
        try:
            child = getattr(actor, 'reward_loop_worker', None) if role == 'agent' else None
            if child is not None:
                try:
                    await asyncio.to_thread(close_actor_group, [child], 'worker', SHUTDOWN_TIMEOUT / 2)
                except Exception:
                    errors.append(traceback.format_exc())
            if role == "vllm":
                try:
                    await _close_vllm(actor)
                except Exception:
                    errors.append(traceback.format_exc())
            try:
                await asyncio.to_thread(_close_loaded_executors)
            except Exception:
                errors.append(traceback.format_exc())
            _attempt(errors, 'process resources', finalize_process_resources)
            if errors:
                raise RuntimeError('\n'.join(errors))
        except Exception:
            state.update(status="failed", error=traceback.format_exc())
        else:
            state["status"] = "closed"

    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    server_task = getattr(actor, '_server_task', None) if role == 'vllm' else None
    if server_task is None and role == 'vllm':
        server_task = getattr(getattr(actor, 'engine', None), 'output_handler', None)
    loop = server_task.get_loop() if server_task is not None else running_loop
    if loop is None:
        asyncio.run(close())
    elif loop is running_loop:
        actor._paraagent_shutdown_task = loop.create_task(close())
    elif loop.is_running():
        actor._paraagent_shutdown_task = asyncio.run_coroutine_threadsafe(close(), loop)
    else:
        state.update(status='failed', error='vLLM actor event loop is no longer running')
    return dict(state)

def actor_shutdown_status(actor):
    return dict(actor._paraagent_shutdown_state)

def _unique_actors(actors):
    result, seen = [], set()
    for actor in actors:
        if actor is None:
            continue
        key = actor._actor_id.hex()
        if key not in seen:
            result.append(actor)
            seen.add(key)
    return result

def close_actor_group(actors, role, timeout=SHUTDOWN_TIMEOUT):
    """Close all owned actors in parallel, then terminate them; bound each stage."""
    import ray

    actors = _unique_actors(actors)
    errors = []
    deadline = time.monotonic() + timeout
    pending = {}
    for actor in actors:
        try:
            pending[actor.__ray_call__.remote(begin_actor_shutdown, role)] = actor
        except Exception as exc:
            errors.append(f"{role} begin: {exc}")
    try:
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                errors.append(f"{role} shutdown timed out for {len(pending)} actors")
                break
            ready, _ = ray.wait(list(pending), num_returns=1, timeout=min(remaining, 0.2))
            if not ready:
                continue
            running = []
            for ref in ready:
                actor = pending.pop(ref)
                try:
                    state = ray.get(ref)
                    if state['status'] == 'running':
                        running.append(actor)
                    elif state['status'] != 'closed':
                        errors.append(f"{role}: {state['error']}")
                except Exception as exc:
                    errors.append(f"{role} RPC: {exc}")
            if running:
                time.sleep(min(0.05, max(0, deadline - time.monotonic())))
                for actor in running:
                    try:
                        pending[actor.__ray_call__.remote(actor_shutdown_status)] = actor
                    except Exception as exc:
                        errors.append(f"{role} status: {exc}")
    finally:

        for actor in actors:
            try:
                ray.kill(actor, no_restart=True)
            except Exception as exc:
                errors.append(f"{role} terminate: {exc}")
    if errors:
        raise RuntimeError("\n".join(errors))

def _attempt(errors, label, fn):
    try:
        fn()
    except Exception:
        errors.append(f"{label}: {traceback.format_exc()}")

def close_rollout_manager(manager):
    if getattr(manager, '_paraagent_closed', False):
        return
    manager._paraagent_closed = True
    errors = []
    
    agents = list(getattr(manager, 'agent_flow_workers', []))
    _attempt(errors, 'agent workers', lambda: close_actor_group(agents, 'agent'))
    replicas = list(getattr(manager, 'rollout_replicas', []))
    servers = [s for replica in replicas for s in getattr(replica, 'servers', [])]
    backend = manager.config.actor_rollout_ref.rollout.name
    if servers:
        if backend == 'vllm':
            _attempt(errors, 'rollout servers', lambda: close_actor_group(servers, 'vllm'))
        else:

            _attempt(errors, 'rollout servers', lambda: close_actor_group(servers, 'worker'))
    balancer = getattr(manager, 'global_load_balancer', None)
    if balancer is not None:
        _attempt(errors, 'load balancer', lambda: close_actor_group([balancer], 'worker'))
    if errors:
        raise RuntimeError('\n'.join(errors))

def close_trainer(trainer):
    if getattr(trainer, '_paraagent_closed', False):
        return
    trainer._paraagent_closed = True
    errors = []
    manager = getattr(trainer, 'async_rollout_manager', None)
    if manager is not None:
        _attempt(errors, 'rollouts', lambda: close_rollout_manager(manager))
    groups = getattr(trainer, '_shutdown_worker_groups', [])
    actors = [actor for group in groups for actor in group.workers]
    _attempt(errors, 'training workers', lambda: close_actor_group(actors, 'worker'))
    _attempt(errors, 'TaskRunner resources', finalize_process_resources)
    if errors:
        raise RuntimeError('\n'.join(errors))

@contextmanager
def cleanup_after_training(trainer):
    """Always close after init/fit; never mask the original training exception."""
    try:
        yield
    finally:
        failed = sys.exc_info()[0] is not None
        try:
            close_trainer(trainer)
        except Exception:
            if not failed:
                raise
            logger.exception('Terminal cleanup also failed; preserving training exception')
