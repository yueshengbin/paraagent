"""Define the Simia training data and execution contract."""

import os
from pathlib import Path

PROFILE = "paraagent-rl"
BUNDLE_NAME = "paraagent-rl"
MODES = {
    "TAU_TRAINING_PROFILE": PROFILE,
    "TAU_NATIVE_WRITE_SCHEMA_MODE": "strict_v1",
    "TAU_NATIVE_TEMPORAL_MODE": "predeparture_v1",
    "TAU_NATIVE_BUSINESS_MODE": "guard_v2",
    "TAU_NATIVE_TRANSACTION_MODE": "all_atomic_v1",
    "TAU_NATIVE_INVENTORY_MODE": "seats_v1",
    "TAU_NATIVE_ISOLATION_MODE": "rollout_v1",
    "TAU_NATIVE_FEEDBACK_MODE": "guard_v1",
    "TAU_NATIVE_CONNECTION_MODE": "connections_v1",
    "TOOLENV_JUDGE_EXEC_TRACE": "1",
    "TOOLENV_JUDGE_TOOL_DESC": "1",
}
PATH_KEYS = {
    "TAU_DATA_BUNDLE",
    "TOOLENV_TRAIN_FILE",
    "TOOLENV_VAL_FILE",
    "TAU_INITIAL_STATE_DIRS",
    "TAU_FULL_DB_ROOT",
}


def profile(bundle):
    bundle = Path(bundle).resolve()
    return {
        **MODES,
        "TAU_DATA_BUNDLE": str(bundle),
        "TOOLENV_TRAIN_FILE": str(bundle / "train.parquet"),
        "TOOLENV_VAL_FILE": str(bundle / "validation.parquet"),
        "TAU_INITIAL_STATE_DIRS": str(bundle / "state"),
        "TAU_FULL_DB_ROOT": str(bundle / "full_db"),
    }


def check_values(expected, actual, *, allow_missing=False):
    for key, value in expected.items():
        if key not in actual and allow_missing:
            continue
        got = actual.get(key)
        equal = bool(got) and (
            Path(got).resolve() == Path(value).resolve() if key in PATH_KEYS else got == value
        )
        if not equal:
            raise ValueError(f"TAU training profile mismatch: {key}")


def validate_training_inputs(train_files, val_files, overrides=None, environ=None):
    env = os.environ if environ is None else environ
    selected = env.get("TAU_TRAINING_PROFILE")
    if selected != PROFILE:
        raise ValueError("Unknown TAU training profile")
    expected = profile(env["TAU_DATA_BUNDLE"])
    check_values(expected, env)
    for files, key in ((train_files, "TOOLENV_TRAIN_FILE"), (val_files, "TOOLENV_VAL_FILE")):
        values = [files] if isinstance(files, (str, Path)) else list(files)
        if len(values) != 1 or Path(values[0]).resolve() != Path(expected[key]).resolve():
            raise ValueError(f"Hydra data override conflicts with TAU profile: {key}")
        path = Path(values[0])
        if path.is_file():
            _validate_env_kwargs_column(path)
    check_values(expected, overrides or {}, allow_missing=True)
    return expected


def _validate_env_kwargs_column(path: Path) -> None:
    """Fail before Ray starts if any row lacks explicit environment routing."""
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["env_kwargs"])
    bad = []
    for index, value in enumerate(table.column("env_kwargs").to_pylist()):
        if not isinstance(value, dict) or value.get("env_type") != "toolenv":
            bad.append(index)
            if len(bad) == 5:
                break
    if bad:
        raise ValueError(
            f"Invalid env_kwargs in {path}: row indices {bad}; every row must contain a supported env_type"
        )


def validate_worker_mode(config):
    if config.trainer.get("use_legacy_worker_impl", "disable") != "disable":
        raise ValueError("ParaAgent requires trainer.use_legacy_worker_impl=disable")
