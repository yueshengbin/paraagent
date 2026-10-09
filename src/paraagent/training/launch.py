"""Configure portable services and launch the ParaAgent RL trainer."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

import yaml


def prepare_environment(root):
    from paraagent.training.data_profile import profile
    from paraagent.data.validate import verify_runtime_bundle

    root = Path(root).resolve()
    bundle = root / "data/rl/paraagent-rl"
    verify_runtime_bundle(bundle)
    defaults = yaml.safe_load((root / "configs/toolenv/runtime.yaml").read_text())["environment"]
    env = dict(os.environ)
    for key, value in defaults.items():
        env.setdefault(key, str(value))
    try:
        if int(env.get("TOOL_SEARCH_DEFAULT_TOP_K", "3")) < 1:
            raise ValueError
    except ValueError as exc:
        raise ValueError("TOOL_SEARCH_DEFAULT_TOP_K must be a positive integer") from exc
    env.update(profile(bundle))
    paths = {
        "TOOL_NAME_API_TSV": "data/toolenv/name_tool.tsv",
        "TOOL_DEP_UNIFIED_PATH": "data/toolenv/dependency-relations.jsonl",
        "MIRRORAPI_CACHE_PKL": "data/toolenv/cache_flat.pkl",
        "MIRRORAPI_VARIANTS_PKL": "data/toolenv/cache_variants.pkl",
        "MIRRORAPI_CONFIG_PATH": "configs/toolenv/simulator.yaml",
    }
    for key, rel in paths.items():
        path = Path(env.get(key, str(root / rel)))
        if not path.is_file():
            raise FileNotFoundError(path)
        env[key] = str(path.resolve())
    env.setdefault("TOOL_REWARD_OPENAI_API_KEY", "EMPTY")
    env.setdefault("TOOLENV_API_KEY", "EMPTY")
    env.setdefault("VERL_FILE_LOGGER_ROOT", str(root / "outputs/logs"))
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--check", action="store_true", help="Check paths and data without starting training."
    )
    args, overrides = parser.parse_known_args()
    root = args.root.resolve()
    env = prepare_environment(root)
    if args.check:
        print("ParaAgent RL data and resource paths are valid. No training or services started.")
        return
    cmd = [
        sys.executable,
        "-m",
        "paraagent.training.main",
        "--config-path",
        str(root / "configs/train"),
        "--config-name",
        "paraagent-rl",
        *overrides,
    ]
    raise SystemExit(subprocess.call(cmd, cwd=root, env=env))


if __name__ == "__main__":
    main()
