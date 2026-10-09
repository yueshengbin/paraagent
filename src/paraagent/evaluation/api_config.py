import json
import os
from pathlib import Path


def load_api_config(path):
    config = json.loads(Path(path).read_text())
    if not isinstance(config, dict) or set(config) != {"model", "base_url", "api_key_env"}:
        raise ValueError("API config must contain only model, base_url and api_key_env")
    if any(not isinstance(config[name], str) or not config[name].strip() for name in config):
        raise ValueError("API config values must be nonempty strings")
    key = os.environ.get(config["api_key_env"])
    if not key:
        raise ValueError(f"Set {config['api_key_env']} for {path}")
    return {**config, "api_key": key}


def public_api_config(config):
    return {name: config[name] for name in ("model", "base_url", "api_key_env")}
