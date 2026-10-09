"""Check the RL resources required at training startup."""
from pathlib import Path


def verify_runtime_bundle(directory):
    import pyarrow.parquet as pq

    root = Path(directory)
    for split in ("train", "validation"):
        path = root / f"{split}.parquet"
        table = pq.read_table(path, columns=["data_source", "extra_info", "env_kwargs"])
        for row in table.to_pylist():
            if row["env_kwargs"].get("env_type") != "toolenv":
                raise ValueError(f"Unsupported environment in {path}")
            if row["data_source"] == "Simia":
                extra = row["extra_info"]
                if not extra.get("is_tau_task"):
                    raise ValueError("Simia samples must retain the native execution flag.")
                sample_id = extra["sample_id"]
                if not (root / "state" / f"{sample_id}.json").is_file():
                    raise FileNotFoundError(f"Missing initial state for {sample_id}")
    if not (root / "full_db").is_dir():
        raise FileNotFoundError(root / "full_db")
    return True
