# Data and resources

[ParaAgent SFT](https://huggingface.co/datasets/ShengbinYue/paraagent-sft) and [ParaAgent RL](https://huggingface.co/datasets/ShengbinYue/paraagent-rl) are published. [ToolEnv SFT](https://huggingface.co/datasets/ShengbinYue/ToolEnv-sft) currently provides a dataset card; its data files and remaining runtime downloads are pending. Large resources are distributed separately from Git and the wheel.

| Path | Contents |
|---|---|
| `data/sft/paraagent-sft.jsonl` | ParaAgent SFT trajectories |
| `data/sft/toolenv-sft.jsonl` | ToolEnv simulator examples |
| `data/rl/paraagent-rl/{train,validation}.{jsonl,parquet}` | RL source files |
| `data/rl/paraagent-rl/{state,full_db,policies}/` | Simia task states, databases and policies |
| `data/toolenv/` | Tool catalog, index, caches and dependency graph |

Use matching schemas, indexes, caches and graphs. See [bundle notes](../data/README.md) and [index building](inference.md#build-retrieval-indexes).

## SFT data

Run from the repository root:

```bash
hf download ShengbinYue/paraagent-sft paraagent-sft.jsonl \
  --repo-type dataset --local-dir data/sft
```

`data/dataset_info.json` registers ParaAgent's ShareGPT conversations and ToolEnv's Alpaca examples.

## ToolEnv runtime resources

Place the following files directly in `data/toolenv/` (relative to the repository root). Keep the catalog and its index together; the two response caches must come from the matching prepared companion bundle.

| File | Purpose |
|---|---|
| `toolcorpus_all.tsv` | Retrieval corpus |
| `name_tool.tsv` | Tool-name catalog used by the simulator |
| `tool-corpus_all_index_HNSW64.bin` and `.bin.meta` | Retrieval index and its corpus/index checksums |
| `dependency-relations.jsonl` | Tool-dependency graph |
| `cache_flat.pkl` and `cache_variants.pkl` | Simulator response caches |

The catalog, index and dependency graph are staged for the [ToolEnv resources dataset](https://huggingface.co/datasets/ShengbinYue/ToolEnv-resources), but its public download is pending. The response caches are not in that staged dataset. Use the prepared local companion bundle for all seven files until their distribution is available. Do not mix an index or cache with a different catalog snapshot.

## RL data

Install the Hugging Face CLI, then download and convert:

```bash
python -m pip install pyarrow
hf download ShengbinYue/paraagent-rl \
  --repo-type dataset --local-dir data/rl/paraagent-rl
python scripts/data/prepare_rl.py --data-dir data/rl/paraagent-rl
```

The converter creates both Parquet splits using [`configs/prompts/paraagent.txt`](../configs/prompts/paraagent.txt). Options: `--system-prompt /path/to/prompt.txt` and `--output-dir /path/to/output`. Keep `state/`, `full_db/` and `policies/` with the Parquet files; conversion does not copy them. Prepared Parquet bundles can be used directly.

## Validation

```bash
bash scripts/train/paraagent-rl.sh --check
```

This checks required resources without training. Use serialized caches from trusted sources.
