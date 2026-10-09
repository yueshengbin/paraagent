# ParaAgent Data

The training entry points read the files below. [ParaAgent SFT](https://huggingface.co/datasets/ShengbinYue/paraagent-sft) and [ParaAgent RL](https://huggingface.co/datasets/ShengbinYue/paraagent-rl) are published on Hugging Face; see the [download instructions](../docs/data.md#sft-data). [ToolEnv SFT](https://huggingface.co/datasets/ShengbinYue/ToolEnv-sft) currently provides a dataset card; its data files are pending. The remaining runtime resources are present in the prepared local companion bundle, with public download locations pending. Large data files are distributed separately from the source package.

| Stage | Path relative to the repository root | Examples |
|---|---|---:|
| ParaAgent SFT | `data/sft/paraagent-sft.jsonl` | 10,896 |
| ToolEnv SFT | `data/sft/toolenv-sft.jsonl` | 243,445 |
| ParaAgent RL training | `data/rl/paraagent-rl/train.parquet` | 16,384 |
| ParaAgent RL validation | `data/rl/paraagent-rl/validation.parquet` | 124 |

| Source | `paraagent-sft` | `paraagent-rl` |
|---|---:|---:|
| ToolBench | 4,461 | 8,665 |
| Toucan | 4,924 | 4,840 |
| AFM | 1,511 | — |
| Simia | — | 2,879 |
| **Total** | **10,896** | **16,384** |

The RL validation split contains 124 examples: 49 ToolBench, 42 Toucan, and 33 Simia examples.

ToolBench training and validation questions exclude normalized text matches to the 765-task ToolBench evaluation set used in this study. Tool references use the released catalog names.

In `paraagent-sft.jsonl`, each line contains a system prompt and a complete multi-turn conversation. Embedded newlines are escaped. `task_id` is a stable release-local string; filtered records leave gaps in its numeric sequence; `dataset` is `toolbench`, `toucan`, or `afm`. SFT IDs are independent of RL task IDs.

`toolenv-sft` contains 243,445 simulator request–response pairs. Responses are simulated API outputs. The dataset and response caches have unresolved quality issues and are not a factuality-certified record of real API execution.

The RL Parquet fields are `data_source`, `agent_name`, `prompt`, `reward_model`, `extra_info`, and `env_kwargs`. Simia tasks use the `Simia` source label and include native execution metadata. The 2,912 Simia task IDs use `simia_airline_` or `simia_retail_` and match the state filenames. The HF source files `train.jsonl` and `validation.jsonl` are retained beside the generated Parquet files.

The ToolEnv SFT training cutoff is 4,096 tokens. Source records retain complete text; records exceeding the configured training window may be truncated during preprocessing.

Keep each RL bundle together with its `state/`, `full_db/`, and `policies/` directories. Place the matching tool catalog, retrieval index, response caches, and dependency graph in `data/toolenv/`. See [ToolEnv runtime resources](../docs/data.md#toolenv-runtime-resources) for the exact filenames and current availability.

Run `bash scripts/train/paraagent-rl.sh --check` to validate required resources without training. See [training](../docs/training.md) for launch commands.
