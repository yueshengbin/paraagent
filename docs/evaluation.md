# Evaluation

Run from the repository root in the `paraagent` environment after [installation](installation.md) and [benchmark data setup](data.md). Full runs also need the [trained/exported checkpoints](training.md); the public data bundle and model weights are not yet complete. Installation creates `paraact` and `paraagent-eval`; without it, use `PYTHONPATH=src python -m paraagent.evaluation.run` and `PYTHONPATH=src python -m paraagent.evaluation.score`. ToolBench has 765 tasks; API-Bank has 50.

## Benchmark data

Download [paraagent-benchmarks.zip](https://drive.google.com/file/d/1meaOTdnT_R-ZESisIQpU60lJq554oFJq/view?usp=drive_link) from Google Drive, then extract it from the repository root:

```bash
unzip paraagent-benchmarks.zip -d data/
```

This provides the ToolBench and API-Bank evaluation resources under `data/benchmarks/toolbench/` and `data/benchmarks/apibank/`.

## Start services

For default ParaAgent + ToolBench evaluation, start these in separate terminals:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/serve/toolenv.sh /path/to/toolenv-sft
CUDA_VISIBLE_DEVICES=1 bash scripts/serve/paraagent.sh /path/to/paraagent-rl-hf
toolenv-retriever --catalog toolbench --port 30401
```

Replace both checkpoint placeholders with existing model directories. ToolEnv serves `toolenv-sft` on port 12345; ToolBench cache misses call it using `configs/eval/toolbench-simulator.yaml`. The policy serves on port 8000. For API-Bank, use `toolenv-retriever --catalog apibank --port 30403` and the policy; its tools run locally. GPT-ReAct uses its inference API instead of the policy service. ToolBench `virtual`, `mirrorapi` and `live` use external tool services instead of the local ToolEnv simulator. See [service details](inference.md).

## API configuration

Inference and ToolBench scoring use separate JSON files with `model`, `base_url` and `api_key_env`: `configs/eval/inference-react.example.json` and `configs/eval/scoring.example.json`. ParaAgent can use `configs/eval/inference-paraagent.example.json` or its default local endpoint. Copy the examples to change providers; keep keys in environment variables. GPT-ReAct requires `--inference-config`, ToolBench scoring requires `--scoring-config`, and API-Bank scoring requires neither.

Fill in the needed keys in `.env`, then load it:

```bash
cp .env.example .env
set -a
source .env
set +a
```

Predictions and scoring summaries record model, base URL and key variable name, never the key. API and retrieval URLs must use HTTP(S), without embedded credentials, query parameters or fragments.

## ToolBench observation backend

ToolBench defaults to a local cache and ToolEnv simulator. To use StableToolBench's GPT/cache `/virtual` service, get a ToolBench key from the [original instructions](https://github.com/THUNLP-MT/StableToolBench#running-the-server-directly) ([application form](https://forms.gle/S4hqVLtnqeXcNTCJA)):

```bash
export TOOLBENCH_SERVICE_URL=http://127.0.0.1:8080/virtual
export TOOLBENCH_KEY=your-toolbench-key
paraact --benchmark toolbench --toolbench-backend virtual \
  --retriever-url http://127.0.0.1:30401 \
  --output outputs/toolbench-virtual --runs 1
```

For the trained [MirrorAPI simulator](https://huggingface.co/stabletoolbench/MirrorAPI), [start its server](https://github.com/THUNLP-MT/StableToolBench#the-mirrorapi-server) with the model and tools, then run:

```bash
export TOOLBENCH_SERVICE_URL=http://127.0.0.1:8080/virtual
paraact --benchmark toolbench --toolbench-backend mirrorapi \
  --retriever-url http://127.0.0.1:30401 \
  --output outputs/toolbench-mirrorapi --runs 1
```

`virtual` and `mirrorapi` share `/virtual` but are recorded separately. For real calls, use `/rapidapi` with `--toolbench-backend live` and `TOOLBENCH_KEY`. External backends do not fall back to local simulation; scoring rejects mixed sources. Validate tool identity before reporting scores.

Connection/read timeouts default to 15/15 seconds for `live` and 15/300 seconds for `virtual` and `mirrorapi`. Override them with `TOOLBENCH_CONNECT_TIMEOUT` and `TOOLBENCH_READ_TIMEOUT` (positive seconds).

## ParaAgent

```bash
paraact --benchmark toolbench --retriever-url http://127.0.0.1:30401 \
  --output outputs/toolbench --runs 1
paraagent-eval --benchmark toolbench --predictions outputs/toolbench --runs run-1 \
  --scoring-config configs/eval/scoring.example.json

paraact --benchmark apibank --retriever-url http://127.0.0.1:30403 \
  --output outputs/apibank --runs 1
paraagent-eval --benchmark apibank --predictions outputs/apibank --runs run-1
```

## GPT-ReAct

```bash
paraact --agent react --paradigm ETE --benchmark toolbench \
  --inference-config configs/eval/inference-react.example.json \
  --retriever-url http://127.0.0.1:30401 --output outputs/react-ETE --runs 1
```

For the EaE baseline, change `--paradigm` to `EaE` and use a new output directory. `ETE` retrieves before execution; `EaE` retrieves during execution. All agents accept `--top-k N` (default: 3).

## Results

- ToolBench: GPT completeness judgment of a final `Finish` answer and tool-hit metrics. `success_per_run` is binary; `judge_pass_rate_per_run` and its per-split breakdown give `Unsure` half credit as in the original pass-rate script.
- API-Bank: one-to-one API-name/input matching; all required calls must match. Outputs and extra calls are ignored.
- Predictions: `run-N/<split>/<query_id>_Agent@1.json`. Use a new output directory when changing experiment settings or benchmark tasks. Resume and scoring reject mismatched results. Scoring requires complete runs; `--limit` is for smoke tests.
- Multiple runs: generate with `--runs 3` and score with `--runs run-1 run-2 run-3`. `pass_at_k` is the fraction solved in any run.
