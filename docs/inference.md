# ToolEnv services and ParaAct inference

Activate `paraagent`; see [installation](installation.md). Run each service in a separate terminal with its own GPU allocation.

```bash
# ToolEnv simulator, port 12345
CUDA_VISIBLE_DEVICES=0 bash scripts/serve/toolenv.sh /path/to/toolenv-sft

# Policy for ParaAct, port 8000
CUDA_VISIBLE_DEVICES=1 bash scripts/serve/paraagent.sh /path/to/paraagent-rl-hf

# RL answer judge, port 22456; tensor parallelism defaults to 8
TP_SIZE=8 bash scripts/serve/judge.sh Qwen/Qwen3-235B-A22B-Instruct-2507
```

Service scripts accept extra vLLM arguments. Override defaults with `TP_SIZE`, `PORT`, `HOST` and `MAX_MODEL_LEN`. For an external judge, set `TOOL_REWARD_OPENAI_BASE_URL`, `TOOL_REWARD_MODEL` and `TOOL_REWARD_OPENAI_API_KEY`.

Choose the retrieval catalog for the task:

```bash
toolenv-retriever --catalog toolenv --port 30400
toolenv-retriever --catalog toolbench --port 30401
toolenv-retriever --catalog apibank --port 30403
```

Retrieval uses BGE-large-en-v1.5, a matching FAISS index and CPU by default. Options: `--model /path/to/model`, `--device cuda:0`, `--top-k 5`. Endpoints: `/health`, `GET /search`, `POST /search`.

The default is 3 tools. Set `paraact --top-k N` or `TOOL_SEARCH_DEFAULT_TOP_K` in the client process; RL also reads `configs/toolenv/runtime.yaml`. Client requests override the server default.

## Build retrieval indexes

Use the supplied indexes or rebuild:

```bash
python scripts/data/build_retrieval_index.py --catalog toolenv --device cuda:0
python scripts/data/build_retrieval_index.py --catalog toolbench --device cuda:0
python scripts/data/build_retrieval_index.py --catalog apibank --device cuda:0
```

| Catalog | Input corpus | Output index |
|---|---|---|
| ToolEnv | `data/toolenv/toolcorpus_all.tsv` | `tool-corpus_all_index_HNSW64.bin` |
| ToolBench | `data/benchmarks/toolbench/corpus.tsv` | `index.bin` |
| API-Bank | `data/benchmarks/apibank/corpus.json` | `index.bin` |

Indexes are written beside the corpus. Use the same `--model` for building and serving. Options: `--device cpu`, `--batch-size N`, `--output-index /path/to/index.bin`, `--overwrite`. A custom `--corpus` requires `--output-index`.

Stop the retriever before replacing the corpus, index and `.meta` together, then restart it. Rebuilt indexes have new checksums in their `.meta` files.

## ParaAct inference

```bash
paraact --benchmark toolbench --retriever-url http://127.0.0.1:30401   --output outputs/toolbench --limit 1
paraact --benchmark apibank --retriever-url http://127.0.0.1:30403   --output outputs/apibank --limit 1
```

Select the policy with `--base-url`, `--model` and `POLICY_API_KEY`. ToolEnv uses `TOOLENV_BASE_URL` and `TOOLENV_API_KEY`; ToolBench settings are in `configs/eval/toolbench-simulator.yaml`.

API-Bank executes local Python tools. ToolBench uses cached responses and calls ToolEnv on misses. Inference defaults: temperature 0, 2,048 tokens per call (`AGENT_MAX_TOKENS`), trace depth 60 (`--max-depth`). Use [evaluation](evaluation.md) for complete runs and scoring.
