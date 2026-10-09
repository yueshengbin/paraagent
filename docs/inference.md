# ToolEnv and policy services

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

Start the ToolEnv retriever:

```bash
toolenv-retriever --catalog toolenv --port 30400
```

Retrieval uses BGE-large-en-v1.5, a matching FAISS index and CPU by default. Options: `--model /path/to/model`, `--device cuda:0`, `--top-k 5`. Endpoints: `/health`, `GET /search`, `POST /search`.

The default is 3 tools. Set `TOOL_SEARCH_DEFAULT_TOP_K` in the client process or `configs/toolenv/runtime.yaml` for RL. Client requests override the server default.

## Build retrieval indexes

Use the supplied indexes or rebuild:

```bash
python scripts/data/build_retrieval_index.py --catalog toolenv --device cuda:0
```

| Catalog | Input corpus | Output index |
|---|---|---|
| ToolEnv | `data/toolenv/toolcorpus_all.tsv` | `tool-corpus_all_index_HNSW64.bin` |

Indexes are written beside the corpus. Use the same `--model` for building and serving. Options: `--device cpu`, `--batch-size N`, `--output-index /path/to/index.bin`, `--overwrite`. A custom `--corpus` requires `--output-index`.

Stop the retriever before replacing the corpus, index and `.meta` together, then restart it. Rebuilt indexes have new checksums in their `.meta` files.
