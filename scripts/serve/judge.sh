#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
# Pass a checkpoint as the first argument; remaining arguments go to vLLM.
checkpoint="${1:-Qwen/Qwen3-235B-A22B-Instruct-2507}"
if (( $# )); then shift; fi
exec vllm serve "$checkpoint" \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-22456}" \
  --served-model-name Qwen3-235B-A22B-Instruct-2507 \
  --tensor-parallel-size "${TP_SIZE:-8}" \
  --max-model-len "${MAX_MODEL_LEN:-18192}" "$@"
