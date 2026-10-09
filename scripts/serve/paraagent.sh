#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
# Pass a checkpoint as the first argument; remaining arguments go to vLLM.
checkpoint="${1:-outputs/paraagent-rl-hf}"
if (( $# )); then shift; fi
exec vllm serve "$checkpoint" \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" \
  --served-model-name paraagent-rl \
  --tensor-parallel-size "${TP_SIZE:-1}" \
  --max-model-len "${MAX_MODEL_LEN:-18048}" "$@"
