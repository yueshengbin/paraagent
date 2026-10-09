#!/usr/bin/env bash
set -euo pipefail
if (( $# < 1 )); then
  echo "Usage: $0 /path/to/global_step_N/actor [output-directory]" >&2
  exit 2
fi
exec python -m verl.model_merger merge --backend fsdp \
  --local_dir "$1" --target_dir "${2:-outputs/paraagent-rl-hf}"
