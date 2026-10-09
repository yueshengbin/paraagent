#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
export FORCE_TORCHRUN=1
exec llamafactory-cli train configs/train/toolenv-sft.yaml "$@"
