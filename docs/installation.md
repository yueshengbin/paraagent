# Installation

Run from the repository root on Linux x86_64 with Python 3.12 and a CUDA 12.8-compatible NVIDIA driver. RL builds also require the CUDA toolkit (`nvcc`), a C++ compiler and Ninja.

| Environment | Purpose | PyTorch | vLLM |
|---|---|---|---|
| `paraagent` | LLaMA-Factory SFT, serving and retrieval | 2.10.0 | 0.19.0 |
| `paraagent-rl` | verl training and policy rollouts | 2.8.0 | 0.11.0 |

## SFT and serving

```bash
conda create -n paraagent python=3.12 pip -y
conda activate paraagent
python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/sft-serve.txt
python -m pip check
llamafactory-cli version
```

Both SFT stages use the bundled [LLaMA-Factory snapshot](../sft/README.md).

## RL

```bash
mkdir -p third_party
git clone https://github.com/verl-project/verl.git third_party/verl
git -C third_party/verl checkout 6eeb5711babe178ca3cc3caba7c9f158efa9fcd6
git -C third_party/verl apply ../../patches/verl-runtime.patch

conda create -n paraagent-rl python=3.12 pip -y
conda activate paraagent-rl
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/rl.txt
MAX_JOBS=4 python -m pip install --no-build-isolation \
  flash-attn==2.8.1
python -m pip check
python -c 'import torch, vllm, verl; print(torch.__version__, vllm.__version__, verl.__file__)'
```

`verl.__file__` should point inside `third_party/verl`.

### CuMem shutdown patch

Required before RL training with vLLM 0.11.0 / PyTorch 2.8.0. Replace the paths below; the output must be a new directory outside this repository.

```bash
git clone --depth 1 --branch v0.11.0 https://github.com/vllm-project/vllm.git /path/to/vllm-0.11.0
python scripts/setup/prepare_vllm_cumem.py \
  --source /path/to/vllm-0.11.0 \
  --output /path/to/vllm-cumem-fixed \
  --cuda-home /usr/local/cuda
export PYTHONPATH="/path/to/vllm-cumem-fixed${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import vllm, vllm.cumem_allocator; print(vllm.__file__); print(vllm.cumem_allocator.__file__)'
```

Both printed paths must point into the overlay. Keep this `PYTHONPATH` only in the RL environment; rebuild the overlay after dependency changes.

## Next steps

Prepare [data](data.md), then follow [training](training.md) or [runtime services](inference.md).

Key dependencies are pinned; transitive dependencies are resolved by pip.
