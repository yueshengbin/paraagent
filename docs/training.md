# Training

Follow [installation](installation.md) and [data preparation](data.md), then run ToolEnv SFT → ParaAgent SFT → ParaAgent RL from the repository root.

## Supervised fine-tuning

```bash
conda activate paraagent
bash scripts/train/toolenv-sft.sh
bash scripts/train/paraagent-sft.sh
```

Both use full-parameter SFT with assistant-only loss across all turns and no packing. Settings are in `configs/train/`; adjust batch size and DeepSpeed settings for your GPUs.

Override a setting with `bash scripts/train/paraagent-sft.sh model_name_or_path=/path/to/Qwen3-4B`. Outputs: `outputs/toolenv-sft` and `outputs/paraagent-sft`.

## Reinforcement learning

Prepare the [RL Parquet bundle](data.md#rl-data). Start the ToolEnv simulator, `toolenv` retriever and answer judge using the [service commands](inference.md). Set endpoints in `configs/toolenv/runtime.yaml` or environment variables; supply credentials through environment variables.

```bash
conda activate paraagent-rl
bash scripts/train/paraagent-rl.sh --check
bash scripts/train/paraagent-rl.sh
```

`configs/train/paraagent-rl.yaml` defaults to eight H200 GPUs, batch 256, two epochs, eight rollouts per prompt and 15 turns: 128 outer updates. Validation is disabled. Adjust batch divisibility and memory settings when changing GPU count.

### Launch overrides

Use Hydra overrides:

```bash
bash scripts/train/paraagent-rl.sh actor_rollout_ref.model.path=/path/to/paraagent-sft
```

Checkpoints: `outputs/paraagent-rl`. Metrics: `outputs/logs/paraagent/`. To preserve logs across resumed jobs, set a new `VERL_FILE_LOGGER_PATH` per invocation; existing log files are overwritten.

Export a checkpoint for serving (default output: `outputs/paraagent-rl-hf`):

```bash
bash scripts/train/export-paraagent-rl.sh outputs/paraagent-rl/global_step_N/actor
```
