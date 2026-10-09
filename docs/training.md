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

### Reward

The reward entry point is `paraagent.rewards.tool_reward.compute_score`.

- Outcome: one Q/F/U judge with a consistency recheck for `110`. Simia uses Q/F with native state, required calls and answer evidence; utility remains diagnostic.
- Process: tool/search coverage and phase scores for grounding, dependencies, layers and execution. The phase score is multiplied by `clip(R_outcome / 1.5, 0, 1)`.
- Penalties: protocol violations, repetition and dependency-order violations, scaled together by `0.25`.

`R_traj = R_outcome + R_step - penalty`; `R_format_score = 1 + R_format`. GDPO normalizes `R_traj`, the gated `R_phase` and `R_format_score` separately, then combines them with weights `1.0`, `0.5` and `0.25`. Format scores and penalties are unconditional. The scalar diagnostic score is `clip(R_format + R_outcome + R_step + R_phase - penalty, -1, 3)`.

Reward hyperparameters use `REWARD_*` names in `configs/toolenv/runtime.yaml`; GDPO weights are in `configs/train/paraagent-rl.yaml`.

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
