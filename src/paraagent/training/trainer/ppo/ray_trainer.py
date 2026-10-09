# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import math
import os
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from functools import reduce
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from paraagent.training.trainer.ppo.metric_utils import compute_data_metrics
from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor
from verl.single_controller.ray import RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    apply_kl_penalty,
    compute_response_mask,
)

try:
    from verl.trainer.ppo.reward import compute_reward_async
except ImportError:
    compute_reward_async = None
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import Role
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip


def get_valid_data(data: DataProto) -> tuple[DataProto, torch.Tensor]:
    """Extract valid (non-padded) data from a DataProto object.

    Args:
        data (DataProto): The data potentially containing padded samples.

    Returns:
        tuple[DataProto, torch.Tensor]: A tuple containing the valid data and a boolean mask
            of valid indices.
    """
    sample_mask = data.batch.get("sample_mask", None)
    if sample_mask is not None:
        valid_mask = sample_mask.to(dtype=torch.bool)
        valid_data = data.select_idxs(valid_mask)
        return valid_data, valid_mask

    valid_mask = torch.ones(len(data), dtype=torch.bool, device=data.batch.device)
    valid_data = data
    return valid_data, valid_mask


def _refresh_global_token_num(data: DataProto) -> None:
    """Recompute global_token_num after changing batch rows or lengths.
    Throughput and MFU metrics use this count.
    """
    if "attention_mask" in data.batch.keys():
        data.meta_info["global_token_num"] = torch.sum(data.batch["attention_mask"], dim=-1).tolist()


def _filter_extra_infos_by_mask(extra_infos: dict, valid_mask: torch.Tensor) -> dict:
    """Remove padded rows from per-sample reward metadata.
    Pass through scalars and values whose lengths do not match the mask.
    """
    mask = valid_mask.detach().cpu().numpy().astype(bool)
    filtered = {}
    for key, values in extra_infos.items():
        try:
            if len(values) == len(mask):
                arr = np.asarray(values, dtype=object)
                filtered[key] = arr[mask].tolist()
            else:
                filtered[key] = values
        except TypeError:
            filtered[key] = values
    return filtered


def _apply_judge_validity_mask(
    data: DataProto,
    reward_extra_infos: dict,
) -> tuple[int, int]:
    """Mask technical judge failures while preserving the padding mask for audit output.
    Valid judge results and deterministic bypasses remain active.
    Return (newly_masked_rows, non_padding_rows).
    """
    values = reward_extra_infos.get("judge_valid_or_bypass")
    if values is None:
        return 0, int(len(data))
    if len(values) != len(data):
        raise ValueError(
            "judge_valid_or_bypass must align with the padded reward batch: "
            f"got {len(values)} values for {len(data)} rows"
        )

    def _coerce(value) -> bool:

        if value is None:
            return True
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "y"}:
                return True
            if normalized in {"0", "false", "no", "n"}:
                return False
        raise ValueError(f"Invalid judge_valid_or_bypass value: {value!r}")

    device = data.batch["prompts"].device
    padding_mask = data.batch.get("padding_sample_mask", None)
    if padding_mask is None:
        existing_mask = data.batch.get("sample_mask", None)
        if existing_mask is None:
            padding_mask = torch.ones(len(data), dtype=torch.bool, device=device)
        else:
            padding_mask = existing_mask.to(device=device, dtype=torch.bool).clone()
        data.batch["padding_sample_mask"] = padding_mask
    else:
        padding_mask = padding_mask.to(device=device, dtype=torch.bool)

    judge_mask = torch.tensor(
        [_coerce(value) for value in values],
        dtype=torch.bool,
        device=device,
    )
    combined_mask = padding_mask & judge_mask
    data.batch["sample_mask"] = combined_mask
    if "response_mask" in data.batch:
        data.batch["response_mask"][~combined_mask] = 0

    newly_masked = int((padding_mask & ~judge_mask).sum().item())
    non_padding = int(padding_mask.sum().item())
    return newly_masked, non_padding


def assign_global_mini_batch_ids(batch: DataProto, mini_batch_size: int, dp_size: int) -> None:
    """Assign global PPO mini-batch ids while preserving the existing DP dispatch layout."""
    if dp_size <= 0:
        raise ValueError(f"dp_size must be positive, got {dp_size}")
    if mini_batch_size % dp_size != 0:
        raise ValueError(f"mini_batch_size ({mini_batch_size}) must be divisible by dp_size ({dp_size})")

    batch_size = len(batch)
    if batch_size % dp_size != 0:
        raise ValueError(f"batch_size ({batch_size}) must be divisible by dp_size ({dp_size})")

    local_batch_size = batch_size // dp_size
    local_mini_batch_size = mini_batch_size // dp_size
    if local_mini_batch_size == 0:
        raise ValueError(f"local_mini_batch_size must be positive, got {local_mini_batch_size}")

    num_mini_batches = math.ceil(local_batch_size / local_mini_batch_size)
    device = batch.batch["prompts"].device
    local_ids = torch.arange(num_mini_batches, dtype=torch.long, device=device).repeat_interleave(
        local_mini_batch_size
    )
    local_ids = local_ids[:local_batch_size]
    mini_batch_ids = local_ids.repeat(dp_size)
    batch.batch["mini_batch_id"] = mini_batch_ids

    response_mask = batch.batch.get("response_mask", None)
    if response_mask is None:
        seq_mask = torch.ones(batch_size, dtype=torch.long, device=device)
    else:
        seq_mask = response_mask.any(dim=-1).to(dtype=torch.long)
    mini_batch_sizes = torch.zeros(num_mini_batches, dtype=torch.long, device=device)
    mini_batch_sizes.scatter_add_(0, mini_batch_ids, seq_mask)
    batch.batch["mini_batch_global_size"] = mini_batch_sizes[mini_batch_ids]

    token_counts = batch.batch["attention_mask"].sum(dim=-1).to(dtype=torch.long)
    mini_batch_token_nums = torch.zeros((batch_size, mini_batch_size), dtype=torch.long, device=device)
    for mini_batch_id in range(num_mini_batches):
        indices = torch.nonzero(mini_batch_ids == mini_batch_id, as_tuple=False).flatten()
        token_nums = token_counts[indices]
        mini_batch_token_nums[indices, : token_nums.numel()] = token_nums.unsqueeze(0).expand(
            indices.numel(), -1
        )
    batch.batch["mini_batch_global_token_num"] = mini_batch_token_nums


def make_json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return make_json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {key: make_json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [make_json_safe(item) for item in value]
    return value


def build_trajectory_dump_entries(
    *,
    inputs,
    outputs,
    gts,
    scores,
    reward_extra_infos_dict,
    trajectory_uids,
    step_indices,
    global_step,
):
    n = len(inputs)
    if not all(len(values) == n for values in (outputs, gts, scores, trajectory_uids, step_indices)):
        raise ValueError(
            "inputs, outputs, gts, scores, trajectory_uids, and step_indices must have the same length"
        )

    aligned_reward_infos = {
        key: values for key, values in reward_extra_infos_dict.items() if len(values) == n
    }

    grouped_steps = {}
    ordered_uids = []

    for idx in range(n):
        trajectory_uid = trajectory_uids[idx]
        if trajectory_uid not in grouped_steps:
            grouped_steps[trajectory_uid] = []
            ordered_uids.append(trajectory_uid)

        step_entry = {
            "step_index": step_indices[idx],
            "input": inputs[idx],
            "output": outputs[idx],
            "gts": gts[idx],
            "score": scores[idx],
        }
        for key, values in aligned_reward_infos.items():
            step_entry[key] = values[idx]

        grouped_steps[trajectory_uid].append(step_entry)

    entries = []
    for trajectory_uid in ordered_uids:
        steps = sorted(grouped_steps[trajectory_uid], key=lambda item: item["step_index"])
        first_step = steps[0]
        last_step = steps[-1]
        entry = {
            "trajectory_uid": trajectory_uid,
            "input": first_step["input"],
            "output": last_step["output"],
            "gts": first_step["gts"],
            "score": sum(step["score"] for step in steps),
            "step": global_step,
            "num_steps": len(steps),
            "steps": steps,
        }
        entries.append(make_json_safe(entry))

    return entries


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Add advantages and returns to the batch using the configured estimator.

    Gamma and lam control GAE; GRPO may normalize by group standard deviation.
    """

    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    advantages = torch.zeros_like(data.batch["token_level_rewards"])
    returns = torch.zeros_like(data.batch["token_level_rewards"])

    valid_data, valid_mask = get_valid_data(data)

    if len(valid_data) == 0:
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        return data

    if adv_estimator == AdvantageEstimator.GAE:
        from paraagent.training.trainer.ppo.core_algos import compute_gae_advantage_return

        valid_advantages, valid_returns = compute_gae_advantage_return(
            token_level_rewards=valid_data.batch["token_level_rewards"],
            values=valid_data.batch["values"],
            response_mask=valid_data.batch["response_mask"],
            trajectory_uids=valid_data.non_tensor_batch["trajectory_uids"],
            step_indices=valid_data.non_tensor_batch["step_indices"],
            gamma=gamma,
            lam=lam,
        )
        advantages[valid_mask] = valid_advantages
        returns[valid_mask] = valid_returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        from paraagent.training.trainer.ppo.core_algos import compute_grpo_outcome_advantage

        valid_advantages, valid_returns = compute_grpo_outcome_advantage(
            token_level_rewards=valid_data.batch["token_level_rewards"],
            response_mask=valid_data.batch["response_mask"],
            index=valid_data.non_tensor_batch["uid"],
            trajectory_uids=valid_data.non_tensor_batch["trajectory_uids"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        advantages[valid_mask] = valid_advantages
        returns[valid_mask] = valid_returns
    elif adv_estimator in (AdvantageEstimator.GDPO, "gdpo"):
        from paraagent.training.trainer.ppo.core_algos import compute_gdpo_outcome_advantage

        gdpo_diagnostics: dict[str, torch.Tensor] = {}
        valid_advantages, valid_returns = compute_gdpo_outcome_advantage(
            token_level_rewards=valid_data.batch["token_level_rewards"],
            response_mask=valid_data.batch["response_mask"],
            index=valid_data.non_tensor_batch["uid"],
            trajectory_uids=valid_data.non_tensor_batch["trajectory_uids"],
            config=config,
            non_tensor_batch=valid_data.non_tensor_batch,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            diagnostics=gdpo_diagnostics,
        )
        advantages[valid_mask] = valid_advantages
        returns[valid_mask] = valid_returns

        valid_mask_np = valid_mask.detach().cpu().numpy().astype(bool)
        for name, valid_values in gdpo_diagnostics.items():
            row_values = np.full(advantages.shape[0], np.nan, dtype=np.float32)
            row_values[valid_mask_np] = valid_values.detach().float().cpu().numpy()
            data.non_tensor_batch[f"gdpo_{name}"] = row_values

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


class RayAgentTrainer(RayPPOTrainer):
    """Coordinate distributed actor rollouts, critic training, and reward computation."""

    def __init__(self, *args, **kwargs):

        from paraagent.training.data_profile import validate_worker_mode

        config = kwargs.get("config", args[0] if args else None)
        validate_worker_mode(config)
        self.reward_fn = kwargs.pop("reward_fn", None)
        self.val_reward_fn = kwargs.pop("val_reward_fn", None)
        super().__init__(*args, **kwargs)
        self.use_reward_loop = True

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        batch.meta_info["temperature"] = rollout_config.temperature
        from verl.utils import tensordict_utils as tu
        from verl.utils.py_functional import rename_dict
        from verl.workers.utils.padding import left_right_2_no_padding

        calculate_entropy = self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
        ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")
        assign_global_mini_batch_ids(batch, mini_batch_size=ppo_mini_batch_size, dp_size=dp_size)
        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
        seed = self.config.actor_rollout_ref.actor.data_loader_seed
        shuffle = self.config.actor_rollout_ref.actor.shuffle
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=calculate_entropy,
            mini_batch_size=ppo_mini_batch_size,
            epochs=ppo_epochs,
            seed=seed,
            dataloader_kwargs={"shuffle": shuffle},
        )

        actor_output = self.actor_rollout_wg.update_actor(batch_td)
        actor_output = tu.get(actor_output, "metrics")
        actor_output = rename_dict(actor_output, "actor/")
        actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
        actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
        return actor_output

    def _sync_rollout_weights(self, global_steps: int) -> None:
        checkpoint_manager = getattr(self, "checkpoint_manager", None)
        if checkpoint_manager is not None:
            checkpoint_manager.update_weights(global_steps)
            return

        ray.get(self.actor_rollout_wg.update_weights(global_steps))

    def _sleep_hybrid_rollout_replicas(self) -> None:
        checkpoint_manager = getattr(self, "checkpoint_manager", None)
        if checkpoint_manager is not None:
            checkpoint_manager.sleep_replicas()
            return

        rollout_config = self.config.actor_rollout_ref.rollout
        if rollout_config.free_cache_engine and self.config.actor_rollout_ref.get("hybrid_engine", False):
            self.async_rollout_manager.sleep()

    def _update_critic(self, batch: DataProto) -> DataProto:
        from verl.utils import tensordict_utils as tu
        from verl.utils.py_functional import rename_dict
        from verl.workers.utils.padding import left_right_2_no_padding

        ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
        ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        dp_size = self._get_worker_group_dp_size(self.critic_wg, ("train", "critic"))
        assign_global_mini_batch_ids(batch, mini_batch_size=ppo_mini_batch_size, dp_size=dp_size)
        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        ppo_epochs = self.config.critic.ppo_epochs
        seed = self.config.critic.data_loader_seed
        shuffle = self.config.critic.shuffle
        tu.assign_non_tensor(
            batch_td,
            mini_batch_size=ppo_mini_batch_size,
            epochs=ppo_epochs,
            seed=seed,
            dataloader_kwargs={"shuffle": shuffle},
        )

        output = self.critic_wg.train_mini_batch(batch_td)
        output = output.get()
        output = tu.get(output, "metrics")
        output = rename_dict(output, "critic/")
        output["perf/mfu/critic"] = output.pop("critic/mfu")
        output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        return output

    def _get_worker_group_dp_size(self, worker_group, roles: Sequence[str]) -> int:
        """Return DP size for the first registered mesh role, falling back to world size."""
        for role in roles:
            try:
                return self._get_dp_size(worker_group, role)
            except (AssertionError, KeyError, ValueError):
                continue
        return worker_group.world_size

    def _dump_generations(
        self,
        inputs,
        outputs,
        gts,
        scores,
        reward_extra_infos_dict,
        dump_path,
        trajectory_uids,
        step_indices,
    ):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        entries = build_trajectory_dump_entries(
            inputs=inputs,
            outputs=outputs,
            gts=gts,
            scores=scores,
            reward_extra_infos_dict=reward_extra_infos_dict,
            trajectory_uids=trajectory_uids,
            step_indices=step_indices,
            global_step=self.global_steps,
        )
        lines = [json.dumps(entry, ensure_ascii=False) for entry in entries]

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _compute_or_extract_reward(self, batch: DataProto, reward_fn, return_dict: bool = False):
        """Extract precomputed rm_scores from rollout, or invoke reward_fn if absent."""
        if "rm_scores" in batch.batch.keys():
            reward_tensor, reward_extra_infos_dict = extract_reward(batch)
            if return_dict:
                return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_infos_dict}
            return reward_tensor, reward_extra_infos_dict

        if reward_fn is None:
            raise ValueError("reward_fn is required when batch does not contain rm_scores.")

        result = reward_fn(batch, return_dict=return_dict)
        if return_dict:
            if isinstance(result, dict):
                return result
            return {"reward_tensor": result, "reward_extra_info": {}}
        if isinstance(result, dict):
            return result["reward_tensor"], result.get("reward_extra_info", {})
        return result, {}

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch
            ]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_to_dump.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
                trajectory_uids=batch.non_tensor_batch["trajectory_uids"],
                step_indices=batch.non_tensor_batch["step_indices"],
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])

        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[:generations_to_log]

        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_uids = []
        dump_inputs = []
        dump_outputs = []
        dump_gts = []
        dump_scores = []
        dump_trajectory_uids = []
        dump_step_indices = []
        dump_reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            if (
                self.config.reward_model.enable
                and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model"
            ):
                return {}

            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            test_output_gen_batch = self.async_rollout_manager.generate_sequences(test_gen_batch)

            print("validation generation end")

            test_output_gen_batch.meta_info["validate"] = True

            result = self._compute_or_extract_reward(
                test_output_gen_batch, reward_fn=self.val_reward_fn, return_dict=True
            )
            reward_tensor = result["reward_tensor"]
            step_scores = reward_tensor.sum(-1).detach().cpu().tolist()
            reward_extra_info = result.get("reward_extra_info", {})
            step_inputs = self.tokenizer.batch_decode(
                test_output_gen_batch.batch["input_ids"], skip_special_tokens=True
            )
            step_outputs = self.tokenizer.batch_decode(
                test_output_gen_batch.batch["responses"], skip_special_tokens=True
            )

            if "num_steps" in test_output_gen_batch.meta_info:
                num_steps = test_output_gen_batch.meta_info.pop("num_steps")
            else:
                num_steps = [1] * len(test_output_gen_batch)

            dump_inputs.extend(step_inputs)
            dump_outputs.extend(step_outputs)
            dump_scores.extend(step_scores)
            for key, values in reward_extra_info.items():
                dump_reward_extra_infos_dict[key].extend(make_json_safe(values))

            start = 0
            batch_traj_scores = []
            batch_traj_inputs = []
            batch_traj_outputs = []
            batch_traj_extra_info = defaultdict(list)
            for trajectory_uid, ground_truth, n in zip(
                test_batch.non_tensor_batch["uid"], ground_truths, num_steps, strict=True
            ):
                dump_trajectory_uids.extend([trajectory_uid] * n)
                dump_step_indices.extend(range(n))
                dump_gts.extend([ground_truth] * n)

                traj_score = sum(step_scores[start : start + n])
                batch_traj_scores.append(traj_score)

                last_step_idx_in_traj = start + n - 1

                for key, values in reward_extra_info.items():
                    batch_traj_extra_info[key].append(make_json_safe(values[last_step_idx_in_traj]))

                input_ids = test_output_gen_batch.batch["input_ids"][start]
                input_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
                batch_traj_inputs.append(input_text)

                output_ids = test_output_gen_batch.batch["responses"][last_step_idx_in_traj]
                output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                batch_traj_outputs.append(output_text)

                start += n

            sample_scores.extend(batch_traj_scores)
            sample_inputs.extend(batch_traj_inputs)
            sample_outputs.extend(batch_traj_outputs)

            reward_extra_infos_dict["reward"].extend(batch_traj_scores)
            if "reward_extra_info" in result:
                for key, vals in batch_traj_extra_info.items():
                    reward_extra_infos_dict[key].extend(make_json_safe(vals))

            data_source_lst.append(
                test_batch.non_tensor_batch.get("data_source", ["unknown"] * len(test_batch))
            )

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=dump_inputs,
                outputs=dump_outputs,
                gts=dump_gts,
                scores=dump_scores,
                reward_extra_infos_dict=dump_reward_extra_infos_dict,
                dump_path=val_data_dir,
                trajectory_uids=dump_trajectory_uids,
                step_indices=dump_step_indices,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), (
                f"{key_info}: {len(lst)=}, {len(sample_scores)=}"
            )

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(
            data_sources, sample_uids, reward_extra_infos_dict
        )
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = make_json_safe(metric_val)

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {
            pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()
        }

        actor_role = (
            Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        )
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            from verl.workers.config.engine import FSDPEngineConfig
            from verl.workers.engine_workers import TrainingWorkerConfig

            orig_critic_cfg = critic_cfg
            if orig_critic_cfg.strategy == "fsdp":
                engine_config: FSDPEngineConfig = orig_critic_cfg.model.fsdp_config
                engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
                engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu
            else:
                raise NotImplementedError(f"Unknown strategy {orig_critic_cfg.strategy=}")

            critic_cfg = TrainingWorkerConfig(
                model_type="value_model",
                model_config=orig_critic_cfg.model_config,
                engine_config=engine_config,
                optimizer_config=orig_critic_cfg.optim,
                checkpoint_config=orig_critic_cfg.checkpoint,
            )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        all_wg = {}

        self._shutdown_worker_groups = []
        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = (
                self.config.trainer.ray_wait_register_center_timeout
            )
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")

            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(
                        self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options"
                    )
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(
                        self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options"
                    )
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            self._shutdown_worker_groups.append(wg_dict)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.reset()

            from functools import partial

            from paraagent.training.workers.utils.losses import value_loss

            value_loss_ = partial(value_loss, config=orig_critic_cfg)
            self.critic_wg.set_loss_fn(value_loss_)

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        self.async_rollout_mode = True

        from paraagent.training.rollout import AgentFlowManager

        if self.config.reward_model.enable:
            rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
        else:
            rm_resource_pool = None

        self.async_rollout_manager = AgentFlowManager(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rm_resource_pool=rm_resource_pool,
        )
        self.checkpoint_manager = None
        checkpoint_engine_config = omega_conf_to_dataclass(
            self.config.actor_rollout_ref.rollout.checkpoint_engine
        )
        checkpoint_manager_class_fqn = self.config.actor_rollout_ref.rollout.get("checkpoint_manager_class")
        if checkpoint_manager_class_fqn:
            CheckpointEngineManager = load_class_from_fqn(
                checkpoint_manager_class_fqn, "CheckpointEngineManager"
            )
        else:
            from verl.checkpoint_engine import CheckpointEngineManager

        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            trainer=self.actor_rollout_wg,
            replicas=self.async_rollout_manager.rollout_replicas,
        )

        self.checkpoint_manager.sleep_replicas()

    def fit(self):
        """Run PPO training through worker RPCs and compute advantages on the driver."""
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        self._load_checkpoint()

        self._sync_rollout_weights(self.global_steps)

        current_epoch = self.global_steps // len(self.train_dataloader)

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=make_json_safe(val_metrics), step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        progress_bar = tqdm(
            total=self.total_training_steps, initial=self.global_steps, desc="Training Progress"
        )

        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    with marked_timer("gen", timing_raw, color="red"):
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                        self._sleep_hybrid_rollout_replicas()

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        raise NotImplementedError(
                            "REMAX advantage estimation is not supported for agent flow."
                        )

                    batch = batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                    )
                    num_steps = gen_batch_output.meta_info.pop("num_steps")
                    batch = batch.sample_level_repeat(num_steps)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    batch = self._pad_dataproto_to_world_size(batch)

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    _refresh_global_token_num(batch)

                    with marked_timer("reward", timing_raw, color="yellow"):
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                            reward_tensor = self.reward_loop_manager.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.get("launch_reward_fn_async", False):
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, return_dict=False
                            )

                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get(
                        "bypass_mode", False
                    )
                    if bypass_recomputing_logprobs:
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.get("launch_reward_fn_async", False):
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )
                            judge_masked, non_padding = _apply_judge_validity_mask(
                                batch, reward_extra_infos_dict
                            )
                            metrics["reward/judge_invalid_masked"] = judge_masked
                            metrics["reward/judge_invalid_fraction"] = (
                                judge_masked / non_padding if non_padding else 0.0
                            )

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import (
                                compute_rollout_correction_and_add_to_batch,
                            )

                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(
                                batch, rollout_corr_config
                            )

                            metrics.update(is_metrics)

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            response_mask = batch.batch["response_mask"]

                            value_mask = torch.zeros_like(response_mask)
                            value_mask[:, 0] = 1
                            sample_mask = batch.batch.get("sample_mask", None)
                            if sample_mask is not None:
                                value_mask[~sample_mask.to(dtype=torch.bool)] = 0
                            batch.batch["response_mask"] = value_mask

                            critic_output = self._update_critic(batch)

                            batch.batch["response_mask"] = response_mask
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ):
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()

                        with marked_timer("update_weights", timing_raw, color="red"):
                            self._sync_rollout_weights(self.global_steps)

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        valid_mask = batch.batch.get("padding_sample_mask", None)
                        if valid_mask is None:
                            valid_batch, valid_mask = get_valid_data(batch)
                        else:
                            valid_mask = valid_mask.to(dtype=torch.bool)
                            valid_batch = batch.select_idxs(valid_mask)
                        valid_reward_extra_infos = _filter_extra_infos_by_mask(
                            reward_extra_infos_dict, valid_mask
                        )
                        self._log_rollout_data(
                            valid_batch, valid_reward_extra_infos, timing_raw, rollout_data_dir
                        )

                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )

                valid_batch, _ = get_valid_data(batch)

                metrics.update(compute_data_metrics(batch=valid_batch, use_critic=self.use_critic))
                gdpo_reward_keys = self.config.algorithm.get("gdpo_reward_keys", None)
                if gdpo_reward_keys and self.config.algorithm.adv_estimator in (
                    AdvantageEstimator.GDPO,
                    "gdpo",
                ):
                    from paraagent.training.trainer.ppo.core_algos import summarize_gdpo_trajectory_advantages

                    for key in gdpo_reward_keys:
                        if key in valid_batch.non_tensor_batch:
                            vals = np.asarray(
                                [
                                    0.0 if value is None else value
                                    for value in valid_batch.non_tensor_batch[key]
                                ],
                                dtype=np.float32,
                            )
                            metrics[f"gdpo/{key}/mean"] = float(np.mean(vals))
                            metrics[f"gdpo/{key}/std"] = float(np.std(vals))
                            metrics[f"gdpo/{key}/max"] = float(np.max(vals))
                            metrics[f"gdpo/{key}/min"] = float(np.min(vals))
                    component_diagnostics = [f"component_advantage/{key}" for key in gdpo_reward_keys]
                    for diagnostic_name in (
                        "pre_bn_advantage",
                        "final_advantage",
                        "advantage_gate",
                        *component_diagnostics,
                    ):
                        batch_key = f"gdpo_{diagnostic_name}"
                        if batch_key not in valid_batch.non_tensor_batch:
                            continue
                        summary = summarize_gdpo_trajectory_advantages(
                            valid_batch.non_tensor_batch[batch_key],
                            valid_batch.non_tensor_batch["trajectory_uids"],
                        )
                        metrics.update(
                            {f"gdpo/{diagnostic_name}/{stat}": value for stat, value in summary.items()}
                        )
                metrics.update(compute_timing_metrics(batch=valid_batch, timing_raw=timing_raw))

                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(
                    compute_throughout_metrics(batch=valid_batch, timing_raw=timing_raw, n_gpus=n_gpus)
                )

                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=valid_batch)

                logger.log(data=make_json_safe(metrics), step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                if hasattr(self.train_dataset, "on_batch_end"):
                    self.train_dataset.on_batch_end(batch=valid_batch)

    def _pad_dataproto_to_world_size(self, batch):
        dp_sizes = []
        if self.use_critic and self.critic_wg.world_size != 0:
            critic_roles = ("train", "critic")
            dp_sizes.append(self._get_worker_group_dp_size(self.critic_wg, critic_roles))
        if self.use_reference_policy and self.ref_policy_wg.world_size != 0:
            ref_roles = ("ref", "actor")
            dp_sizes.append(self._get_worker_group_dp_size(self.ref_policy_wg, ref_roles))
        if self.hybrid_engine:
            if self.actor_rollout_wg.world_size != 0:
                dp_sizes.append(self._get_worker_group_dp_size(self.actor_rollout_wg, ("actor",)))
        else:
            if self.actor_wg.world_size != 0:
                dp_sizes.append(self._get_worker_group_dp_size(self.actor_wg, ("actor",)))
        if not dp_sizes:
            return batch

        size_divisor = reduce(math.lcm, dp_sizes)

        original_batch_size = batch.batch["prompts"].shape[0]
        batch, pad_size = pad_dataproto_to_divisor(batch, size_divisor)
        sample_mask = torch.ones(len(batch), dtype=torch.bool, device=batch.batch["prompts"].device)
        if pad_size > 0:
            sample_mask[original_batch_size:] = False
        batch.batch["sample_mask"] = sample_mask

        if "response_mask" in batch.batch:
            batch.batch["response_mask"][~sample_mask] = 0

        return batch
