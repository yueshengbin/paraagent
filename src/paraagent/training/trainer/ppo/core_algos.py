# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO-like algorithms.
"""

from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch

import verl.utils.torch_functional as verl_F


def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    trajectory_uids: np.ndarray,
    step_indices: np.ndarray,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Compute step-level GAE and returns, masked to response tokens.
    Adapted from Hugging Face TRL's PPO trainer.
    """
    device = token_level_rewards.device

    with torch.no_grad():
        rewards = (token_level_rewards * response_mask).sum(dim=1)

        values = values[:, 0]

        unique_traj_np, traj_inv_np = np.unique(trajectory_uids, return_inverse=True)
        num_traj = len(unique_traj_np)
        traj_inv = torch.as_tensor(traj_inv_np, dtype=torch.long, device=device)
        step_ids = torch.as_tensor(step_indices, device=device)
        max_step = int(step_ids.max().item()) + 1

        rewards_map = torch.zeros((num_traj, max_step), dtype=rewards.dtype, device=device)
        values_map = torch.zeros((num_traj, max_step), dtype=values.dtype, device=device)

        rewards_map[traj_inv, step_ids] = rewards
        values_map[traj_inv, step_ids] = values

        lastgaelam = 0
        advantages_reversed = []

        for t in reversed(range(max_step)):
            nextvalues = values_map[:, t + 1] if t < max_step - 1 else 0.0
            delta = rewards_map[:, t] + gamma * nextvalues - values_map[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages_map = torch.stack(advantages_reversed[::-1], dim=1)

        advantages = advantages_map[traj_inv, step_ids]
        returns = advantages + values

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        advantages = advantages.unsqueeze(1) * response_mask
        returns = returns.unsqueeze(1) * response_mask

    return advantages, returns


def compute_token_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    trajectory_uids: np.ndarray,
    step_indices: np.ndarray,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Compute token-level GAE across ordered steps of each trajectory.
    response_mask excludes tool and padding tokens without advancing recursion;
    critic values align with the state preceding each generated token.
    """
    device = token_level_rewards.device
    bsz, resp_len = token_level_rewards.shape

    with torch.no_grad():
        unique_traj_np, traj_inv_np = np.unique(trajectory_uids, return_inverse=True)
        num_traj = len(unique_traj_np)
        traj_inv = torch.as_tensor(traj_inv_np, dtype=torch.long, device=device)
        step_ids = torch.as_tensor(step_indices, dtype=torch.long, device=device)
        max_step = int(step_ids.max().item()) + 1 if bsz > 0 else 0

        row_map = torch.full((num_traj, max_step), -1, dtype=torch.long, device=device)
        row_map[traj_inv, step_ids] = torch.arange(bsz, device=device, dtype=torch.long)

        advantages = torch.zeros_like(token_level_rewards)
        returns = torch.zeros_like(token_level_rewards)

        gae_dtype = token_level_rewards.dtype
        bootstrap_value = torch.zeros((num_traj,), dtype=gae_dtype, device=device)
        lastgaelam = torch.zeros((num_traj,), dtype=gae_dtype, device=device)

        for t in reversed(range(max_step)):
            rows = row_map[:, t]
            active = rows >= 0
            if not torch.any(active):
                continue

            idx = rows[active]
            r = token_level_rewards[idx]
            v = values[idx]
            m = response_mask[idx]
            m_bool = m.to(dtype=torch.bool)

            r = r * m

            nextvalues = bootstrap_value[active].clone()
            lastgaelam_active = lastgaelam[active].clone()

            adv_step = torch.zeros_like(r)

            for j in reversed(range(resp_len)):
                delta = r[:, j] + gamma * nextvalues - v[:, j]
                lastgaelam_ = delta + gamma * lam * lastgaelam_active

                mj = m[:, j].to(dtype=nextvalues.dtype)
                vj = v[:, j].to(dtype=nextvalues.dtype)
                nextvalues = vj * mj + (1 - mj) * nextvalues
                lastgaelam_active = lastgaelam_ * mj + (1 - mj) * lastgaelam_active
                adv_step[:, j] = lastgaelam_active

            adv_step = adv_step * m
            ret_step = (adv_step + v) * m

            advantages[idx] = adv_step
            returns[idx] = ret_step

            has_action = m_bool.any(dim=-1)
            bootstrap_value_active = bootstrap_value[active]
            bootstrap_value_active = torch.where(has_action, nextvalues, bootstrap_value_active)
            bootstrap_value[active] = bootstrap_value_active
            lastgaelam[active] = lastgaelam_active

        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    trajectory_uids: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute trajectory-grouped GRPO advantages from scalar response rewards.
    Optionally divide by group standard deviation; disabling it follows Dr.GRPO.
    """

    step_scores = (token_level_rewards * response_mask).sum(dim=-1)

    traj2total_score: dict[object, torch.Tensor] = {}
    traj2index: dict[object, object] = {}

    id2score = defaultdict(list)
    id2mean: dict[object, torch.Tensor] = {}
    id2std: dict[object, torch.Tensor] = {}

    with torch.no_grad():
        bsz = step_scores.shape[0]

        for i in range(bsz):
            traj_uid = trajectory_uids[i]
            if traj_uid in traj2total_score:
                traj2total_score[traj_uid] = traj2total_score[traj_uid] + step_scores[i]
            else:
                traj2total_score[traj_uid] = step_scores[i]
                traj2index[traj_uid] = index[i]

        for traj_uid, total_score in traj2total_score.items():
            id2score[traj2index[traj_uid]].append(total_score)

        for idx in id2score:
            if len(id2score[idx]) == 1:
                ref = id2score[idx][0]
                id2mean[idx] = torch.zeros_like(ref)
                id2std[idx] = torch.ones_like(ref)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")

        traj2adv: dict[object, torch.Tensor] = {}
        for traj_uid, total_score in traj2total_score.items():
            idx = traj2index[traj_uid]
            if norm_adv_by_std_in_grpo:
                traj2adv[traj_uid] = (total_score - id2mean[idx]) / (id2std[idx] + epsilon)
            else:
                traj2adv[traj_uid] = total_score - id2mean[idx]

        scores = step_scores.clone()
        for i in range(bsz):
            scores[i] = traj2adv[trajectory_uids[i]]

        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def _get_config_value(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _component_scores_to_token_level(
    *,
    component_values: Any,
    response_mask: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    device = response_mask.device
    values: list[float] = []
    for row_index, value in enumerate(component_values):
        if value is None:
            raise ValueError(
                f"GDPO component value is missing at batch row {row_index}; "
                "reward producers must provide every configured component"
            )
        try:
            scalar = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"GDPO component value at batch row {row_index} must be a scalar, got {value!r}"
            ) from exc
        if not np.isfinite(scalar):
            raise ValueError(f"GDPO component value at batch row {row_index} must be finite, got {scalar!r}")
        values.append(scalar)
    scalar_scores = torch.as_tensor(np.asarray(values, dtype=np.float32), dtype=dtype, device=device)
    if scalar_scores.shape[0] != response_mask.shape[0]:
        raise ValueError(
            f"GDPO component length mismatch: got {scalar_scores.shape[0]} values for "
            f"{response_mask.shape[0]} batch rows"
        )

    token_scores = torch.zeros_like(response_mask, dtype=dtype)
    action_mask = response_mask.to(dtype=torch.bool)
    has_action = action_mask.any(dim=-1)
    if torch.any(has_action):
        positions = torch.arange(response_mask.shape[-1], device=device).unsqueeze(0).expand_as(response_mask)
        last_action_pos = (
            torch.where(action_mask, positions, torch.full_like(positions, -1)).max(dim=-1).values
        )
        rows = torch.nonzero(has_action, as_tuple=False).flatten()
        token_scores[rows, last_action_pos[rows]] = scalar_scores[rows]
    return token_scores


def _whiten_gdpo_advantages_by_trajectory(
    combined_advantage: torch.Tensor,
    response_mask: torch.Tensor,
    trajectory_uids: np.ndarray,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize GDPO with one equally weighted value per trajectory.
    Aggregate across tokens and rows first so longer trajectories receive no extra weight.
    """
    if combined_advantage.shape != response_mask.shape:
        raise ValueError(
            "GDPO combined advantage and response mask must have the same shape: "
            f"got {tuple(combined_advantage.shape)} and {tuple(response_mask.shape)}"
        )
    if len(trajectory_uids) != combined_advantage.shape[0]:
        raise ValueError(
            f"GDPO trajectory uid length mismatch: got {len(trajectory_uids)} values for "
            f"{combined_advantage.shape[0]} batch rows"
        )

    action_mask = response_mask.to(dtype=torch.bool)
    trajectory_rows: dict[object, list[int]] = {}
    for row, trajectory_uid in enumerate(trajectory_uids):
        trajectory_rows.setdefault(trajectory_uid, []).append(row)

    trajectory_values = []
    for trajectory_uid, rows in trajectory_rows.items():
        value = None
        for row in rows:
            row_values = combined_advantage[row][action_mask[row]]
            if row_values.numel() > 0:
                value = row_values[0]
                break
        if value is None:
            raise ValueError(f"GDPO trajectory {trajectory_uid!r} contains no action tokens")
        trajectory_values.append(value)

    values = torch.stack(trajectory_values)
    if values.numel() == 1:
        normalized_values = torch.zeros_like(values)
    else:
        normalized_values = (values - values.mean()) / (values.std() + epsilon)

    advantages = torch.zeros_like(combined_advantage)
    pre_bn_by_row = torch.zeros(
        combined_advantage.shape[0], dtype=combined_advantage.dtype, device=combined_advantage.device
    )
    final_by_row = torch.zeros_like(pre_bn_by_row)
    for pre_bn_value, normalized_value, rows in zip(
        values, normalized_values, trajectory_rows.values(), strict=True
    ):
        for row in rows:
            advantages[row] = normalized_value * response_mask[row]
            pre_bn_by_row[row] = pre_bn_value
            final_by_row[row] = normalized_value
    return advantages, pre_bn_by_row, final_by_row


def compute_gdpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    trajectory_uids: np.ndarray,
    config: Optional[Any] = None,
    non_tensor_batch: Optional[dict] = None,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    diagnostics: Optional[dict[str, torch.Tensor]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize reward channels per prompt, combine weights, then normalize the batch.
    Packed rows belonging to one trajectory receive the same advantage.
    """
    gdpo_reward_keys = _get_config_value(config, "gdpo_reward_keys", None)
    if not gdpo_reward_keys:
        raise ValueError("GDPO requires algorithm.gdpo_reward_keys.")
    gdpo_reward_keys = list(gdpo_reward_keys)

    if non_tensor_batch is None:
        raise ValueError("GDPO requires non_tensor_batch containing reward component values.")

    missing_keys = [key for key in gdpo_reward_keys if key not in non_tensor_batch]
    if missing_keys:
        raise ValueError(
            f"GDPO reward keys missing from non_tensor_batch: {missing_keys}. "
            f"Available keys: {list(non_tensor_batch.keys())}"
        )

    gdpo_weights = _get_config_value(config, "gdpo_reward_weights", None)
    if gdpo_weights is None:
        weights = torch.ones(
            len(gdpo_reward_keys), dtype=token_level_rewards.dtype, device=token_level_rewards.device
        )
    else:
        gdpo_weights = list(gdpo_weights)
        if len(gdpo_weights) != len(gdpo_reward_keys):
            raise ValueError(
                f"algorithm.gdpo_reward_weights length ({len(gdpo_weights)}) must match "
                f"algorithm.gdpo_reward_keys length ({len(gdpo_reward_keys)})."
            )
        weights = torch.as_tensor(
            gdpo_weights, dtype=token_level_rewards.dtype, device=token_level_rewards.device
        )

    combined_advantage = torch.zeros_like(token_level_rewards)
    for weight, key in zip(weights, gdpo_reward_keys, strict=True):
        component_rewards = _component_scores_to_token_level(
            component_values=non_tensor_batch[key],
            response_mask=response_mask,
            dtype=token_level_rewards.dtype,
        )
        component_advantage, _ = compute_grpo_outcome_advantage(
            token_level_rewards=component_rewards,
            response_mask=response_mask,
            index=index,
            trajectory_uids=trajectory_uids,
            epsilon=epsilon,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        combined_advantage = combined_advantage + weight * component_advantage
        if diagnostics is not None:
            component_by_row = torch.zeros(
                response_mask.shape[0],
                dtype=component_advantage.dtype,
                device=component_advantage.device,
            )
            action_mask = response_mask.to(dtype=torch.bool)
            for row in range(response_mask.shape[0]):
                row_values = component_advantage[row][action_mask[row]]
                if row_values.numel() > 0:
                    component_by_row[row] = row_values[0]
            diagnostics[f"component_advantage/{key}"] = component_by_row

    advantages, pre_bn_by_row, final_by_row = _whiten_gdpo_advantages_by_trajectory(
        combined_advantage=combined_advantage,
        response_mask=response_mask,
        trajectory_uids=trajectory_uids,
        epsilon=epsilon,
    )
    if diagnostics is not None:
        diagnostics["pre_bn_advantage"] = pre_bn_by_row
        diagnostics["final_advantage"] = final_by_row
    return advantages, advantages


def summarize_gdpo_trajectory_advantages(
    row_values: Any,
    trajectory_uids: np.ndarray,
) -> dict[str, float]:
    """Summarize one equally weighted GDPO advantage value per trajectory."""
    values = np.asarray(row_values, dtype=np.float64).reshape(-1)
    uids = np.asarray(trajectory_uids, dtype=object).reshape(-1)
    if values.shape[0] != uids.shape[0]:
        raise ValueError(
            f"GDPO diagnostic length mismatch: got {values.shape[0]} values for {uids.shape[0]} trajectory rows"
        )

    trajectory_values: dict[object, float] = {}
    for trajectory_uid, value in zip(uids, values, strict=True):
        if trajectory_uid not in trajectory_values and np.isfinite(value):
            trajectory_values[trajectory_uid] = float(value)
    if not trajectory_values:
        return {}

    unique_values = np.asarray(list(trajectory_values.values()), dtype=np.float64)
    return {
        "mean": float(np.mean(unique_values)),
        "std": float(np.std(unique_values)),
        "min": float(np.min(unique_values)),
        "max": float(np.max(unique_values)),
        "abs_mean": float(np.mean(np.abs(unique_values))),
        "positive_frac": float(np.mean(unique_values > 0.0)),
    }


def agg_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_agg_mode: str,
    dp_size: int = 1,
    batch_num_tokens: Optional[int] = None,
    global_batch_size: Optional[int] = None,
    loss_scale_factor: Optional[int] = None,
):
    """Aggregate loss with pad-aware sequence counting and zero-safe empty-batch handling."""

    def is_zero_denom(denom):
        if torch.is_tensor(denom):
            return denom.detach().item() == 0
        return denom == 0

    if loss_agg_mode == "token-mean":
        if batch_num_tokens is None:
            denom = loss_mask.sum()
        else:
            denom = batch_num_tokens
        if is_zero_denom(denom):
            return loss_mat.sum() * 0.0
        loss = verl_F.masked_sum(loss_mat, loss_mask) / denom * dp_size
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        seq_mask = (torch.sum(loss_mask, dim=-1) > 0).float()
        if global_batch_size is None:
            denom = seq_mask.sum()
        else:
            denom = global_batch_size
        if is_zero_denom(denom):
            return loss_mat.sum() * 0.0
        loss = verl_F.masked_sum(seq_losses, seq_mask) / denom * dp_size
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_token_count = torch.sum(loss_mask, dim=-1)
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / (seq_token_count + 1e-8)
        seq_mask = (seq_token_count > 0).float()
        if global_batch_size is None:
            denom = seq_mask.sum()
        else:
            denom = global_batch_size
        if is_zero_denom(denom):
            return loss_mat.sum() * 0.0
        loss = verl_F.masked_sum(seq_losses, seq_mask) / denom * dp_size
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        if loss_scale_factor is None:
            loss_scale_factor = loss_mask.shape[-1]
        if loss_scale_factor == 0:
            return loss_mat.sum() * 0.0
        loss = torch.sum(seq_losses) / loss_scale_factor
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_agg_mode: str = "token-mean",
):
    """Local value loss that uses pad-aware `agg_loss`."""
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = 0.5 * agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[Any] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    assert config is not None

    clip_ratio = config.clip_ratio
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get("clip_ratio_c", 3.0)

    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **getattr(config, "global_batch_info", {}),
    )
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


def compute_policy_loss_reinforce(
    rollout_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-sum",
    config: Optional[Any] = None,
    rollout_is_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    assert config is not None, "ActorConfig must be provided for REINFORCE loss"

    if rollout_is_weights is not None:
        pg_losses = -advantages * log_prob * rollout_is_weights
    else:
        pg_losses = -advantages * log_prob

    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **getattr(config, "global_batch_info", {}),
    )

    negative_approx_kl = log_prob - rollout_log_prob
    kl_divergence = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_metrics = {
        "actor/ppo_kl": kl_divergence.detach().item(),
    }
    return pg_loss, pg_metrics


def compute_policy_loss_bypass_mode(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[Any] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_rejection_mask

    assert config is not None, "config is required for bypass_mode loss"
    del rollout_is_weights

    rollout_corr_config = (
        config.policy_loss.get("rollout_correction", None) if hasattr(config, "policy_loss") else None
    )
    if rollout_corr_config is None:
        raise ValueError(
            "rollout_correction config not found in policy_loss. "
            "When using loss_mode='bypass_mode', ensure rollout_correction config is passed."
        )

    loss_type = rollout_corr_config.get("loss_type", "ppo_clip")
    rollout_is = rollout_corr_config.get("rollout_is", None)
    rollout_is_threshold = rollout_corr_config.get("rollout_is_threshold", 2.0)
    rollout_rs = rollout_corr_config.get("rollout_rs", None)
    rollout_rs_threshold = rollout_corr_config.get("rollout_rs_threshold", None)
    rollout_rs_threshold_lower = rollout_corr_config.get("rollout_rs_threshold_lower", None)
    rollout_token_veto_threshold = rollout_corr_config.get("rollout_token_veto_threshold", None)
    rollout_is_batch_normalize = rollout_corr_config.get("rollout_is_batch_normalize", True)

    rollout_log_prob = old_log_prob
    rollout_metrics, modified_response_mask, rollout_is_weights_proto = (
        compute_rollout_correction_and_rejection_mask(
            old_log_prob=log_prob,
            rollout_log_prob=rollout_log_prob,
            response_mask=response_mask,
            rollout_is=rollout_is,
            rollout_is_threshold=rollout_is_threshold,
            rollout_rs=rollout_rs,
            rollout_rs_threshold=rollout_rs_threshold,
            rollout_rs_threshold_lower=rollout_rs_threshold_lower,
            rollout_token_veto_threshold=rollout_token_veto_threshold,
            rollout_is_batch_normalize=rollout_is_batch_normalize,
        )
    )

    computed_is_weights = (
        rollout_is_weights_proto.batch["rollout_is_weights"] if rollout_is_weights_proto else None
    )
    effective_mask = modified_response_mask

    if loss_type == "reinforce":
        pg_loss, pg_metrics = compute_policy_loss_reinforce(
            rollout_log_prob=rollout_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=effective_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=computed_is_weights,
        )
    elif loss_type == "ppo_clip":
        pg_loss, pg_metrics = compute_policy_loss_vanilla(
            old_log_prob=rollout_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=effective_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=None,
        )
    else:
        raise ValueError(f"Invalid loss_type: {loss_type}. Must be 'reinforce' or 'ppo_clip'.")

    pg_metrics.update(rollout_metrics)
    return pg_loss, pg_metrics


def get_policy_loss_fn(name: str):
    local_policy_loss_fns = {
        "vanilla": compute_policy_loss_vanilla,
        "reinforce": compute_policy_loss_reinforce,
        "bypass_mode": compute_policy_loss_bypass_mode,
    }
    if name in local_policy_loss_fns:
        return local_policy_loss_fns[name]

    from verl.trainer.ppo.core_algos import get_policy_loss_fn as upstream_get_policy_loss_fn

    return upstream_get_policy_loss_fn(name)
