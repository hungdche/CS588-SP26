from __future__ import annotations

import numpy as np
from waymax import datatypes

from planner.types import AgentPredictions
from planner.types import TrajectorySample

HARD_COLLISION_BUFFER_M = 0.25


def extract_agent_predictions(
    state: datatypes.SimulatorState,
    horizon_steps: int,
) -> AgentPredictions:
  """Extracts future ground-truth agent states for discrete collision checks.

  Args:
    state: Current simulator state.
    horizon_steps: Number of future discrete steps to extract.

  Returns:
    Future agent positions, box sizes, validity mask, and ids.
  """
  timestep = int(state.timestep)
  end_t = min(timestep + horizon_steps, state.log_trajectory.num_timesteps)
  sl = slice(timestep, end_t)
  return AgentPredictions(
      xy=np.asarray(state.log_trajectory.xy[:, sl]).transpose(1, 0, 2),
      yaw=np.asarray(state.log_trajectory.yaw[:, sl]).transpose(1, 0),
      length=np.asarray(state.log_trajectory.length[:, sl]).transpose(1, 0),
      width=np.asarray(state.log_trajectory.width[:, sl]).transpose(1, 0),
      valid=np.asarray(state.log_trajectory.valid[:, sl]).transpose(1, 0),
      ids=np.asarray(state.object_metadata.ids),
  )


def geometric_collision_mask(
    trajectory: TrajectorySample,
    predictions: AgentPredictions,
    ego_size: tuple[float, float],
    ego_id: int,
) -> np.ndarray:
  """Checks oriented-box overlap against future agents at each step.

  Args:
    trajectory: Candidate ego trajectory over `T` samples.
    predictions: Future agent predictions with shape `(T, A, ...)`.
    ego_size: Ego `(length, width)` in meters.
    ego_id: Ego object id to exclude from collision checks.

  Returns:
    Boolean collision mask of shape `(T,)`.
  """
  return geometric_collision_mask_batch(
      [trajectory],
      predictions,
      ego_size=ego_size,
      ego_id=ego_id,
  )[0]


def geometric_collision_mask_batch(
    trajectories: list[TrajectorySample],
    predictions: AgentPredictions,
    ego_size: tuple[float, float],
    ego_id: int,
) -> np.ndarray:
  """Checks oriented-box overlap for a trajectory batch with vectorized SAT.

  Args:
    trajectories: Candidate ego trajectories over the same horizon.
    predictions: Future agent predictions with shape `(T, A, ...)`.
    ego_size: Ego `(length, width)` in meters.
    ego_id: Ego object id to exclude from collision checks.

  Returns:
    Boolean collision mask with shape `(C, T)`.
  """
  if not trajectories:
    return np.zeros((0, 0), dtype=bool)

  horizon_steps = min(len(trajectories[0].x), predictions.xy.shape[0])
  ego_xy = np.stack(
      [np.stack([traj.x[:horizon_steps], traj.y[:horizon_steps]], axis=-1) for traj in trajectories],
      axis=1,
  )
  ego_yaw = np.stack([traj.yaw[:horizon_steps] for traj in trajectories], axis=1)
  other_xy = predictions.xy[:horizon_steps]
  other_yaw = predictions.yaw[:horizon_steps]
  other_length = predictions.length[:horizon_steps]
  other_width = predictions.width[:horizon_steps]
  valid = predictions.valid[:horizon_steps] & (predictions.ids[None, :] != ego_id)
  if horizon_steps:
    valid[0] = False

  ego_half_length = 0.5 * float(ego_size[0])
  ego_half_width = 0.5 * float(ego_size[1])
  ego_half_diag = 0.5 * float(np.hypot(*ego_size))
  other_half_length = 0.5 * other_length
  other_half_width = 0.5 * other_width
  other_half_diag = 0.5 * np.hypot(other_length, other_width)

  deltas = other_xy[:, None, :, :] - ego_xy[:, :, None, :]
  dist = np.linalg.norm(deltas, axis=-1)
  nearby = dist <= (ego_half_diag + other_half_diag[:, None, :] + 1.0)
  candidate_mask = valid[:, None, :] & nearby
  if not candidate_mask.any():
    return np.zeros((len(trajectories), horizon_steps), dtype=bool)

  ego_cos = np.cos(ego_yaw)
  ego_sin = np.sin(ego_yaw)
  ego_forward = np.stack([ego_cos, ego_sin], axis=-1)
  ego_left = np.stack([-ego_sin, ego_cos], axis=-1)

  other_cos = np.cos(other_yaw)
  other_sin = np.sin(other_yaw)
  other_forward = np.stack([other_cos, other_sin], axis=-1)
  other_left = np.stack([-other_sin, other_cos], axis=-1)

  t_a0 = np.sum(deltas * ego_forward[:, :, None, :], axis=-1)
  t_a1 = np.sum(deltas * ego_left[:, :, None, :], axis=-1)
  r00 = np.sum(ego_forward[:, :, None, :] * other_forward[:, None, :, :], axis=-1)
  r01 = np.sum(ego_forward[:, :, None, :] * other_left[:, None, :, :], axis=-1)
  r10 = np.sum(ego_left[:, :, None, :] * other_forward[:, None, :, :], axis=-1)
  r11 = np.sum(ego_left[:, :, None, :] * other_left[:, None, :, :], axis=-1)
  abs_r00 = np.abs(r00) + 1e-6
  abs_r01 = np.abs(r01) + 1e-6
  abs_r10 = np.abs(r10) + 1e-6
  abs_r11 = np.abs(r11) + 1e-6

  sep_a0 = np.abs(t_a0) > (
      ego_half_length
      + other_half_length[:, None, :] * abs_r00
      + other_half_width[:, None, :] * abs_r01
      + HARD_COLLISION_BUFFER_M
  )
  sep_a1 = np.abs(t_a1) > (
      ego_half_width
      + other_half_length[:, None, :] * abs_r10
      + other_half_width[:, None, :] * abs_r11
      + HARD_COLLISION_BUFFER_M
  )
  sep_b0 = np.abs(t_a0 * r00 + t_a1 * r10) > (
      ego_half_length * abs_r00
      + ego_half_width * abs_r10
      + other_half_length[:, None, :]
      + HARD_COLLISION_BUFFER_M
  )
  sep_b1 = np.abs(t_a0 * r01 + t_a1 * r11) > (
      ego_half_length * abs_r01
      + ego_half_width * abs_r11
      + other_half_width[:, None, :]
      + HARD_COLLISION_BUFFER_M
  )

  overlaps = candidate_mask & ~(sep_a0 | sep_a1 | sep_b0 | sep_b1)
  return np.any(overlaps, axis=-1).transpose(1, 0)
