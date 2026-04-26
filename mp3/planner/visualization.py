from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from waymax import datatypes

from planner import scene_viz
from planner.types import ReferencePath
from planner.types import TrajectoryCost
from planner.types import TrajectorySample
from planner.types import AgentPredictions

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


GHOST_SAMPLE_STEP = 4
EGO_DEBUG_COLOR = "#7ec8ff"
OTHER_DEBUG_COLOR = "#e88a8a"
EGO_DEBUG_EDGE = "#2a7fb8"
OTHER_DEBUG_EDGE = "#9c3f3f"


def _ego_geometry(state: datatypes.SimulatorState) -> tuple[int, tuple[float, float]]:
  """Returns ego id and current `(length, width)`."""
  sdc_mask = np.asarray(state.object_metadata.is_sdc).astype(bool)
  ego_idx = int(np.flatnonzero(sdc_mask)[0])
  timestep = int(state.timestep)
  return (
      int(np.asarray(state.object_metadata.ids[ego_idx])),
      (
          float(np.asarray(state.sim_trajectory.length[ego_idx, timestep])),
          float(np.asarray(state.sim_trajectory.width[ego_idx, timestep])),
      ),
  )


def _agent_indices_from_ids(
    predictions: AgentPredictions,
    focus_agent_ids: tuple[int, ...] | None,
) -> np.ndarray:
  """Maps explicit agent ids to prediction indices."""
  if not focus_agent_ids:
    return np.array([], dtype=int)
  ids = np.asarray(predictions.ids)
  indices = [int(np.flatnonzero(ids == agent_id)[0]) for agent_id in focus_agent_ids if np.any(ids == agent_id)]
  if not indices:
    return np.array([], dtype=int)
  return np.asarray(indices, dtype=int)


def _project_to_frenet_signed(
    reference_path: ReferencePath,
    xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Projects points to debug Frenet coordinates without clamping before route start."""
  ref_xy = reference_path.xy
  diffs = xy[:, None, :] - ref_xy[None, :, :]
  nearest_idx = np.argmin(np.linalg.norm(diffs, axis=-1), axis=1)
  nearest_xy = ref_xy[nearest_idx]
  nearest_heading = reference_path.heading[nearest_idx]
  tangent = np.stack([np.cos(nearest_heading), np.sin(nearest_heading)], axis=-1)
  normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=-1)
  local = xy - nearest_xy
  longitudinal = reference_path.arc_length[nearest_idx] + np.sum(local * tangent, axis=-1)
  lateral = np.sum(local * normal, axis=-1)
  return longitudinal, lateral


def _lead_agent_indices(
    reference_path: ReferencePath,
    chosen_trajectory: TrajectorySample,
    chosen_start_index: int,
    predictions: AgentPredictions,
    ego_id: int,
    ego_size: tuple[float, float],
    max_agents: int,
) -> np.ndarray:
  """Returns current-frame forward actors visible in the `s-d` window."""
  start_idx = max(0, min(chosen_start_index, len(chosen_trajectory.x) - 1))
  horizon_steps = min(len(chosen_trajectory.x) - start_idx, predictions.xy.shape[0])
  if horizon_steps == 0:
    return np.array([], dtype=int)

  ego_length = float(ego_size[0])
  ego_s0 = float(chosen_trajectory.s[start_idx])
  ego_d0 = float(chosen_trajectory.d[start_idx])
  pred_s0, pred_d0 = _project_to_frenet_signed(reference_path, predictions.xy[0])
  valid0 = predictions.valid[0] & (predictions.ids != ego_id)
  ds = pred_s0 - ego_s0
  dd = pred_d0 - ego_d0
  front_gap = ds - 0.5 * (ego_length + predictions.length[0])
  in_window = valid0 & (dd >= -5.0) & (dd <= 5.0) & (ds >= -5.0) & (ds <= 35.0)
  ahead = front_gap >= 0.0
  relevant = in_window & ahead
  if not np.any(relevant):
    return np.array([], dtype=int)
  candidate_indices = np.flatnonzero(relevant)
  order = candidate_indices[np.argsort(front_gap[candidate_indices])]
  return order[:max_agents]


def _plot_future_overlay(
    ax: plt.Axes,
    reference_path: ReferencePath,
    state: datatypes.SimulatorState,
    chosen_trajectory: TrajectorySample,
    chosen_start_index: int,
    predictions: AgentPredictions | None,
    focus_agent_ids: tuple[int, ...] | None = None,
) -> None:
  """Plots lightweight future trajectory overlays."""
  if predictions is None:
    return
  ego_id, ego_size = _ego_geometry(state)
  start_idx = max(0, min(chosen_start_index, len(chosen_trajectory.x) - 1))
  chosen_xy = np.stack(
      [chosen_trajectory.x[start_idx:], chosen_trajectory.y[start_idx:]],
      axis=-1,
  )
  horizon_steps = min(len(chosen_xy), predictions.xy.shape[0])
  if horizon_steps == 0:
    return

  chosen_xy = chosen_xy[:horizon_steps]
  if focus_agent_ids:
    nearby_agents = _agent_indices_from_ids(predictions, focus_agent_ids)
  else:
    nearby_agents = _lead_agent_indices(
        reference_path,
        chosen_trajectory,
        chosen_start_index,
        predictions,
        ego_id,
        ego_size,
        max_agents=1,
    )

  for agent_idx in nearby_agents:
    valid = predictions.valid[:horizon_steps, agent_idx]
    if not np.any(valid):
      continue
    pts = predictions.xy[:horizon_steps, agent_idx][valid]
    color = OTHER_DEBUG_COLOR
    ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2.2, alpha=0.95, zorder=6)
    sample_idx = np.arange(0, pts.shape[0], GHOST_SAMPLE_STEP, dtype=int)
    if sample_idx.size == 0 or sample_idx[-1] != pts.shape[0] - 1:
      sample_idx = np.concatenate([sample_idx, np.array([pts.shape[0] - 1], dtype=int)])
    ax.scatter(
        pts[sample_idx, 0],
        pts[sample_idx, 1],
        s=18,
        color=color,
        alpha=0.95,
        edgecolors="none",
        zorder=7,
    )
    for local_idx in sample_idx:
      ax.text(
          pts[local_idx, 0] + 0.2,
          pts[local_idx, 1] + 0.2,
          str(int(local_idx)),
          color="black",
          fontsize=7,
          zorder=8,
      )
    ax.text(
        pts[-1, 0] + 0.4,
        pts[-1, 1] + 0.4,
        f"id {int(predictions.ids[agent_idx])}",
        color=OTHER_DEBUG_EDGE,
        fontsize=8,
        zorder=7,
    )


def render_planner_debug_image(
    state: datatypes.SimulatorState,
    reference_path: ReferencePath,
    trajectories: list[TrajectorySample] | None = None,
    costs: list[TrajectoryCost] | None = None,
    best_index: int | None = None,
    chosen_trajectory: TrajectorySample | None = None,
    chosen_start_index: int = 0,
    chosen_predictions: AgentPredictions | None = None,
    focus_agent_ids: tuple[int, ...] | None = None,
    radius_m: float = 30.0,
    pixels_per_meter: float = 20.0,
) -> np.ndarray:
  """Renders a planner debug frame with path and candidate overlays.

  Args:
    state: Current simulator state.
    reference_path: Lane-level reference path and goal.
    trajectories: Optional candidate trajectory list.
    costs: Optional per-candidate costs aligned with `trajectories`.
    best_index: Optional best-trajectory index.
    chosen_trajectory: Optional currently active trajectory to highlight.
    chosen_start_index: Starting sample index into `chosen_trajectory` for the
      highlighted suffix.
    chosen_predictions: Optional future predicted agents aligned with the
      current state for ghosted same-time overlays.
    radius_m: View radius in meters around ego.
    pixels_per_meter: Raster density in pixels/meter.

  Returns:
    RGB image of shape `(H, W, 3)` as `uint8`.
  """
  viz_config = scene_viz.viz_utils.VizConfig(
      front_x=radius_m,
      back_x=radius_m,
      front_y=radius_m,
      back_y=radius_m,
      px_per_meter=pixels_per_meter,
      show_agent_id=True,
  )
  fig, ax = scene_viz.viz_utils.init_fig_ax(viz_config)
  ax.set_facecolor("white")
  scene_viz._plot_roadgraph_lines(ax, state.roadgraph_points)
  scene_viz._plot_agents(ax, state, use_log_traj=False, show_agent_id=True)

  if trajectories is not None:
    speed_min = min(traj.target_speed for traj in trajectories)
    speed_max = max(traj.target_speed for traj in trajectories)
    speed_span = max(speed_max - speed_min, 1e-6)
    for idx, traj in enumerate(trajectories):
      speed_scale = (traj.target_speed - speed_min) / speed_span
      color = plt.cm.turbo(0.12 + 0.76 * speed_scale)
      alpha = 0.2 if best_index != idx else 0.98
      linewidth = 1.6 if best_index != idx else 4.2
      ax.plot(traj.x, traj.y, color=color, alpha=alpha, linewidth=linewidth, zorder=4)
      if best_index == idx:
        ax.scatter(
            traj.x[-1],
            traj.y[-1],
            s=30,
            color=color,
            alpha=alpha,
            edgecolors="none",
            zorder=5,
        )
  if chosen_trajectory is None and trajectories is not None and best_index is not None:
    chosen_trajectory = trajectories[best_index]
    chosen_start_index = 0

  if chosen_trajectory is not None:
    start_idx = max(0, min(chosen_start_index, len(chosen_trajectory.x) - 1))
    chosen_x = chosen_trajectory.x[start_idx:]
    chosen_y = chosen_trajectory.y[start_idx:]
    if len(chosen_x) > 0:
      ax.plot(chosen_x, chosen_y, color="black", linewidth=6.4, alpha=0.18, zorder=4)
      ax.plot(chosen_x, chosen_y, color="#7a3cff", linewidth=4.8, alpha=0.98, zorder=6)
      ax.scatter(
          chosen_x[-1],
          chosen_y[-1],
          s=34,
          color="#7a3cff",
          alpha=0.98,
          edgecolors="none",
          zorder=7,
      )

  if chosen_trajectory is not None and chosen_predictions is not None:
    _plot_future_overlay(
        ax,
        reference_path,
        state,
        chosen_trajectory,
        chosen_start_index,
        chosen_predictions,
        focus_agent_ids,
    )

  ax.plot(
      reference_path.xy[:, 0],
      reference_path.xy[:, 1],
      color="#1f6fe5",
      linewidth=3.0,
      alpha=0.95,
      zorder=5,
  )

  if best_index is not None and costs is not None:
    cost = costs[best_index]
    best = trajectories[best_index]
    ax.set_title(
        f"bank={len(trajectories)} best={best_index} total={cost.total:.2f} goal={cost.goal:.2f} "
        f"jerk={cost.jerk:.2f} collision={cost.collision:.2f} long={cost.longitudinal:.2f} "
        f"track={cost.tracking:.2f} "
        f"vT={best.target_speed:.1f} aT={best.target_accel:.1f} dT={best.target_offset:.1f}",
        fontsize=10,
    )
  scene_viz._center_axes(ax, state, viz_config, use_log_traj=False)
  fig.canvas.draw()
  width, height = fig.canvas.get_width_height()
  image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(
      height, width, 4
  )[:, :, :3].copy()
  plt.close(fig)
  return image


def save_image(image: np.ndarray, path: Path) -> None:
  """Saves one RGB image to disk.

  Args:
    image: RGB image of shape `(H, W, 3)` as `uint8`.
    path: Output image path.
  """
  path.parent.mkdir(parents=True, exist_ok=True)
  Image.fromarray(image).save(path)


def save_mp4(frames: list[np.ndarray], path: Path, fps: int = 10) -> None:
  """Saves an MP4 video from RGB frames.

  Args:
    frames: RGB frames of shape `(H, W, 3)` as `uint8`.
    path: Output MP4 path.
    fps: Video frame rate in frames per second.
  """
  path.parent.mkdir(parents=True, exist_ok=True)
  import imageio.v3 as iio
  iio.imwrite(str(path), np.stack(frames, axis=0), fps=fps, plugin="pyav", codec="libx264")


def write_metrics_text(lines: list[str], path: Path) -> None:
  """Writes plain-text milestone metrics to disk.

  Args:
    lines: Metric lines to write.
    path: Output text-file path.
  """
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text("\n".join(lines) + "\n")
