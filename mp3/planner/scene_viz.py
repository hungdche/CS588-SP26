from __future__ import annotations

import dataclasses
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib.patches import Circle
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np
from waymax import config as waymax_config
from waymax import dataloader
from waymax import datatypes
from waymax.datatypes import SimulatorState
from waymax.visualization import utils as viz_utils

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_DATA = PROJECT_ROOT / "data"
DEFAULT_DATASET_PATH = (
    Path("/content/drive/MyDrive/CS588/womd")
    if Path("/content/drive/MyDrive/CS588/womd").is_dir()
    else _LOCAL_DATA
)

DARK_GRAY = np.array([0.42, 0.42, 0.42])
LIGHT_RED = np.array([0.93, 0.62, 0.62])
DARK_RED = np.array([0.45, 0.08, 0.08])
LIGHT_BLUE = np.array([0.50, 0.79, 0.98])
BLACK = np.array([0.0, 0.0, 0.0])
LANE_SHADE = np.array([0.70, 0.70, 0.70])
ROAD_STYLE = {
    datatypes.MapElementIds.LANE_FREEWAY: (BLACK, 1.8),
    datatypes.MapElementIds.LANE_SURFACE_STREET: (BLACK, 1.8),
    datatypes.MapElementIds.LANE_BIKE_LANE: (BLACK, 1.8),
    datatypes.MapElementIds.ROAD_LINE_BROKEN_SINGLE_WHITE: (DARK_GRAY, 2.0),
    datatypes.MapElementIds.ROAD_LINE_SOLID_SINGLE_WHITE: (DARK_GRAY, 2.2),
    datatypes.MapElementIds.ROAD_LINE_SOLID_DOUBLE_WHITE: (DARK_GRAY, 2.2),
    datatypes.MapElementIds.ROAD_LINE_BROKEN_SINGLE_YELLOW: (DARK_GRAY, 2.0),
    datatypes.MapElementIds.ROAD_LINE_BROKEN_DOUBLE_YELLOW: (DARK_GRAY, 2.0),
    datatypes.MapElementIds.ROAD_LINE_SOLID_SINGLE_YELLOW: (DARK_GRAY, 2.2),
    datatypes.MapElementIds.ROAD_LINE_SOLID_DOUBLE_YELLOW: (DARK_GRAY, 2.2),
    datatypes.MapElementIds.ROAD_LINE_PASSING_DOUBLE_YELLOW: (DARK_GRAY, 2.2),
    datatypes.MapElementIds.ROAD_EDGE_BOUNDARY: (BLACK, 3.2),
    datatypes.MapElementIds.ROAD_EDGE_MEDIAN: (BLACK, 3.2),
    datatypes.MapElementIds.CROSSWALK: (DARK_RED, 2.8),
    datatypes.MapElementIds.SPEED_BUMP: (DARK_RED, 2.8),
}
POLYGON_TYPES = {
    datatypes.MapElementIds.CROSSWALK,
    datatypes.MapElementIds.SPEED_BUMP,
}
LANE_CENTER_TYPES = {
    datatypes.MapElementIds.LANE_FREEWAY,
    datatypes.MapElementIds.LANE_SURFACE_STREET,
    datatypes.MapElementIds.LANE_BIKE_LANE,
}
SYNTHETIC_EGO_SPEED_MPS = 8.0
SYNTHETIC_EGO_LENGTH = 4.8
SYNTHETIC_EGO_WIDTH = 2.0
WOMD_START_TIMESTEP = 10
PATH_JUMP_THRESHOLD_M = 3.0
PATH_TRANSITION_DISTANCE_M = 10.0
PATH_TRANSITION_SAMPLES = 25
GOAL_SPACING_M = 10.0


@dataclasses.dataclass(frozen=True)
class DemoConfig:
  dataset_path: Path = DEFAULT_DATASET_PATH
  max_num_objects: int = 100
  max_steps: int = 30
  viewport_meters: float = 40.0
  px_per_meter: float = 30.0
  scenario_index: int = 0


def _compute_box_corners(box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  x, y, length, width, yaw = box
  c = np.cos(yaw)
  s = np.sin(yaw)
  u = np.array([c, s])
  ut = np.array([s, -c])
  center = np.array([x, y])
  tl = center + length / 2 * u - width / 2 * ut
  tr = center + length / 2 * u + width / 2 * ut
  br = center - length / 2 * u + width / 2 * ut
  bl = center - length / 2 * u - width / 2 * ut
  return np.vstack([tl, tr, br, bl]), center


def _ordered_polygon_points(points: np.ndarray) -> np.ndarray:
  center = points.mean(axis=0)
  angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
  ordered = points[np.argsort(angles)]
  return np.vstack([ordered, ordered[0]])


def _normalize(vec: np.ndarray) -> np.ndarray:
  norm = np.linalg.norm(vec)
  if norm < 1e-6:
    return np.zeros_like(vec)
  return vec / norm


def _make_smooth_transition(
    path_xy: np.ndarray,
    left_idx: int,
    right_idx: int,
) -> np.ndarray:
  start = path_xy[left_idx]
  end = path_xy[right_idx]

  left_dir = _normalize(start - path_xy[max(0, left_idx - 1)])
  right_dir = _normalize(path_xy[min(len(path_xy) - 1, right_idx + 1)] - end)
  if not left_dir.any():
    left_dir = _normalize(end - start)
  if not right_dir.any():
    right_dir = _normalize(end - start)

  tangent_scale = min(PATH_TRANSITION_DISTANCE_M, np.linalg.norm(end - start) * 0.5)
  p0 = start
  p1 = start + left_dir * tangent_scale
  p2 = end - right_dir * tangent_scale
  p3 = end

  ts = np.linspace(0.0, 1.0, PATH_TRANSITION_SAMPLES)
  curve = (
      ((1 - ts) ** 3)[:, None] * p0
      + (3 * ((1 - ts) ** 2) * ts)[:, None] * p1
      + (3 * (1 - ts) * (ts**2))[:, None] * p2
      + (ts**3)[:, None] * p3
  )
  return curve


def _smooth_path_jumps(path_xy: np.ndarray) -> np.ndarray:
  if path_xy.shape[0] < 3:
    return path_xy

  result = [path_xy[0]]
  for idx in range(1, path_xy.shape[0]):
    prev_pt = path_xy[idx - 1]
    curr_pt = path_xy[idx]
    if np.linalg.norm(curr_pt - prev_pt) <= PATH_JUMP_THRESHOLD_M:
      result.append(curr_pt)
      continue

    transition = _make_smooth_transition(path_xy, idx - 1, idx)
    result.extend(transition[1:])
  return np.asarray(result)


def _resample_polyline(path_xy: np.ndarray, spacing_m: float) -> np.ndarray:
  if path_xy.shape[0] < 2:
    return path_xy
  segment_lengths = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
  cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
  total_length = cumulative[-1]
  if total_length < 1e-6:
    return path_xy[:1]

  sample_distances = np.arange(0.0, total_length + spacing_m, spacing_m)
  sample_distances[-1] = min(sample_distances[-1], total_length)
  sample_x = np.interp(sample_distances, cumulative, path_xy[:, 0])
  sample_y = np.interp(sample_distances, cumulative, path_xy[:, 1])
  sampled = np.column_stack([sample_x, sample_y])
  if np.linalg.norm(sampled[-1] - path_xy[-1]) > 1e-3:
    sampled = np.vstack([sampled, path_xy[-1]])
  return sampled


def _current_xy(state: SimulatorState, *, use_log_traj: bool) -> np.ndarray:
  traj = state.log_trajectory if use_log_traj else state.sim_trajectory
  timestep = int(state.timestep)
  return np.asarray(traj.xy[:, timestep, :])


def _has_true_sdc(state: SimulatorState) -> bool:
  sdc_mask = np.asarray(state.object_metadata.is_sdc).astype(bool)
  return bool(sdc_mask.any())


def _get_primary_agent_index(state: SimulatorState) -> int:
  sdc_mask = np.asarray(state.object_metadata.is_sdc).astype(bool)
  if sdc_mask.any():
    return int(np.flatnonzero(sdc_mask)[0])

  valid_mask = np.asarray(state.object_metadata.is_valid).reshape(-1).astype(bool)
  valid_indices = np.flatnonzero(valid_mask)
  if valid_indices.size == 0:
    return 0

  if state.sdc_paths is not None:
    current_xy = _current_xy(state, use_log_traj=False)
    on_route = np.asarray(state.sdc_paths.on_route).reshape(-1).astype(bool)
    path_valid = np.asarray(state.sdc_paths.valid).astype(bool)
    path_xy = np.asarray(state.sdc_paths.xy)
    route_points = []
    for path_index in np.flatnonzero(on_route):
      pts = path_xy[path_index][path_valid[path_index]]
      if pts.shape[0]:
        route_points.append(pts)
    if route_points:
      route_points = np.concatenate(route_points, axis=0)
      distances = np.linalg.norm(
          current_xy[valid_indices, None, :] - route_points[None, :, :], axis=-1
      )
      return int(valid_indices[np.argmin(np.min(distances, axis=1))])

  return int(valid_indices[0])


def _get_primary_agent_mask(state: SimulatorState) -> np.ndarray:
  mask = np.zeros(state.num_objects, dtype=bool)
  mask[_get_primary_agent_index(state)] = True
  return mask


def _get_reference_agent_index(state: SimulatorState) -> int:
  if _has_true_sdc(state):
    return _get_primary_agent_index(state)
  return _get_primary_agent_index(state)


def _plot_roadgraph_lines(
    ax: plt.Axes,
    roadgraph: datatypes.RoadgraphPoints,
) -> None:
  valid = np.asarray(roadgraph.valid).astype(bool)
  if not valid.any():
    return

  xy = np.asarray(roadgraph.xy)[valid]
  ids = np.asarray(roadgraph.ids)[valid]
  types = np.asarray(roadgraph.types)[valid]

  for feature_type, (line_color, line_width) in ROAD_STYLE.items():
    type_mask = types == int(feature_type)
    if not type_mask.any():
      continue
    type_xy = xy[type_mask]
    type_ids = ids[type_mask]
    for feature_id in np.unique(type_ids):
      pts = type_xy[type_ids == feature_id]
      if pts.shape[0] < 2:
        continue
      if feature_type in LANE_CENTER_TYPES:
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color=LANE_SHADE,
            linewidth=8.0,
            alpha=0.22,
            linestyle="-",
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=0,
        )
      if feature_type in POLYGON_TYPES:
        polygon_pts = _ordered_polygon_points(pts)
        if feature_type == datatypes.MapElementIds.CROSSWALK:
          ax.fill(
              polygon_pts[:, 0],
              polygon_pts[:, 1],
              color="white",
              alpha=0.35,
              zorder=0,
          )
        pts = polygon_pts
      ax.plot(
          pts[:, 0],
          pts[:, 1],
          color=line_color,
          linewidth=line_width,
          linestyle="--" if feature_type in LANE_CENTER_TYPES else "-",
          dashes=(6, 6) if feature_type in LANE_CENTER_TYPES else (),
          solid_capstyle="round",
          solid_joinstyle="round",
          zorder=1,
      )


def _get_route_path_points(state: SimulatorState) -> np.ndarray | None:
  reference_index = _get_reference_agent_index(state)
  reference_xy = _current_xy(state, use_log_traj=False)[reference_index]

  if state.sdc_paths is not None:
    paths_xy = np.asarray(state.sdc_paths.xy)
    paths_valid = np.asarray(state.sdc_paths.valid).astype(bool)
    on_route = np.asarray(state.sdc_paths.on_route).astype(bool).reshape(-1)

    candidate_indices = np.flatnonzero(on_route)
    if candidate_indices.size:
      best_path = None
      best_score = None
      for path_index in candidate_indices:
        valid_points = paths_xy[path_index][paths_valid[path_index]]
        if valid_points.shape[0] < 2:
          continue
        dists = np.linalg.norm(valid_points - reference_xy, axis=1)
        distance = float(np.min(dists))
        jump_sizes = np.linalg.norm(np.diff(valid_points, axis=0), axis=1)
        jump_penalty = float(np.sum(np.maximum(jump_sizes - PATH_JUMP_THRESHOLD_M, 0.0)))
        score = distance + 2.0 * jump_penalty
        if best_score is None or score < best_score:
          best_score = score
          best_path = valid_points
      if best_path is not None:
        return _smooth_path_jumps(best_path)

  timestep = int(state.timestep)
  primary_index = _get_primary_agent_index(state)
  future_valid = np.asarray(
      state.sim_trajectory.valid[primary_index, timestep:]
  ).astype(
      bool
  )
  if not future_valid.any():
    return None
  future_xy = np.asarray(state.sim_trajectory.xy[primary_index, timestep:])[
      future_valid
  ]
  if future_xy.shape[0] < 2:
    return None
  return _smooth_path_jumps(future_xy)


def _get_synthetic_progress_index(state: SimulatorState, path_xy: np.ndarray) -> int:
  if path_xy.shape[0] < 2:
    return 0
  step_offset = max(0, int(state.timestep) - WOMD_START_TIMESTEP)
  segment_lengths = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
  cum_length = np.concatenate([[0.0], np.cumsum(segment_lengths)])
  target_distance = step_offset * datatypes.TIME_INTERVAL * SYNTHETIC_EGO_SPEED_MPS
  return int(np.clip(np.searchsorted(cum_length, target_distance), 0, len(path_xy) - 1))


def _get_ego_pose(state: SimulatorState, *, use_log_traj: bool) -> tuple[np.ndarray, np.ndarray] | None:
  if _has_true_sdc(state):
    ego_index = _get_primary_agent_index(state)
    traj = state.log_trajectory if use_log_traj else state.sim_trajectory
    timestep = int(state.timestep)
    if not np.asarray(traj.valid[ego_index, timestep]).item():
      return None
    box = np.asarray(
        traj.stack_fields(["x", "y", "length", "width", "yaw"])[ego_index, timestep]
    )
    corners, center = _compute_box_corners(box)
    return corners, center

  path_xy = _get_route_path_points(state)
  if path_xy is None or path_xy.shape[0] < 2:
    return None
  path_idx = _get_synthetic_progress_index(state, path_xy)
  if path_idx >= path_xy.shape[0] - 1:
    prev_idx = max(0, path_idx - 1)
    next_idx = path_idx
  else:
    prev_idx = path_idx
    next_idx = path_idx + 1
  center = path_xy[path_idx]
  direction = path_xy[next_idx] - path_xy[prev_idx]
  yaw = float(np.arctan2(direction[1], direction[0]))
  box = np.array([center[0], center[1], SYNTHETIC_EGO_LENGTH, SYNTHETIC_EGO_WIDTH, yaw])
  corners, center = _compute_box_corners(box)
  return corners, center


def _get_planned_centerline_xy(state: SimulatorState) -> np.ndarray | None:
  if _has_true_sdc(state):
    ego_index = _get_primary_agent_index(state)
    timestep = int(state.timestep)
    log_valid = np.asarray(state.log_trajectory.valid[ego_index, timestep:]).astype(bool)
    if not log_valid.any():
      return None
    log_future_xy = np.asarray(state.log_trajectory.xy[ego_index, timestep:])[log_valid]
    if log_future_xy.shape[0] < 2:
      return log_future_xy
    return _resample_polyline(_smooth_path_jumps(log_future_xy), spacing_m=1.0)

  route_xy = _get_route_path_points(state)
  if route_xy is None:
    return None

  path_idx = _get_synthetic_progress_index(state, route_xy)
  suffix = route_xy[path_idx:]
  planned_xy = suffix if suffix.shape[0] >= 2 else route_xy
  return _resample_polyline(planned_xy, spacing_m=1.0)


def _plot_agents(
    ax: plt.Axes,
    state: SimulatorState,
    *,
    use_log_traj: bool,
    show_agent_id: bool,
) -> None:
  traj = state.log_trajectory if use_log_traj else state.sim_trajectory
  timestep = int(state.timestep)
  current_valid = np.asarray(traj.valid[:, timestep]).astype(bool)
  history_valid = np.asarray(traj.valid[:, :timestep]).astype(bool)
  history_xy = np.asarray(traj.xy[:, :timestep])
  has_true_sdc = _has_true_sdc(state)
  is_primary = _get_primary_agent_mask(state) if has_true_sdc else np.zeros(state.num_objects, dtype=bool)
  agent_ids = np.asarray(state.object_metadata.ids)
  boxes = np.asarray(
      traj.stack_fields(["x", "y", "length", "width", "yaw"])
  )

  for obj_index in range(traj.num_objects):
    face_color = LIGHT_BLUE if is_primary[obj_index] else LIGHT_RED
    if history_valid.shape[1] and history_valid[obj_index].any():
      pts = history_xy[obj_index][history_valid[obj_index]]
      ax.plot(
          pts[:, 0],
          pts[:, 1],
          color=face_color,
          linewidth=1.8,
          alpha=0.45,
          zorder=2,
      )

    if not current_valid[obj_index]:
      continue

    corners, center = _compute_box_corners(boxes[obj_index, timestep])
    polygon = Polygon(
        corners,
        closed=True,
        facecolor=face_color,
        edgecolor=BLACK,
        linewidth=2.6,
        zorder=6,
    )
    ax.add_patch(polygon)

    front_center = (corners[0] + corners[1]) / 2
    ax.plot(
        [center[0], front_center[0]],
        [center[1], front_center[1]],
        color=BLACK,
        linewidth=2.2,
        zorder=7,
    )

    if show_agent_id:
      ax.text(
          center[0] - 2.0,
          center[1] + 2.2,
          f"{agent_ids[obj_index]}",
          color=BLACK,
          fontsize=8,
          zorder=8,
      )

  if not has_true_sdc:
    route_xy = _get_route_path_points(state)
    ego_pose = _get_ego_pose(state, use_log_traj=use_log_traj)
    if route_xy is not None:
      ego_idx = _get_synthetic_progress_index(state, route_xy)
      history_xy = route_xy[: ego_idx + 1]
      if history_xy.shape[0] >= 2:
        ax.plot(
            history_xy[:, 0],
            history_xy[:, 1],
            color=LIGHT_BLUE,
            linewidth=1.8,
            alpha=0.45,
            zorder=3,
        )
    if ego_pose is not None:
      corners, center = ego_pose
      polygon = Polygon(
          corners,
          closed=True,
          facecolor=LIGHT_BLUE,
          edgecolor=BLACK,
          linewidth=2.8,
          zorder=8,
      )
      ax.add_patch(polygon)
      front_center = (corners[0] + corners[1]) / 2
      ax.plot(
          [center[0], front_center[0]],
          [center[1], front_center[1]],
          color=BLACK,
          linewidth=2.3,
          zorder=9,
      )


def _center_axes(
    ax: plt.Axes,
    state: SimulatorState,
    viz_config: viz_utils.VizConfig,
    *,
    use_log_traj: bool,
) -> None:
  ego_pose = _get_ego_pose(state, use_log_traj=use_log_traj)
  if ego_pose is not None:
    _, center_xy = ego_pose
  else:
    current_xy = _current_xy(state, use_log_traj=use_log_traj)
    center_xy = current_xy[_get_primary_agent_index(state)]
  ax.axis((
      center_xy[0] - viz_config.back_x,
      center_xy[0] + viz_config.front_x,
      center_xy[1] - viz_config.back_y,
      center_xy[1] + viz_config.front_y,
  ))
  ax.set_aspect("equal", adjustable="box")
  ax.set_xticks([])
  ax.set_yticks([])
  for spine in ax.spines.values():
    spine.set_visible(False)


def _plot_current_planned_path(
    ax: plt.Axes,
    state: SimulatorState,
) -> None:
  planned_xy = _get_planned_centerline_xy(state)
  if planned_xy is None:
    return

  ax.plot(
      planned_xy[:, 0],
      planned_xy[:, 1],
      color=LIGHT_BLUE,
      linewidth=6.0,
      alpha=0.95,
      solid_capstyle="round",
      solid_joinstyle="round",
      zorder=9,
  )


def _plot_goal(
    ax: plt.Axes,
    state: SimulatorState,
) -> None:
  planned_xy = _get_planned_centerline_xy(state)
  if planned_xy is None or planned_xy.shape[0] == 0:
    return
  goal_points = _resample_polyline(planned_xy, spacing_m=GOAL_SPACING_M)
  for goal_idx, goal_xy in enumerate(goal_points[1:], start=1):
    is_terminal_goal = goal_idx == len(goal_points) - 1
    goal = Circle(
        (goal_xy[0], goal_xy[1]),
        radius=5.0 if is_terminal_goal else 3.0,
        facecolor=(0.1, 0.8, 0.2, 0.28 if is_terminal_goal else 0.18),
        edgecolor=(0.05, 0.55, 0.12, 0.75 if is_terminal_goal else 0.45),
        linewidth=2.0 if is_terminal_goal else 1.2,
        zorder=5,
    )
    ax.add_patch(goal)


def _dataset_config(config: DemoConfig) -> waymax_config.DatasetConfig:
  return waymax_config.DatasetConfig(
      path=str(config.dataset_path),
      max_num_objects=config.max_num_objects,
      data_format=waymax_config.DataFormat.TFRECORD,
      include_sdc_paths=True,
      num_paths=30,
      num_points_per_path=200,
  )


def load_scenario(config: DemoConfig) -> SimulatorState:
  dataset_iter = dataloader.simulator_state_generator(config=_dataset_config(config))
  scenario = None
  for _ in range(config.scenario_index + 1):
    scenario = next(dataset_iter)
  if scenario is None:
    raise ValueError("No scenario was loaded from the configured dataset path.")
  return scenario


