from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import uniform_filter1d
from waymax import datatypes

from planner.types import ReferencePath


LANE_WIDTH_M = 3.6
DEFAULT_LANE_OFFSETS = np.array(
    [-LANE_WIDTH_M, -2.4, -1.2, -0.6, 0.0, 0.6, 1.2, 2.4, LANE_WIDTH_M],
    dtype=float,
)


def resample_polyline(xy: np.ndarray, spacing_m: float = 1.0) -> np.ndarray:
  """Resamples a polyline at approximately uniform arc-length spacing.

  Args:
    xy: Polyline points of shape `(N, 2)` in meters.
    spacing_m: Desired sample spacing in meters.

  Returns:
    Resampled polyline of shape `(M, 2)` in meters.
  """
  if xy.shape[0] < 2:
    return xy
  seg_len = np.linalg.norm(np.diff(xy, axis=0), axis=1)
  arc = np.concatenate([[0.0], np.cumsum(seg_len)])
  if arc[-1] <= 1e-6:
    return xy[:1]
  samples = np.arange(0.0, arc[-1] + spacing_m, spacing_m)
  samples[-1] = min(samples[-1], arc[-1])
  resampled_x = np.interp(samples, arc, xy[:, 0])
  resampled_y = np.interp(samples, arc, xy[:, 1])
  resampled = np.column_stack([resampled_x, resampled_y])
  if np.linalg.norm(resampled[-1] - xy[-1]) > 1e-3:
    resampled = np.vstack([resampled, xy[-1]])
  return resampled


def smooth_path_jumps(xy: np.ndarray, jump_threshold_m: float = 3.0) -> np.ndarray:
  """Smooths large discontinuities in a route polyline with cubic transitions.

  Args:
    xy: Polyline points of shape `(N, 2)` in meters.
    jump_threshold_m: Distance threshold above which adjacent points are
      treated as a discontinuity.

  Returns:
    Smoothed polyline in meters with shape `(M, 2)`.
  """
  if xy.shape[0] < 3:
    return xy

  def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-6 else np.zeros_like(vec)

  out = [xy[0]]
  for idx in range(1, xy.shape[0]):
    prev_pt = xy[idx - 1]
    cur_pt = xy[idx]
    if np.linalg.norm(cur_pt - prev_pt) <= jump_threshold_m:
      out.append(cur_pt)
      continue

    left_dir = _normalize(prev_pt - xy[max(0, idx - 2)])
    right_dir = _normalize(xy[min(xy.shape[0] - 1, idx + 1)] - cur_pt)
    if not left_dir.any():
      left_dir = _normalize(cur_pt - prev_pt)
    if not right_dir.any():
      right_dir = _normalize(cur_pt - prev_pt)
    tangent_scale = min(10.0, 0.5 * np.linalg.norm(cur_pt - prev_pt))
    p0 = prev_pt
    p1 = prev_pt + left_dir * tangent_scale
    p2 = cur_pt - right_dir * tangent_scale
    p3 = cur_pt
    ts = np.linspace(0.0, 1.0, 25)
    curve = (
        ((1 - ts) ** 3)[:, None] * p0
        + (3 * (1 - ts) ** 2 * ts)[:, None] * p1
        + (3 * (1 - ts) * ts**2)[:, None] * p2
        + (ts**3)[:, None] * p3
    )
    out.extend(curve[1:])
  return np.asarray(out)


def build_reference_path(
    state: datatypes.SimulatorState,
    lane_offsets: Iterable[float] | None = None,
) -> ReferencePath:
  """Builds a lane-level reference path from Waymax route paths.

  Args:
    state: Simulator state at the current planning step with optional
      `sdc_paths` route information.
    lane_offsets: Optional candidate lane offsets in meters.

  Returns:
    ReferencePath containing smooth centerline samples, arc length, heading,
    lane offsets, and a goal point.
  """
  if state.sdc_paths is None:
    raise ValueError("state.sdc_paths is required for reference-path extraction.")

  current_xy = np.asarray(state.current_sim_trajectory.xy).reshape(-1, 2)
  sdc_mask = np.asarray(state.object_metadata.is_sdc).astype(bool)
  if not sdc_mask.any():
    raise ValueError("Reference-path extraction assumes an SDC is present.")
  ego_xy = current_xy[sdc_mask][0]
  ego_yaw = float(
      np.asarray(state.current_sim_trajectory.yaw).reshape(-1)[sdc_mask][0]
  )
  ego_fwd = np.array([np.cos(ego_yaw), np.sin(ego_yaw)])

  path_xy = np.asarray(state.sdc_paths.xy)
  path_valid = np.asarray(state.sdc_paths.valid).astype(bool)
  on_route = np.asarray(state.sdc_paths.on_route).reshape(-1).astype(bool)

  candidate_indices = np.flatnonzero(on_route)
  if candidate_indices.size == 0:
    candidate_indices = np.arange(path_xy.shape[0])

  def _truncate_backward(path: np.ndarray) -> np.ndarray:
    """Truncate path at the first large backward jump (parallel lane merge)."""
    segs = np.diff(path, axis=0)
    lens = np.linalg.norm(segs, axis=1)
    for k in range(1, len(segs)):
      if lens[k] > 3.0 and np.dot(segs[k], segs[k - 1]) < 0:
        return path[: k + 1]
    return path

  # One pass: track best same-lane path (≤ half lane width) and best overall.
  best_path = best_uncapped = None
  best_score = best_uncapped_score = None
  for path_idx in candidate_indices:
    valid_points = path_xy[path_idx][path_valid[path_idx]]
    if valid_points.shape[0] < 2:
      continue
    dists = np.linalg.norm(valid_points - ego_xy, axis=1)
    fwd_proj = (valid_points - ego_xy) @ ego_fwd
    if np.any(fwd_proj >= 0):
      nearest_idx = int(np.argmin(np.where(fwd_proj >= 0, dists, np.inf)))
    else:
      nearest_idx = int(np.argmin(dists))
    suffix = valid_points[nearest_idx:]
    if suffix.shape[0] < 2:
      continue
    min_dist = float(dists[nearest_idx])
    jump_penalty = float(
        np.maximum(np.linalg.norm(np.diff(suffix, axis=0), axis=1) - 3.0, 0.0).sum()
    )
    score = min_dist + 2.0 * jump_penalty
    if best_uncapped_score is None or score < best_uncapped_score:
      best_uncapped_score, best_uncapped = score, suffix
    if min_dist <= 0.5 * LANE_WIDTH_M and (best_score is None or score < best_score):
      best_score, best_path = score, suffix

  if best_path is None and best_uncapped is None:
    raise ValueError("No valid route path found for the current state.")

  # Prefer same-lane path; fall back to uncapped if it truncates too short
  # (HD-map sdc_paths data quality issue in that lane).
  chosen = best_path if best_path is not None else best_uncapped
  chosen = _truncate_backward(chosen)
  if len(chosen) < 20 and best_uncapped is not None:
    chosen = _truncate_backward(best_uncapped)

  xy = resample_polyline(smooth_path_jumps(chosen), spacing_m=1.0)
  # Smooth high-frequency oscillations in intersection geometry without
  # blurring intentional large-scale turns (9m window on 1m-spaced points).
  # Pin endpoints so smoothing doesn't shift the path start away from the
  # vehicle (which would corrupt the Frenet projection at trajectory[0]).
  if xy.shape[0] > 9:
    first, last = xy[0].copy(), xy[-1].copy()
    xy[:, 0] = uniform_filter1d(xy[:, 0], size=9, mode="nearest")
    xy[:, 1] = uniform_filter1d(xy[:, 1], size=9, mode="nearest")
    xy[0], xy[-1] = first, last
  seg_len = np.linalg.norm(np.diff(xy, axis=0), axis=1)
  arc = np.concatenate([[0.0], np.cumsum(seg_len)])
  dx = np.gradient(xy[:, 0], arc, edge_order=1)
  dy = np.gradient(xy[:, 1], arc, edge_order=1)
  heading = np.unwrap(np.arctan2(dy, dx))
  offsets = np.asarray(
      DEFAULT_LANE_OFFSETS if lane_offsets is None else list(lane_offsets),
      dtype=float,
  )
  return ReferencePath(
      xy=xy,
      arc_length=arc,
      heading=heading,
      lane_offsets=offsets,
      goal_xy=xy[-1],
      goal_s=float(arc[-1]),
      spline_x=CubicSpline(arc, xy[:, 0], bc_type="natural"),
      spline_y=CubicSpline(arc, xy[:, 1], bc_type="natural"),
  )
