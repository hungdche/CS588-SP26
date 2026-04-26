from __future__ import annotations

import numpy as np

from planner.types import ReferencePath


def evaluate_reference_path(
    reference_path: ReferencePath,
    s_query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Evaluates the reference path at arc-length positions.

  Args:
    reference_path: Smooth centerline parameterized by arc length.
    s_query: Query arc lengths of shape `(N,)` in meters.

  Returns:
    Tuple `(xy, heading, normal)` with shapes `(N, 2)`, `(N,)`, and `(N, 2)`.
  """
  s_clipped = np.clip(s_query, 0.0, reference_path.goal_s)
  x = reference_path.spline_x(s_clipped)
  y = reference_path.spline_y(s_clipped)
  dx = reference_path.spline_x(s_clipped, 1)
  dy = reference_path.spline_y(s_clipped, 1)
  heading = np.unwrap(np.arctan2(dy, dx))
  tangent = np.stack([np.cos(heading), np.sin(heading)], axis=-1)
  normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=-1)
  return np.stack([x, y], axis=-1), heading, normal


def project_to_frenet(
    reference_path: ReferencePath,
    xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Projects Cartesian points to approximate Frenet `(s, d)` coordinates.

  Args:
    reference_path: Smooth centerline parameterized by arc length.
    xy: Cartesian points with shape `(N, 2)` in meters.

  Returns:
    Tuple `(s, d)` with shapes `(N,)` and `(N,)` in meters.
  """
  ref_xy = reference_path.xy
  diffs = xy[:, None, :] - ref_xy[None, :, :]
  nearest_idx = np.argmin(np.linalg.norm(diffs, axis=-1), axis=1)
  nearest_xy = ref_xy[nearest_idx]
  nearest_heading = reference_path.heading[nearest_idx]
  normal = np.stack([-np.sin(nearest_heading), np.cos(nearest_heading)], axis=-1)
  lateral = np.sum((xy - nearest_xy) * normal, axis=-1)
  return reference_path.arc_length[nearest_idx], lateral


def frenet_to_cartesian(
    reference_path: ReferencePath,
    s: np.ndarray,
    d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Converts Frenet `(s, d)` samples back to world coordinates.

  Args:
    reference_path: Smooth centerline parameterized by arc length.
    s: Longitudinal Frenet positions of shape `(N,)` in meters.
    d: Lateral Frenet offsets of shape `(N,)` in meters.

  Returns:
    Tuple `(xy, heading)` with shapes `(N, 2)` and `(N,)`.
  """
  center_xy, center_heading, normal = evaluate_reference_path(reference_path, s)
  xy = center_xy + normal * d[:, None]
  return xy, center_heading

