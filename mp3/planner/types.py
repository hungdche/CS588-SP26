from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ReferencePath:
  """Lane-level reference path sampled in Cartesian space.

  Attributes:
    xy: Sampled polyline in world coordinates of shape `(N, 2)` in meters.
    arc_length: Cumulative path arc length of shape `(N,)` in meters.
    heading: Tangent heading along the path of shape `(N,)` in radians.
    lane_offsets: Candidate target lane offsets of shape `(K,)` in meters.
    goal_xy: Goal point in world coordinates of shape `(2,)` in meters.
    goal_s: Goal arc-length position in meters.
    spline_x: Cubic spline over arc length for x(s).
    spline_y: Cubic spline over arc length for y(s).
  """

  xy: np.ndarray
  arc_length: np.ndarray
  heading: np.ndarray
  lane_offsets: np.ndarray
  goal_xy: np.ndarray
  goal_s: float
  spline_x: Any
  spline_y: Any


@dataclass(frozen=True)
class TrajectorySample:
  """One sampled candidate trajectory.

  Attributes:
    times: Sample times of shape `(T,)` in seconds.
    s: Longitudinal Frenet positions of shape `(T,)` in meters.
    d: Lateral Frenet offsets of shape `(T,)` in meters.
    x: World x positions of shape `(T,)` in meters.
    y: World y positions of shape `(T,)` in meters.
    yaw: World yaw of shape `(T,)` in radians.
    speed: Speed profile of shape `(T,)` in meters/second.
    accel: Longitudinal acceleration proxy of shape `(T,)` in meters/second^2.
    curvature: Path curvature proxy of shape `(T,)` in 1/meters.
    s_jerk: Longitudinal jerk of shape `(T,)` in meters/second^3.
    d_jerk: Lateral jerk of shape `(T,)` in meters/second^3.
    target_offset: Terminal lane offset in meters.
    target_speed: Terminal longitudinal speed in meters/second.
    target_accel: Terminal longitudinal acceleration in meters/second^2.
  """

  times: np.ndarray
  s: np.ndarray
  d: np.ndarray
  x: np.ndarray
  y: np.ndarray
  yaw: np.ndarray
  speed: np.ndarray
  accel: np.ndarray
  curvature: np.ndarray
  s_jerk: np.ndarray
  d_jerk: np.ndarray
  target_offset: float
  target_speed: float
  target_accel: float


@dataclass(frozen=True)
class TrajectoryCost:
  """Scalar trajectory cost with explicit components.

  Attributes:
    total: Total scalar cost.
    goal: Goal progress / remaining distance cost.
    jerk: Combined longitudinal and lateral jerk cost.
    collision: Collision cost.
    longitudinal: Longitudinal headway / spacing cost.
    centerline: Reference path deviation cost.
    tracking: Logged-future tracking cost.
    feasibility: Dynamic-feasibility penalty.
    collided: Whether the candidate overlaps another agent.
  """

  total: float
  goal: float
  jerk: float
  collision: float
  longitudinal: float
  centerline: float
  tracking: float
  feasibility: float
  collided: bool


@dataclass(frozen=True)
class AgentPredictions:
  """Ground-truth future agent states for discrete collision checking.

  Attributes:
    xy: World positions with shape `(T, A, 2)` in meters.
    yaw: Agent yaw with shape `(T, A)` in radians.
    length: Agent lengths with shape `(T, A)` in meters.
    width: Agent widths with shape `(T, A)` in meters.
    valid: Validity mask with shape `(T, A)`.
    ids: Agent ids with shape `(A,)`.
  """

  xy: np.ndarray
  yaw: np.ndarray
  length: np.ndarray
  width: np.ndarray
  valid: np.ndarray
  ids: np.ndarray
