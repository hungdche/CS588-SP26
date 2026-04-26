"""
Task 3: Pure Pursuit Controller
================================

Implement a Pure Pursuit geometric path tracker that follows the best-cost
Frenet trajectory under the InvertibleBicycleModel dynamics.

Pure Pursuit equations:
    1. Lookahead point L: first waypoint at distance >= lookahead_dist from ego.
    2. Local-frame offset:
           dx_loc =  (Lx - x)*cos(psi) + (Ly - y)*sin(psi)
           dy_loc = -(Lx - x)*sin(psi) + (Ly - y)*cos(psi)
    3. alpha = atan2(dy_loc, dx_loc)
    4. kappa = 2*sin(alpha) / lookahead_dist
    5. a     = clip(k_p*(v_target - v), a_min, a_max)

Action format for Waymax InvertibleBicycleModel:
    action.data  : (num_objects, 2)  = [acceleration (m/s2), steering_curvature (1/m)]
    action.valid : (num_objects, 1)  = True for the SDC
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PurePursuitConfig:
    lookahead_dist: float = 10.0
    speed_gain:     float = 1.0
    max_accel:      float = 3.0
    min_accel:      float = -5.0
    max_curvature:  float = 0.5
    dt:             float = 0.1


class PurePursuitController:
    """Pure Pursuit path tracker for the SDC."""

    def __init__(self, config: PurePursuitConfig = None):
        self.cfg = config or PurePursuitConfig()

    def find_lookahead_point(
        self,
        ego_xy:      np.ndarray,
        path_xy:     np.ndarray,
        path_speeds: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """TODO(Task 3.1): Return the first waypoint at least lookahead_dist away.

        Compute the Euclidean distance from the ego position to every waypoint
        in path_xy, then select the first one whose distance is at least
        self.cfg.lookahead_dist.  If no waypoint is far enough, fall back to
        the last waypoint in path_xy.

        Args:
            ego_xy: Current ego position (2,) in global frame.
            path_xy: (T, 2) future waypoints in global frame.
            path_speeds: (T,) speed at each waypoint (m/s).

        Returns:
            lookahead_xy:    (2,)  global [x, y] of the lookahead point.
            lookahead_speed: float speed (m/s) at that waypoint.
        """
        # ======= STUDENT TODO START (edit only inside this block) =======
        # TODO(student): implement find_lookahead_point
        #
        # Steps:
        #   1. Compute Euclidean distances from ego_xy to every row of path_xy.
        #   2. Find all indices where the distance is at least lookahead_dist.
        #   3. Select the first such index; if none exists, use the last waypoint.
        #   4. Return the waypoint position and speed at that index.

        # placeholder -- always returns the last waypoint
        return path_xy[-1], float(path_speeds[-1])
        # ======= STUDENT TODO END (do not change code outside this block) =======

    def compute_steering(
        self,
        ego_xy:       np.ndarray,
        ego_yaw:      float,
        lookahead_xy: np.ndarray,
    ) -> float:
        """TODO(Task 3.2): Compute steering curvature using the Pure Pursuit formula.

        Translate the lookahead point into the ego's local frame by rotating the
        displacement vector by -ego_yaw.  Compute the lateral angle alpha between
        the ego's heading and the lookahead direction, then apply the Pure Pursuit
        geometric relationship kappa = 2·sin(alpha) / lookahead_dist.  Clamp the
        result to [-max_curvature, +max_curvature].

        Args:
            ego_xy: Ego position (2,) in global frame.
            ego_yaw: Ego heading (rad).
            lookahead_xy: Lookahead point (2,) in global frame.

        Returns:
            Steering curvature (1/m); positive = turn left.
        """
        # ======= STUDENT TODO START (edit only inside this block) =======
        # TODO(student): implement compute_steering
        #
        # Steps:
        #   1. Compute the displacement vector from ego_xy to lookahead_xy.
        #   2. Rotate that vector into the ego's local frame using ego_yaw
        #      (so that straight ahead becomes the positive x-axis locally).
        #   3. Compute the lateral angle alpha using atan2 of the local y over x.
        #   4. Apply the Pure Pursuit formula: kappa = 2 * sin(alpha) / lookahead_dist.
        #   5. Clamp kappa to [-max_curvature, +max_curvature] and return it.

        # placeholder -- zero steering (vehicle drives straight)
        return 0.0
        # ======= STUDENT TODO END (do not change code outside this block) =======

    def compute_acceleration(
        self,
        current_speed: float,
        target_speed:  float,
    ) -> float:
        """TODO(Task 3.3): Proportional speed controller.

        Compute an acceleration command proportional to the error between
        target_speed and current_speed, scaled by self.cfg.speed_gain, then
        clamp it to [self.cfg.min_accel, self.cfg.max_accel].

        Args:
            current_speed: Current ego speed (m/s).
            target_speed: Target speed at the next waypoint (m/s).

        Returns:
            Acceleration command (m/s²); positive = speed up, negative = brake.
        """
        # ======= STUDENT TODO START (edit only inside this block) =======
        # TODO(student): implement compute_acceleration
        #
        # Steps:
        #   1. Compute the speed error as (target_speed - current_speed).
        #   2. Scale by speed_gain to get a raw acceleration command.
        #   3. Clamp to [min_accel, max_accel] and return.

        # placeholder -- zero acceleration (vehicle coasts to a stop)
        return 0.0
        # ======= STUDENT TODO END (do not change code outside this block) =======
