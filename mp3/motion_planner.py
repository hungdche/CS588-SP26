"""
motion_planner.py — Frenet-based motion planner integration
============================================================

Integrates Tasks 1, 2, and 3 into a single planning loop.

The planner drives the SDC via Frenet-based sampling (Tasks 1+2) and
executes the chosen trajectory with Pure Pursuit (Task 3).

Waymax action format (StateDynamics):
    action.data  : (num_objects, 5)  = [x, y, yaw, vel_x, vel_y]
    action.valid : (num_objects, 1)
"""

import os
import sys
import numpy as np
import jax.numpy as jnp
from waymax import datatypes

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from planner.reference_path import build_reference_path
from planner.collision import extract_agent_predictions
from student.trajectory_sampler import sample_trajectories
from student.cost_evaluator import evaluate 


def _score_trajectories(trajectories, ref_path, predictions, state):
    return evaluate(trajectories, predictions, ref_path, state)

# ── Frenet action conversion ───────────────────────────────────────────────────

def _frenet_traj_to_action(
    trajectory,      # TrajectorySample
    sample_index: int,
) -> np.ndarray:
    """Extract [x, y, yaw, vx, vy] from a TrajectorySample at sample_index."""
    idx = min(sample_index, len(trajectory.x) - 1)
    yaw = float(trajectory.yaw[idx])
    spd = float(trajectory.speed[idx])
    return np.array([
        trajectory.x[idx],
        trajectory.y[idx],
        yaw,
        spd * np.cos(yaw),
        spd * np.sin(yaw),
    ], dtype=np.float32)


# ── Main planner ───────────────────────────────────────────────────────────────

class MotionPlanner:
    """
    Frenet sampling-based planner for the SDC.

    Parameters
    ----------
    planning_horizon_s : float
        Trajectory planning horizon in seconds.
    replan_every : int
        Re-plan every this many Waymax steps (1 = every step).
    """

    def __init__(
        self,
        planning_horizon_s: float = 3.0,
        replan_every: int = 5,
    ):
        self.planning_horizon_s = planning_horizon_s
        self.replan_every = replan_every
        self._step_count = 0
        self._active_trajectory = None
        self._active_sample_index = 0

        # Cache for visualization
        self.last_trajectories    = None
        self.last_costs           = None
        self.last_best_index      = None
        self.last_reference_path  = None
        self.last_active_index    = 0
        self.last_chosen_traj     = None
        self.last_is_replan       = False
        self.last_raw_costs       = None

    def plan(self, state) -> datatypes.Action:
        """
        Compute a Waymax-compatible Action for the current state.

        Returns
        -------
        action : waymax.datatypes.Action
            data  : (num_objects, 5)  float32
            valid : (num_objects, 1)  bool
        """
        num_objects  = int(state.object_metadata.is_sdc.shape[0])
        action_data  = np.zeros((num_objects, 5), dtype=np.float32)
        action_valid = np.zeros((num_objects, 1), dtype=bool)

        sdc_mask = np.asarray(state.object_metadata.is_sdc).astype(bool)
        sdc_idx  = int(np.flatnonzero(sdc_mask)[0])

        self._plan(state, sdc_idx, action_data, action_valid)
        self._step_count += 1
        return datatypes.Action(
            data  = jnp.array(action_data),
            valid = jnp.array(action_valid),
        )

    def _plan(self, state, sdc_idx, action_data, action_valid):
        should_replan = (
            self._active_trajectory is None
            or self._step_count % self.replan_every == 0
            or self._active_sample_index >= len(self._active_trajectory.times) - 1
        )
        if should_replan:
            try:
                ref_path = build_reference_path(state)
            except ValueError:
                ref_path = self.last_reference_path
            if ref_path is None:
                return

            num_samples = int(round(self.planning_horizon_s / 0.1)) + 1
            trajectories = sample_trajectories(
                state, ref_path,
                horizon_s=self.planning_horizon_s,
                num_samples=num_samples,
            )
            if not trajectories:
                return

            predictions = extract_agent_predictions(state, horizon_steps=num_samples)
            totals      = _score_trajectories(trajectories, ref_path, predictions, state)
            best_idx    = int(np.argmin(totals))

            self.last_trajectories    = trajectories
            self.last_costs           = None
            self.last_best_index      = best_idx
            self.last_reference_path  = ref_path
            self.last_raw_costs       = totals
            self._active_trajectory   = trajectories[best_idx]
            self._active_sample_index = 0
            self.last_is_replan       = True
        else:
            self.last_is_replan = False

        self.last_chosen_traj    = self._active_trajectory
        self.last_active_index   = self._active_sample_index
        self._active_sample_index += 1
        idx = min(self._active_sample_index, len(self._active_trajectory.times) - 1)
        action_data[sdc_idx]  = _frenet_traj_to_action(self._active_trajectory, idx)
        action_valid[sdc_idx] = True
