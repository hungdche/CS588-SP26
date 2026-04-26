"""
Task 2: Cost Evaluator
=======================

Implement cost functions that score the 504 candidate Frenet trajectories.
The evaluator returns a scalar cost for each candidate; the planner selects
the trajectory with the lowest total cost.

Required cost terms (6 pts total):
    collision   — hard penalty when the trajectory overlaps an agent bounding box
    goal        — remaining arc-length to a local goal; penalises short progress
    jerk        — mean squared Frenet jerk; penalises abrupt control changes
    feasibility — soft quadratic penalty for speed / accel / curvature limit violations
    centerline  — penalises lateral drift from the lane centre
    evaluate    — weighted assembly of all terms into a (N,) cost array

Total cost formula:
    C(τ) = w_collision   · c_collision(τ)
         + w_goal        · c_goal(τ)
         + w_jerk        · c_jerk(τ)
         + w_feasibility · c_feasibility(τ)
         + w_centerline  · c_centerline(τ)

Helpers in planner/collision.py (do NOT re-implement):
    geometric_collision_mask(trajectory, predictions, ego_size, ego_id)
        → boolean mask (T,) — True where SDC box overlaps any agent box.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from waymax import datatypes

from planner.collision import geometric_collision_mask       # noqa: F401 — used in TODO 2.1
from planner.types import AgentPredictions
from planner.types import ReferencePath
from planner.types import TrajectorySample


@dataclass
class CostConfig:
    """Scalar weights for each cost term.

    Tune these after implementing the individual terms.
    A good starting point is to make w_collision very large so the safety
    filter dominates, and w_goal large enough to drive forward progress.

    Attributes:
        w_collision:   Weight for the hard collision cost.
        w_goal:        Weight for the goal-progress cost.
        w_jerk:        Weight for the jerk cost.
        w_feasibility: Weight for the feasibility cost.
        w_centerline:  Weight for the centerline deviation cost.
    """
    w_collision:   float = 1.0
    w_goal:        float = 1.0
    w_jerk:        float = 1.0
    w_feasibility: float = 1.0
    w_centerline:  float = 1.0


# ── TODO 2.1 — Collision cost ──────────────────────────────────────────────────

def collision_cost(
    trajectory:  TrajectorySample,
    predictions: AgentPredictions,
    ego_size:    tuple[float, float],
    ego_id:      int,
) -> float:
    """TODO(Task 2.1): Hard collision cost — 1.0 if any overlap occurs, else 0.0.

    Use geometric_collision_mask from planner.collision to obtain a boolean
    mask over all trajectory timesteps indicating oriented bounding-box overlap
    with predicted agent boxes.  Do NOT re-implement box geometry.
    Return 1.0 if any timestep has an overlap, 0.0 otherwise.

    Args:
        trajectory: One candidate trajectory.
        predictions: Future agent states (T, A, ...).
        ego_size: Ego (length, width) in metres.
        ego_id: Ego object ID to exclude from self-collision checks.

    Returns:
        1.0 if any overlap is detected, 0.0 otherwise.
    """
    # ======= STUDENT TODO START (edit only inside this block) =======
    # TODO(student): implement collision_cost
    #
    # Steps:
    #   1. Call geometric_collision_mask with the trajectory, predictions,
    #      ego_size, and ego_id to get a boolean mask over timesteps.
    #   2. Return 1.0 if any element of the mask is True, else 0.0.

    # placeholder — always reports no collision (unsafe — implement this first!)
    return 0.0
    # ======= STUDENT TODO END (do not change code outside this block) =======


# ── TODO 2.2 — Goal progress cost ─────────────────────────────────────────────

def goal_cost(
    trajectory:     TrajectorySample,
    reference_path: ReferencePath,
) -> float:
    """TODO(Task 2.2): Goal progress cost — remaining arc-length to local goal.

    Clip the local goal to at most 35 m ahead of the trajectory start position
    (to avoid penalising long-range goals the horizon cannot reach), then
    return the non-negative shortfall: how far the trajectory end falls short
    of that clipped goal.  A trajectory that reaches or passes the goal gets
    a cost of zero.

    Args:
        trajectory: One candidate trajectory (Frenet s in trajectory.s).
        reference_path: Lane-level reference path (goal arc-length in goal_s).

    Returns:
        Non-negative remaining distance to the local goal (m).
    """
    # ======= STUDENT TODO START (edit only inside this block) =======
    # TODO(student): implement goal_cost
    #
    # Steps:
    #   1. Read the start and end arc-length values from trajectory.s.
    #   2. Clip reference_path.goal_s to at most 35 m ahead of the start.
    #   3. Return the non-negative shortfall between the clipped goal and the end.

    # placeholder — zero cost (planner receives no incentive to make progress)
    return 0.0
    # ======= STUDENT TODO END (do not change code outside this block) =======


# ── TODO 2.3 — Jerk cost ──────────────────────────────────────────────────────

def jerk_cost(trajectory: TrajectorySample) -> float:
    """TODO(Task 2.3): Mean squared Frenet jerk.

    Penalise abrupt longitudinal and lateral control changes.  The
    trajectory already stores pre-computed third-derivative arrays
    trajectory.s_jerk (longitudinal) and trajectory.d_jerk (lateral).
    Return the mean of their squared sum across all timesteps.

    Args:
        trajectory: One candidate trajectory.

    Returns:
        Mean squared jerk scalar (m²/s⁶).
    """
    # ======= STUDENT TODO START (edit only inside this block) =======
    # TODO(student): implement jerk_cost
    #
    # Steps:
    #   1. Square both trajectory.s_jerk and trajectory.d_jerk element-wise.
    #   2. Sum the squared arrays and take the mean across all timesteps.
    #   3. Return the result as a float.

    # placeholder — zero jerk cost
    return 0.0
    # ======= STUDENT TODO END (do not change code outside this block) =======


# ── TODO 2.4 — Feasibility cost ───────────────────────────────────────────────

def feasibility_cost(trajectory: TrajectorySample) -> float:
    """TODO(Task 2.4): Soft quadratic penalty for physical limit violations.

    Apply a separate mean squared penalty for each of three physical limits:
    speed (max 25 m/s), absolute acceleration (max 6 m/s²), and absolute
    curvature (max 0.25 m⁻¹).  Each penalty is zero when the limit is
    satisfied and grows quadratically with the excess.  Return the sum of
    all three penalties.  Use trajectory.speed, trajectory.accel, and
    trajectory.curvature.

    Args:
        trajectory: One candidate trajectory (speed, accel, curvature arrays).

    Returns:
        Non-negative feasibility penalty scalar.
    """
    # ======= STUDENT TODO START (edit only inside this block) =======
    # TODO(student): implement feasibility_cost
    #
    # Steps:
    #   1. For each limit (speed > 25 m/s, |accel| > 6 m/s², |curvature| > 0.25 m⁻¹),
    #      compute the excess over the limit (clamp negatives to zero).
    #   2. Square each excess and take the mean over all timesteps.
    #   3. Return the sum of all three mean-squared penalties.

    # placeholder — zero feasibility cost
    return 0.0
    # ======= STUDENT TODO END (do not change code outside this block) =======


# ── TODO 2.5 — Centerline cost ────────────────────────────────────────────────

def centerline_cost(
    trajectory:     TrajectorySample,
    reference_path: ReferencePath,
) -> float:
    """TODO(Task 2.6): Centerline deviation cost — penalises lateral drift.

    Penalise trajectories that drift away from the lane centre using the
    Frenet lateral offset d (trajectory.d) at each timestep.  Add a second
    term that penalises large terminal lane-change targets (trajectory.target_offset)
    to discourage aggressive offset commands.  The second term is weighted at 0.5.

    Args:
        trajectory: One candidate trajectory (Frenet d in trajectory.d,
                    target lateral offset in trajectory.target_offset).
        reference_path: Unused; included for API consistency.

    Returns:
        Non-negative centerline deviation scalar (m²).
    """
    # ======= STUDENT TODO START (edit only inside this block) =======
    # TODO(student): implement centerline_cost
    #
    # Steps:
    #   1. Compute the mean of trajectory.d squared over all timesteps.
    #   2. Add 0.5 times trajectory.target_offset squared as a terminal penalty.
    #   3. Return the sum as a float.

    # placeholder — zero centerline cost
    return 0.0
    # ======= STUDENT TODO END (do not change code outside this block) =======


# ── TODO 2.6 — Weighted cost assembly ─────────────────────────────────────────

def evaluate(
    trajectories:   list[TrajectorySample],
    predictions:    AgentPredictions,
    reference_path: ReferencePath,
    state:          datatypes.SimulatorState,
    weights:        CostConfig = None,
) -> np.ndarray:
    """TODO(Task 2.7): Assemble the weighted total cost for each candidate.

    Call each of the six cost functions above for every trajectory, multiply
    each term by its weight from CostConfig, sum them, and return a 1-D array
    of length N.  The planner calls argmin on this array to select the best
    trajectory.

    To read the ego geometry needed by collision_cost,
    extract the SDC index from state.object_metadata.is_sdc, then read length
    and width from state.sim_trajectory at the current timestep, and the
    object ID from state.object_metadata.ids.

    Args:
        trajectories: List of N candidate TrajectorySample objects.
        predictions: Future agent states for collision checking.
        reference_path: Lane-level reference path (needed for goal_cost and centerline_cost).
        state: Current simulator state (used to read ego geometry and id).
        weights: Cost weights (uses CostConfig defaults if None).

    Returns:
        costs: NumPy array of shape (N,) with one scalar cost per candidate.
               Lower is better.
    """
    # ======= STUDENT TODO START (edit only inside this block) =======
    # TODO(student): implement evaluate
    #
    # Steps:
    #   1. Default weights to CostConfig() if none are provided.
    #   2. Extract ego_size (length, width) and ego_id from state:
    #      find the SDC index via object_metadata.is_sdc, then read length and
    #      width from sim_trajectory at the current timestep, and the id from
    #      object_metadata.ids.
    #   3. For each trajectory, call all six cost functions and multiply each
    #      result by its corresponding weight from the CostConfig.
    #   4. Sum the six weighted terms to get the total cost for that trajectory.
    #   5. Return a NumPy array of shape (N,) with one total cost per candidate.

    # placeholder
    return np.zeros(len(trajectories), dtype=float)
    # ======= STUDENT TODO END (do not change code outside this block) =======
