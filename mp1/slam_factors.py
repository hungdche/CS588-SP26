from __future__ import annotations

import math

import numpy as np

from utils.geometry_utils import normalize_angle

def motion_error_and_jacobians(
    pose_i: np.ndarray,
    pose_j: np.ndarray,
    T_ji: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    TODO(Task 2.1): Computes the motion residual and Jacobians using the forward kinematics 
    function in the global frame.

    You might find normalize_angle useful.

    Inputs:
        pose_i (3,): Pose i [x, y, theta].
        pose_j (3,): Pose j [x, y, theta].
        T_ji (3,): ICP measurement [dx, dy, dtheta] from i to j.

    Returns:
        residual: Motion residual .
        J_i: Jacobian matrix of the residual wrt pose_i.
        J_j: Jacobian matrix of the residual wrt pose_j.
    """

    # ======= STUDENT TODO START (edit only inside this block) =======
    # TODO(student): implement the motion error and Jacobians analytically

    # placeholder
    raise NotImplementedError("Not implemented")

    # ======= STUDENT TODO END (do not change code outside this block) =======

    return residual, J_i, J_j


def motion_factor(
    pose_i: np.ndarray,
    pose_j: np.ndarray,
    T_ji: np.ndarray,
    sigma_xy: float,
    sigma_theta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    # Compute the raw global-frame error and Jacobians
    residual, J_i, J_j = motion_error_and_jacobians(pose_i, pose_j, T_ji)

    # Apply covariance weighting (Information matrix)
    residual[:2] /= sigma_xy
    residual[2] /= sigma_theta
    J_i[:2] /= sigma_xy
    J_i[2] /= sigma_theta
    J_j[:2] /= sigma_xy
    J_j[2] /= sigma_theta
    
    return residual, J_i, J_j

def distance_factor(
    pose_t: np.ndarray,
    landmark_i: np.ndarray,
    z_ti: float,
    sigma_d: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx = landmark_i[0] - pose_t[0]
    dy = landmark_i[1] - pose_t[1]
    dist = max(math.hypot(dx, dy), 1e-9)
    residual = np.array([(dist - z_ti) / sigma_d], dtype=np.float64)

    inv_dist = 1.0 / dist
    J_pose = np.array([[-dx * inv_dist, -dy * inv_dist, 0.0]], dtype=np.float64) / sigma_d
    J_land = np.array([[dx * inv_dist, dy * inv_dist]], dtype=np.float64) / sigma_d
    return residual, J_pose, J_land
