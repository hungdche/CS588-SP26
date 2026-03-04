from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slam_factors import distance_factor, motion_factor


@dataclass
class GraphSlamProblem:
    initial_poses: np.ndarray
    initial_landmarks: np.ndarray
    motion_edges: np.ndarray
    observations: np.ndarray
    sigma_xy: float
    sigma_theta: float
    sigma_d: float

    @property
    def num_poses(self) -> int:
        return int(self.initial_poses.shape[0])

    @property
    def num_landmarks(self) -> int:
        return int(self.initial_landmarks.shape[0])

    @property
    def state_dim(self) -> int:
        return 3 * (self.num_poses - 1) + 2 * self.num_landmarks

    @property
    def residual_dim(self) -> int:
        return (
            3 * int(self.motion_edges.shape[0])
            # + 3 * max(self.num_poses - 2, 0)
            + int(self.observations.shape[0])
        )

    def pack_state(self, poses: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        return np.concatenate([poses[1:].reshape(-1), landmarks.reshape(-1)], axis=0)

    def unpack_state(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pose_block = 3 * (self.num_poses - 1)
        poses = self.initial_poses.copy()
        if pose_block > 0:
            poses[1:] = state[:pose_block].reshape(self.num_poses - 1, 3)
        landmarks = state[pose_block:].reshape(self.num_landmarks, 2).copy()
        return poses, landmarks

    def pose_col(self, pose_idx: int) -> int | None:
        if pose_idx == 0:
            return None
        return 3 * (pose_idx - 1)

    def landmark_col(self, landmark_idx: int) -> int:
        return 3 * (self.num_poses - 1) + 2 * landmark_idx


def build_linear_system(problem: GraphSlamProblem, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    poses, landmarks = problem.unpack_state(state)
    J = np.zeros((problem.residual_dim, problem.state_dim), dtype=np.float64)
    r = np.zeros((problem.residual_dim,), dtype=np.float64)

    row = 0

    # Motion edges: [t, t+1, dx, dy, dtheta]
    # where t is the frame index and t+1 is the next frame index
    # dx, dy, dtheta is the relative SE(2) transform from frame t to t+1, i.e., T_t+1,t
    for edge in problem.motion_edges:
        pose_i = int(edge[0])
        pose_j = int(edge[1])
        T_ji = edge[2:5]
        res, J_i, J_j = motion_factor(
            poses[pose_i],
            poses[pose_j],
            T_ji,
            sigma_xy=problem.sigma_xy,
            sigma_theta=problem.sigma_theta,
        )
        r[row:row + 3] = res
        col_i = problem.pose_col(pose_i)
        col_j = problem.pose_col(pose_j)
        if col_i is not None:
            J[row:row + 3, col_i:col_i + 3] = J_i
        if col_j is not None:
            J[row:row + 3, col_j:col_j + 3] = J_j
        row += 3

    # Observations: [landmark_idx, pose_idx, distance]
    # where landmark_idx is the index of the landmark and pose_idx is the index of the pose
    # distance is the distance between the landmark and the pose, i.e., z_ti
    for obs in problem.observations:
        landmark_idx = int(obs[0])
        pose_idx = int(obs[1])
        z_ti = float(obs[2])
        res, J_pose, J_land = distance_factor(
            poses[pose_idx],
            landmarks[landmark_idx],
            z_ti,
            sigma_d=problem.sigma_d,
        )
        r[row] = res[0]
        col_pose = problem.pose_col(pose_idx)
        col_land = problem.landmark_col(landmark_idx)
        if col_pose is not None:
            J[row:row + 1, col_pose:col_pose + 3] = J_pose
        J[row:row + 1, col_land:col_land + 2] = J_land
        row += 1

    return r, J, poses, landmarks


def solve_graph_slam(
    problem: GraphSlamProblem,
    max_iterations: int = 20,
    cost_tol: float = 1e-6,
    damping: float = 1e-6,
) -> dict[str, np.ndarray | list[float]]:
    """
    TODO(Task 2.2): Implement Gauss-Newton Graph-SLAM optimization.

    Inputs:
        problem: GraphSlamProblem instance containing the problem data.
        max_iterations: Maximum number of iterations.
        cost_tol: Relative cost tolerance for convergence.
        damping: Damping factor for the normal equations.

    Returns:
        Dictionary containing the optimized poses, landmarks, cost history, and Jacobian.
        - poses: Array of shape (T, 3) containing the optimized poses in world (first frame) coordinates.
        - landmarks: Array of shape (L, 2) containing the optimized landmarks.
        - cost_history: List of floats containing the cost history.
        - jacobian: Array of shape (R, C) containing the Jacobian.
        - final_residual: Array of shape (R,) containing the final residual.
    """
    state = problem.pack_state(problem.initial_poses, problem.initial_landmarks)
    cost_history: list[float] = []
    first_jacobian = None

    for _ in range(max_iterations):
        r, J, poses, landmarks = build_linear_system(problem, state)
        if first_jacobian is None:
            first_jacobian = J.copy()

        # Compute cost at current state
        cost = 0.5 * float(np.sum(r * r))
        cost_history.append(cost)

        # ======= STUDENT TODO START (edit only inside this block) =======
        # TODO(student): implement one Gauss-Newton update step
        # 1) Build linear system Adx = b, where A = J^T J and b = -J^T r
        # 2) Add damping to A diagonal, A = A + damping * I
        # 3) Solve for dx , you might find np.linalg.solve useful
        # 4) Form candidate_state = state + dx and evaluate candidate cost 
        # 5) Done for you: Assign candidate_state to state and break early if the new cost is similar to the current cost
        # 6) Optional: Add a small step size to dx if needed 
        
        # placeholder
        candidate_state = state
        new_cost = cost + 1.0

        # ======= STUDENT TODO END (do not change code outside this block) =======
        
        state = candidate_state
        if abs(cost - new_cost) / max(cost, 1.0) < cost_tol:
            cost_history.append(new_cost)
            break

    final_r, final_J, final_poses, final_landmarks = build_linear_system(problem, state)
    final_cost = 0.5 * float(np.sum(final_r * final_r))
    if not cost_history or cost_history[-1] != final_cost:
        cost_history.append(final_cost)

    return {
        "poses": final_poses,
        "landmarks": final_landmarks,
        "cost_history": np.array(cost_history, dtype=np.float64),
        "jacobian": first_jacobian if first_jacobian is not None else final_J,
        "final_residual": final_r,
    }