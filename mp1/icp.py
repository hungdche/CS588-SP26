from __future__ import annotations

import math

import numpy as np
import open3d as o3d

from utils.geometry_utils import se3_to_se2


def _to_open3d_cloud(points_xyz: np.ndarray):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
    return cloud


def _prepare_cloud(points: np.ndarray, voxel_size: float, estimate_normals: bool = False):
    finite = np.isfinite(points).all(axis=1)
    xyz = points[finite, :3]
    if len(xyz) == 0:
        return _to_open3d_cloud(np.zeros((0, 3), dtype=np.float64))
    cloud = _to_open3d_cloud(xyz)
    cloud = cloud.voxel_down_sample(voxel_size=max(voxel_size, 0.05))
    if estimate_normals and len(cloud.points) > 0:
        radius = max(voxel_size * 2.5, 0.25)
        cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
        )
    return cloud


def run_pairwise_icp(
    source_points: np.ndarray,
    target_points: np.ndarray,
    voxel_size: float = 0.2,
    max_correspondence_distance: float = 1.5,
    max_iterations: int = 60,
) -> np.ndarray:
    """
    TODO(Task 1): Run pairwise ICP between two point clouds using Open3D.

    Inputs:
        source_points (N, 3): Source point cloud.
        target_points (M, 3): Target point cloud.
        voxel_size: Voxel size for downsampling.
        max_correspondence_distance: Maximum correspondence distance.
        max_iterations: Maximum number of iterations.

    Returns:
        np.ndarray (3,): T_source,target: Relative SE(2) transform [dx, dy, dtheta] from target to source.
    """

    source = _prepare_cloud(source_points, voxel_size, estimate_normals=False)
    target = _prepare_cloud(target_points, voxel_size, estimate_normals=False)
    if len(source.points) == 0 or len(target.points) == 0:
        raise ValueError("ICP received an empty source or target cloud after filtering/downsampling.")

    # ======= STUDENT TODO START (edit only inside this block) =======
    # TODO(student): implement pairwise ICP similar to MP0
    
    # placeholder
    result = o3d.pipelines.registration.RegistrationResult()
    result.transformation = np.eye(4, dtype=np.float32)

    
    # ======= STUDENT TODO END (do not change code outside this block) =======

    rel_se3 = np.asarray(result.transformation, dtype=np.float64)
    rel_se2 = se3_to_se2(rel_se3)
    return rel_se2


def compute_icp_chains(
    static_points: list[np.ndarray],
    voxel_size: float,
    max_correspondence_distance: float,
    max_iterations: int,
) -> dict[str, np.ndarray]:
    """
    TODO(Task 1): Compute ICP chains and motion edges between all frames in the static point cloud sequence.

    Inputs:
        static_points (T, N, 3): List of static point clouds.
        voxel_size: Voxel size for downsampling.
        max_correspondence_distance: Maximum correspondence distance.  
        max_iterations: Maximum number of iterations.

    Returns:
        Dictionary containing:
            "motion_edges": np.ndarray (T-1, 5): Motion edges [t, t+1, dx, dy, dtheta], 
                i.e., list of T_(t+1)t: the relative SE(2) transform from frame t to t+1.
            "trajectory": np.ndarray (T, 3): ICP trajectory in world (first frame) coordinates.
                i.e., list of T_t0: the absolute SE(2) transform from frame 0 to t.
    """ 

    num_frames = len(static_points)
    edges: list[tuple[int, int, float, float, float]] = []

    # ======= STUDENT TODO START (edit only inside this block) =======
    # TODO(student): implement ICP chaining and motion edges
    
    # placeholders
    icp_poses = np.zeros((num_frames, 3), dtype=np.float64)

    # ======= STUDENT TODO END (do not change code outside this block) =======

    return {
        "motion_edges": np.array(edges, dtype=np.float64),
        "trajectory": icp_poses,
    }
