from __future__ import annotations

import numpy as np
from utils.geometry_utils import MapSpec

def world_to_pixel(xy: np.ndarray, map_spec: MapSpec) -> np.ndarray:
    """TODO(Task 0): Convert world coordinates to pixel coordinates.
    
    Inputs:
        xy: (N, 2) array: Points in world (first frame) coordinates [x, y].
        map_spec: Map specification.
        - map_spec.x_min: Minimum x-coordinate of the map.
        - map_spec.y_max: Maximum y-coordinate of the map.
        - map_spec.resolution: Resolution of the map.
    
    Returns:
        (N, 2) array: Points in pixel coordinates [row, col].
    """

    # ======= STUDENT TODO START (edit only inside this block) =======
    # TODO(student): implement world_to_pixel
    # 1) recenter the points by subtracting the x_min and y_max
    #   1) compute cols by subtracting wtih the x_min 
    #   2) compute rows by subtracting from the y_max
    # 2) divide by the result by the resolution
    # 3) floor the normalized coordinates to get the pixel coordinates, np.floor might be helpful
    # 4) use np.int64 for the pixel coordinates
    # 5) make sure to avoid using for loops

    # placeholders 
    cols = np.zeros_like(xy[:, 0], dtype=np.int64)
    rows = np.zeros_like(xy[:, 1], dtype=np.int64)

    # ======= STUDENT TODO END (do not change code outside this block) =======

    return np.stack([rows, cols], axis=1)

def rasterize_topdown(points_world: list[np.ndarray], map_spec: MapSpec) -> dict[str, np.ndarray]:
    """TODO(Task 0): Rasterize points into a top-down map.
    
    Inputs:
        points_world: List of points in world (first frame) coordinates.
        map_spec: Map specification.
    
    Returns:
        Dictionary containing density map.
    """
    shape = (map_spec.height, map_spec.width)
    density = np.zeros(shape, dtype=np.float32)

    for pts in points_world:
        if pts.size == 0:
            continue

        xy = pts[:, :2].astype(np.float64, copy=False)

        # ======= STUDENT TODO START (edit only inside this block) =======
        # TODO(student): compute the valid rows and columns
        # 1) implement world_to_pixel: compute the rows and columns in pixel coordinates
        # 2) filter out the points that are outside the map
        
        # placeholders
        rc = world_to_pixel(xy, map_spec)
        rows = rc[:, 0]
        cols = rc[:, 1]

        # ======= STUDENT TODO END (do not change code outside this block) =======
        
        np.add.at(density, (rows, cols), 1.0)

    return {
        "density": density,
    }

def build_accumulated_map(static_points: list[np.ndarray], poses_se2: np.ndarray) -> list[np.ndarray]:
    """TODO(Task 0): Build accumulated map.
    
    Inputs:
        static_points: List of point clouds in current frame coordinates.
        poses_se2: List of SE(2) poses from current frame coordinates to world (first frame) coordinates.
    
    Returns:
        List of accumulated points in world (first frame) coordinates.
    """
    
    # ======= STUDENT TODO START (edit only inside this block) =======
    # TODO(student): implement build_accumulated_map
    
    # placeholders
    world_points = [static_points[0]]

    # ======= STUDENT TODO END (do not change code outside this block) =======

    return world_points

def accumulate_and_rasterize(static_points: list[np.ndarray], poses_se2: np.ndarray, map_spec: MapSpec) -> dict[str, np.ndarray]:
    world_points = build_accumulated_map(static_points, poses_se2)
    return rasterize_topdown(world_points, map_spec)