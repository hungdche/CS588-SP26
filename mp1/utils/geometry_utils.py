from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def normalize_angle(theta: float | np.ndarray) -> float | np.ndarray:
    """Wrap angles to (-pi, pi]."""
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def rotation_matrix_z(yaw: float) -> np.ndarray:
    """Build a 3x3 rotation matrix for a yaw angle about +Z.

    Inputs:
        yaw: Rotation angle in radians.

    Returns:
        np.ndarray: 3x3 SO(3) rotation matrix for yaw-only rotation.
    """
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rotx(angle: float) -> np.ndarray:
    """Build a 3x3 rotation matrix for rotation about +X axis.

    Inputs:
        angle: Rotation angle in radians.

    Returns:
        np.ndarray: 3x3 SO(3) rotation matrix Rx(angle).
    """
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float64,
    )


def roty(angle: float) -> np.ndarray:
    """Build a 3x3 rotation matrix for rotation about +Y axis.

    Inputs:
        angle: Rotation angle in radians.

    Returns:
        np.ndarray: 3x3 SO(3) rotation matrix Ry(angle).
    """
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )


def rotz(angle: float) -> np.ndarray:
    """Build a 3x3 rotation matrix for rotation about +Z axis.

    Inputs:
        angle: Rotation angle in radians.

    Returns:
        np.ndarray: 3x3 SO(3) rotation matrix Rz(angle).
    """
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rigid_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Construct a homogeneous SE(3) transform from rotation and translation.

    Inputs:
        R: 3x3 rotation matrix.
        t: Translation vector-like with 3 values [tx, ty, tz].

    Returns:
        np.ndarray: 4x4 homogeneous transform matrix.
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Compute inverse of a rigid 4x4 homogeneous transform.

    Inputs:
        T: 4x4 SE(3) transform matrix.

    Returns:
        np.ndarray: 4x4 inverse transform matrix.
    """
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def transform_points_se3(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply an SE(3) transform to point cloud coordinates.

    Inputs:
        points: Array with at least 3 columns (x, y, z, ...).
        T: 4x4 SE(3) transform matrix.

    Returns:
        np.ndarray: Transformed copy of finite points; xyz columns are updated
        and extra columns (if any) are preserved.
    """
    if points.size == 0:
        return points.copy()
    finite = np.isfinite(points[:, :3]).all(axis=1)
    points = points[finite].copy()
    xyz = points[:, :3]
    R = T[:3, :3]
    t = T[:3, 3]
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    xyz_t = np.stack(
        [
            R[0, 0] * x + R[0, 1] * y + R[0, 2] * z + t[0],
            R[1, 0] * x + R[1, 1] * y + R[1, 2] * z + t[1],
            R[2, 0] * x + R[2, 1] * y + R[2, 2] * z + t[2],
        ],
        axis=1,
    )
    out = points
    out[:, :3] = xyz_t
    return out


def extract_yaw(T: np.ndarray) -> float:
    """Extract yaw angle from a rotation/pose matrix.

    Inputs:
        T: Matrix containing 2D heading in top-left rotation block.

    Returns:
        float: Yaw angle in radians computed as atan2(T[1,0], T[0,0]).
    """
    return math.atan2(T[1, 0], T[0, 0])


def se2_to_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert SE(2) pose vector [x, y, theta] to 3x3 homogeneous matrix.

    Inputs:
        pose: Array-like [x, y, theta].

    Returns:
        np.ndarray: 3x3 homogeneous SE(2) matrix.
    """
    x, y, theta = pose
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array(
        [
            [c, -s, x],
            [s, c, y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

def matrix_to_se2(T: np.ndarray) -> np.ndarray:
    """Convert 3x3 homogeneous SE(2) matrix to [x, y, theta].

    Inputs:
        T: 3x3 homogeneous SE(2) matrix.

    Returns:
        np.ndarray: Pose vector [x, y, theta].
    """
    return np.array([T[0, 2], T[1, 2], math.atan2(T[1, 0], T[0, 0])], dtype=np.float64)


def se3_to_se2(T: np.ndarray) -> np.ndarray:
    """Project an SE(3) transform to planar SE(2) components.

    Inputs:
        T: 4x4 SE(3) transform.

    Returns:
        np.ndarray: [x, y, yaw] extracted from the SE(3) matrix.
    """
    return np.array([T[0, 3], T[1, 3], extract_yaw(T)], dtype=np.float64)


@dataclass(frozen=True)
class MapSpec:
    resolution: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def width(self) -> int:
        return int(math.ceil((self.x_max - self.x_min) / self.resolution))

    @property
    def height(self) -> int:
        return int(math.ceil((self.y_max - self.y_min) / self.resolution))

    def to_dict(self) -> dict[str, float]:
        return {
            "resolution": self.resolution,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, float]) -> "MapSpec":
        return cls(
            resolution=float(payload["resolution"]),
            x_min=float(payload["x_min"]),
            x_max=float(payload["x_max"]),
            y_min=float(payload["y_min"]),
            y_max=float(payload["y_max"]),
        )


def pixel_to_world(rc: np.ndarray, map_spec: MapSpec) -> np.ndarray:
    rows = rc[:, 0].astype(np.float64)
    cols = rc[:, 1].astype(np.float64)
    x = map_spec.x_min + (cols + 0.5) * map_spec.resolution
    y = map_spec.y_max - (rows + 0.5) * map_spec.resolution
    return np.stack([x, y], axis=1)
