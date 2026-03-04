from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import math
import xml.etree.ElementTree as ET

import numpy as np

from .geometry_utils import invert_transform, rigid_transform, rotx, roty, rotz


@dataclass(frozen=True)
class Box3D:
    track_id: int
    object_type: str
    center: np.ndarray
    size: np.ndarray
    yaw: float


class KittiTrackingSequence:
    """Minimal KITTI raw loader that exposes LiDAR, GT tracklets, and GT poses."""

    def __init__(
        self,
        data_root: str | Path,
        seq_date: str = "2011_09_26",
        drive: str = "0005",
    ) -> None:
        self.root_dir = self._resolve_root(Path(data_root), seq_date, drive)
        self.seq_date = seq_date
        self.drive = drive
        self.seq_dir = self.root_dir / seq_date / f"{seq_date}_drive_{drive}_sync"
        self.velo_dir = self.seq_dir / "velodyne_points" / "data"
        self.oxts_dir = self.seq_dir / "oxts" / "data"
        self.tracklet_path = self.seq_dir / "tracklet_labels.xml"
        self.calib_imu_to_velo_path = self.root_dir / seq_date / "calib_imu_to_velo.txt"

        self._assert_exists(self.seq_dir, "sequence directory")
        self._assert_exists(self.velo_dir, "velodyne directory")
        self._assert_exists(self.oxts_dir, "oxts directory")
        self._assert_exists(self.tracklet_path, "tracklet labels")
        self._assert_exists(self.calib_imu_to_velo_path, "imu-to-velo calibration")

        self.velo_files = sorted(self.velo_dir.glob("*.bin"))
        self.oxts_files = sorted(self.oxts_dir.glob("*.txt"))
        if not self.velo_files:
            raise RuntimeError(f"No LiDAR frames found in {self.velo_dir}")
        if len(self.velo_files) != len(self.oxts_files):
            raise RuntimeError(
                f"LiDAR/OXTS mismatch: {len(self.velo_files)} velodyne files vs "
                f"{len(self.oxts_files)} oxts files."
            )

        self.num_frames = len(self.velo_files)
        self.frame_to_boxes = self._parse_tracklets()
        self.poses_w_lidar = self._load_lidar_poses()

    @staticmethod
    def _resolve_root(data_root: Path, seq_date: str, drive: str) -> Path:
        candidates = []
        if data_root.exists():
            candidates.extend(
                [
                    data_root,
                    data_root / "data" / "kitti_raw",
                    data_root / "kitti_raw",
                ]
            )
        for candidate in candidates:
            seq_dir = candidate / seq_date / f"{seq_date}_drive_{drive}_sync"
            if seq_dir.is_dir():
                return candidate
        raise FileNotFoundError(
            f"Could not resolve KITTI raw root from {data_root}. "
            "Expected either <root>/<date>/<date>_drive_<drive>_sync or "
            "<root>/data/kitti_raw/<date>/..."
        )

    @staticmethod
    def _assert_exists(path: Path, label: str) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Missing KITTI {label}: {path}")

    def frame_indices(self, start: int = 0, end: int | None = None) -> list[int]:
        stop = self.num_frames if end is None else min(end, self.num_frames)
        return list(range(max(0, start), stop))

    def load_pointcloud(self, frame_idx: int) -> np.ndarray:
        self._check_index(frame_idx)
        points = np.fromfile(self.velo_files[frame_idx], dtype=np.float32).reshape(-1, 4)
        finite = np.isfinite(points[:, :3]).all(axis=1)
        return points[finite]

    def get_boxes(self, frame_idx: int) -> list[Box3D]:
        self._check_index(frame_idx)
        return self.frame_to_boxes.get(frame_idx, [])

    def get_pose(self, frame_idx: int) -> np.ndarray:
        self._check_index(frame_idx)
        return self.poses_w_lidar[frame_idx].copy()

    def iter_frames(self, indices: Iterable[int]) -> Iterable[tuple[int, np.ndarray, np.ndarray, list[Box3D]]]:
        for frame_idx in indices:
            yield frame_idx, self.load_pointcloud(frame_idx), self.get_pose(frame_idx), self.get_boxes(frame_idx)

    def _check_index(self, frame_idx: int) -> None:
        if frame_idx < 0 or frame_idx >= self.num_frames:
            raise IndexError(f"frame_idx={frame_idx} outside [0, {self.num_frames})")

    def _parse_tracklets(self) -> dict[int, list[Box3D]]:
        frame_to_boxes = {i: [] for i in range(self.num_frames)}
        root = ET.parse(self.tracklet_path).getroot()
        tracklets_node = root.find("tracklets")
        if tracklets_node is None:
            return frame_to_boxes

        for track_id, tracklet in enumerate(tracklets_node.findall("item")):
            object_type = (tracklet.findtext("objectType") or "unknown").strip()
            h = float(tracklet.findtext("h", default="0"))
            w = float(tracklet.findtext("w", default="0"))
            l = float(tracklet.findtext("l", default="0"))
            first_frame = int(tracklet.findtext("first_frame", default="0"))
            poses_node = tracklet.find("poses")
            if poses_node is None:
                continue

            for local_idx, pose_node in enumerate(poses_node.findall("item")):
                frame_idx = first_frame + local_idx
                if frame_idx < 0 or frame_idx >= self.num_frames:
                    continue

                tx = float(pose_node.findtext("tx", default="0"))
                ty = float(pose_node.findtext("ty", default="0"))
                tz = float(pose_node.findtext("tz", default="0"))
                rz = float(pose_node.findtext("rz", default="0"))
                center = np.array([tx, ty, tz + 0.5 * h], dtype=np.float64)
                size = np.array([l, w, h], dtype=np.float64)
                frame_to_boxes[frame_idx].append(
                    Box3D(
                        track_id=track_id,
                        object_type=object_type,
                        center=center,
                        size=size,
                        yaw=rz,
                    )
                )
        return frame_to_boxes

    @staticmethod
    def _parse_calib_file(path: Path) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if ":" not in line:
                    continue
                key, raw = line.split(":", 1)
                tokens = raw.strip().split()
                try:
                    values = np.array([float(tok) for tok in tokens], dtype=np.float64)
                except ValueError:
                    continue
                out[key] = values
        return out

    def _load_lidar_poses(self) -> np.ndarray:
        calib = self._parse_calib_file(self.calib_imu_to_velo_path)
        T_velo_imu = rigid_transform(calib["R"].reshape(3, 3), calib["T"])
        T_imu_velo = invert_transform(T_velo_imu)

        oxts_packets = [np.loadtxt(path, dtype=np.float64) for path in self.oxts_files]
        lat0 = float(oxts_packets[0][0])
        scale = math.cos(math.radians(lat0))
        T_w_imu_abs = [self._pose_from_oxts(packet, scale) for packet in oxts_packets]
        T_rel_imu = [invert_transform(T_w_imu_abs[0]) @ T_w_imu_abs_t for T_w_imu_abs_t in T_w_imu_abs]
        T_rel_lidar = [T_velo_imu @ T_rel_imu_t @ T_imu_velo for T_rel_imu_t in T_rel_imu]
        return np.stack(T_rel_lidar, axis=0)

    @staticmethod
    def _pose_from_oxts(packet: np.ndarray, scale: float) -> np.ndarray:
        lat, lon, alt, roll, pitch, yaw = packet[:6]
        earth_radius = 6378137.0
        tx = scale * math.radians(lon) * earth_radius
        ty = scale * earth_radius * math.log(math.tan(math.pi * (90.0 + lat) / 360.0))
        tz = alt
        R = rotx(roll) @ roty(pitch) @ rotz(yaw)
        return rigid_transform(R, np.array([tx, ty, tz], dtype=np.float64))

# for filtering dynamic points
def points_in_oriented_box(points_xyz: np.ndarray, box: Box3D) -> np.ndarray:
    if points_xyz.size == 0:
        return np.zeros((0,), dtype=bool)
    centered = points_xyz - box.center.reshape(1, 3)
    c = float(np.cos(box.yaw))
    s = float(np.sin(box.yaw))
    local_x = centered[:, 0] * c + centered[:, 1] * s
    local_y = -centered[:, 0] * s + centered[:, 1] * c
    local_z = centered[:, 2]
    local = np.stack([local_x, local_y, local_z], axis=1)
    half_extents = 0.5 * box.size.reshape(1, 3)
    return np.all(np.abs(local) <= half_extents + 1e-6, axis=1)


def label_dynamic_points(points: np.ndarray, boxes: list[Box3D]) -> np.ndarray:
    if points.size == 0 or not boxes:
        return np.zeros((points.shape[0],), dtype=bool)
    xyz = points[:, :3].astype(np.float64, copy=False)
    finite = np.isfinite(xyz).all(axis=1)
    dynamic_mask = np.zeros((points.shape[0],), dtype=bool)
    for box in boxes:
        box_mask = np.zeros_like(dynamic_mask)
        box_mask[finite] = points_in_oriented_box(xyz[finite], box)
        dynamic_mask |= box_mask
    return dynamic_mask
