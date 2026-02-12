from __future__ import annotations

import os
import shutil
import xml.etree.ElementTree as ET

import cv2
import numpy as np


def ensure_kitti_root(data_root: str, date: str, drive: str, dataset: str) -> str:
    """Copy KITTI calib + drive data into data/kitti_raw for distribution."""
    calib_dir = os.path.join(data_root, f"{date}_calib", date)
    drive_dir = os.path.join(
        data_root,
        f"{date}_drive_{drive}_{dataset}",
        date,
        f"{date}_drive_{drive}_{dataset}",
    )

    if not os.path.isdir(calib_dir):
        raise FileNotFoundError(f"Missing calib dir: {calib_dir}")
    if not os.path.isdir(drive_dir):
        raise FileNotFoundError(f"Missing drive dir: {drive_dir}")

    raw_root = os.path.join(data_root, "kitti_raw")
    raw_date_dir = os.path.join(raw_root, date)
    os.makedirs(raw_date_dir, exist_ok=True)

    for fname in ["calib_cam_to_cam.txt", "calib_imu_to_velo.txt", "calib_velo_to_cam.txt"]:
        src = os.path.join(calib_dir, fname)
        dst = os.path.join(raw_date_dir, fname)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    drive_name = f"{date}_drive_{drive}_{dataset}"
    dst_drive = os.path.join(raw_date_dir, drive_name)
    if not os.path.exists(dst_drive):
        shutil.copytree(drive_dir, dst_drive, dirs_exist_ok=True)

    return raw_root


def resolve_kitti_root(data_root: str, date: str, drive: str, dataset: str) -> str:
    """Resolve the KITTI raw root for pykitti, supporting merged kitti_raw layouts."""
    raw_candidate = os.path.join(data_root, date, f"{date}_drive_{drive}_{dataset}")
    if os.path.isdir(raw_candidate):
        return data_root

    kitti_root = os.path.join(data_root, "kitti_raw")
    raw_candidate = os.path.join(kitti_root, date, f"{date}_drive_{drive}_{dataset}")
    if os.path.isdir(raw_candidate):
        return kitti_root

    return ensure_kitti_root(data_root, date, drive, dataset)


def load_tracklets(tracklet_path: str):
    """Load tracklet XML into a frame-indexed dict of object annotations."""
    if not os.path.isfile(tracklet_path):
        raise FileNotFoundError(f"Missing tracklets file: {tracklet_path}")

    tree = ET.parse(tracklet_path)
    root = tree.getroot()

    frames = {}
    tracklets_node = root.find("tracklets")
    if tracklets_node is None:
        return frames

    for tracklet_id, trk in enumerate(tracklets_node.findall("item")):
        obj_type = trk.findtext("objectType", default="unknown")
        h = float(trk.findtext("h"))
        w = float(trk.findtext("w"))
        l = float(trk.findtext("l"))
        first_frame = int(trk.findtext("first_frame"))
        poses = trk.find("poses")
        if poses is None:
            continue

        for i, pose in enumerate(poses.findall("item")):
            frame_idx = first_frame + i
            tx = float(pose.findtext("tx"))
            ty = float(pose.findtext("ty"))
            tz = float(pose.findtext("tz"))
            rx = float(pose.findtext("rx"))
            ry = float(pose.findtext("ry"))
            rz = float(pose.findtext("rz"))
            frames.setdefault(frame_idx, []).append(
                {
                    "id": tracklet_id,
                    "type": obj_type,
                    "size": np.array([l, w, h], dtype=np.float32),
                    "pos": np.array([tx, ty, tz], dtype=np.float32),
                    "rpy": np.array([rx, ry, rz], dtype=np.float32),
                }
            )

    return frames


def ensure_output_dir(path: str) -> None:
    """Create output directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def to_4x4(T: np.ndarray) -> np.ndarray:
    """Convert a 3x4 rigid transform to 4x4 homogeneous form (or pass through 4x4)."""
    if T.shape == (4, 4):
        return T
    if T.shape == (3, 4):
        return np.vstack([T, [0, 0, 0, 1]])
    raise ValueError(f"Unexpected transform shape: {T.shape}")


def depth_to_color(depths: np.ndarray) -> np.ndarray:
    """Map depth values to a jet-like RGB colormap (uint8)."""
    if depths.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    d_min, d_max = np.percentile(depths, [5, 95])
    if d_max <= d_min:
        d_min, d_max = depths.min(), depths.max() + 1e-6
    t = np.clip((depths - d_min) / (d_max - d_min), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * (t - 0.75)), 0, 1)
    g = np.clip(1.5 - np.abs(4 * (t - 0.5)), 0, 1)
    b = np.clip(1.5 - np.abs(4 * (t - 0.25)), 0, 1)
    return (np.stack([b, g, r], axis=-1) * 255).astype(np.uint8)


def distance_to_color(distances: np.ndarray) -> np.ndarray:
    """Map distances to a turbo-like RGB colormap (uint8)."""
    if distances.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    d_min, d_max = np.percentile(distances, [5, 95])
    if d_max <= d_min:
        d_min, d_max = distances.min(), distances.max() + 1e-6
    t = np.clip((distances - d_min) / (d_max - d_min), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * (t - 0.75)), 0, 1)
    g = np.clip(1.5 - np.abs(4 * (t - 0.5)), 0, 1)
    b = np.clip(1.5 - np.abs(4 * (t - 0.25)), 0, 1)
    return (np.stack([b, g, r], axis=-1) * 255).astype(np.uint8)


def color_for_id(tracklet_id: int) -> np.ndarray:
    """Generate a bright, stable RGB color for a tracklet id (uint8)."""
    hue = (tracklet_id * 0.61803398875) % 1.0
    h = int(hue * 179)
    s = 230
    v = 255
    bgr = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return np.array([int(bgr[2]), int(bgr[1]), int(bgr[0])], dtype=np.uint8)


def box_corners(size: np.ndarray) -> np.ndarray:
    """Return the 8 corners of a 3D box centered at origin, given [l, w, h]."""
    l, w, h = size
    x = l / 2
    y = w / 2
    z = h / 2
    return np.array(
        [
            [x, y, z],
            [x, -y, z],
            [-x, -y, z],
            [-x, y, z],
            [x, y, -z],
            [x, -y, -z],
            [-x, -y, -z],
            [-x, y, -z],
        ],
        dtype=np.float32,
    )


def project_points_cam(points_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project camera-frame points into the image; return 2D points + valid mask."""
    z = points_cam[:, 2]
    valid = z > 0.1
    pts = points_cam[valid]
    proj = (K @ pts.T).T
    proj[:, 0] /= proj[:, 2]
    proj[:, 1] /= proj[:, 2]
    return proj[:, :2], valid


def draw_3d_boxes_on_image(img: np.ndarray, boxes_cam0, T_camx_cam0, K_camx):
    """Draw 3D box wireframes (cam0) onto a camera image."""
    overlay = img.copy()
    R = T_camx_cam0[:3, :3]
    t = T_camx_cam0[:3, 3]
    for b in boxes_cam0:
        corners = box_corners(b["size"])
        corners = (b["R"] @ corners.T).T + b["center"]
        corners_camx = (R @ corners.T).T + t
        proj, valid = project_points_cam(corners_camx, K_camx)
        if proj.shape[0] != 8:
            continue
        pts = proj.astype(np.int32)
        color = b["color"].tolist()
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for i, j in edges:
            cv2.line(overlay, pts[i], pts[j], color, 1, lineType=cv2.LINE_AA)
    return overlay


def overlay_lidar_on_image(img: np.ndarray, proj_xy: np.ndarray, depths: np.ndarray):
    """Overlay projected lidar points on an image, colored by depth."""
    h, w = img.shape[:2]
    overlay = img.copy()
    xy = proj_xy.astype(np.int32)
    mask = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < w)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < h)
    )
    xy = xy[mask]
    depths = depths[mask]
    colors = depth_to_color(depths)
    for (u, v), c in zip(xy[::2], colors[::2]):
        cv2.circle(overlay, (u, v), 1, c.tolist(), -1, lineType=cv2.LINE_AA)
    return overlay


def rpy_to_quat(rpy: np.ndarray) -> np.ndarray:
    """Convert roll/pitch/yaw (radians) to quaternion [x, y, z, w]."""
    roll, pitch, yaw = rpy
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return np.array([qx, qy, qz, qw], dtype=np.float32)


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to a 3x3 rotation matrix."""
    x, y, z, w = q
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to quaternion [x, y, z, w]."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return np.array([x, y, z, w], dtype=np.float32)
