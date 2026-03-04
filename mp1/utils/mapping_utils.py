from __future__ import annotations

import json
from pathlib import Path
import warnings

import numpy as np

from .geometry_utils import MapSpec, pixel_to_world

def estimate_map_spec(points_world: list[np.ndarray], resolution: float = 0.05, padding: float = 2.0) -> MapSpec:
    xyz = [pts[:, :3] for pts in points_world if pts.size]
    if not xyz:
        raise ValueError("Cannot estimate map bounds from zero points.")
    all_xyz = np.concatenate(xyz, axis=0)
    mins = all_xyz[:, :2].min(axis=0) - padding
    maxs = all_xyz[:, :2].max(axis=0) + padding
    return MapSpec(
        resolution=resolution,
        x_min=float(mins[0]),
        x_max=float(maxs[0]),
        y_min=float(mins[1]),
        y_max=float(maxs[1]),
    )


def save_map_spec(map_spec: MapSpec, path: str | Path) -> None:
    path = Path(path)
    path.write_text(json.dumps(map_spec.to_dict(), indent=2), encoding="utf-8")


def load_map_spec(path: str | Path) -> MapSpec:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return MapSpec.from_dict(payload)

# For computing landmarks 

def _sliding_window_max(arr: np.ndarray, window: int) -> np.ndarray:
    pad = window // 2
    padded = np.pad(arr, pad, mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(padded, (window, window))
    return view.max(axis=(-1, -2))


def extract_landmarks_from_density(
    density_map: np.ndarray,
    map_spec: MapSpec,
    gt_poses_se2: np.ndarray,
    target_count: int = 100,
    nms_window: int = 7,
    t_ref: int = 10,
    ref_distance_limit: float = 20.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if density_map.size == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float64)

    ref_idx = min(max(t_ref, 0), len(gt_poses_se2) - 1)
    ref_xy = gt_poses_se2[ref_idx, :2]
    percentiles = [99.9, 99.7, 99.5, 99.0, 98.0, 97.0, 95.0]
    local_max = _sliding_window_max(density_map, nms_window)
    valid_mask = density_map == local_max

    best_world = np.zeros((0, 2), dtype=np.float64)
    best_pixels = np.zeros((0, 2), dtype=np.int64)
    best_scores = np.zeros((0,), dtype=np.float64)

    for percentile in percentiles:
        thresh = float(np.percentile(density_map[density_map > 0], percentile)) if np.any(density_map > 0) else 0.0
        mask = valid_mask & (density_map >= thresh) & (density_map > 0)
        pixels = np.argwhere(mask)
        if pixels.size == 0:
            continue

        scores = density_map[pixels[:, 0], pixels[:, 1]].astype(np.float64)
        order = np.argsort(scores)[::-1]
        pixels = pixels[order]
        scores = scores[order]
        world = pixel_to_world(pixels, map_spec)
        d_ref = np.linalg.norm(world - ref_xy.reshape(1, 2), axis=1)
        visible = d_ref <= ref_distance_limit
        pixels = pixels[visible]
        scores = scores[visible]
        world = world[visible]

        if len(world) == 0:
            continue
        if len(world) >= target_count:
            return world[:target_count], pixels[:target_count], scores[:target_count]
        best_world, best_pixels, best_scores = world, pixels, scores

    if len(best_world) < max(20, target_count // 4):
        warnings.warn(
            f"Only found {len(best_world)} candidate landmarks near frame {ref_idx}. "
            "Later observation filtering may reduce this further."
        )
    return best_world, best_pixels, best_scores


# For computing observations

def compute_distance_observations(
    landmarks_xy: np.ndarray,
    gt_poses_se2: np.ndarray,
    max_range: float = 40.0,
) -> tuple[np.ndarray, np.ndarray]:
    if len(landmarks_xy) == 0:
        dense = np.zeros((0, len(gt_poses_se2)), dtype=np.float64)
        return dense, np.zeros((0, 3), dtype=np.float64)

    pose_xy = gt_poses_se2[:, :2]
    deltas = landmarks_xy[:, None, :] - pose_xy[None, :, :]
    dists = np.linalg.norm(deltas, axis=2)
    dense = np.where(dists <= max_range, dists, -1.0)

    obs_rows = []
    for landmark_idx in range(dense.shape[0]):
        valid_ts = np.where(dense[landmark_idx] >= 0.0)[0]
        for frame_idx in valid_ts:
            obs_rows.append((landmark_idx, frame_idx, dense[landmark_idx, frame_idx]))

    if not obs_rows:
        return dense, np.zeros((0, 3), dtype=np.float64)
    return dense, np.array(obs_rows, dtype=np.float64)


def filter_landmarks_by_observations(
    landmarks_xy: np.ndarray,
    landmark_pixels: np.ndarray,
    scores: np.ndarray,
    dense_observations: np.ndarray,
    max_valid_frames: int = 10,
    target_count: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(landmarks_xy) == 0:
        return (
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0, 2), dtype=np.int64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0, dense_observations.shape[1]), dtype=np.float64),
        )

    capped_dense = np.full_like(dense_observations, -1.0)
    for idx in range(dense_observations.shape[0]):
        valid_ts = np.where(dense_observations[idx] >= 0.0)[0]
        if len(valid_ts) == 0:
            continue
        if len(valid_ts) > max_valid_frames:
            order = np.argsort(dense_observations[idx, valid_ts])
            valid_ts = valid_ts[order[:max_valid_frames]]
        capped_dense[idx, valid_ts] = dense_observations[idx, valid_ts]

    valid_counts = np.sum(capped_dense >= 0.0, axis=1)
    keep = valid_counts > 0
    kept_xy = landmarks_xy[keep]
    kept_pixels = landmark_pixels[keep]
    kept_scores = scores[keep]
    kept_dense = capped_dense[keep]

    order = np.argsort(kept_scores)[::-1]
    kept_xy = kept_xy[order]
    kept_pixels = kept_pixels[order]
    kept_scores = kept_scores[order]
    kept_dense = kept_dense[order]

    if len(kept_xy) > target_count:
        kept_xy = kept_xy[:target_count]
        kept_pixels = kept_pixels[:target_count]
        kept_scores = kept_scores[:target_count]
        kept_dense = kept_dense[:target_count]

    return kept_xy, kept_pixels, kept_scores, kept_dense


def sparse_observations_from_dense(dense: np.ndarray) -> np.ndarray:
    rows = []
    for landmark_idx in range(dense.shape[0]):
        valid_ts = np.where(dense[landmark_idx] >= 0.0)[0]
        for frame_idx in valid_ts:
            rows.append((landmark_idx, frame_idx, dense[landmark_idx, frame_idx]))
    if not rows:
        return np.zeros((0, 3), dtype=np.float64)
    return np.array(rows, dtype=np.float64)
