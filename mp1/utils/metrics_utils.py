from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import numpy as np

from .kitti_utils import KittiTrackingSequence

def write_report(
    out_paths: dict[str, Path],
    sequence: KittiTrackingSequence,
    frame_indices: list[int],
    static_data: dict[str, Any],
    landmarks_data: dict[str, Any],
    metrics: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    report_path = out_paths["report"] / "report.md"
    mean_points = float(np.mean(static_data["point_counts"])) if len(static_data["point_counts"]) else 0.0
    report = f"""# CS588 KITTI Static Mapping + ICP + Graph-SLAM

## Dataset

- Sequence: `{sequence.seq_date}_drive_{sequence.drive}_sync`
- Frames used: `{len(frame_indices)}` (`{frame_indices[0]}` to `{frame_indices[-1]}`)
- Mean static points per frame: `{mean_points:.1f}`
- Mean dynamic-point ratio: `{static_data["dynamic_ratios"].mean():.4f}`
- Final landmarks: `{len(landmarks_data["landmarks_xy"])}`

## Parameters

- Map resolution: `{args.map_resolution:.2f} m/pixel`
- ICP voxel size: `{args.voxel_size:.2f} m`
- ICP max correspondence: `{args.icp_max_corr:.2f} m`
- ICP iterations: `{args.icp_iters}`
- Graph-SLAM sigmas: `sigma_xy={args.sigma_xy}`, `sigma_theta={args.sigma_theta}`, `sigma_d={args.sigma_d}`
- Gauss-Newton iterations: `{args.gn_iters}`
- Gauss-Newton damping: `{args.gn_damping}`

## Metrics
- ICP ATE RMSE: `{metrics["ate_icp_rmse"]:.4f}` m
- Graph-SLAM ATE RMSE: `{metrics["ate_slam_rmse"]:.4f}` m

## Figures

![GT map](../maps/map_gt.png)

![ICP map](../maps/map_icp.png)

![Graph-SLAM map](../maps/map_slam.png)

![Trajectory overlay](../traj/traj_overlay.png)

![Jacobian sparsity](../jacobian/jacobian_spy.png)

![Graph SLAM Cost History](../jacobian/gauss_newton_cost.png)
"""
    report_path.write_text(report, encoding="utf-8")


def print_runtime_summary(args: argparse.Namespace, data_root: Path, sequence: KittiTrackingSequence, frame_indices: list[int]) -> None:
    print(f"data_root={data_root}")
    print(f"sequence={sequence.seq_date}_drive_{sequence.drive}_sync")
    print(f"frames=[{frame_indices[0]}, {frame_indices[-1] + 1}) count={len(frame_indices)}")
    print(
        f"params: voxel_size={args.voxel_size} icp_max_corr={args.icp_max_corr} "
        f"icp_iters={args.icp_iters}"
        f"sigma_xy={args.sigma_xy} sigma_theta={args.sigma_theta} "
        f"sigma_d={args.sigma_d} gn_iters={args.gn_iters}"
    )

def to_jsonable_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        elif isinstance(value, (np.floating, np.integer)):
            out[key] = value.item()
        else:
            out[key] = value
    return out

def ate_rmse(est_xy: np.ndarray, gt_xy: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((est_xy - gt_xy) ** 2, axis=1))))
