from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from utils.kitti_utils import label_dynamic_points, KittiTrackingSequence
from utils.geometry_utils import MapSpec, se3_to_se2
from utils.viz_utils import save_cost_plot, save_jacobian_spy, save_layer_png, save_map_with_trajectory, save_trajectory_overlay
from utils.metrics_utils import write_report, print_runtime_summary, to_jsonable_metrics, ate_rmse
from utils.mapping_utils import (
  extract_landmarks_from_density, 
  estimate_map_spec, load_map_spec, save_map_spec,
  compute_distance_observations, 
  filter_landmarks_by_observations, 
  sparse_observations_from_dense,
)

from icp import compute_icp_chains
from graph_slam import GraphSlamProblem, solve_graph_slam
from alignment import rasterize_topdown, accumulate_and_rasterize, build_accumulated_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CS588 KITTI static mapping + ICP + 2D Graph-SLAM")
    parser.add_argument("command", nargs="?", default="all", choices=["gt_align", "icp", "all"])
    
    # dataset 
    parser.add_argument("--data_root", default="data/kitti_raw", help="KITTI root")
    parser.add_argument("--out_dir", default="./outputs")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=100, help="Exclusive end frame.")

    # recompute flags
    parser.add_argument("--recompute_labels", action="store_true", help="Ignore cached dynamic/static labels and recompute them.")
    parser.add_argument("--recompute_map", action="store_true", help="Ignore cached GT map and recompute it.")
    parser.add_argument("--recompute_landmarks", action="store_true", help="Ignore cached landmarks/observations and recompute them.")

    # ICP parameters
    parser.add_argument("--voxel_size", type=float, default=0.25)
    parser.add_argument("--icp_max_corr", type=float, default=1.5)
    parser.add_argument("--icp_iters", type=int, default=60)

    # Graph-SLAM parameters
    parser.add_argument("--sigma_xy", type=float, default=0.24)
    parser.add_argument("--sigma_theta", type=float, default=0.04)
    parser.add_argument("--sigma_d", type=float, default=0.08)
    parser.add_argument("--gn_iters", type=int, default=50)
    parser.add_argument("--gn_damping", type=float, default=1e-6)

    # Map parameters
    parser.add_argument("--map_resolution", type=float, default=0.05)
    parser.add_argument("--landmark_target", type=int, default=100)
    return parser.parse_args()


def ensure_output_dirs(out_dir: Path) -> dict[str, Path]:
    paths = {
        "root": out_dir,
        "maps": out_dir / "maps",
        "traj": out_dir / "traj",
        "jacobian": out_dir / "jacobian",
        "metrics": out_dir / "metrics",
        "report": out_dir / "report",
        "cache": out_dir / "cache",
        "labels": out_dir / "cache" / "labels",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save_map_layers(prefix: str, layers: dict[str, np.ndarray], map_spec: MapSpec, out_dir: Path, trajectory_xy: np.ndarray, landmarks_xy: np.ndarray | None = None) -> None:
    save_layer_png(layers["density"], out_dir / f"{prefix}_density.png", f"{prefix.upper()} density", cmap="magma", log_scale=True)
    save_map_with_trajectory(
        layers["density"],
        map_spec,
        trajectory_xy,
        out_dir / f"map_{prefix}.png",
        f"{prefix.upper()} accumulated map",
        landmarks_xy=landmarks_xy,
    )


def compute_static_points(
    sequence: KittiTrackingSequence,
    frame_indices: list[int],
    out_paths: dict[str, Path],
    recompute_labels: bool,
) -> dict[str, Any]:
    """Load frames, remove dynamic points, and collect GT pose data.

    Inputs:
        sequence: KITTI tracking sequence reader.
        frame_indices: Ordered frame indices to process.
        out_paths: Output directory dictionary from `ensure_output_dirs`.
        recompute_labels: If True, ignore label cache and recompute labels.

    Returns:
        dict[str, Any]: Static point clouds, per-frame dynamic ratios,
        per-frame static point counts, GT SE(3) poses, and canonical GT SE(2)
        trajectory.
    """
    static_points: list[np.ndarray] = []
    dynamic_ratios: list[float] = []
    gt_poses_w_lidar = []
    gt_poses_se2 = []
    point_counts = []

    for frame_idx in frame_indices:
        cache_path = out_paths["labels"] / f"frame_{frame_idx:06d}.npz"
        if cache_path.exists() and not recompute_labels:
            payload = np.load(cache_path)
            points = payload["static_points"]
            dynamic_ratio = float(payload["dynamic_ratio"])
        else:
            points_all = sequence.load_pointcloud(frame_idx)
            boxes = sequence.get_boxes(frame_idx)
            dynamic_mask = label_dynamic_points(points_all, boxes)
            points = points_all[~dynamic_mask]
            dynamic_ratio = float(dynamic_mask.mean()) if len(dynamic_mask) else 0.0
            np.savez_compressed(
                cache_path,
                static_points=points,
                dynamic_mask=dynamic_mask,
                dynamic_ratio=np.array(dynamic_ratio, dtype=np.float64),
            )
        static_points.append(points)
        dynamic_ratios.append(dynamic_ratio)
        point_counts.append(int(len(points)))
        pose_w_lidar = sequence.get_pose(frame_idx)
        gt_poses_w_lidar.append(pose_w_lidar)
        gt_poses_se2.append(se3_to_se2(pose_w_lidar))

    return {
        "static_points": static_points,
        "dynamic_ratios": np.array(dynamic_ratios, dtype=np.float64),
        "point_counts": np.array(point_counts, dtype=np.int64),
        "gt_poses_w_lidar": np.stack(gt_poses_w_lidar, axis=0),
        "gt_poses_se2": np.stack(gt_poses_se2, axis=0),
    }


def compute_gt_map(
    static_data: dict[str, Any],
    out_paths: dict[str, Path],
    map_resolution: float,
    recompute_map: bool,
) -> dict[str, Any]:
    """Build or load the ground-truth static map from GT-aligned points.

    Inputs:
        static_data: Output dictionary from `compute_static_points`.
        map_resolution: Grid resolution in meters per pixel.
        out_paths: Output directory dictionary from `ensure_output_dirs`.
        recompute_map: If True, ignore GT map cache and rebuild map assets.

    Returns:
        dict[str, Any]: `map_spec`, GT world points, and GT map layers.
    """
    map_spec_path = out_paths["cache"] / "map_spec.json"
    map_cache_path = out_paths["cache"] / "gt_map_layers.npz"
    gt_poses_se2 = [se3_to_se2(pose) for pose in static_data["gt_poses_w_lidar"]]
    gt_points_world = build_accumulated_map(static_data["static_points"], gt_poses_se2)

    if map_spec_path.exists() and map_cache_path.exists() and not recompute_map:
        map_spec = load_map_spec(map_spec_path)
        payload = np.load(map_cache_path)
        layers = {key: payload[key] for key in ["density"]}
    else:
        map_spec = estimate_map_spec(gt_points_world, resolution=map_resolution)
        save_map_spec(map_spec, map_spec_path)
        layers = rasterize_topdown(gt_points_world, map_spec)
        np.savez_compressed(map_cache_path, **layers)

    save_map_layers(
        "gt",
        layers,
        map_spec,
        out_paths["maps"],
        static_data["gt_poses_se2"][:, :2],
    )
    return {
        "map_spec": map_spec,
        "gt_points_world": gt_points_world,
        "gt_layers": layers,
    }


def compute_landmarks_and_observations(
    static_data: dict[str, Any],
    gt_map: dict[str, Any],
    out_paths: dict[str, Path],
    landmark_target: int,
    recompute_landmarks: bool,
) -> dict[str, Any]:
    """Extract landmarks from density map and build distance observations.

    Inputs:
        static_data: Static points and GT trajectory data.
        gt_map: Ground-truth map dictionary from `compute_gt_map`.
        out_paths: Output directory dictionary from `ensure_output_dirs`.
        landmark_target: Desired final number of landmarks.
        recompute_landmarks: If True, ignore landmark/observation cache.

    Returns:
        dict[str, Any]: Landmark positions/pixels/scores and both dense and
        sparse landmark observation tensors used by Graph-SLAM.
    """
    cache_path = out_paths["cache"] / "landmarks_observations.npz"
    if cache_path.exists() and not recompute_landmarks:
        payload = np.load(cache_path)
        return {
            "landmarks_xy": payload["landmarks_xy"],
            "landmark_pixels": payload["landmark_pixels"],
            "scores": payload["scores"],
            "dense_observations": payload["dense_observations"],
            "sparse_observations": payload["sparse_observations"],
        }

    candidates_xy, candidate_pixels, scores = extract_landmarks_from_density(
        gt_map["gt_layers"]["density"],
        gt_map["map_spec"],
        static_data["gt_poses_se2"],
        target_count=max(landmark_target * 3, landmark_target),
    )
    dense_obs, _ = compute_distance_observations(candidates_xy, static_data["gt_poses_se2"], max_range=40.0)
    landmarks_xy, landmark_pixels, scores, dense_obs = filter_landmarks_by_observations(
        candidates_xy,
        candidate_pixels,
        scores,
        dense_obs,
        max_valid_frames=10,
        target_count=landmark_target,
    )
    sparse_obs = sparse_observations_from_dense(dense_obs)
    np.savez_compressed(
        cache_path,
        landmarks_xy=landmarks_xy,
        landmark_pixels=landmark_pixels,
        scores=scores,
        dense_observations=dense_obs,
        sparse_observations=sparse_obs,
    )
    return {
        "landmarks_xy": landmarks_xy,
        "landmark_pixels": landmark_pixels,
        "scores": scores,
        "dense_observations": dense_obs,
        "sparse_observations": sparse_obs,
    }


def compute_slam(
    static_data: dict[str, Any],
    landmarks_data: dict[str, Any],
    icp_data: dict[str, Any],
    out_paths: dict[str, Path],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run or load Gauss-Newton Graph-SLAM optimization.

    Inputs:
        static_data: Static points and GT trajectory information.
        landmarks_data: Landmarks and sparse distance observations.
        icp_data: Motion constraints and initial trajectory from ICP.
        out_paths: Output directory dictionary from `ensure_output_dirs`.
        args: Parsed command-line options containing GN and noise settings.
        (No disk cache is used; optimization is rerun each run.)

    Returns:
        dict[str, Any]: Optimized poses, optimized landmarks, GN cost history,
        and the Jacobian snapshot used for sparsity visualization.
    """
    if len(landmarks_data["landmarks_xy"]) == 0:
        raise RuntimeError("No landmarks survived observation filtering; Graph-SLAM cannot run.")

    slam_problem = GraphSlamProblem(
        initial_poses=icp_data["trajectory"],
        initial_landmarks=landmarks_data["landmarks_xy"],
        motion_edges=icp_data["motion_edges"],
        observations=landmarks_data["sparse_observations"],
        sigma_xy=args.sigma_xy,
        sigma_theta=args.sigma_theta,
        sigma_d=args.sigma_d,
    )
    result = solve_graph_slam(slam_problem, max_iterations=args.gn_iters, damping=args.gn_damping)
    # result["poses"] = canonicalize_pose_sequence(result["poses"])
    save_jacobian_spy(result["jacobian"], out_paths["jacobian"] / "jacobian_spy.png")
    save_cost_plot(result["cost_history"], out_paths["jacobian"] / "gauss_newton_cost.png")
    return result


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    out_dir = Path(args.out_dir)
    out_paths = ensure_output_dirs(out_dir)

    # --- Load data --- # 
    sequence = KittiTrackingSequence(data_root)
    frame_indices = sequence.frame_indices(args.start, args.end)
    if len(frame_indices) < 2:
        raise RuntimeError("Need at least 2 frames for ICP and Graph-SLAM.")
    print_runtime_summary(args, data_root, sequence, frame_indices)

    # --- Filter dynamic points --- # 
    static_data = compute_static_points(
        sequence,
        frame_indices,
        out_paths,
        recompute_labels=args.recompute_labels,
    )

    # --- Build GT map --- # 
    recompute_map = True if args.command == "gt_align" else args.recompute_map
    gt_map = compute_gt_map(
        static_data,
        out_paths,
        map_resolution=args.map_resolution,
        recompute_map=recompute_map,
    )
    save_map_with_trajectory(
        gt_map["gt_layers"]["density"],
        gt_map["map_spec"],
        static_data["gt_poses_se2"][:, :2],
        out_paths["traj"] / "gt_trajectory_on_map.png",
        "GT trajectory on static map",
    )
    if args.command == "gt_align":
        exit(0)

    # --- Extract landmarks and build observations --- # 
    landmarks_data = compute_landmarks_and_observations(
        static_data,
        gt_map,
        out_paths,
        landmark_target=args.landmark_target,
        recompute_landmarks=args.recompute_landmarks,
    )

    # --- Run and save ICP --- # 
    icp_data = compute_icp_chains(static_data["static_points"], args.voxel_size, args.icp_max_corr, args.icp_iters)
    icp_layers = accumulate_and_rasterize(static_data["static_points"], icp_data["trajectory"], gt_map["map_spec"])
    save_map_layers("icp", icp_layers, gt_map["map_spec"], out_paths["maps"], icp_data["trajectory"][:, :2])
    if args.command == "icp":
        icp_ate_rmse = ate_rmse(icp_data["trajectory"][:, :2], static_data["gt_poses_se2"][:, :2])
        print(f"ICP ATE RMSE: {icp_ate_rmse:.4f} m")
        exit(0)

    # --- Run and save Graph-SLAM --- # 
    slam_result = compute_slam(static_data, landmarks_data, icp_data, out_paths, args)
    slam_layers = accumulate_and_rasterize(static_data["static_points"], slam_result["poses"], gt_map["map_spec"])
    save_map_layers(
        "slam",
        slam_layers,
        gt_map["map_spec"],
        out_paths["maps"],
        slam_result["poses"][:, :2],
        landmarks_xy=slam_result["landmarks"],
    )
    save_trajectory_overlay(
        static_data["gt_poses_se2"][:, :2],
        icp_data["trajectory"][:, :2],
        slam_result["poses"][:, :2],
        out_paths["traj"] / "traj_overlay.png",
    )

    # --- Compute and save metrics --- # 
    metrics = {
        "sequence": f"{sequence.seq_date}_drive_{sequence.drive}_sync",
        "frame_start": frame_indices[0],
        "frame_end_exclusive": frame_indices[-1] + 1,
        "num_frames": len(frame_indices),
        "num_landmarks": int(len(landmarks_data["landmarks_xy"])),
        "num_motion_edges": int(icp_data["motion_edges"].shape[0]),
        "ate_icp_rmse": ate_rmse(icp_data["trajectory"][:, :2], static_data["gt_poses_se2"][:, :2]),
        "ate_slam_rmse": ate_rmse(slam_result["poses"][:, :2], static_data["gt_poses_se2"][:, :2]),
        "dynamic_ratio_mean": float(static_data["dynamic_ratios"].mean()),
        "sigma_xy": float(args.sigma_xy),
        "sigma_theta": float(args.sigma_theta),
        "sigma_d": float(args.sigma_d),
        "gn_iters": int(args.gn_iters),
        "gn_damping": float(args.gn_damping),
        "gauss_newton_cost_history": np.asarray(slam_result["cost_history"], dtype=np.float64),
    }
    (out_paths["metrics"] / "metrics.json").write_text(
        json.dumps(to_jsonable_metrics(metrics), indent=2),
        encoding="utf-8",
    )
    write_report(out_paths, sequence, frame_indices, static_data, landmarks_data, metrics, args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
