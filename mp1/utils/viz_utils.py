from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .geometry_utils import MapSpec
from alignment import world_to_pixel


def _normalize_for_display(layer: np.ndarray, log_scale: bool = False) -> np.ndarray:
    arr = layer.astype(np.float64)
    if log_scale:
        arr = np.log1p(np.maximum(arr, 0.0))
    arr = arr - np.nanmin(arr)
    max_val = np.nanmax(arr)
    if max_val <= 1e-9:
        return np.zeros_like(arr, dtype=np.float64)
    return arr / max_val


def save_layer_png(layer: np.ndarray, path: str | Path, title: str, cmap: str = "viridis", log_scale: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(_normalize_for_display(layer, log_scale=log_scale), cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("x pixel")
    ax.set_ylabel("y pixel")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_map_with_trajectory(
    density_map: np.ndarray,
    map_spec: MapSpec,
    trajectory_xy: np.ndarray,
    path: str | Path,
    title: str,
    color: str = "red",
    landmarks_xy: np.ndarray | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(_normalize_for_display(density_map, log_scale=True), cmap="gray")
    traj_px = world_to_pixel(trajectory_xy, map_spec)
    ax.plot(traj_px[:, 1], traj_px[:, 0], color=color, linewidth=2.0, label="trajectory")
    if landmarks_xy is not None and len(landmarks_xy):
        lm_px = world_to_pixel(landmarks_xy, map_spec)
        ax.scatter(lm_px[:, 1], lm_px[:, 0], s=12, c="cyan", label="landmarks")
    ax.set_title(title)
    ax.set_xlabel("x pixel")
    ax.set_ylabel("y pixel")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_trajectory_overlay(
    gt_xy: np.ndarray,
    icp_xy: np.ndarray,
    slam_xy: np.ndarray,
    path: str | Path,
    posegraph_xy: np.ndarray | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(gt_xy[:, 0], gt_xy[:, 1], label="GT", linewidth=2.0)
    ax.plot(icp_xy[:, 0], icp_xy[:, 1], label="ICP", linewidth=1.8)
    ax.plot(slam_xy[:, 0], slam_xy[:, 1], label="Graph-SLAM", linewidth=1.8)
    if posegraph_xy is not None:
        ax.plot(posegraph_xy[:, 0], posegraph_xy[:, 1], label="PoseGraph", linewidth=1.8)
    ax.set_title("Trajectory Overlay")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_cost_plot(cost_history: np.ndarray, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(len(cost_history)), cost_history, marker="o")
    ax.set_title("Gauss-Newton Cost")
    ax.set_xlabel("iteration")
    ax.set_ylabel("0.5 * ||r||^2")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_jacobian_spy(J: np.ndarray, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.spy(J, markersize=0.25)
    ax.set_title("Jacobian Sparsity")
    ax.set_xlabel("state columns")
    ax.set_ylabel("residual rows")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
