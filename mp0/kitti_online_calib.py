from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import open3d as o3d
import pykitti
import rerun as rr
import rerun.blueprint as bp

from utils import to_4x4


def disparity_to_pointcloud(disp: np.ndarray, K: np.ndarray, baseline: float):
    """
    TASK 3.1 (Student): Unproject Stereo Disparity -> 3D
    Implement the following steps:
      1) Use fx, fy, cx, cy from K.
      2) Convert disparity to depth with Z = fx * baseline / disp.
      3) Back-project to X, Y using pinhole model.
    """
    # ======= STUDENT TODO START (edit only inside this block) =======
    # TODO(student): Implement unprojection from disparity to 3D points in camera frame.
    
    # Placeholder (keeps script runnable):
    stereo_pts = np.array([[0.0, 0.0, 5.0]], dtype=np.float32)
    valid = np.zeros_like(disp, dtype=bool)
    valid[0, 0] = True

    # ======= STUDENT TODO END (do not change code outside this block) =======
    return stereo_pts, valid


def make_o3d_pointcloud(points: np.ndarray, color: tuple[float, float, float]):
    """Create a constant-color Open3D point cloud."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.tile(np.array(color, dtype=np.float32), (points.shape[0], 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def make_o3d_pointcloud_rgb(points: np.ndarray, rgb: np.ndarray):
    """Create an Open3D point cloud with per-point RGB colors in [0, 1]."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd


def filter_cam_frustum(points_cam: np.ndarray, K: np.ndarray, width: int, height: int, max_range: float):
    """Keep points in front of the camera, within range, and inside the image bounds."""
    z = points_cam[:, 2]
    valid = (z > 0.1) & (z < max_range)
    pts = points_cam[valid]
    uv = (K @ pts.T).T
    uv[:, 0] /= uv[:, 2]
    uv[:, 1] /= uv[:, 2]
    in_img = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    return pts[in_img], valid.nonzero()[0][in_img]


def icp_align(
  source: o3d.geometry.PointCloud, 
  target: o3d.geometry.PointCloud, 
  threshold: float, 
  max_iteration: int
) -> o3d.pipelines.registration.RegistrationResult:
    """
    TASK 3.2 (Student): ICP Alignment
    Call Open3D ICP registration to align source -> target:
      1) Use point-to-point ICP.
      2) Use identity as the init in this aligned frame.
      3) Use threshold 0.3 and max_iteration 200.
    """
    # ======= STUDENT TODO START (edit only inside this block) =======
    # TODO(student): Call Open3D ICP to align source -> target.

    # Placeholder (keeps script runnable):
    reg = o3d.pipelines.registration.RegistrationResult()
    reg.transformation = np.eye(4, dtype=np.float32)
    reg.fitness = 0.0
    reg.inlier_rmse = 0.0

    # ======= STUDENT TODO END (do not change code outside this block) =======
    return reg


def rotation_error_deg(R):
    """Compute rotation error in degrees from a rotation matrix."""
    trace = np.clip((np.trace(R) - 1) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(trace))


def project_points(points_cam: np.ndarray, K: np.ndarray):
    """Project 3D camera-frame points into the image plane."""
    z = points_cam[:, 2]
    valid = z > 0.1
    pts = points_cam[valid]
    proj = (K @ pts.T).T
    proj[:, 0] /= proj[:, 2]
    proj[:, 1] /= proj[:, 2]
    return proj[:, :2], valid


def draw_overlay(img: np.ndarray, proj_xy: np.ndarray, color=(0, 255, 0)):
    """Overlay projected points on an image for visualization."""
    overlay = img.copy()
    h, w = overlay.shape[:2]
    xy = proj_xy.astype(np.int32)
    mask = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    xy = xy[mask]
    for u, v in xy[::2]:
        cv2.circle(overlay, (u, v), 1, color, -1, lineType=cv2.LINE_AA)
    return overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/kitti_raw")
    parser.add_argument("--date", default="2011_09_26")
    parser.add_argument("--drive", default="0005")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--disp", default=None)
    args = parser.parse_args()

    # Instruction: Your goal is to calibrate LiDAR and camera by aligning
    # stereo-inferred point cloud with LiDAR point cloud. You must:
    # 1) Unproject stereo disparity into a 3D point cloud.
    # 2) Call ICP alignment (open3d) to align LiDAR to camera.
    # 3) Report rotational/translation error and outlier ratio.

    dataset = pykitti.raw(args.data_root, args.date, args.drive)
    if args.disp is None:
        disp_name = f"{args.frame:010d}.npz"
        args.disp = os.path.join(
            args.data_root,
            args.date,
            f"{args.date}_drive_{args.drive}_sync",
            "disp_02",
            disp_name,
        )
    left = np.array(dataset.get_cam2(args.frame))
    right = np.array(dataset.get_cam3(args.frame))
    velo = dataset.get_velo(args.frame)
    calib = dataset.calib

    K = calib.K_cam2
    b2 = -calib.P_rect_20[0, 3] / calib.P_rect_20[0, 0]
    b3 = -calib.P_rect_30[0, 3] / calib.P_rect_30[0, 0]
    baseline = abs(b3 - b2)

    disp_npz = np.load(args.disp)
    disp = disp_npz["disp"].astype(np.float32)
    scale_used = float(disp_npz["scale"]) if "scale" in disp_npz else None
    if disp.shape != left.shape[:2]:
        if scale_used is None:
            scale_x = left.shape[1] / disp.shape[1]
            scale_y = left.shape[0] / disp.shape[0]
            scale_used = scale_x
        disp = cv2.resize(disp, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_LINEAR)
        disp *= scale_used

    # Unproject stereo disparity into a 3D point cloud
    stereo_pts, valid = disparity_to_pointcloud(disp, K, baseline)
    left_rgb = left.astype(np.float32) / 255.0
    colors = left_rgb[valid]

    h, w = left.shape[:2]
    stereo_pts, idx = filter_cam_frustum(stereo_pts, K, w, h, max_range=20.0)
    colors = colors[idx]
    stereo_pcd = make_o3d_pointcloud_rgb(stereo_pts, colors)

    lidar_velo = velo[:, :3]
    T_cam2_velo = to_4x4(calib.T_cam2_velo)

    # Axis-alignment init: velodyne (x fwd, y left, z up) -> cam2 (x right, y down, z fwd)
    R_init = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    rng = np.random.default_rng(0)
    t_noise = rng.uniform(-1.5, 1.5, size=3).astype(np.float32)
    T_init = np.eye(4, dtype=np.float32)
    T_init[:3, :3] = R_init
    T_init[:3, 3] = t_noise

    # Prepare ICP by axis-aligning lidar into cam2-ish coordinates.
    lidar_init_cam2 = (T_init @ np.hstack([lidar_velo, np.ones((lidar_velo.shape[0], 1))]).T).T[:, :3]
    lidar_init_cam2, _ = filter_cam_frustum(lidar_init_cam2, K, w, h, max_range=20.0)

    source = make_o3d_pointcloud(lidar_init_cam2, (0.0, 0.0, 1.0)).voxel_down_sample(voxel_size=0.05)
    target = stereo_pcd.voxel_down_sample(voxel_size=0.05)

    # ICP alignment (identity init in this aligned frame)
    reg = icp_align(source, target, threshold=0.3, max_iteration=200)
    T_icp = reg.transformation @ T_init

    # Precompute aligned LiDAR clouds for eval + viz
    lidar_init = (T_init @ np.hstack([lidar_velo, np.ones((lidar_velo.shape[0], 1))]).T).T[:, :3]
    lidar_est = (T_icp @ np.hstack([lidar_velo, np.ones((lidar_velo.shape[0], 1))]).T).T[:, :3]
    lidar_gt = (T_cam2_velo @ np.hstack([lidar_velo, np.ones((lidar_velo.shape[0], 1))]).T).T[:, :3]

    lidar_init, _ = filter_cam_frustum(lidar_init, K, w, h, max_range=20.0)
    lidar_est, _ = filter_cam_frustum(lidar_est, K, w, h, max_range=20.0)
    lidar_gt, _ = filter_cam_frustum(lidar_gt, K, w, h, max_range=20.0)

    # Inlier ratios using already aligned point clouds (same frame)
    eval_threshold = 0.3
    stereo_eval = make_o3d_pointcloud(stereo_pts, (1.0, 0.0, 0.0)).voxel_down_sample(voxel_size=0.05)
    init_eval = o3d.pipelines.registration.evaluate_registration(
        make_o3d_pointcloud(lidar_init, (0.0, 0.0, 1.0)).voxel_down_sample(voxel_size=0.05),
        stereo_eval,
        eval_threshold,
        np.eye(4),
    )
    gt_eval = o3d.pipelines.registration.evaluate_registration(
        make_o3d_pointcloud(lidar_gt, (0.0, 0.0, 1.0)).voxel_down_sample(voxel_size=0.05),
        stereo_eval,
        eval_threshold,
        np.eye(4),
    )
    est_eval = o3d.pipelines.registration.evaluate_registration(
        make_o3d_pointcloud(lidar_est, (0.0, 0.0, 1.0)).voxel_down_sample(voxel_size=0.05),
        stereo_eval,
        eval_threshold,
        np.eye(4),
    )

    # Errors vs GT
    T_err = T_icp @ np.linalg.inv(T_cam2_velo)
    rot_err = rotation_error_deg(T_err[:3, :3])
    trans_err = np.linalg.norm(T_err[:3, 3])
    init_err = T_init @ np.linalg.inv(T_cam2_velo)
    init_rot_err = rotation_error_deg(init_err[:3, :3])
    init_trans_err = np.linalg.norm(init_err[:3, 3])

    print("GT T_cam2_velo:\n", T_cam2_velo)
    print("ICP T_cam2_velo (estimated):\n", T_icp)
    print(f"INIT rotation error (deg): {init_rot_err:.3f}")
    print(f"INIT translation error (m): {init_trans_err:.3f}")
    print(f"INIT inlier ratio: {init_eval.fitness:.4f}")
    print(f"INIT outlier ratio: {1.0 - init_eval.fitness:.4f}")
    print(f"ICP rotation error (deg): {rot_err:.3f}")
    print(f"ICP translation error (m): {trans_err:.3f}")
    print(f"ICP inlier ratio: {reg.fitness:.4f}")
    print(f"ICP outlier ratio: {1.0 - reg.fitness:.4f}")
    print(f"GT inlier ratio: {gt_eval.fitness:.4f}")
    print(f"GT outlier ratio: {1.0 - gt_eval.fitness:.4f}")
    print(
        f"ERRORS vs GT -> INIT: rot {init_rot_err:.3f} deg, trans {init_trans_err:.3f} m | "
        f"ICP: rot {rot_err:.3f} deg, trans {trans_err:.3f} m"
    )
    # Save metrics for grading.
    os.makedirs("output", exist_ok=True)
    with open(os.path.join("output", "calib_results.txt"), "w", encoding="utf-8") as f:
        f.write("GT T_cam2_velo:\n")
        f.write(f"{T_cam2_velo}\n")
        f.write("Estimated T_cam2_velo (ICP):\n")
        f.write(f"{T_icp}\n")
        f.write(f"INIT rotation error (deg): {init_rot_err:.3f}\n")
        f.write(f"INIT translation error (m): {init_trans_err:.3f}\n")
        f.write(f"INIT inlier ratio: {init_eval.fitness:.4f}\n")
        f.write(f"INIT outlier ratio: {1.0 - init_eval.fitness:.4f}\n")
        f.write(f"ICP rotation error (deg): {rot_err:.3f}\n")
        f.write(f"ICP translation error (m): {trans_err:.3f}\n")
        f.write(f"ICP inlier ratio: {reg.fitness:.4f}\n")
        f.write(f"ICP outlier ratio: {1.0 - reg.fitness:.4f}\n")
        f.write(f"GT inlier ratio: {gt_eval.fitness:.4f}\n")
        f.write(f"GT outlier ratio: {1.0 - gt_eval.fitness:.4f}\n")

    # -------------------------
    # Rerun Visualization
    # -------------------------
    rr.init("kitti_online_calib", spawn=True)
    rr.send_blueprint(
        bp.Blueprint(
            bp.Horizontal(
                bp.Vertical(
                    bp.Spatial3DView(origin="clouds/stereo_only", contents="$origin/**", name="Stereo Only"),
                    bp.Spatial3DView(origin="align/combined", contents="$origin/**", name="Align (Est)"),
                    row_shares=[1, 1],
                ),
                bp.Vertical(
                    bp.Spatial3DView(origin="align/init", contents="$origin/**", name="Align Init"),
                    bp.Spatial3DView(origin="align/gt", contents="$origin/**", name="Align GT"),
                    row_shares=[1, 1],
                ),
                bp.Vertical(
                    bp.Spatial2DView(origin="images/left", contents="$origin", name="Left"),
                    bp.Spatial2DView(origin="images/right", contents="$origin", name="Right"),
                    bp.Spatial2DView(origin="images/disparity", contents="$origin", name="Disparity"),
                    bp.Spatial2DView(origin="overlays/est", contents="$origin", name="Overlay Est"),
                    row_shares=[1, 1, 1, 1],
                ),
                column_shares=[1, 1, 1],
            )
        )
    )
    rr.set_time("frame", sequence=args.frame)

    rr.log("images/left", rr.Image(left))
    rr.log("images/right", rr.Image(right))

    disp_vis = disp.copy()
    disp_norm = np.clip(disp_vis / (np.percentile(disp_vis, 95) + 1e-6), 0, 1)
    disp_color = cv2.applyColorMap((disp_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    rr.log("images/disparity", rr.Image(cv2.cvtColor(disp_color, cv2.COLOR_BGR2RGB)))

    rr.log("clouds/stereo_only", rr.Points3D(stereo_pts, colors=(colors * 255).astype(np.uint8)))

    # Alignment clouds (red/blue)
    stereo_red = (np.tile(np.array([255, 0, 0], dtype=np.uint8), (stereo_pts.shape[0], 1)))

    rr.log("align/init/stereo", rr.Points3D(stereo_pts, colors=stereo_red))
    rr.log("align/init/lidar", rr.Points3D(lidar_init, colors=[0, 0, 255]))
    rr.log("align/est/stereo", rr.Points3D(stereo_pts, colors=stereo_red))
    rr.log("align/est/lidar", rr.Points3D(lidar_est, colors=[0, 0, 255]))
    rr.log("align/gt/stereo", rr.Points3D(stereo_pts, colors=stereo_red))
    rr.log("align/gt/lidar", rr.Points3D(lidar_gt, colors=[0, 0, 255]))

    # Combined red/blue alignment view
    rr.log("align/combined/stereo", rr.Points3D(stereo_pts, colors=stereo_red))
    rr.log("align/combined/lidar", rr.Points3D(lidar_est, colors=[0, 0, 255]))

    # Lidar-image overlays
    proj_init, _ = project_points(lidar_init, K)
    proj_est, _ = project_points(lidar_est, K)
    proj_gt, _ = project_points(lidar_gt, K)

    overlay_init = draw_overlay(left, proj_init, color=(0, 255, 0))
    overlay_est = draw_overlay(left, proj_est, color=(255, 255, 0))
    overlay_gt = draw_overlay(left, proj_gt, color=(255, 0, 0))

    rr.log("overlays/init", rr.Image(overlay_init))
    rr.log("overlays/est", rr.Image(overlay_est))
    rr.log("overlays/gt", rr.Image(overlay_gt))

    # Dump before/after overlays for grading.
    os.makedirs("output", exist_ok=True)
    cv2.imwrite("output/online_calib_overlay_init.png", cv2.cvtColor(overlay_init, cv2.COLOR_RGB2BGR))
    cv2.imwrite("output/online_calib_overlay_est.png", cv2.cvtColor(overlay_est, cv2.COLOR_RGB2BGR))
    cv2.imwrite("output/online_calib_overlay_gt.png", cv2.cvtColor(overlay_gt, cv2.COLOR_RGB2BGR))

    metrics_msg = (
        f"INIT vs GT: rot {init_rot_err:.3f} deg, trans {init_trans_err:.3f} m | "
        f"ICP vs GT: rot {rot_err:.3f} deg, trans {trans_err:.3f} m | "
        f"Inlier: {reg.fitness:.4f}, Outlier: {1.0 - reg.fitness:.4f}"
    )
    rr.log("metrics", rr.TextLog(metrics_msg))


if __name__ == "__main__":
    main()
