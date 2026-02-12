from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import pykitti
import rerun as rr
import rerun.blueprint as bp

from utils import (
    color_for_id,
    distance_to_color,
    draw_3d_boxes_on_image,
    ensure_output_dir,
    load_tracklets,
    overlay_lidar_on_image,
    quat_to_rot,
    resolve_kitti_root,
    rot_to_quat,
    rpy_to_quat,
    to_4x4,
)


def project_point_cloud_to_image(points: np.ndarray, T_cam_velo: np.ndarray, K_cam: np.ndarray):
    """
    TASK 1 (Student): Lidar -> Camera projection
    Implement three steps:
      1) Transform velodyne points into the camera frame.
      2) Apply perspective projection with intrinsics.
      3) Filter out invalid points (e.g., z <= 0).
    """
    # ======= STUDENT TODO START (edit only inside this block) =======
    # TODO(student): implement projection.

    # Placeholder to keep the script runnable.
    n = points.shape[0]
    return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((n,), dtype=bool)

    # ======= STUDENT TODO END (do not change code outside this block) =======


def build_blueprint() -> bp.Blueprint:
    """
    TASK 2 (Student): Add blueprint visualization
    Create a layout with a large 3D view, four overlay image views, and right-side
    panels for GPS + raw images.
    """
    # ======= STUDENT TODO START (edit only inside this block) =======
    # TODO(student): implement the full layout.
    # Placeholder minimal layout to keep the script runnable.
    view_3d = bp.Spatial3DView(
        origin="world",
        contents=["world/velo", "world/tracklets", "world/cam0", "world/cam2", "world/cam3"],
        name="3D",
    )
    cam2 = bp.Spatial2DView(origin="world/cam2", contents="$origin/image", name="cam2")
    cam3 = bp.Spatial2DView(origin="world/cam3", contents="$origin/image", name="cam3")
    layout = bp.Horizontal(view_3d, bp.Vertical(cam2, cam3, row_shares=[1, 1]), column_shares=[3, 1])
    return bp.Blueprint(layout)
    # ======= STUDENT TODO END (do not change code outside this block) =======


def transform_box_to_cam0(
    center_velo: np.ndarray, # center of bbox in velodyne coordinates
    R_box_velo: np.ndarray, # rotation matrix of bbox in velodyne coordinates
    T_cam0_velo: np.ndarray
) -> tuple[np.ndarray, np.ndarray]: 

    """
    TASK 2 (Student): Tracklets -> World visualization
    Tracklets are in velodyne coordinates, so you must transform into cam0 rectified world
    """
    # ======= STUDENT TODO START (edit only inside this block) =======
    # TODO(student): Tracklets are in velodyne coordinates, so you must
    # transform into cam0 rectified world

    # Edit `center_cam0` and `R_box_cam0` to the correct values
    center_cam0 = center_velo
    R_box_cam0 = R_box_velo

    # ======= STUDENT TODO END (do not change code outside this block) =======
    return center_cam0, R_box_cam0


def transform_lidar_to_cam0(pts_velo: np.ndarray, T_cam0_velo: np.ndarray) -> np.ndarray:
    """
    TASK 2 (Student): Lidar -> World visualization
    Transform lidar from velodyne coordinates to cam0 rectified world coordinates.
    """
    # ======= STUDENT TODO START (edit only inside this block) =======
    # TODO(student): Transform lidar into world (cam0) coordinates and colorize.

    # Placeholder to keep the script runnable.
    pts_cam0 = pts_velo

    # ======= STUDENT TODO END (do not change code outside this block) =======
    return pts_cam0[:, :3]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--date", default="2011_09_26")
    parser.add_argument("--drive", default="0005")
    parser.add_argument("--dataset", default="sync")
    parser.add_argument("--frames", type=int, default=15)
    args = parser.parse_args()

    merged_root = resolve_kitti_root(args.data_root, args.date, args.drive, args.dataset)
    raw = pykitti.raw(merged_root, args.date, args.drive, dataset=args.dataset)

    tracklet_path = os.path.join(
        args.data_root,
        f"{args.date}_drive_{args.drive}_tracklets",
        args.date,
        f"{args.date}_drive_{args.drive}_sync",
        "tracklet_labels.xml",
    )
    if not os.path.isfile(tracklet_path):
        tracklet_path = os.path.join(
            merged_root,
            args.date,
            f"{args.date}_drive_{args.drive}_{args.dataset}",
            "tracklet_labels.xml",
        )
    if os.path.isfile(tracklet_path):
        tracklets = load_tracklets(tracklet_path)
    else:
        tracklets = {}
        print(f"No tracklets found at {tracklet_path} (3D boxes will be empty).")
    output_dir = "output"
    ensure_output_dir(output_dir)

    rr.init("kitti_raw", spawn=True)
    rr.send_blueprint(build_blueprint())
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN)

    # ===========================
    # 1) Parse data and calib
    # ===========================
    calib = raw.calib
    T_cam2_velo = to_4x4(calib.T_cam2_velo)
    T_cam3_velo = to_4x4(calib.T_cam3_velo)
    K_cam2 = calib.K_cam2
    K_cam3 = calib.K_cam3
    T_cam0_velo = to_4x4(calib.T_cam0_velo)

    T_velo_cam0 = np.linalg.inv(T_cam0_velo)
    T_cam2_cam0 = T_cam2_velo @ T_velo_cam0
    T_cam3_cam0 = T_cam3_velo @ T_velo_cam0

    rr.log("world/cam0", rr.Transform3D(translation=[0, 0, 0]))
    rr.log("world/cam2", rr.Transform3D(translation=T_cam2_cam0[:3, 3], mat3x3=T_cam2_cam0[:3, :3]))
    rr.log("world/cam3", rr.Transform3D(translation=T_cam3_cam0[:3, 3], mat3x3=T_cam3_cam0[:3, :3]))

    num_frames = min(len(raw), args.frames)
    if num_frames <= 0:
        print("No frames found.")
        return 1

    gps_history = []

    # ===========================
    # 2) Per-frame processing
    # ===========================
    for i in range(num_frames):
        rr.set_time("frame", sequence=i)

        # 2.1) Process frame data
        img2 = np.array(raw.get_cam2(i))
        img3 = np.array(raw.get_cam3(i))
        velo = raw.get_velo(i)
        boxes = tracklets.get(i, [])
        packet, _pose = raw.oxts[i]

        # 2.2) Overlay computation
        proj2, z2, _ = project_point_cloud_to_image(velo, T_cam2_velo[:3, :], K_cam2)
        proj3, z3, _ = project_point_cloud_to_image(velo, T_cam3_velo[:3, :], K_cam3)
        overlay2 = overlay_lidar_on_image(img2, proj2, z2)
        overlay3 = overlay_lidar_on_image(img3, proj3, z3)

        boxes_cam0 = []
        centers = []
        half_sizes = []
        quats = []
        labels = []

        for b in boxes:
            pos_velo = b["pos"]
            size = b["size"]
            quat_velo = rpy_to_quat(b["rpy"]) # bbox at velodyne coordinates
            R_box_velo = quat_to_rot(quat_velo)

            # Shift from bottom-center to box center,
            center_velo = pos_velo + np.array([0.0, 0.0, size[2] * 0.5], dtype=np.float32)
            
            # Transform bbox to cam0 coordinates
            center_cam0, R_box_cam0 = transform_box_to_cam0(center_velo, R_box_velo, T_cam0_velo)

            quat_cam0 = rot_to_quat(R_box_cam0)
            centers.append(center_cam0)
            half_sizes.append(size * 0.5)
            quats.append(quat_cam0) 
            labels.append(b["type"])
            boxes_cam0.append(
                {
                    "center": center_cam0,
                    "size": size,
                    "R": R_box_cam0,
                    "color": color_for_id(b["id"]),
                }
            )

        overlay2_boxes = draw_3d_boxes_on_image(img2, boxes_cam0, T_cam2_cam0, K_cam2)
        overlay3_boxes = draw_3d_boxes_on_image(img3, boxes_cam0, T_cam3_cam0, K_cam3)

        # 2.3) Log visualization
        pts_velo = velo[:, :3]
        # Transform lidar into world (cam0) coordinates
        pts_cam0 = transform_lidar_to_cam0(pts_velo, T_cam0_velo)
        distances = np.linalg.norm(pts_cam0, axis=1)
        rr.log("world/velo", rr.Points3D(pts_cam0, colors=distance_to_color(distances)))

        h2, w2 = img2.shape[:2]
        h3, w3 = img3.shape[:2]
        rr.log(
            "world/cam2",
            rr.Pinhole(
                image_from_camera=K_cam2,
                resolution=[w2, h2],
                camera_xyz=rr.ViewCoordinates.RIGHT_HAND_Y_DOWN,
            ),
        )
        rr.log(
            "world/cam3",
            rr.Pinhole(
                image_from_camera=K_cam3,
                resolution=[w3, h3],
                camera_xyz=rr.ViewCoordinates.RIGHT_HAND_Y_DOWN,
            ),
        )
        rr.log("world/cam2/image", rr.Image(img2))
        rr.log("world/cam3/image", rr.Image(img3))
        rr.log("world/cam2/overlay", rr.Image(overlay2))
        rr.log("world/cam3/overlay", rr.Image(overlay3))
        rr.log("world/cam2/boxes_overlay", rr.Image(overlay2_boxes))
        rr.log("world/cam3/boxes_overlay", rr.Image(overlay3_boxes))

        if i == 0:
            cv2.imwrite(
                os.path.join(output_dir, "cam2_overlay_000000.png"),
                cv2.cvtColor(overlay2, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                os.path.join(output_dir, "cam3_overlay_000000.png"),
                cv2.cvtColor(overlay3, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                os.path.join(output_dir, "cam2_boxes_000000.png"),
                cv2.cvtColor(overlay2_boxes, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                os.path.join(output_dir, "cam3_boxes_000000.png"),
                cv2.cvtColor(overlay3_boxes, cv2.COLOR_RGB2BGR),
            )

        colors = [color_for_id(b["id"]) for b in boxes]
        rr.log(
            "world/tracklets",
            rr.Boxes3D(
                centers=centers,
                half_sizes=half_sizes,
                quaternions=quats,
                colors=colors,
                labels=labels,
                show_labels=True,
            ),
        )

        gps_history.append([packet.lat, packet.lon])
        rr.log("world/gps", rr.GeoPoints(lat_lon=[[packet.lat, packet.lon]]))
        rr.log("world/gps_history", rr.GeoLineStrings(lat_lon=[gps_history], colors=[0, 170, 255]))

    return 0


if __name__ == "__main__":
    sys.exit(main())
