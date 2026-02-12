# CS 588 MP0: Data Loading, Visualization and Calibration

## Goal

In this MP you’ll bring a real self-driving dataset to life: load and visualize a KITTI raw log, wire up sensor-frame conversions, re-derive camera projection from first principles, and explore multimodal fusion with Rerun. This is the foundation for sensor alignment and 3D perception; later in the MP (not implemented yet) you’ll extend this by performing target-free camera–LiDAR alignment using ICP.

## Setup

```bash
git clone https://github.com/hungdche/CS588-SP26.git
cd CS588-SP26/mp0

conda create -n cs588 python=3.11.0
conda activate cs588
pip install -r requirements.txt
```

## Download data
Download the data zip file from [here](https://uofi.box.com/s/1h1m7u0ob895b2o7rqklyyw8l4qumere). You might be required to log in with your illinois.edu email. 

Once the dataset has been downloaded, unzip it and place it in the `mp0` directory. Verify that it is in this structure
```
mp0
├── data
│   ├── extra_credit
│   │   ├── cam2.png
│   │   └── cam3.png
│   └── kitti_raw
│       └── 2011_09_26
│           ├── 2011_09_26_drive_0005_sync
│           │   ├── disp_02
│           │   ├── image_00
│           │   ├── image_01
│           │   ├── image_02
│           │   ├── image_03
│           │   ├── oxts
│           │   ├── tracklet_labels.xml
│           │   └── velodyne_points
│           ├── calib_cam_to_cam.txt
│           ├── calib_imu_to_velo.txt
│           └── calib_velo_to_cam.txt
```

## How To Run

```bash
# For task 1 and 2
# Reduce number of frames if laggy
python3 kitti_viz.py --frames 150 

# For task 3
python3 kitti_online_calib.py
```

## Codebase Structure

1. `data/kitti_raw/`
   - `calib_cam_to_cam.txt`: camera intrinsics/extrinsics.
   - `calib_imu_to_velo.txt`: IMU → Velodyne transform.
   - `calib_velo_to_cam.txt`: Velodyne → camera transform.
   - `image_02/data/`: rectified color camera 2 images.
   - `image_03/data/`: rectified color camera 3 images.
   - `velodyne_points/data/`: LiDAR point clouds.
   - `oxts/data/`: GPS/IMU packets.

2. `utils.py` 
   - File I/O helpers for dataset setup and tracklets parsing.
   - Geometry helpers (quaternions, box corners, projections).
   - Visualization helpers (color maps, overlay drawing).
   - **IMPORTANT**: You do not need to modify anything in here!! 

3. `kitti_viz.py`
   - Where you will implement task 1 and 2. Please refer to the doc and comments for more details. 

4. `kitti_online_calib.py`
   - Where you will implement task 3. Please refer to the doc and comments for more details.
