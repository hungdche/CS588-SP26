# CS588 MP1: Static Mapping + ICP + Graph-SLAM

## Goal

SLAM (Simultaneous Localization and Mapping) is one of the most fundamental problem for robotics and autonomous driving. In this MP you will solve localizatoin and mapping with lidar. You will implement two basic method. One is odometry only approach which will chain relative pose estimated from lidar. Second one is a simulatansious localization and mapping which jointly solve poses and map with graph-SLAM.

## Setup

```bash
git clone https://github.com/hungdche/CS588-SP26.git
cd CS588-SP26/mp1

conda create -n cs588 python=3.11.0
conda activate cs588
pip install -r requirements.txt
```

## Download data
We will use the same data as MP0, so you can just move the folder over.

Download the data zip file from [here](https://uofi.box.com/s/1h1m7u0ob895b2o7rqklyyw8l4qumere). You might be required to log in with your illinois.edu email. 

Once the dataset has been downloaded, unzip it and place it in the `mp1` directory. Verify that it is in this structure
```
mp1
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
│           └── calib
```

## How To Run

```bash
# For task 0
python run.py gt_align

# For task 1
python run.py icp

# For task 2
python run.py all
```

## Notes

- World frame is the LiDAR frame at `t=0`.
- Convention for pose: T_ab means the transformation from b to a