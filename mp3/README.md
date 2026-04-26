# CS588 MP3: Sampling-Based Motion Planning

## Goal

In this MP you will build a complete sampling-based motion planner for an autonomous vehicle simulated inside Waymax, Waymo's open-source autonomous-driving simulator backed by the Waymo Open Dataset You will implement three self-contained components that together form a closed-loop planning stack: a Frenet-frame trajectory sampler (Task~1), a multi-term cost evaluator (Task~2), and a geometric path tracker (Task~3). Each component builds on the previous one and is evaluated in a shared closed-loop simulation harness.

## Setup

```bash
git clone https://github.com/hungdche/CS588-SP26.git
cd CS588-SP26/mp3

conda create -n cs588 python=3.11.0
conda activate cs588
pip install git+https://github.com/waymo-research/waymax.git@main#egg=waymo-waymax
pip install -r requirements.txt
```

## Note: 

You do not need a GPU to do this MP. It is perfectly fine for Waymax to complain things like

```
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1777215668.929391  450463 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
I0000 00:00:1777215668.929972  450463 cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1777215669.881269  450463 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
I0000 00:00:1777215669.881617  450463 cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
```

or

```
E0000 00:00:1777215672.471936  450463 cuda_executor.cc:1737] INTERNAL: CUDA Runtime error: Failed call to cudaGetRuntimeVersion: Error loading CUDA libraries. GPU will not be used.: Error loading CUDA libraries. GPU will not be used.
W0000 00:00:1777215672.472197  450555 cuda_executor.cc:1755] Failed to determine cuDNN version (Note that this is expected if the application doesn't link the cuDNN plugin): INTERNAL: cuDNN error: CUDNN_STATUS_INTERNAL_ERROR
W0000 00:00:1777215672.487975  450463 gpu_device.cc:2365] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
```

As long as your code runs and produces results, it is good. 

## Data

Please refer to the Canvas announcement for the link to download the data. Please put the unzipped `data` folder into `mp2` as shown below.
```text
mp3
└── data
    └── training_tfexample.tfrecord-00000-of-01000
```

**IMPORTANT**: Please do not distribute the data. 

## How To Run

### Task 1

```bash
python evaluate.py --task 1
```

### Task 2

```bash
python evaluate.py --task 2
```


### Task 3

```bash
python evaluate.py --task 3
```