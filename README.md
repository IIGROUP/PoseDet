# PoseDet: Towards Real-time Multi-person Pose Estimation

## Description

##### A simple system that perform real-time multi-person pose estimation at real-time inference speed.

## Installation

This repository is based on [mmdetection2.1.0](https://mmdetection.readthedocs.io/en/v2.1.0/).  To run PoseDet, you are supposed to install mmdetection.

Requirements

- Linux
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2
- GCC 5+
- Annoconda

```bash
# A script with Annoconda
conda create -name PoseDet
conda install pytorch=1.3.1 cudatoolkit=9.2 torchvision=0.4.2 -c pytorch
pip install mmcv==0.6.2
git clone **
cd mmdetection
pip install -r requirements/build.txt
pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools" --user
python setup.py develop
```



## Quick Start

## Evaluation

