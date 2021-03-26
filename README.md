# PoseDet: Towards Real-time Multi-person Pose Estimation

## Description

##### A simple system that performs real-time multi-person pose estimation at real-time inference speed.

## Installation

This repository is based on [mmdetection2.1.0](https://mmdetection.readthedocs.io/en/v2.1.0/).  To run PoseDet, you are supposed to install mmdetection.

##### Requirements

- Linux
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2
- GCC 5+
- Annoconda

```bash
# A script with Annoconda
cd ./PoseDet_mmdetection
# Create env of conda
conda create -n PoseDet python=3.7
conda activate PoseDet

# Install pytorch
conda install pytorch=1.3.1 cudatoolkit=9.2 torchvision=0.4.2 -c pytorch

# Install mmcv, requirements, and coco api
pip install mmcv==0.6.2 --user
pip install -r requirements/build.txt
#TO isntall coco api, you can also download the package and install it localy via python setup.py develop
pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools" --user

# Install mmdet by develop mode
python setup.py develop
```

## Quick Start

#### For model training 

You are supposed to download the imagenet pre-trained models, then put them into `PoseDet_mmdetection/pretrained`:

- [DLA 34](http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth)
- [HRNet-W32](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/pretrain/third_party/hrnetv2_w32-dc9eeb4f.pth)
- [HRNer-W48](https://drive.google.com/file/d/1xk3tevawZ-XOK0y5DJi3TUsleM6B6e6p/view?usp=sharing)

#### Dataset preparation

Download the required dataset and modify the path to the dataset (data_root) in `PoseDet/configs/_base_/dataset_coco.py`

#### Train

```bash
#Replace config_file with the path to the config file (e.g., ./PoseDet/config/COCO/PoseDet_DLA34_coco.py)
python -m torch.distributed.launch --nproc_per_node=8  tools/train.py --launcher pytorch --config config_file
```

#### Test

```bash
#Replace config_file and checkpoint_file
python tools/test.py --config config_file --checkpoint checkpoint_file

#Visulize
#The output image with predicted pose will be saved in show_dir
 python tools/test.py --config config_file --checkpoint checkpoint_file --show-pose --show-dir show_dir
```

For multi-scale testing and flip, you are supposed to modify the config file (`cfg.test_pipeline.MultiScaleFlipAug`)



