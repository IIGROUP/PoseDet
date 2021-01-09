import albumentations as A
import cv2
import os.path as osp
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch
from collections.abc import Sequence
from mmcv.parallel import DataContainer as DC
from mmdet.core import BitmapMasks, PolygonMasks
from mmdet.datasets.builder import PIPELINES
from PoseDet.utils import xfirst2yfirst

# Augment an image

@PIPELINES.register_module()
class Transform_package(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """
    def __init__(self):

        self.train_augmentator = A.Compose([
                # HorizontalFlip(p=0.5),
                # ShiftScaleRotate(shift_limit=0.06, scale_limit=0.06,
                #                  rotate_limit=(-9, 9), border_mode=0, p=0.8),
                # A.RandomResizedCrop(args.width, args.width, scale=(0.9, 1.05), ratio=(0.9, 1.1)),
                A.RandomBrightness(limit=(-0.18, 0.18), p=0.65),
                A.RandomContrast(limit=(-0.15, 0.15), p=0.65),
                A.RGBShift(r_shift_limit=9, g_shift_limit=9, b_shift_limit=9, p=0.65),
                A.IAAAdditiveGaussianNoise(scale=(1, 3), p=0.2)
            ], p=1.0)

    def __call__(self, results):

        transformed = self.train_augmentator(image=results['img'])
        results['img'] = transformed["image"]

        return results

    def __repr__(self):
        return self.__class__.__name__
