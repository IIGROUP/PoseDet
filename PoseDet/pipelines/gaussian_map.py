import os.path as osp

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch
from collections.abc import Sequence
from mmcv.parallel import DataContainer as DC
from mmdet.core import BitmapMasks, PolygonMasks
from mmdet.datasets.builder import PIPELINES
import cv2
from PoseDet.utils import xfirst2yfirst

@PIPELINES.register_module()
class GaussianMap(object):

    def __init__(self, sigma=7.0, num_keypoints=17, pos_weight=1, SCALE=False, scale_factor=100):
        self.sigma = sigma
        self.num_keypoints = num_keypoints
        self.count=0
        self.pos_weight=pos_weight
        self.SCALE = SCALE
        self.scale_factor = scale_factor

    def __call__(self, results):
        img_shape = results['pad_shape']
        keypoints = results['gt_keypoints'] #List: [N,1,51]
        boxes = results['gt_bboxes']
        gt_num_keypoints = results['gt_num_keypoints']

        GaussianMap, heatmap_weight = self.get_map_weight(img_shape, keypoints, boxes, gt_num_keypoints)

        #old version
        # GaussianMap = np.zeros([img_shape[0], img_shape[1], self.num_keypoints])
        # for keypoint in keypoints:
        #     for i in range(self.num_keypoints):
        #         k = keypoint[0][i*3:(i+1)*3]
        #         if k[2] == 2:
        #             GaussianMap[:,:,i] += self.put_gaussian_map(k[:2], img_shape)

        # GaussianMap[GaussianMap > 1.0] = 1.0

        # heatmap_weight = np.ones([img_shape[0], img_shape[1]])
        # for i, n in enumerate(gt_num_keypoints):
        #     if n==0:
        #         x, y, x2, y2 = boxes[i]
        #         heatmap_weight[int(y):int(y2), int(x):int(x2)] = 0

        # pos = (GaussianMap.sum(axis=-1)>0)
        # heatmap_weight[pos] *= self.pos_weight

        # visulize
        # img_single = results['img']
        # max_, min_ = img_single.max(), img_single.min()
        # img_single = (img_single-min_)/(max_ - min_) * 255
        # img_single = img_single.astype(np.uint8)
        # cv2.imwrite('./debug_img/img%d.jpg'%self.count, img_single)

        # m = GaussianMap.sum(axis=-1)
        # m *= 50
        # m = m.astype(np.uint8)
        # cv2.imwrite('./debug_img/gausssionmap%d.jpg'%self.count, m)
        # m = heatmap_weight
        # m *= 50
        # m = m.astype(np.uint8)
        # # cv2.imwrite('./debug_img/heatmap_weight%d.jpg'%self.count, m)       
        # self.count += 1
        # if self.count==10:
        #     exit()


        results['heatmap'] = GaussianMap
        results['heatmap_weight'] = heatmap_weight
        return results

    def get_map_weight(self, img_shape, keypoints, boxes, gt_num_keypoints):
        GaussianMap = np.zeros([img_shape[0], img_shape[1], self.num_keypoints])

        for n, keypoint in enumerate(keypoints):
            x, y, x2, y2 = boxes[n]
            size = ((x2-x)*(y2-y))**0.5
            for i in range(self.num_keypoints):
                k = keypoint[0][i*3:(i+1)*3]
                if k[2] == 2:
                    GaussianMap[:,:,i] += self.put_gaussian_map(k[:2], img_shape, size)

        GaussianMap[GaussianMap > 1.0] = 1.0

        heatmap_weight = np.ones([img_shape[0], img_shape[1]])
        for i, n in enumerate(gt_num_keypoints):
            if n==0:
                x, y, x2, y2 = boxes[i]
                heatmap_weight[int(y):int(y2), int(x):int(x2)] = 0

        pos = (GaussianMap.sum(axis=-1)>0)
        heatmap_weight[pos] *= self.pos_weight
        return GaussianMap, heatmap_weight

    def put_gaussian_map(self, center, img_shape, size=None):
        grid_y = img_shape[0]
        grid_x = img_shape[1]
        start = 0
        y_range = [i for i in range(grid_y)]
        x_range = [i for i in range(grid_x)]
        xx, yy = np.meshgrid(x_range, y_range)
        d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
        exponent = d2 / 2.0 / self.sigma / self.sigma
        if self.SCALE:
            # print(exponent)
            exponent /= size/self.scale_factor
            # print(exponent)
            # exit()
        mask = exponent <= 4.6052
        cofid_map = np.exp(-exponent)
        cofid_map = np.multiply(mask, cofid_map)
        return cofid_map