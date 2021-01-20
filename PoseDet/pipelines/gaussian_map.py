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

@PIPELINES.register_module()
class GaussianMap(object):

    # def __init__(self, sigma=7.0, num_keypoints=17, pos_weight=1, SCALE=False, scale_factor=100):
    #     self.sigma = sigma
    #     self.num_keypoints = num_keypoints
    #     self.count=0
    #     self.pos_weight=pos_weight
    #     self.SCALE = SCALE
    #     self.scale_factor = scale_factor



    def __init__(self, strides, num_keypoints=17, sigma=2):
        self.strides = strides
        self.num_keypoints = num_keypoints
        self.sigma = sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        self.count=0


    def __call__(self, results):
        img_shape = results['pad_shape']
        #TODO: remove this
        assert img_shape[0] == img_shape[1]
        keypoints = results['gt_keypoints'] #List: [N,1,51]
        boxes = results['gt_bboxes']
        gt_num_keypoints = results['gt_num_keypoints']
        heatmap_weight_list = []
        GaussianMap_list = []
        for stride in self.strides:
            output_scale = int(img_shape[0]/stride)
            heatmap_weight = self.get_map_weight(output_scale, boxes, gt_num_keypoints, stride)
            GaussianMap = self.get_gaussian_map(output_scale, keypoints, stride)
            heatmap_weight_list.append(heatmap_weight)
            GaussianMap_list.append(GaussianMap)


        # visulize
        # img_single = results['img']
        # max_, min_ = img_single.max(), img_single.min()
        # img_single = (img_single-min_)/(max_ - min_) * 255
        # img_single = img_single.astype(np.uint8)
        # cv2.imwrite('./debug_img/img%d.jpg'%self.count, img_single)

        # m = GaussianMap_list[0].sum(axis=0)
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


        results['heatmap'] = GaussianMap_list
        results['heatmap_weight'] = heatmap_weight_list
        return results

    def get_map_weight(self, output_scale, boxes, gt_num_keypoints, stride):
        heatmap_weight = np.ones([1, output_scale, output_scale])
        boxes = np.array(boxes)/stride
        for i, n in enumerate(gt_num_keypoints):
            if n==0:
                x, y, x2, y2 = boxes[i]
                heatmap_weight[:, int(y):int(y2), int(x):int(x2)] = 0

        return heatmap_weight

    def get_gaussian_map(self, output_scale, keypoints, stride):
        hms = np.zeros((self.num_keypoints, output_scale, output_scale),
                       dtype=np.float32)
        sigma = self.sigma
        joints = np.array(keypoints).reshape((len(keypoints), -1, 3))
        joints[:,:,:2] = joints[:,:,:2]/stride

        for p in joints:
            for idx, pt in enumerate(p):
                if pt[2] > 0:
                    x, y = int(pt[0]), int(pt[1])
                    if x < 0 or y < 0 or \
                       x >= output_scale or y >= output_scale:
                        continue

                    ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                    br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

                    c, d = max(0, -ul[0]), min(br[0], output_scale) - ul[0]
                    a, b = max(0, -ul[1]), min(br[1], output_scale) - ul[1]

                    cc, dd = max(0, ul[0]), min(br[0], output_scale)
                    aa, bb = max(0, ul[1]), min(br[1], output_scale)
                    hms[idx, aa:bb, cc:dd] = np.maximum(
                        hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])
        return hms

# class HeatmapGenerator():
#     def __init__(self, output_res, num_joints, sigma=-1):
#         self.output_res = output_res
#         self.num_joints = num_joints
#         if sigma < 0:
#             sigma = self.output_res/64
#         self.sigma = sigma
#         size = 6*sigma + 3
#         x = np.arange(0, size, 1, float)
#         y = x[:, np.newaxis]
#         x0, y0 = 3*sigma + 1, 3*sigma + 1
#         self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

#     def __call__(self, joints):
#         hms = np.zeros((self.num_joints, self.output_res, self.output_res),
#                        dtype=np.float32)
#         sigma = self.sigma
#         for p in joints:
#             for idx, pt in enumerate(p):
#                 if pt[2] > 0:
#                     x, y = int(pt[0]), int(pt[1])
#                     if x < 0 or y < 0 or \
                       # x >= self.output_res or y >= self.output_res:
#                         continue

#                     ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
#                     br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

#                     c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
#                     a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

#                     cc, dd = max(0, ul[0]), min(br[0], self.output_res)
#                     aa, bb = max(0, ul[1]), min(br[1], self.output_res)
#                     hms[idx, aa:bb, cc:dd] = np.maximum(
#                         hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])
#         return hms