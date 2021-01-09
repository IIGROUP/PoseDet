import math
import torch
import numpy as np
from torch.nn import functional as F
import os
import time
from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import LOSSES

@LOSSES.register_module()
class KeypointsLoss(nn.Module):
    '''
    parameters:
        d_type: distance type to evaluater hausdroff distance
    input:
        pts_pred :Tensor[N1,2], normalized coord
        pts_target :Tensor[N2,2], normalized coord
    '''
    def __init__(self, d_type='smooth_L1', beta=1, 
                    alpha=0.5, modified_L2_th=0.2, weight=1, 
                    area_nor=False, stage='init', normalize_factor=1,
                    with_joint_weight=False, visible=False):
        super(KeypointsLoss, self).__init__()
        self.beta = beta
        self.alpha = alpha
        self.weight = weight
        self.area_nor = area_nor
        self.d_type = d_type
        # self.loss_num_keypoints = [[] for i in range(18)]
        self.total_count = 0
        self.stage = stage
        self.count = 0
        self.normalize_factor = normalize_factor
        self.joint_loss = np.zeros(17)
        self.count = np.zeros(17)
        self.with_joint_weight = with_joint_weight
        self.visible = visible

        self.debug_loss = [ [] for i in range(18)]
        self.count = 0

    def area(self, pts):
        # pts : tensor[n, (x,y)]
        pts_ = pts.view((-1,2))
        w = pts_[:,0].max() - pts_[:,0].min() + 1
        h = pts_[:,1].max() - pts_[:,1].min() + 1

        #防止骨骼点为1或者2时area太小
        w = w.clamp(32)
        h = h.clamp(32)
        area = (w*h).sqrt()

        return area

    def forward(self, pts_pred, pts_target, avg_factor=1, visible=None):
        '''
        pts_pred: tensor[n, point_number, 2]
        pts_target: tensor[n, point_number, 3]
        normalize_factor: 输入的pts经过归一化之后很小，smooth l1的梯度也很小，需要将其放大
        '''

        pts_pred *= self.normalize_factor
        pts_target *= self.normalize_factor
        point_loss = []
        for i, pts_pred_single in enumerate(pts_pred):
            pts_target_single = pts_target[i]
            valid_index = (pts_target_single[:,2]!=0) #为0的骨骼点不参与训练
            distance = pts_pred_single[valid_index] - pts_target_single[valid_index,:2]
            if len(distance)==0:
                return pts_pred.new_zeros(1)
            #L2
            if self.d_type=='L2':
                point_loss_single = torch.sum(distance**2, -1).sqrt()
            #smooth L1
            elif self.d_type=='smooth_L1':
                distance = torch.abs(distance)
                distance = torch.sum(distance, -1)
                point_loss_single = torch.where(distance < self.beta, self.alpha * distance * distance / self.beta,
                                   distance - self.alpha * self.beta)
            else:
                raise 'wrong distance type'

            if self.with_joint_weight:
                joint_weight = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5], device=pts_pred_single.device)
                point_loss_single = point_loss_single*joint_weight[valid_index]

            point_loss_single = point_loss_single.clamp(0, 5)

            # point_loss.append(point_loss_single.mean())
            point_loss.append(point_loss_single.sum())

            #analyze loss distribution
            # v_num = len(point_loss_single)
            # self.debug_loss[v_num].append(point_loss_single.mean()) 
            # self.count += 1
            # if self.count%100==0:
            #     for i, l in enumerate(self.debug_loss):
            #         if len(l) != 0:
            #             l2 = torch.stack(l)
            #             print(i, l2.mean())
            #         else:
            #             print(i, 0)


            # check loss of each joint
            # distance_all = pts_pred_single - pts_target_single[:,:2]
            # point_loss_single_all = torch.sum(distance_all**2, -1).sqrt()
            # point_loss_single_all[~valid_index] = 0
            # self.joint_loss += point_loss_single_all.detach().cpu().numpy()
            # self.count += valid_index.detach().cpu().numpy()
            # if self.count[0]%500==0:
            #     l = (self.joint_loss/self.count)*100
            #     l = l.astype(np.uint8)
            #     print(self.count)
            #     if self.weight == 1:
            #         print('Average joint loss in init stage:', l)
            #     else:
            #         print('Average joint loss in refine stage:', l)

        point_loss = torch.stack(point_loss).sum() * self.weight / avg_factor

        if self.visible:
            visible_target = pts_target[:,:,2:]
            visible_target[visible_target!=0] = 1
            visible_loss = F.binary_cross_entropy_with_logits(visible, visible_target) * self.weight * visible_target.size()[0] / avg_factor
            return point_loss, visible_loss
        else:
            return point_loss
