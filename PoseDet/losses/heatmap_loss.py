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
class HeatmapLoss(nn.Module):
    def __init__(self, weight=1, with_sigmas=False):
        super(HeatmapLoss, self).__init__()
        self.weight = weight
        self.with_sigmas = with_sigmas

    def forward(self, heatmap_feat, heatmap, heatmap_weight):

        ''' Modified focal loss
        heatmap_feat (batch x c x h x w)
        heatmap (batch x c x h x w)
        '''
        # pos_inds = heatmap.eq(1).float() * heatmap_weight
        # neg_inds = heatmap.lt(1).float() * heatmap_weight
        
        heatmap_feat = heatmap_feat.sigmoid()
        pos_inds = heatmap.eq(1).float() * heatmap_weight.float()
        neg_inds = heatmap.lt(1).float() * heatmap_weight.float()

        neg_weights = torch.pow(1 - heatmap, 4)

        loss = 0

        pos_loss = torch.log(heatmap_feat) * torch.pow(1 - heatmap_feat, 2) * pos_inds
        neg_loss = torch.log(1 - heatmap_feat) * torch.pow(heatmap_feat, 2) * neg_weights * neg_inds

        if self.with_sigmas:
            sigmas = torch.tensor([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89], device=heatmap_feat.device)/10.0
            sigmas = sigmas[None,...,None,None]
            pos_loss *= sigmas
            neg_loss *= sigmas

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss * self.weight

