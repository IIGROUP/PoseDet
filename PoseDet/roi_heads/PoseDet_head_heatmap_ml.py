import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init

from mmdet.core import (PointGenerator, build_assigner, build_sampler,
                        images_to_levels, multi_apply, unmap,
                        )
from PoseDet.keypoints_nms import keypoints_nms

from mmdet.ops import DeformConv
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from .PoseDet_head import PoseDetHead

@HEADS.register_module()
class PoseDetHeadHeatMapMl(PoseDetHead):
    def __init__(self, heatmap_convs=1, loss_heatmap=None, **args):
        self.heatmap_convs = heatmap_convs      
        super(PoseDetHeadHeatMapMl, self).__init__(**args)
        if loss_heatmap:
            self.with_loss_heatmap = True
            self.loss_heatmap = build_loss(loss_heatmap)
        else:
            self.with_loss_heatmap = False

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_pre_convs = nn.ModuleList()
        self.init_pre_convs = nn.ModuleList()
        self.refine_pre_convs = nn.ModuleList()
        self.heatmap_pre_convs = nn.ModuleList()

        for i in range(self.init_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            # chn = self.in_channels + self.num_keypoints if i == 0 else self.feat_channels
            self.init_pre_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))

        for i in range(self.cls_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            # chn = self.in_channels + self.num_keypoints if i == 0 else self.feat_channels
            self.cls_pre_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))

        for i in range(self.refine_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            # chn = self.in_channels + self.num_keypoints if i == 0 else self.feat_channels
            self.refine_pre_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))

        for i in range(self.heatmap_convs):
            self.heatmap_pre_convs.append(
                ConvModule(
                    self.in_channels,
                    self.in_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))

        self.cls_embedding_conv = DeformConv(self.feat_channels,
                                             self.embedding_feat_channels,
                                             self.dcn_kernel, 1, self.dcn_pad)
        self.cls_out = nn.Conv2d(self.embedding_feat_channels, self.cls_out_channels, 1, 1, 0)
 
        self.init_out = nn.Conv2d(self.feat_channels, self.num_keypoints*2, 1, 1, 0)

        self.refine_embedding_conv = DeformConv(self.feat_channels,
                                                    self.embedding_feat_channels,
                                                    self.dcn_kernel, 1,
                                                    self.dcn_pad)
        self.refine_out = nn.Conv2d(self.embedding_feat_channels, self.num_keypoints*2, 1, 1, 0)

        self.heatmap_out = nn.Conv2d(self.in_channels, self.num_keypoints, 1, 1, 0)


    def init_weights(self):
        super(PoseDetHeadHeatMapMl, self).init_weights()
        for m in self.heatmap_pre_convs:
            normal_init(m.conv, std=0.01)
        normal_init(self.heatmap_out, std=0.01)

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_keypoints=None,
                      gt_num_keypoints=None,
                      heatmap=None,
                      heatmap_weight=None):
        '''
        input:
        gt_keypoints:list[tensor], tensor:[N, 1, 51]
        ''' 

        outs = self(x)  
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, 
                            gt_keypoints=gt_keypoints,
                            gt_num_keypoints=gt_num_keypoints,
                            heatmap=heatmap,
                            heatmap_weight=heatmap_weight)
        return losses

    def forward(self, feats):

        return multi_apply(self.forward_single, feats)

    def forward_single(self, x):

        heatmap_feat = x
        for conv in self.heatmap_pre_convs:
            heatmap_feat = conv(heatmap_feat)
        heatmap = self.heatmap_out(heatmap_feat)

        dcn_base_offset = self.dcn_base_offset.type_as(x)

        # x = torch.cat((x, heatmap), dim=1)

        #fusion
        # attention = heatmap.detach().sigmoid().max(dim=1, keepdim=True)[0]
        # x = x*attention
        
        cls_feat = x
        init_feat = x
        refine_feat = x
        # score_feat = x

        for cls_conv in self.cls_pre_convs:
            cls_feat = cls_conv(cls_feat)
        for init_conv in self.init_pre_convs:
            init_feat = init_conv(init_feat)
        for refine_conv in self.refine_pre_convs:
            refine_feat = refine_conv(refine_feat)
        # for score_conv in self.score_pre_convs:
        #     score_feat = score_conv(score_feat)

        ######################### stage1 reg ###########################
        offset_init = self.init_out(init_feat)

        ######################### stage2 reg ###########################

        dcn_offset_init = (1 - self.gradient_mul) * offset_init.detach(
        ) + self.gradient_mul * offset_init


        dcn_offset_init = dcn_offset_init - dcn_base_offset

        #foward
        offset_refine = self.refine_embedding_conv(refine_feat, dcn_offset_init)
        offset_refine = self.relu(offset_refine)
        offset_refine = self.refine_out(offset_refine)
        offset_refine = offset_refine + offset_init.detach()
        # offset_refine = offset_refine + offset_init

        dcn_offset_refine = offset_refine.detach()

        dcn_offset_refine = dcn_offset_refine - dcn_base_offset

        for i in range(self.refine_num-1):
            offset_refine_ = self.reppoints_pts_refine_conv(refine_feat, dcn_offset_refine)
            offset_refine_ = self.relu(offset_refine_)
            offset_refine_ = self.reppoints_pts_refine_out(offset_refine_)

            offset_refine = offset_refine_.detach() + offset_refine.detach()
            dcn_offset_refine = offset_refine - dcn_base_offset

        ######################### cls ###########################
        # cls_out = self.reppoints_cls_conv(cls_feat, dcn_offset_refine)
        cls_out = self.cls_embedding_conv(cls_feat, dcn_offset_init)
        cls_out = self.relu(cls_out)
        cls_out = self.cls_out(cls_out)


        ####################### score ###########################
        # score_out = self.reppoints_score_conv(score_feat, dcn_offset_refine)
        # score_out = self.score_embedding_conv(score_feat, dcn_offset_init.detach())
        # score_out = self.reppoints_score_conv(cls_feat, dcn_offset_init)
        # score_out = self.relu(score_out)
        # score_out = self.score_out(score_out)

        return cls_out, offset_init, offset_refine, heatmap

    def loss(self,
             cls_scores,
             offset_init,
             offset_refine,
             heatmap_pred_list,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_keypoints=None,
             gt_num_keypoints=None,
             heatmap=None,
             heatmap_weight=None):
        loss_dict_all = super(PoseDetHeadHeatMapMl, self).loss(cls_scores,
                                                             offset_init,
                                                             offset_refine,
                                                             gt_bboxes,
                                                             gt_labels,
                                                             img_metas,
                                                             gt_keypoints,
                                                             gt_num_keypoints)
        # loss_dict_all['loss_keypoints_init'] = loss_keypoints_init
        heatmap_loss = torch.zeros(1, device=heatmap[0].device)
        for i, heatmap_pred in enumerate(heatmap_pred_list):
            heatmap_loss += self.loss_heatmap(heatmap_pred, heatmap[i], heatmap_weight[i])

        loss_dict_all['heatmap_loss'] = heatmap_loss/len(heatmap_pred_list)
        return loss_dict_all

    def get_bboxes(self,
                   cls_scores,
                   offset_preds_init,
                   offset_preds_refine,
                   heatmap_pred_list,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   nms=True):
        return super(PoseDetHeadHeatMapMl, self).get_bboxes(
                                                       cls_scores,
                                                       offset_preds_init,
                                                       offset_preds_refine,
                                                       img_metas,
                                                       cfg,
                                                       rescale,
                                                       nms)
