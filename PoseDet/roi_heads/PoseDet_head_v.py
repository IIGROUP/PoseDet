import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init

from mmdet.core import (PointGenerator, build_assigner, build_sampler,
                        images_to_levels, multi_apply, unmap,
                            # multiclass_nms
                        )
from PoseDet.keypoints_nms import keypoints_nms

from mmdet.ops import DeformConv
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from .PoseDet_head import PoseDetHead

@HEADS.register_module()
class PoseDetHeadV(PoseDetHead):

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_pre_convs = nn.ModuleList()
        self.init_pre_convs = nn.ModuleList()
        self.refine_pre_convs = nn.ModuleList()
        self.heatmap_pre_convs = nn.ModuleList()

        for i in range(self.init_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
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
            self.refine_pre_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))


        self.cls_embedding_conv = DeformConv(self.feat_channels,
                                             self.embedding_feat_channels,
                                             self.dcn_kernel, 1, self.dcn_pad)
        self.cls_out = nn.Conv2d(self.embedding_feat_channels, self.cls_out_channels, 1, 1, 0)
 
        self.init_out = nn.Conv2d(self.feat_channels, self.num_keypoints*3, 1, 1, 0)

        self.refine_embedding_conv = DeformConv(self.feat_channels,
                                                    self.embedding_feat_channels,
                                                    self.dcn_kernel, 1,
                                                    self.dcn_pad)
        self.refine_out = nn.Conv2d(self.embedding_feat_channels, self.num_keypoints*2, 1, 1, 0)

    def forward_single(self, x, heatmap_feat=None):
        dcn_base_offset = self.dcn_base_offset.type_as(x)
        cls_feat = x
        init_feat = x
        refine_feat = x

        for cls_conv in self.cls_pre_convs:
            cls_feat = cls_conv(cls_feat)
        for init_conv in self.init_pre_convs:
            init_feat = init_conv(init_feat)
        for refine_conv in self.refine_pre_convs:
            refine_feat = refine_conv(refine_feat)

        ######################### stage1 reg ###########################
        init_out = self.init_out(init_feat)
        offset_init = init_out[:,:self.num_keypoints*2]
        visible = init_out[:,self.num_keypoints*2:]

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

        return cls_out, offset_init, offset_refine, visible
    def loss(self,
             cls_scores,
             offset_init,
             offset_refine,
             visible,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_keypoints=None,
             gt_num_keypoints=None,):

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.point_generators)

        # Generate candidates and proposals for target assign
        # Three kinds of candidates/proposals are needed for stage init,refine and cls
        center_list, valid_flag_list = self.get_points(featmap_sizes, img_metas)
        pts_init_pred = self.offset_to_pts(center_list, offset_init, num_points=self.num_keypoints)
        pts_refine_pred = self.offset_to_pts(center_list, offset_refine, num_points=self.num_keypoints)
        candidate_list = center_list

        pts_init_pred_for_assign = []
        pts_refine_pred_for_assign = []
        for i_img, center in enumerate(center_list):
            pts_init_pred_img = []
            pts_refine_pred_img = []
            for i_lvl in range(len(pts_init_pred)):
                pts_init_pred_img.append(pts_init_pred[i_lvl][i_img].detach())
                pts_refine_pred_img.append(pts_refine_pred[i_lvl][i_img].detach())
            pts_init_pred_for_assign.append(pts_init_pred_img)
            pts_refine_pred_for_assign.append(pts_refine_pred_img)

        targets_init = self.get_targets(
            valid_flag_list.copy(),
            img_metas,
            gt_bboxes,
            gt_labels_list=gt_labels,
            stage='init',
            gt_keypoints_list=gt_keypoints,
            proposals_list=candidate_list)

        (_, _, pts_weights_list_init,
         num_total_pos_init, num_total_neg_init, keypoints_gt_list_init, ious_list_init) = targets_init

        num_total_samples_init = num_total_pos_init

        targets_refine = self.get_targets(
            valid_flag_list.copy(),
            img_metas,
            gt_bboxes,
            gt_labels_list=gt_labels,
            stage='refine',
            gt_keypoints_list=gt_keypoints,
            proposals_list=pts_init_pred_for_assign.copy(),)

        (_, _, pts_weights_list_refine, num_total_pos_refine,
         num_total_neg_refine, keypoints_gt_list_refine, ious_list_refine) = targets_refine
        num_total_samples_refine = num_total_pos_refine

        targets_cls = self.get_targets(
            valid_flag_list.copy(),
            img_metas,
            gt_bboxes,
            gt_labels_list=gt_labels,
            stage='cls',
            gt_keypoints_list=gt_keypoints,
            proposals_list=pts_init_pred_for_assign.copy(),
            # proposals_list=pts_refine_pred_for_assign.copy(),
            )

        (labels_list_cls, label_weights_list_cls, _, num_total_pos_cls, num_total_neg_cls, _, _) = targets_cls
        num_total_samples_cls = num_total_pos_cls

        # print('gt_keypoints', gt_keypoints[0].size())
        # compute loss
        losses = multi_apply(
            self.loss_single,
            cls_scores,
            pts_init_pred,
            pts_refine_pred,
            visible,
            labels_list_cls,
            label_weights_list_cls,
            pts_weights_list_init,
            keypoints_gt_list_init,
            pts_weights_list_refine,
            keypoints_gt_list_refine,
            self.point_strides,
            num_total_samples_cls=num_total_samples_cls,
            num_total_samples_init=num_total_samples_init,
            num_total_samples_refine=num_total_samples_refine,)
        # print('num_total_samples_cls', num_total_samples_cls)
        # print('num_total_samples_init', num_total_samples_init)
        # print('num_total_samples_refine', num_total_samples_cls)

        (losses_cls, loss_keypoints_init, loss_keypoints_refine, visible_loss) = losses
        loss_dict_all = {
            'loss_cls': losses_cls,
        }

        if self.with_loss_keypoints_init:
            loss_dict_all['loss_keypoints_init'] = loss_keypoints_init

        if self.with_loss_keypoints_refine:
            loss_dict_all['loss_keypoints_refine'] = loss_keypoints_refine

        loss_dict_all['loss_visible'] = visible_loss

        return loss_dict_all

    def loss_single(self, cls_score, pts_pred_init, pts_pred_refine, visible, labels_cls, label_weights_cls,
                    pts_weights_init, keypoints_gt_init, pts_weights_refine, keypoints_gt_refine, stride,
                    num_total_samples_cls, num_total_samples_init, num_total_samples_refine):

        labels_cls = labels_cls.reshape(-1)
        label_weights_cls = label_weights_cls.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score,
            labels_cls,
            label_weights_cls,
            avg_factor=num_total_samples_cls
            )

        pos_index_init = pts_weights_init.reshape(-1, 1)[:,0].bool()
        pos_index_refine = pts_weights_refine.reshape(-1, 1)[:,0].bool()


        #use assigne resluts of init for refine
        # pos_index_refine = pos_index_init
        # num_total_samples_refine = num_total_samples_init

        normalize_term = self.point_base_scale * stride
        
        loss_keypoints_init = torch.zeros_like(loss_cls)
        visible_loss = torch.zeros_like(loss_cls)
        if self.with_loss_keypoints_init:
            if pos_index_init.any() > 0:
                pos_num = pos_index_init.sum()
                pos_pts_pred = pts_pred_init.reshape(-1, 2 * self.num_keypoints)[pos_index_init]
                pos_pts_pred = pos_pts_pred.reshape(pos_num, -1, 2)
                visible_pred = visible.reshape(-1, 1 * self.num_keypoints)[pos_index_init]
                visible_pred = visible_pred.reshape(pos_num, -1, 1)
                pos_keypoints_target = keypoints_gt_init.reshape(-1, self.num_keypoints*3)[pos_index_init]
                pos_keypoints_target = pos_keypoints_target.reshape(pos_num, -1, 3)
                loss_keypoints_init, visible_loss = self.loss_keypoints_init(pos_pts_pred/normalize_term, 
                                                                pos_keypoints_target/normalize_term,
                                                                avg_factor=num_total_samples_init,
                                                                visible=visible_pred
                                                                )


        # hausdroff distance loss refine
        loss_keypoints_refine = torch.zeros_like(loss_cls)
        if self.with_loss_keypoints_refine:
            if pos_index_refine.any() > 0:
                pos_num = pos_index_refine.sum()
                pos_pts_pred = pts_pred_refine.reshape(-1, 2 * self.num_keypoints)[pos_index_refine]
                pos_pts_pred = pos_pts_pred.reshape(pos_num, -1, 2)
                pos_keypoints_target = keypoints_gt_refine.reshape(-1, self.num_keypoints*3)[pos_index_refine]
                pos_keypoints_target = pos_keypoints_target.reshape(pos_num, -1, 3)
                loss_keypoints_refine = self.loss_keypoints_refine(pos_pts_pred/normalize_term, 
                                                                    pos_keypoints_target/normalize_term,
                                                                    avg_factor=num_total_samples_refine
                                                                    )

        return loss_cls, loss_keypoints_init, loss_keypoints_refine, visible_loss

    def get_bboxes(self,
                   cls_scores,
                   offset_preds_init,
                   offset_preds_refine,
                   visible,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   nms=True):
        return super(PoseDetHeadV, self).get_bboxes(
                                                       cls_scores,
                                                       offset_preds_init,
                                                       offset_preds_refine,
                                                       img_metas,
                                                       cfg,
                                                       rescale,
                                                       nms)
