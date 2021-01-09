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
class PoseDetHeadHeatMap(PoseDetHead):
    def __init__(self, heatmap_convs=1, loss_heatmap=None, **args):
        self.heatmap_convs = heatmap_convs      
        super(PoseDetHeadHeatMap, self).__init__(**args)
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
            # chn = self.in_channels if i == 0 else self.feat_channels
            chn = self.in_channels + self.num_keypoints if i == 0 else self.feat_channels
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
            # chn = self.in_channels if i == 0 else self.feat_channels
            chn = self.in_channels + self.num_keypoints if i == 0 else self.feat_channels
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
            # chn = self.in_channels if i == 0 else self.feat_channels
            chn = self.in_channels + self.num_keypoints if i == 0 else self.feat_channels
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
        super(PoseDetHeadHeatMap, self).init_weights()
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
        heatmap_feat = feats[0]
        for conv in self.heatmap_pre_convs:
            heatmap_feat = conv(heatmap_feat)
        heatmap_feat = self.heatmap_out(heatmap_feat)

        cls_out_list, offset_init_list, offset_refine_list = multi_apply(self.forward_single, feats, heatmap_feat=heatmap_feat)

        return cls_out_list, offset_init_list, offset_refine_list, heatmap_feat

    def forward_single(self, x, heatmap_feat=None):
        dcn_base_offset = self.dcn_base_offset.type_as(x)
        heatmap_feat_stride = torch.nn.functional.interpolate(heatmap_feat.detach(), x.size()[-2:])

        x = torch.cat((x, heatmap_feat_stride), dim=1)
        
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

        return cls_out, offset_init, offset_refine

    def loss(self,
             cls_scores,
             offset_init,
             offset_refine,
             heatmap_feat,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_keypoints=None,
             gt_num_keypoints=None,
             heatmap=None,
             heatmap_weight=None):
        loss_dict_all = super(PoseDetHeadHeatMap, self).loss(cls_scores,
                                                             offset_init,
                                                             offset_refine,
                                                             gt_bboxes,
                                                             gt_labels,
                                                             img_metas,
                                                             gt_keypoints,
                                                             gt_num_keypoints)
        # loss_dict_all['loss_keypoints_init'] = loss_keypoints_init

        heatmap = torch.nn.functional.interpolate(heatmap, heatmap_feat.size()[-2:])
        heatmap_weight = torch.nn.functional.interpolate(heatmap_weight, heatmap_feat.size()[-2:])
        # heatmap_feat = torch.zeros_like(heatmap)
        # heatmap_loss = (heatmap - heatmap_feat) * heatmap_weight
        # heatmap_loss = heatmap_loss**2
        # heatmap_loss = heatmap_loss.mean()

        heatmap_loss = self.loss_heatmap(heatmap_feat, heatmap, heatmap_weight)

        loss_dict_all['heatmap_loss'] = heatmap_loss
        return loss_dict_all

    # def get_bboxes(self,
    #                cls_scores,
    #                offset_preds_init,
    #                offset_preds_refine,
    #                heatmap,
    #                img_metas,
    #                cfg=None,
    #                rescale=False,
    #                nms=True):
    #     return super(PoseDetHeadHeatMap, self).get_bboxes(
    #                                                    cls_scores,
    #                                                    offset_preds_init,
    #                                                    offset_preds_refine,
    #                                                    img_metas,
    #                                                    cfg,
    #                                                    rescale,
    #                                                    nms)
    def get_bboxes(self,
                   cls_scores,
                   offset_preds_init,
                   offset_preds_refine,
                   heatmap,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   nms=True):
        assert len(cls_scores) == len(offset_preds_refine)

        num_levels = len(cls_scores)
        mlvl_points = [
            self.point_generators[i].grid_points(cls_scores[i].size()[-2:],
                                                 self.point_strides[i])
            for i in range(num_levels)
        ]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]

            offset_pred_list = [
                offset_preds_refine[i][img_id].detach()
                for i in range(num_levels)
            ]

            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score_list, heatmap,
                                                mlvl_points, img_shape,
                                                scale_factor, cfg, rescale,
                                                nms, offset_preds=offset_pred_list)

            result_list.append(proposals)

        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           heatmap,
                           mlvl_points,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           nms=True,
                           offset_preds=None):
        cfg = self.test_cfg if cfg is None else cfg

        # assert len(cls_scores) == (bbox_preds) == len(mlvl_points)
        mlvl_scores = []
        mlvl_PointSets = []
        for i_lvl, (cls_score, points, offset_pred) in enumerate(
                zip(cls_scores, mlvl_points, offset_preds)):
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)

            offset_pred = offset_pred.permute(1, 2, 0).reshape(-1, self.num_keypoints*2)

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    # remind that we set FG labels to [0, num_class-1]
                    # since mmdet v2.0
                    # BG cat_id: num_class
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                offset_pred = offset_pred[topk_inds, :]
                scores = scores[topk_inds, :]

            center_points = points[:,:2].repeat((1, self.num_keypoints))
            offset_pred_ = offset_pred.view(-1, self.num_keypoints, 2)
            pts_pred = torch.stack((offset_pred_[:,:,1], offset_pred_[:,:,0]), dim=-1) 
            pts_pred = pts_pred.view(-1, self.num_keypoints*2)
            pts_pred = pts_pred * self.point_strides[i_lvl] + center_points

            for i in range(self.num_keypoints*2):
                shape_choice = (i+1)%2
                pts_pred[:, i] = pts_pred[:, i].clamp(min=0, max=img_shape[shape_choice])

            mlvl_scores.append(scores)
            mlvl_PointSets.append(pts_pred)

        mlvl_PointSets = torch.cat(mlvl_PointSets)
        if rescale:
            mlvl_PointSets /= mlvl_PointSets.new_tensor(scale_factor[:2].repeat((self.num_keypoints)))
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            # Add a dummy background class to the backend when using sigmoid
            # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
            # BG cat_id: num_class
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        if nms:
            det_poses, det_labels = keypoints_nms(mlvl_scores, 
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img, multi_PointSets=mlvl_PointSets,
                                                    num_points=self.num_keypoints)
            #rescore from heatmap
            # heatmap_ = heatmap.sigmoid()
            # H = heatmap.size()[2] * 8
            # W = heatmap.size()[3] * 8
            # # det_poses_score = mlvl_PointSets.clone()
            # # det_poses_score = det_poses_score.reshape(mlvl_PointSets.size()[0], 17, 2)
            # det_poses_score = det_poses[:,:34].clone()
            # det_poses_score = det_poses_score.reshape(det_poses.size()[0], 17, 2)

            # det_poses_score[:,:,0] = det_poses_score[:,:,0]/W*2 - 1 
            # det_poses_score[:,:,1] = det_poses_score[:,:,1]/H*2 - 1 
            # det_poses_score = det_poses_score[None, ...]

            # heatmap_score = torch.nn.functional.grid_sample(heatmap_, det_poses_score)
            # heatmap_score = heatmap_score.max(dim=1)[0]
            # heatmap_score = heatmap_score.max(dim=-1)[0][0]
            # heatmap_score = heatmap_score.clamp(max=1)
            # alpha = 0.5
            # # mlvl_scores[:,0] = alpha*heatmap_score + (1-alpha)*mlvl_scores[:,0]
            # det_poses[:,-1] = alpha*heatmap_score + (1-alpha)*det_poses[:,-1]

            # select_index = (det_poses[:,-1] > 0.1)

            return det_poses
        else:
            return mlvl_PointSets, mlvl_scores
