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

@HEADS.register_module()
class PoseDetHead(BaseDenseHead):
    def __init__(self,
                norm_cfg=None,
                num_classes=1,
                in_channels=256,
                feat_channels=256,
                embedding_feat_channels=256,
                init_convs=1,
                refine_convs=1,
                cls_convs=1,
                gradient_mul=0.1,
                point_strides=[8, 16, 32, 64],
                point_base_scale=4,
                num_keypoints=17,
                refine_num=1,
                dcn_kernel=(3,3), #h,w
                loss_cls=None,
                loss_keypoints_init=None,
                loss_keypoints_refine=None,
                loss_heatmap=None,
                train_cfg=None,
                test_cfg=None
        ):
        super(PoseDetHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.embedding_feat_channels = embedding_feat_channels
        self.init_convs = init_convs
        self.refine_convs = refine_convs
        self.cls_convs = cls_convs
        self.num_keypoints = num_keypoints
        self.gradient_mul = gradient_mul
        self.point_base_scale = point_base_scale
        self.point_strides = point_strides
        self.norm_cfg = norm_cfg
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.background_label = num_classes
        self.cls_out_channels = num_classes
        #TO remove
        self.sampling = False
        self.use_sigmoid_cls = True
        self.use_grid_points = False
        self.center_init = True
        self.conv_cfg = None

        self.point_generators = [PointGenerator() for _ in self.point_strides]
        if self.train_cfg:
            self.init_assigner = build_assigner(self.train_cfg.init.assigner)
            self.refine_assigner = build_assigner(self.train_cfg.refine.assigner)
            self.cls_assigner = build_assigner(self.train_cfg.cls.assigner)
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.loss_cls = build_loss(loss_cls)
        if loss_keypoints_init:
            self.with_loss_keypoints_init = True
            self.loss_keypoints_init = build_loss(loss_keypoints_init)
        else:
            self.with_loss_keypoints_init = False

        if loss_keypoints_refine:
            self.with_loss_keypoints_refine = True
            self.loss_keypoints_refine = build_loss(loss_keypoints_refine)
        else:
            self.with_loss_keypoints_refine = False

        self.count=0
        self.neg = 0
        self.pos = 0
        self.center_proposal_iou = np.zeros(10)
        self.num_keypoints = num_keypoints
        self.refine_num = refine_num

        #initialzie dcn parameters
        self.dcn_kernel = dcn_kernel
        # self.dcn_pad = int((max(dcn_kernel[0], dcn_kernel[1])-1)/2)
        self.dcn_pad = (int((dcn_kernel[0]-1)/2),  int((dcn_kernel[1]-1)/2))

        assert num_keypoints%2==1 #only support odd number of keypoints

        dcn_base_offset = [[i,j] for i in range(-int(self.dcn_kernel[0]/2), int(self.dcn_kernel[0]/2)+1)
                            for j in range(-int(self.dcn_kernel[1]/2), int(self.dcn_kernel[1]/2)+1)]
        self.dcn_base_offset = torch.tensor(dcn_base_offset).float().reshape(1, -1, 1, 1)
        self._init_layers()

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_pre_convs = nn.ModuleList()
        self.init_pre_convs = nn.ModuleList()
        self.refine_pre_convs = nn.ModuleList()
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
 
        self.init_out = nn.Conv2d(self.feat_channels, self.num_keypoints*2, 1, 1, 0)

        self.refine_embedding_conv = DeformConv(self.feat_channels,
                                                    self.embedding_feat_channels,
                                                    self.dcn_kernel, 1,
                                                    self.dcn_pad)
        self.refine_out = nn.Conv2d(self.embedding_feat_channels, self.num_keypoints*2, 1, 1, 0)


    def init_weights(self):
        for m in self.cls_pre_convs:
            normal_init(m.conv, std=0.01)
        for m in self.init_pre_convs:
            normal_init(m.conv, std=0.01)
        for m in self.refine_pre_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.cls_embedding_conv, std=0.01)
        normal_init(self.cls_out, std=0.01, bias=bias_cls)
        normal_init(self.init_out, std=0.01)
        normal_init(self.refine_embedding_conv, std=0.01)
        normal_init(self.refine_out, std=0.01)

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_keypoints=None,
                      gt_num_keypoints=None,
                      **args,):
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
                            gt_num_keypoints=gt_num_keypoints)
        return losses

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def forward_single(self, x):
        dcn_base_offset = self.dcn_base_offset.type_as(x)
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
        # cls_out = self.cls_embedding_conv(cls_feat, dcn_offset_refine)
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

        (losses_cls, loss_keypoints_init, loss_keypoints_refine) = losses
        loss_dict_all = {
            'loss_cls': losses_cls,
        }

        if self.with_loss_keypoints_init:
            loss_dict_all['loss_keypoints_init'] = loss_keypoints_init

        if self.with_loss_keypoints_refine:
            loss_dict_all['loss_keypoints_refine'] = loss_keypoints_refine

        return loss_dict_all

    def loss_single(self, cls_score, pts_pred_init, pts_pred_refine, labels_cls, label_weights_cls,
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
        if self.with_loss_keypoints_init:
            if pos_index_init.any() > 0:
                pos_num = pos_index_init.sum()
                pos_pts_pred = pts_pred_init.reshape(-1, 2 * self.num_keypoints)[pos_index_init]
                pos_pts_pred = pos_pts_pred.reshape(pos_num, -1, 2)
                pos_keypoints_target = keypoints_gt_init.reshape(-1, self.num_keypoints*3)[pos_index_init]
                pos_keypoints_target = pos_keypoints_target.reshape(pos_num, -1, 3)
                loss_keypoints_init = self.loss_keypoints_init(pos_pts_pred/normalize_term, 
                                                                pos_keypoints_target/normalize_term,
                                                                avg_factor=num_total_samples_init
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

        return loss_cls, loss_keypoints_init, loss_keypoints_refine

    def get_targets(self,
                    valid_flag_list,
                    img_metas,
                    gt_bboxes_list,
                    gt_labels_list=None,
                    stage='init',
                    unmap_outputs=True,
                    gt_keypoints_list=None,
                    proposals_list=None,):
        assert stage in ['init', 'refine', 'cls']
        num_imgs = len(img_metas)
        assert len(proposals_list) == len(valid_flag_list) == num_imgs
        # points number of multi levels
        num_level_proposals = [points.size(0) for points in proposals_list[0]]

        # concat all level points and flags to a single tensor
        for i in range(num_imgs):
            assert len(proposals_list[i]) == len(valid_flag_list[i])
            proposals_list[i] = torch.cat(proposals_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])


        # compute targets for each image
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]

        (all_labels, all_label_weights, all_proposal_weights, 
                pos_inds_list, neg_inds_list, all_keypoints_gt, all_ious) = multi_apply(
             self._get_target_single,
             proposals_list,
             valid_flag_list,
             gt_bboxes_list,
             gt_labels_list,
             gt_keypoints_list,
             stage=stage,
             unmap_outputs=unmap_outputs,)

        # no valid points
        if any([labels is None for labels in all_labels]):
            return None
        # sampled points of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        labels_list = images_to_levels(all_labels, num_level_proposals)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_proposals)
        keypoints_gt_list = images_to_levels(all_keypoints_gt, num_level_proposals)
        proposal_weights_list = images_to_levels(all_proposal_weights,
                                                 num_level_proposals)

        if all_ious[0] is None:
            rois_list = None
        else:
            rois_list = images_to_levels(all_ious, num_level_proposals)

        return (labels_list, label_weights_list, proposal_weights_list, 
                num_total_pos, num_total_neg, keypoints_gt_list, rois_list)

    def _get_target_single(self,
                             flat_proposals,
                             valid_flags,
                             gt_bboxes,
                             gt_labels,
                             gt_keypoints,
                             stage='init',
                             unmap_outputs=True):
        inside_flags = valid_flags
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample proposals
        proposals = flat_proposals[inside_flags, :]

        if stage == 'init':
            assigner = self.init_assigner
            pos_weight = self.train_cfg.init.pos_weight
        elif stage=='refine':
            assigner = self.refine_assigner
            pos_weight = self.train_cfg.refine.pos_weight
        elif stage=='cls':
            assigner = self.cls_assigner
            pos_weight = self.train_cfg.cls.pos_weight

        assign_result, ious = assigner.assign(proposals, gt_labels, gt_keypoints, gt_bboxes=gt_bboxes)

        sampling_result = self.sampler.sample(assign_result, proposals, gt_bboxes)


        num_valid_proposals = proposals.shape[0]
        # bbox_gt = proposals.new_zeros([num_valid_proposals, 4])
        keypoints_gt = proposals.new_zeros([num_valid_proposals, self.num_keypoints*3])
        pos_proposals = torch.zeros_like(proposals)
        proposals_weights = proposals.new_zeros([num_valid_proposals, 1])
        labels = proposals.new_full((num_valid_proposals, ),
                                    self.background_label,
                                    dtype=torch.long)
        label_weights = proposals.new_zeros(
            num_valid_proposals, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        if len(pos_inds) > 0:
            pos_gt_bboxes = sampling_result.pos_gt_bboxes
            # bbox_gt[pos_inds, :] = pos_gt_bboxes
            keypoints_gt[pos_inds, :] = gt_keypoints[sampling_result.pos_assigned_gt_inds,0,:]
            # pos_proposals[pos_inds, :] = proposals[pos_inds, :]
            proposals_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of proposals
        if unmap_outputs:
            num_total_proposals = flat_proposals.size(0)
            labels = unmap(labels, num_total_proposals, inside_flags)
            label_weights = unmap(label_weights, num_total_proposals,
                                  inside_flags)
            # bbox_gt = unmap(bbox_gt, num_total_proposals, inside_flags)
            keypoints_gt = unmap(keypoints_gt, num_total_proposals, inside_flags)
            # pos_proposals = unmap(pos_proposals, num_total_proposals,
            #                       inside_flags)
            proposals_weights = unmap(proposals_weights, num_total_proposals,
                                      inside_flags)
            if ious is not None:
                ious = unmap(ious, num_total_proposals, inside_flags)

        # return (labels, label_weights, bbox_gt, pos_proposals,
        #         proposals_weights, pos_inds, neg_inds, keypoints_gt, ious)

        return (labels, label_weights, proposals_weights, pos_inds, neg_inds, keypoints_gt, ious)

    def get_bboxes(self,
                   cls_scores,
                   # pts_preds_init,
                   # pts_preds_refine,
                   offset_preds_init,
                   offset_preds_refine,
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
            proposals = self._get_bboxes_single(cls_score_list,
                                                mlvl_points, img_shape,
                                                scale_factor, cfg, rescale,
                                                nms, offset_preds=offset_pred_list)

            result_list.append(proposals)

        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
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
        mlvl_poses = []
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

            poses = offset_pred.new_zeros(len(offset_pred), self.num_keypoints*3)
            poses[:,0::3] = offset_pred_[:,:,1] * self.point_strides[i_lvl] + center_points[:,0::2]
            poses[:,1::3] = offset_pred_[:,:,0] * self.point_strides[i_lvl] + center_points[:,1::2]
            poses[:,2::3] = 1 

            mlvl_scores.append(scores)
            mlvl_poses.append(poses)

        mlvl_poses = torch.cat(mlvl_poses)
        mlvl_poses[:,0::3] = mlvl_poses[:,0::3].clamp(min=0, max=img_shape[1])
        mlvl_poses[:,1::3] = mlvl_poses[:,1::3].clamp(min=0, max=img_shape[0])

        if rescale:
            mlvl_poses[:,0::3] = mlvl_poses[:,0::3]/scale_factor[0]
            mlvl_poses[:,1::3] = mlvl_poses[:,1::3]/scale_factor[1]

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
                                                    cfg.max_per_img, 
                                                    multi_poses = mlvl_poses,
                                                    num_points=self.num_keypoints)
            return det_poses
        else:
            return mlvl_poses, mlvl_scores


    def offset_to_pts(self, center_list, pred_list, num_points=None):
        """Change from point offset to point coordinate.
        """
        pts_list = []
        if not num_points:
            num_points = self.num_points

        for i_lvl in range(len(self.point_strides)):
            pts_lvl = []
            for i_img in range(len(center_list)):
                pts_center = center_list[i_img][i_lvl][:, :2].repeat(
                    1, num_points)
                pts_shift = pred_list[i_lvl][i_img]
                yx_pts_shift = pts_shift.permute(1, 2, 0).reshape(
                    -1, 2 * num_points)
                y_pts_shift = yx_pts_shift[..., 0::2]
                x_pts_shift = yx_pts_shift[..., 1::2]
                xy_pts_shift = torch.stack([x_pts_shift, y_pts_shift], -1)
                xy_pts_shift = xy_pts_shift.reshape(*yx_pts_shift.shape[:-1], -1)
                pts = xy_pts_shift * self.point_strides[i_lvl] + pts_center
                pts_lvl.append(pts)
            pts_lvl = torch.stack(pts_lvl, 0)
            pts_list.append(pts_lvl)
        return pts_list

    def get_points(self, featmap_sizes, img_metas):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: points of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # points center for one time
        multi_level_points = []
        for i in range(num_levels):
            points = self.point_generators[i].grid_points(
                featmap_sizes[i], self.point_strides[i])
            multi_level_points.append(points)
        points_list = [[point.clone() for point in multi_level_points]
                       for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level grids
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                point_stride = self.point_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w = img_meta['pad_shape'][:2]
                valid_feat_h = min(int(np.ceil(h / point_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / point_stride)), feat_w)
                flags = self.point_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w))
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)

        return points_list, valid_flag_list

    def offset_to_pts(self, center_list, pred_list, num_points=None):
        """Change from point offset to point coordinate.
        """
        pts_list = []

        for i_lvl in range(len(self.point_strides)):
            pts_lvl = []
            for i_img in range(len(center_list)):
                pts_center = center_list[i_img][i_lvl][:, :2].repeat(
                    1, num_points)
                pts_shift = pred_list[i_lvl][i_img]
                yx_pts_shift = pts_shift.permute(1, 2, 0).reshape(
                    -1, 2 * num_points)
                y_pts_shift = yx_pts_shift[..., 0::2]
                x_pts_shift = yx_pts_shift[..., 1::2]
                xy_pts_shift = torch.stack([x_pts_shift, y_pts_shift], -1)
                xy_pts_shift = xy_pts_shift.reshape(*yx_pts_shift.shape[:-1], -1)
                pts = xy_pts_shift * self.point_strides[i_lvl] + pts_center
                pts_lvl.append(pts)
            pts_lvl = torch.stack(pts_lvl, 0)
            pts_list.append(pts_lvl)
        return pts_list