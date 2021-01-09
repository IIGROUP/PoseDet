import torch
import numpy as np
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner

@BBOX_ASSIGNERS.register_module()
class KeypointsAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each point.

    Each proposals will be assigned with `0`, or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    """

    def __init__(self, scale=4, pos_num=3, number_keypoints_thr=1, num_keypoints=17, center_type='box'):
        self.scale = scale
        self.pos_num = pos_num
        self.number_keypoints_thr = number_keypoints_thr
        self.num_keypoints = num_keypoints
        self.center_type=center_type
        self.count_lvl = np.zeros(7)
        self.count_iter = 0


    def assign(self, points, gt_labels, gt_keypoints, gt_bboxes=None):
        """Assign gt to points.

        This method assign a gt bbox to every points set, each points set
        will be assigned with  the background_label (-1), or a label number.
        -1 is background, and semi-positive number is the index (0-based) of
        assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every points to the background_label (-1)
        2. A point is assigned to some gt bbox if
            (i) the point is within the k closest points to the gt bbox
            (ii) the distance between this point and the gt is smaller than
                other gt bboxes

        Args:
            points (Tensor): points to be assigned, shape(n, 3) while last
                dimension stands for (x, y, stride).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
                NOTE: currently unused.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        num_points = points.shape[0]
        num_gts = gt_labels.shape[0]

        # gt_bboxes:[n, 4]
        # gt_keypoints:[n, 1, 3*keypoints_number]

        if num_gts == 0 or num_points == 0:
            # If no truth assign everything to the background
            assigned_gt_inds = points.new_full((num_points, ),
                                               0,
                                               dtype=torch.long)
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = points.new_full((num_points, ),
                                                  -1,
                                                  dtype=torch.long)
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels), None

        #generate box of keypoints
        gt_keypoints = gt_keypoints
        gt_bboxes = self.pts2box(gt_keypoints)

        #find center of keypoints
        gt_keypoints = gt_keypoints.reshape(len(gt_keypoints), -1, 3)
        gt_keypoints_flag = (gt_keypoints[:,:,2]!=0)
        gt_keypoints_xy = torch.zeros((len(gt_keypoints), 2), device=gt_keypoints.device)
        for i, k in enumerate(gt_keypoints_xy):
            gt_keypoints_xy[i] = gt_keypoints[i,gt_keypoints_flag[i],:2].mean(dim=0)

        gt_keypoints[:,:,2] = (gt_keypoints[:,:,2]!=0)
        keypoints_number = gt_keypoints.sum(dim=1)
        keypoints_number = keypoints_number[:,2]

        true_index = (keypoints_number>=self.number_keypoints_thr)

        points_xy = points[:, :2]
        points_stride = points[:, 2]
        points_lvl = torch.log2(
            points_stride).int()  # [3...,4...,5...,6...,7...]
        lvl_min, lvl_max = points_lvl.min(), points_lvl.max()
        # assign gt box
        if self.center_type == 'box':
            gt_bboxes_xy = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2
        elif self.center_type == 'keypoints':
            gt_bboxes_xy = gt_keypoints_xy

        gt_bboxes_wh = (gt_bboxes[:, 2:] - gt_bboxes[:, :2]).clamp(min=1e-6)
        scale = self.scale
        gt_bboxes_lvl = ((torch.log2(gt_bboxes_wh[:, 0] / scale) +
                          torch.log2(gt_bboxes_wh[:, 1] / scale)) / 2).int()
        # print(gt_bboxes_lvl)
        gt_bboxes_lvl = torch.clamp(gt_bboxes_lvl, min=lvl_min, max=lvl_max)

        #count gt sizes
        # for l in gt_bboxes_lvl:
        #     self.count_lvl[int(l)-1] += 1
        # self.count_iter += 1
        # if self.count_iter%100==0:
        #     print('level count:', self.count_lvl)
        #     print('level ratio:', self.count_lvl/np.sum(self.count_lvl))

        # stores the assigned gt index of each point
        assigned_gt_inds = points.new_zeros((num_points, ), dtype=torch.long)
        # stores the assigned gt dist (to this point) of each point
        assigned_gt_dist = points.new_full((num_points, ), float('inf'))
        points_range = torch.arange(points.shape[0])

        for idx in range(num_gts):
            if not true_index[idx]:
                continue
            gt_lvl = gt_bboxes_lvl[idx]
            # get the index of points in this level
            lvl_idx = gt_lvl == points_lvl
            points_index = points_range[lvl_idx]
            # get the points in this level
            lvl_points = points_xy[lvl_idx, :]
            # get the center point of gt
            gt_point = gt_bboxes_xy[[idx], :]
            # get width and height of gt
            gt_wh = gt_bboxes_wh[[idx], :]
            # compute the distance between gt center and
            #   all points in this level
            points_gt_dist = ((lvl_points - gt_point) / gt_wh).norm(dim=1)
            # find the nearest k points to gt center in this level
            min_dist, min_dist_index = torch.topk(
                points_gt_dist, self.pos_num, largest=False)
            # the index of nearest k points to gt center in this level
            min_dist_points_index = points_index[min_dist_index]
            # The less_than_recorded_index stores the index
            #   of min_dist that is less then the assigned_gt_dist. Where
            #   assigned_gt_dist stores the dist from previous assigned gt
            #   (if exist) to each point.
            less_than_recorded_index = min_dist < assigned_gt_dist[
                min_dist_points_index]
            # The min_dist_points_index stores the index of points satisfy:
            #   (1) it is k nearest to current gt center in this level.
            #   (2) it is closer to current gt center than other gt center.
            min_dist_points_index = min_dist_points_index[
                less_than_recorded_index]
            # assign the result
            assigned_gt_inds[min_dist_points_index] = idx + 1
            assigned_gt_dist[min_dist_points_index] = min_dist[
                less_than_recorded_index]

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_points, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None


        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels), None

    def pts2box(self, pts_gts):
        bbox_gt_pts = torch.zeros((len(pts_gts),4), dtype=pts_gts.dtype, device=pts_gts.device)
        if len(pts_gts) == 0:
            # return gt_bboxes_fake
            return bbox_gt_pts

        pts_gts = pts_gts.reshape((len(pts_gts), -1, 3))
        pts_gts_false = (pts_gts[:,:,2]==0) #skeleton类别为1和2参与计算
        pts_gts_true = (pts_gts[:,:,2]!=0)
        pts_gts = pts_gts[:,:,:2]

        for i in range(len(pts_gts)):
            # assert pts_gts_true.sum()!=0
            if not pts_gts_true[i].any():
                # bbox_gt_pts[i] = gt_bboxes_fake[i]
                continue
            pts_gts_x_max = pts_gts[i,pts_gts_true[i],0].max(dim=-1)[0]
            pts_gts_x_min = pts_gts[i,pts_gts_true[i],0].min(dim=-1)[0]
            pts_gts_y_max = pts_gts[i,pts_gts_true[i],1].max(dim=-1)[0]
            pts_gts_y_min = pts_gts[i,pts_gts_true[i],1].min(dim=-1)[0]
            bbox_gt_pts[i] = torch.stack((pts_gts_x_min, pts_gts_y_min, pts_gts_x_max, pts_gts_y_max), dim=-1)
        return bbox_gt_pts