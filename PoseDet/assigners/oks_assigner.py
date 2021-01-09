import torch

from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.iou_calculators import build_iou_calculator
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner
from mmdet.core.bbox.assigners.max_iou_assigner import MaxIoUAssigner
import numpy as np
from PoseDet.utils import computeOks

@BBOX_ASSIGNERS.register_module()
class OksAssigner(MaxIoUAssigner):
    def __init__(self,
                 pos_PD_thr,
                 neg_PD_thr,
                 min_pos_iou=.0,
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True,
                 match_low_quality=True,
                 gpu_assign_thr=-1,
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 num_keypoints = 17,
                 number_keypoints_thr = 1,
                 normalize_factor=0,
                 **args,):
        self.pos_iou_thr = pos_PD_thr
        self.neg_iou_thr = neg_PD_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.match_low_quality = match_low_quality
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.num_keypoints = num_keypoints
        self.number_keypoints_thr = number_keypoints_thr
        self.normalize_factor = normalize_factor
        self.max_iou_analyze = np.zeros(11)
        self.count = 0
        self.gt_fitted = np.zeros(11)

    def assign(self, pts_preds, gt_labels, gt_pts, gt_bboxes=None):

        overlaps = computeOks(gt_pts, pts_preds, gt_bboxes, 
                                    num_keypoints=self.num_keypoints,
                                    number_keypoints_thr=self.number_keypoints_thr,
                                    normalize_factor=self.normalize_factor,)

        #analyze how much is gt fitted
        # if len(overlaps)!=0:
        #     overlaps_max, overlaps_max_index = overlaps.max(dim=1)
        #     thr = [0.5, 0.75]
        #     # thr = [0.1 + i*0.1 for i in range(9)]
        #     for i in range(len(thr)):
        #         above = len(torch.where(overlaps_max>thr[i])[0])
        #         self.gt_fitted[i] += above
        #     num = len(overlaps_max)
        #     self.gt_fitted[-1] += num
        #     self.count += 1
        #     if self.count%100 == 0:
        #         show = (self.gt_fitted[:-1]/self.gt_fitted[-1])*100
        #         show = show.astype(int)
        #         if self.pos_iou_thr==0.7:
        #             print('overlaps_max init ',thr, show)
        #         else:
        #             print('overlaps_max refine',thr, show)


        #analyze distribution of max_iou
        # if len(overlaps) != 0:
        #     overlaps_max, overlaps_max_index = overlaps.max(dim=0)
        #     overlaps_max_sort, overlaps_max_sort_index = torch.sort(overlaps_max)

        #     for i in range(11):
        #         thr = i*0.1
        #         remain_index = torch.where(overlaps_max_sort>thr)[0]
        #         num = len(overlaps_max_sort) - len(remain_index)
        #         overlaps_max_sort = overlaps_max_sort[remain_index]
        #         self.max_iou_analyze[i] += num
        #     self.count += 1
        #     if self.count%100==0:
        #         # result = self.max_iou_analyze/self.max_iou_analyze.sum()
        #         # result = result*1000
        #         # result = result.astype(int)
        #         result = self.max_iou_analyze/self.count
        #         result = result.astype(int)
        #         if self.pos_iou_thr==0.7:
        #             print('max_iou_analyze init stage', result)
        #         else:
        #             print('max_iou_analyze refine stage', result)

        # debug show keypoints
        # import cv2
        # import numpy as np

        # # overlaps_max_iou, overlaps_max_index_iou = overlaps_iou.max(dim=0)
        # # overlaps_max_sort_iou, overlaps_max_sort_index_iou = torch.sort(overlaps_max_iou)

        # # overlaps_max_oks, overlaps_max_index_oks = overlaps_oks.max(dim=0)
        # # overlaps_max_sort_oks, overlaps_max_sort_index_oks = torch.sort(overlaps_max_oks)

        # overlaps_max, overlaps_max_index = overlaps.max(dim=0)
        # overlaps_max_sort, overlaps_max_sort_index = torch.sort(overlaps_max)

        # np.random.seed(12345)
        # color1 = np.random.randint(0, 256, (3), dtype=int)
        # color2 = np.random.randint(0, 256, (3), dtype=int)
        # iou_neg = 0.65
        # iou_pos = 0.6

        # neg_index = torch.where(overlaps_max_sort<iou_neg)[0][-1]
        # pos_index = torch.where(overlaps_max_sort>iou_pos)[0]
        # print('------------overlaps_max_sort---------------')
        # print('gt num', len(overlaps))
        # print('pos num', len(pos_index))
        # if len(pos_index) != 0:
        #     pos_index = pos_index[0]
        #     for i in range(20):
        #         index = i+pos_index
        #         # print(overlaps_max_sort[index])
        #         if index > len(overlaps_max_sort_index) - 1:
        #             break
        #         if overlaps_max_sort[index] >= 0.7:
        #             break

        #         pts1 = gt_pts[overlaps_max_index[overlaps_max_sort_index[index]]]
        #         pts2 = pts_preds[overlaps_max_sort_index[index]]
        #         print(index, self.count, overlaps_max_sort[index])
        #         img = np.zeros([1200,1200,3])
        #         pts1 = pts1.detach().cpu().numpy()
        #         pts1 = pts1.reshape((-1, 3))
        #         pts2 = pts2.detach().cpu().numpy()
        #         pts2 = pts2.reshape((-1,2))
        #         # print('pts1', pts1.shape)
        #         # print('pts2', pts2.shape)
        #         for j in range(len(pts2)):
        #             cv2.circle(img, (pts2[j][0], pts2[j][1]), radius=3, color=(int(color1[0]),int(color1[1]),int(color1[2])), thickness=4)

        #         for j in range(len(pts1)):
        #             cv2.circle(img, (pts1[j][0], pts1[j][1]), radius=3, color=(int(color2[0]),int(color2[1]),int(color2[2])), thickness=4)
        #         cv2.imwrite('./debug_img/keypoint%d_%d.jpg'%(index,self.count), img)
        #     self.count += 1

        #     if self.count == 10:
        #         exit()

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            if self.ignore_wrt_candidates:
                ignore_overlaps = self.iou_calculator(
                    bboxes, gt_bboxes_ignore, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = self.iou_calculator(
                    gt_bboxes_ignore, bboxes, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1


        assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)

        # test statistic assign result
        # ignore_preds = (assign_result.gt_inds==-1)
        # neg_preds = (assign_result.gt_inds==0)
        # pos_preds = (assign_result.gt_inds>0)
        # if self.pos_iou_thr==0.7:
        #     print('-----refine stage-----')
        # else:
        #     print('------init stage------')
        # print('total preds', len(assign_result.gt_inds))
        # print('neg pred', neg_preds.sum())
        # print('ignore pred', ignore_preds.sum())
        # print('pos pred', pos_preds.sum())


        # if assign_on_cpu:
        #     assign_result.gt_inds = assign_result.gt_inds.to(device)
        #     assign_result.max_overlaps = assign_result.max_overlaps.to(device)
        #     if assign_result.labels is not None:
        #         assign_result.labels = assign_result.labels.to(device)
        if len(overlaps) != 0:
            overlaps_max = overlaps.max(dim=0)[0].detach()
            # overlaps_max = overlaps[0] #get iou in terms of gt0
        else:
            overlaps_max = torch.zeros(len(pts_preds), device=pts_preds.device)
        return assign_result, overlaps_max
        # return assign_result, overlaps

    def enlarge_small_box(self, boxes):
        w = boxes[:,2] - boxes[:,0]
        h = boxes[:,3] - boxes[:,1]
        enlarge_w = (w<32)
        boxes[enlarge_w, 0] -= (18-w[enlarge_w]/2)
        boxes[enlarge_w, 2] += (18-w[enlarge_w]/2)
        enlarge_h = (h<32)
        boxes[enlarge_h, 1] -= (18-h[enlarge_h]/2)
        boxes[enlarge_h, 3] += (18-h[enlarge_h]/2)
        return boxes


