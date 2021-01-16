
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset

import itertools
import logging
import os.path as osp
import tempfile

import mmcv
import numpy as np
from mmcv.utils import print_log
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval

from PoseDet.datasets.CrowdPose_toolkits.coco import COCO
from PoseDet.datasets.CrowdPose_toolkits.cocoeval import COCOeval

from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from collections import defaultdict
import gc
import os
from PoseDet.utils import add_keypoints_flag
import json
import torch

def computeOks(gt_pts, pts_preds, gt_bboxes=None, keypoints_num=17, number_keypoints_thr=1, normalize_factor=0):
    # gt_pts:Tensor[n, points_num, 3]
    # pts_preds:Tensor[m, points_num, 2]
    # return oks:Tensor[n, m]
    oks = torch.zeros((len(gt_pts), len(pts_preds)), device=pts_preds.device)
    if len(gt_pts)==0:
        return oks
    if keypoints_num==17:
        sigmas = torch.tensor([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89], device=pts_preds.device)/10.0
    elif keypoints_num==14: 
        sigmas = torch.tensor([.79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89, .79, .79], device=pts_preds.device)/10.0
        # gt_pts = gt_pts[:,:14*3]
        # pts_preds = pts_preds[:,:14*2]

    # sigmas = torch.from_numpy(self.kpt_oks_sigmas, device=pts_preds.device)
    vars = (sigmas * 2)**2
    vars = vars.unsqueeze(0)

    # if area_interval:
    #     gt_bboxes_wh = (gt_bboxes[:, 2:] - gt_bboxes[:, :2]).clamp(min=1e-6)
    #     gt_bboxes_lvl = ((torch.log2(gt_bboxes_wh[:, 0] / scale) +
    #                       torch.log2(gt_bboxes_wh[:, 1] / scale)) / 2).int()
    #     gt_bboxes_lvl = gt_bboxes_lvl.clamp(stride_minmax[0], stride_minmax[1])
    #     areas = 2**(gt_bboxes_lvl)*scale
    #     areas = areas**2
    # else:
    gt_bboxes_wh = (gt_bboxes[:, 2:] - gt_bboxes[:, :2])
    enlarg_small = (256 - gt_bboxes_wh).clamp(min=0)
    gt_bboxes_wh += enlarg_small*normalize_factor#增加小目标的oks，对小目标友好
    areas = gt_bboxes_wh[:,0]*gt_bboxes_wh[:,1]

    # gt_pts = gt_pts.reshape(len(gt_pts), -1, 3)
    # pts_preds = pts_preds.view(len(pts_preds), -1, 2)
    gt_pts_v = gt_pts[:,:,2]
    gt_pts_v = (gt_pts_v!=0)
    gt_pts = gt_pts[:,:,:2]
    gt_pts = gt_pts.unsqueeze(1)
    pts_preds = pts_preds.unsqueeze(0)
    distance = ((gt_pts - pts_preds)**2).sum(dim=-1)

    for i in range(len(gt_pts)):
        n = len(torch.nonzero(gt_pts_v[i]))
        if n<number_keypoints_thr:
            oks[i,:] = -1 #这种不参与assigner的正负样本分类
        else:
            ditance_ = torch.exp(-distance[i]/vars/areas[i])[:, gt_pts_v[i]]
            oks[i] = ditance_.mean(dim=-1)

    return oks

def pts2box(pts_gts):
    bbox_gt_pts = torch.zeros((len(pts_gts),4), dtype=pts_gts.dtype)
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


def oks_nms(keypoints, thresh):
    keypoints = np.array(keypoints).reshape(-1, keypoints_num*3)
    keypoints = torch.from_numpy(keypoints)
    keypoints = keypoints.view((keypoints.size()[0], -1, 3))
    pointsets = keypoints[:,:,:2]
    scores = torch.ones(pointsets.size()[0])
    bboxes = pts2box(keypoints)

    oks = computeOks(keypoints, pointsets, bboxes, keypoints_num=keypoints_num)
    # vars = (sigmas * 2)**2
    # vars = vars.unsqueeze(0).unsqueeze(0) #[1, 1, 17]

    # w_all = torch.max(pointsets[:,:,0], dim=1)[0] - torch.min(pointsets[:,:,0], dim=1)[0]
    # h_all = torch.max(pointsets[:,:,1], dim=1)[0] - torch.min(pointsets[:,:,1], dim=1)[0]
    # areas = w_all*h_all
    # areas = areas.clamp(32*32)
    # areas = (areas.unsqueeze(0)+areas.unsqueeze(1))/2
    # areas = areas.unsqueeze(-1) #[points_num, points_num, 1]

    # distance = ((pointsets.unsqueeze(0) - pointsets.unsqueeze(1))**2).sum(dim=-1) # [m, m, points_num]
    # oks = torch.exp(-distance/vars/areas).mean(dim=-1)

    keep = []
    index = scores.sort(descending=True)[1]  

    while index.size()[0] >0:
        i = index[0]       # every time the first is the biggst, and add it directly
        keep.append(i)
        if index.size()[0] == 1:
            break       
        oks_selected = torch.index_select(oks[i], 0, index)
        idx = torch.where(oks_selected<thresh)[0]
         
        index = index[idx]

    keep = torch.stack(keep)     
    # if len(keep) < len(keypoints):
    #     print(oks)
    #     print(keep)
    keypoints = keypoints[keep]
    keypoints = keypoints.resize(keypoints.size()[0], keypoints_num*3)
    keypoints = keypoints.numpy().tolist()
    # if len(keep) < len(keypoints):
    #     print(oks)
    #     print(keep)
    # exit()
    return keypoints

def iou_nms(keypoints, thresh):
    keypoints = np.array(keypoints).reshape(-1, keypoints_num*3)
    keypoints = torch.from_numpy(keypoints)
    keypoints = keypoints.view((keypoints.size()[0], -1, 3))
    bboxes = pts2box(keypoints)
    # pointsets = keypoints[:,:,:2]
    scores = torch.ones(keypoints.size()[0])
    # x_min, x_max = pointsets[:,:,0].min(dim=-1)[0], pointsets[:,:,0].max(dim=-1)[0]
    # y_min, y_max = pointsets[:,:,1].min(dim=-1)[0], pointsets[:,:,1].max(dim=-1)[0]
    # bboxes = torch.stack((x_min, y_min, x_max, y_max), dim=-1)
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2-x1)*(y2-y1)   # [N,] 每个bbox的面积
    _, order = scores.sort(0, descending=True)    # 降序排列

    keep = []
    while order.numel() > 0:       # torch.numel()返回张量元素个数
        if order.numel() == 1:     # 保留框只剩一个
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()    # 保留scores最大的那个框box[i]
            keep.append(i)

        # 计算box[i]与其余各框的IOU(思路很好)
        xx1 = x1[order[1:]].clamp(min=x1[i])   # [N-1,]
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        inter = (xx2-xx1).clamp(min=0) * (yy2-yy1).clamp(min=0)   # [N-1,]

        iou = inter / (areas[i]+areas[order[1:]]-inter)  # [N-1,]
        idx = (iou <= thresh).nonzero().squeeze() # 注意此时idx为[N-1,] 而order为[N,]
        if idx.numel() == 0:
            break
        order = order[idx+1]  # 修补索引之间的差值
    keep = torch.LongTensor(keep)   # Pytorch的索引值为LongTensor


    keypoints = keypoints[keep]
    keypoints = keypoints.resize(keypoints.size()[0], keypoints_num*3)
    keypoints = keypoints.numpy().tolist()
    return keypoints

# data_root = '/mnt/data/tcy/coco/'
data_root = '/mnt/data/tcy/CrowdPose/'

# file = 'person_keypoints_val2017.json'
# file = 'person_keypoints_train2017.json'
file = 'crowdpose_test.json'
keypoints_num = 14
ann_file = os.path.join(data_root, 'annotations', file)

with open(ann_file) as f:
    dataset = json.load(f)

images = dataset['images']
annotations = dataset['annotations'] 
gt_keypoints_list = []
image_id_list = []
for image in images:
    image_id = image['id']
    gt_keypoints = []
    for ann in annotations:
        if image_id == ann['image_id'] :
            k = ann['keypoints']
            k = np.array(k).reshape(-1, 3)
            n = np.sum(k[:,2])
            if n > 1:
            # if len(ann['keypoints']) != 0:
                gt_keypoints.append(ann['keypoints'])

    gt_keypoints_list.append(gt_keypoints)
    image_id_list.append(image_id)


for j in range(10):
    print('------------')
    thr = 1 - j*0.1
    print('thr', thr)
    # thr = 0.5
    results_keypoints = []
    for i, gt_keypoints in enumerate(gt_keypoints_list):
        image_id = image_id_list[i]
        results = oks_nms(gt_keypoints.copy(), thr)
        # results = iou_nms(gt_keypoints, thr)
        assert len(results) !=0
        for result in results:
            results_keypoints.append({
                            'image_id': image_id,
                            'category_id': 1,
                            'score': 1,
                            'keypoints': result,
                            })

    result_file = './debug_img/results.json'
    if os.path.exists(result_file):
        os.remove(result_file)
    mmcv.dump(results_keypoints, result_file)

    cocoGt = COCO(ann_file)
    cocoDt = cocoGt.loadRes(result_file)
    coco_eval = COCOeval(cocoGt, cocoDt, 'keypoints')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()






