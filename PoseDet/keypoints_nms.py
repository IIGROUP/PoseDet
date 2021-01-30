import torch
# from mmdet.ops.nms import batched_nms
# from mmdet.ops.nms import nms_ext
import time

def keypoints_nms(multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None,
                   multi_poses=None,
                   num_points=9,
                   ):
    #multi_poses： Tensor[N, num_points*3+1], 1是score channel

    num_classes = multi_scores.size(1) - 1

    pointsets = multi_poses[:, None].expand(-1, num_classes, num_points*3)

    scores = multi_scores[:, :-1]

    valid_mask = scores > score_thr

    pointsets = pointsets[valid_mask]

    scores = scores[valid_mask]

    labels = valid_mask.nonzero()[:, 1]

    if pointsets.numel() == 0:
        pointsets = multi_poses.new_zeros((0, num_points*3 + 1))
        labels = pointsets.new_zeros((0, ), dtype=torch.long)
        return pointsets, labels

    dets, keep = oks_nms(
        torch.cat([pointsets, scores[:, None]], -1), iou_thr=nms_cfg['iou_thr'],num_points=num_points)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]
    
    #dets: Tenosr[N, num_points*3 + 1(score)]
    return dets, labels[keep]


def oks_nms(dets, iou_thr, device_id=None,num_points=17):

    if isinstance(dets, torch.Tensor):
        is_numpy = False
        dets_th = dets
    elif isinstance(dets, np.ndarray):
        is_numpy = True
        device = 'cpu' if device_id is None else f'cuda:{device_id}'
        dets_th = torch.from_numpy(dets).to(device)
    else:
        raise TypeError('dets must be either a Tensor or numpy array, '
                        f'but got {type(dets)}')

    # execute cpu or cuda nms
    if dets_th.shape[0] == 0:
        inds = dets_th.new_zeros(0, dtype=torch.long)
    else:
        # torch.cuda.synchronize()
        # t1 = time.time()
        inds = _oks_nms(dets_th, iou_thr, num_points)

        # import os
        # import numpy as np 
        # dets_np = dets_th.detach().cpu().numpy()
        # for i in range(501):
        #     path = './debug_img2/%d.npy'%i
        #     if not os.path.exists(path):
        #         np.save(path, dets_np)
        #         break

        # inds = _oks_fast_nms(dets_th, iou_thr)

        # torch.cuda.synchronize()
        # t2 = time.time()

    if is_numpy:
        inds = inds.cpu().numpy()

    return dets[inds, :], inds

def _oks_nms(dets, thresh, num_points):
    if num_points==17:
        sigmas = torch.tensor([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89], device=dets.device)/10.0
        pointsets = dets[:,:-1]
        pointsets = pointsets.view((pointsets.size()[0], -1, 3))
        pointsets = pointsets[:,:,:2]
    elif num_points==15:
        sigmas = torch.tensor([.79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89, .79, .79], device=dets.device)/10.0
        pointsets = dets[:,:-4] #the last points did not used in oks computation
        pointsets = pointsets.view((pointsets.size()[0], -1, 3))
        pointsets = pointsets[:,:,:2]
        
    vars = (sigmas * 2)**2
    vars = vars.unsqueeze(0).unsqueeze(0) #[1, 1, 17]

    w_all = torch.max(pointsets[:,:,0], dim=1)[0] - torch.min(pointsets[:,:,0], dim=1)[0]
    h_all = torch.max(pointsets[:,:,1], dim=1)[0] - torch.min(pointsets[:,:,1], dim=1)[0]
    areas = w_all*h_all
    areas = areas.clamp(32*32)
    areas = (areas.unsqueeze(0)+areas.unsqueeze(1))/2
    areas = areas.unsqueeze(-1) #[points_num, points_num, 1]

    distance = ((pointsets.unsqueeze(0) - pointsets.unsqueeze(1))**2).sum(dim=-1) # [m, m, points_num]
    oks = torch.exp(-distance/vars/areas).mean(dim=-1)

    scores = dets[:,-1]

    keep = []
    index = scores.sort(descending=True)[1]  

    while index.size()[0] >0:
        i = index[0]       # every time the first is the biggst, and add it directly
        keep.append(i)
        if index.size()[0] == 1:
            break       
        oks_selected = torch.index_select(oks[i], 0, index)
        idx = torch.where(oks_selected<=thresh)[0]
         
        index = index[idx]

    keep = torch.stack(keep)    
    return keep
    
def _matrix_oks_nms(dets, thresh, num_points):
    if num_points==17:
        sigmas = torch.tensor([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89], device=dets.device)/10.0
        pointsets = dets[:,:-1]
        pointsets = pointsets.view((pointsets.size()[0], -1, 3))
        pointsets = pointsets[:,:,:2]
    elif num_points==15:
        sigmas = torch.tensor([.79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89, .79, .79], device=dets.device)/10.0
        pointsets = dets[:,:-4] #the last points did not used in oks computation
        pointsets = pointsets.view((pointsets.size()[0], -1, 3))
        pointsets = pointsets[:,:,:2]
        
    vars = (sigmas * 2)**2
    vars = vars.unsqueeze(0).unsqueeze(0) #[1, 1, 17]

    w_all = torch.max(pointsets[:,:,0], dim=1)[0] - torch.min(pointsets[:,:,0], dim=1)[0]
    h_all = torch.max(pointsets[:,:,1], dim=1)[0] - torch.min(pointsets[:,:,1], dim=1)[0]
    areas = w_all*h_all
    areas = areas.clamp(32*32)
    areas = (areas.unsqueeze(0)+areas.unsqueeze(1))/2
    areas = areas.unsqueeze(-1) #[points_num, points_num, 1]

    distance = ((pointsets.unsqueeze(0) - pointsets.unsqueeze(1))**2).sum(dim=-1) # [m, m, points_num]
    oks = torch.exp(-distance/vars/areas).mean(dim=-1)

    scores = dets[:,-1]

    keep = []
    index = scores.sort(descending=True)[1]  

    while index.size()[0] >0:
        i = index[0]       # every time the first is the biggst, and add it directly
        keep.append(i)
        if index.size()[0] == 1:
            break       
        oks_selected = torch.index_select(oks[i], 0, index)
        idx = torch.where(oks_selected<=thresh)[0]
         
        index = index[idx]

    keep = torch.stack(keep)    
    return keep

