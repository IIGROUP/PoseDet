import torch
import numpy as np
import cv2

# coco
limbs = [[3,1], [1,0], [0,2], [2,4],
          [0,17], [17,5], [17,6], [5,7], [7,9], [6,8], [8,10],
          [17,11], [17,12], [11,13], [13,15], [17,12], [12,14], [14,16]]

def pose_flip(poses, img_shape, direction='horizontal', num_keypoints=17):
    assert direction == 'horizontal'
    poses_fliped = torch.zeros_like(poses)
    for i, kp in enumerate(poses):
        kp = kp.reshape((-1, 3))
        if direction == 'horizontal':
            w = img_shape[1]
            kp[:,0] = w - kp[:,0]
        elif direction == 'vertical':
            h = img_shape[0]
            kp[:,1] = h - kp[:,1]
        if num_keypoints == 17:
            idnex_map = [0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15]
        elif num_keypoints==15:
            idnex_map = [1,0,3,2,5,4,7,6,9,8,11,10,12,13,14]
        kp = kp[idnex_map]     
        poses_fliped[i] = kp.reshape(-1)

    return poses_fliped

def pose_mapping_back(poses,
                      img_shape,
                      scale_factor,
                      flip,
                      flip_direction='horizontal',
                      num_keypoints=17):
    assert flip_direction=='horizontal'

    # new_poses = poses * poses.new_tensor(scale_factor[0])
    new_poses = poses
    if flip:
        new_poses = pose_flip(new_poses, img_shape, flip_direction, num_keypoints=num_keypoints)
    return new_poses

def computeOks(gt_pts, pts_preds, gt_bboxes=None, num_keypoints=17, number_keypoints_thr=1, normalize_factor=0):
    # gt_pts:Tensor[n, 1, points_num*3]
    # pts_preds:Tensor[m, points_num*2]
    # return oks:Tensor[n, m]
    oks = torch.zeros((len(gt_pts), len(pts_preds)), device=pts_preds.device)
    if len(gt_pts)==0:
        return oks
    if num_keypoints==17:
        sigmas = torch.tensor([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89], device=pts_preds.device)/10.0
    elif num_keypoints==15: 
        sigmas = torch.tensor([.79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89, .79, .79], device=pts_preds.device)/10.0
        gt_pts = gt_pts[:,:,:14*3]
        pts_preds = pts_preds[:,:14*2]

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

    gt_pts = gt_pts.reshape(len(gt_pts), -1, 3)
    pts_preds = pts_preds.view(len(pts_preds), -1, 2)
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

def show_pose(image, poses, thickness=2):
    #poses: array[N, k*3] 
    assert poses.shape[1] == 51 #for coco format only
    image = image.copy()
    for pose in poses:
        color = np.random.rand(3) * 255
        # color = np.array(color).astype(np.uint8)
        color = (int(color[0]), int(color[1]), int(color[2]))
        # print(color)
        #add neck
        keypoints = np.array(pose).reshape(-1, 3)
        keypoints_ = np.zeros((18, 3))
        neck = (keypoints[5,:2]+keypoints[6,:2])/2
        keypoints_[:17] = keypoints
        keypoints_[17,:2] = neck
        keypoints_[17,2] = 2
        #draw pose
        for i in range(len(limbs)):
            limb = limbs[i]
            x1, y1 = keypoints_[limb[0]][:2]
            x2, y2 = keypoints_[limb[1]][:2]
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color=color, thickness=thickness)
    return image

#convert a set of keypoints fron 18 to 17
#input:[18*2]
#output:[17*3]
def eighteen2seventeen(keypoints):
    '''
    coco format 17 keypoints: 
        '0 - nose','1 - left_eye','2 - right_eye','3 - left_ear','4 - right_ear', 
        '5 - left_shoulder','6 - right_shoulder','7 - left_elbow','8 - right_elbow',
        '9 - left_wrist','10 - right_wrist','11 - left_hip','12 - right_hip',
        '13 - left_knee','14 - right_knee','15  - left_ankle','16 - right_ankle'
    18 keypoints: 
            0~8
            left_eye, right_eye, neck, left_hip     , right_hip     , left_elbow, right_elbow, left_knee,   right_knee,
            9~17
            left_ear, right_ear, nose, left_shoulder, right_shoulder, left_wrist, right_wrist, left_ankle, right_ankle
        
    '''
    assert len(keypoints)==18*3
    index_map = [11, 0, 1, 9, 10, 12, 13, 5, 6, 14, 15, 3, 4, 7, 8, 16, 17]
    keypoints_seventeen = []
    for index in index_map:
        keypoints_seventeen += keypoints[index*3:index*3+3]
    return keypoints_seventeen

def add_keypoints_flag(keypoints):
    n = len(keypoints)/2
    n = int(n)
    keypoints_ = []
    for i in range(n):
        keypoints_ += (keypoints[i*2:(i*2+2)] + [2])
    return keypoints_

def yfirst2xfirst(keypoints, v_flag=True):
    assert v_flag==True
    keypoints_ = []
    num = 3 
    n = len(keypoints)/num
    n = int(n)
    for i in range(n):
        keypoints_ += [keypoints[i*num+1]] + [keypoints[i*num]] + [keypoints[i*num+2]]
    return keypoints_

def xfirst2yfirst(keypoints, v_flag=True):
    assert v_flag==True
    keypoints_ = []
    num = 3 
    n = len(keypoints)/num
    n = int(n)
    for i in range(n):
        keypoints_ += [keypoints[i*num+1]] + [keypoints[i*num]] + [keypoints[i*num+2]]
    return keypoints_