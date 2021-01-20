from .affine import gen_patch_image_from_box_cv, trans_points_3d, trans_point2d
from mmdet.datasets.builder import PIPELINES
from numpy import random
import numpy as np
import cv2

@PIPELINES.register_module
class CenterRandomCropXiao(object):
    def __init__(self, scale_factor, rot_factor, patch_width, patch_height):
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor

        self.patch_width = patch_width
        self.patch_height = patch_height

        self.count = 0

    def __call__(self, results):
        img, boxes, labels, keypoints, gt_num_keypoints = [
            results[k] for k in ('img', 'gt_bboxes', 'gt_labels', 'gt_keypoints', 'gt_num_keypoints')
        ]
        # h, w, c = img.shape

        # random choose a box from box whose keypoints number is not the small 50%
        num_box = len(boxes)
        index = random.choice(num_box)
        # if len(keypoints) != 0:
        _keypoints = np.array(keypoints).reshape(num_box, -1, 3)
        _keypoints_v = (_keypoints[:,:,2]!=0)
        num_visible = np.sum(_keypoints_v, axis=1)
        sort_index = np.argsort(num_visible)
        half_min_index = sort_index[:int(num_box/2)]
        while index in half_min_index:
            index = random.choice(num_box)

        the_box = boxes[index]
        new_w = the_box[2] - the_box[0]
        new_h = the_box[3] - the_box[1]
        left = the_box[0]
        top = the_box[1]

        # scale = np.clip(np.random.randn(), -1.0, 1.0) * self.scale_factor + 1.0
        # rot = np.clip(np.random.randn(), -1.0, 1.0) * self.rot_factor
        scale = random.uniform(1.0 - self.scale_factor, 1.0 + self.scale_factor)
        rot = random.uniform(-self.rot_factor, self.rot_factor)

        # expand patch has the same center
        exp_w = self.patch_width * scale
        exp_h = self.patch_height * scale
        exp_left = left + (new_w - exp_w) / 2.0
        exp_top = top + (new_h - exp_h) / 2.0
        c_x = left + new_w / 2.0
        c_y = top + new_h / 2.0
        patch_expand = np.array(
            (int(exp_left), int(exp_top), int(exp_left + exp_w), int(exp_top + exp_h)))

        # center of boxes should inside the crop img
        center = (boxes[:, :2] + boxes[:, 2:]) / 2
        mask = ((center[:, 0] > patch_expand[0]) * (center[:, 1] > patch_expand[1]) *
                (center[:, 0] < patch_expand[2]) * (center[:, 1] < patch_expand[3]))
        if not mask.any():
            assert 0
        boxes = np.array(boxes)[mask]
        labels = np.array(labels)[mask]
        keypoints = np.array(keypoints)[mask]
        gt_num_keypoints = np.array(gt_num_keypoints)[mask]

        # image
        img_patch, trans = gen_patch_image_from_box_cv(img, c_x, c_y, exp_w, exp_h, self.patch_width,
                                                       self.patch_height, False, 1.0, rot)
        # keypoints
        num_kpt = len(keypoints)
        keypoints = trans_points_3d(keypoints.reshape(-1, 3), trans, 1.0)
        inds = keypoints[..., 2] == 0
        keypoints[inds] = 0
        keypoints = keypoints.reshape(num_kpt, -1)
        # boxes
        num_box = len(boxes)
        boxes = boxes.reshape(-1, 2)
        for n_jt in range(len(boxes)):
            boxes[n_jt, 0:2] = trans_point2d(boxes[n_jt, 0:2], trans)
        boxes = boxes.reshape(num_box, -1)

        # if False:
        #     from mmcv_custom.vis import vis_one_image_opencv
        #     import cv2
        #     vis = vis_one_image_opencv(img_patch, boxes=np.concatenate((boxes, np.ones((len(boxes), 1))), 1),
        #                                keypoints=keypoints.reshape(-1, 17, 3).transpose(0, 2, 1), kp_thresh=1)
        #     cv2.imwrite('./result' + 'test' + str(int(boxes.flatten()[0] * 10)) + '.jpg', vis)

        img_shape = img_patch.shape
        results['img'] = img_patch
        results['img_shape'] = img_shape
        results['gt_bboxes'] = boxes
        results['gt_labels'] = labels.tolist()
        results['gt_keypoints'] = keypoints.tolist()
        results['gt_num_keypoints'] = gt_num_keypoints.tolist()

        # visualize
        # for k in keypoints:
        #     color1 = np.random.randint(0, 256, (3), dtype=int)
        #     for j in range(17):
        #         cv2.circle(img_patch, (k[j*3], k[j*3+1]), radius=3, color=(int(color1[0]),int(color1[1]),int(color1[2])), thickness=4)
        # cv2.imwrite('./debug_img/img%d.jpg'%self.count, img)
        # cv2.imwrite('./debug_img/img_patch_%d.jpg'%self.count, img_patch)
        # self.count += 1
        # if self.count == 20:
        #     exit()


        # TODO(Xiao): Very dirty, refine this. Can't use this in testing!
        results['ori_shape'] = img_shape
        if 'scale' in results:
            del results['scale']
        if 'scale_idx' in results:
            del results['scale_idx']
        if 'pad_shape' in results:
            del results['pad_shape']
        if 'scale_factor' in results:
            del results['scale_factor']
        if 'keep_ratio' in results:
            del results['keep_ratio']

        return results

    def __repr__(self):
        return self.__class__.__name__ + '(patch_width={}, patch_height={})'.format(
            self.patch_width, self.patch_height)