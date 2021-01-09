import os.path as osp

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch
from collections.abc import Sequence
from mmcv.parallel import DataContainer as DC
from mmdet.core import BitmapMasks, PolygonMasks
from mmdet.datasets.builder import PIPELINES
import cv2
from PoseDet.utils import xfirst2yfirst

@PIPELINES.register_module()
class LoadKeypoints(object):

    def __init__(self, load_num_keypoints=True, convert2nine=False, convert2eighteen=False):
        self.load_num_keypoints = load_num_keypoints
        self.convert2nine = convert2nine
        self.convert2eighteen = convert2eighteen
        assert (convert2nine and convert2eighteen) is False #only one is allowed

    def __call__(self, results):
        keypoints = results['ann_info']['keypoints']
        
        # for i, k in enumerate(keypoints):
        #     keypoints[i] = xfirst2yfirst(k)

        num_keypoints = results['ann_info']['num_keypoints']
        if self.convert2nine:
            # print('---------------------')
            # print('1', keypoints[-1][:3*4])
            keypoints = self._convert2nine(keypoints)
            # print('2', keypoints[-1][:3])
        if self.convert2eighteen:
            keypoints = self._convert2eighteen(keypoints)

        results['gt_keypoints'] = keypoints
        results['gt_num_keypoints'] = num_keypoints

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str
    def _convert2eighteen(self, keypoints):
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
        if len(keypoints) == 0:
            return keypoints
        keypoints = np.array(keypoints)
        keypoints = keypoints.reshape((len(keypoints), -1, 3))
        keypoints_eighteen = np.zeros((len(keypoints), 18, 3))
        true_index = (keypoints[:,:,2]!=0)

        keypoints_eighteen[:,0] = keypoints[:,1]
        keypoints_eighteen[:,1] = keypoints[:,2]

        index = (true_index[:,6].astype(np.uint8) + true_index[:,5].astype(np.uint8))
        index[index==0] = 1
        keypoints_eighteen[:,2] = (keypoints[:,5] + keypoints[:,6])/index[..., None]

        keypoints_eighteen[:,3] = keypoints[:,11]
        keypoints_eighteen[:,4] = keypoints[:,12]

        keypoints_eighteen[:,5] = keypoints[:,7]
        keypoints_eighteen[:,6] = keypoints[:,8]

        keypoints_eighteen[:,7] = keypoints[:,13]
        keypoints_eighteen[:,8] = keypoints[:,14]

        keypoints_eighteen[:,9] = keypoints[:,3]
        keypoints_eighteen[:,10] = keypoints[:,4]

        keypoints_eighteen[:,11] = keypoints[:,0]

        keypoints_eighteen[:,12] = keypoints[:,5]
        keypoints_eighteen[:,13] = keypoints[:,6]

        keypoints_eighteen[:,14] = keypoints[:,9]
        keypoints_eighteen[:,15] = keypoints[:,10]

        keypoints_eighteen[:,16] = keypoints[:,15]
        keypoints_eighteen[:,17] = keypoints[:,16]


        # img = np.zeros([1000,1000,3])
        # pts_pred = keypoints_eighteen[0].astype(np.uint8)
        # pts_target = keypoints[0].astype(np.uint8)
        # # print('pts_pred', pts_pred.shape)
        # # print('pts_target', pts_target.shape)
        # # print('hausdroff_1', torch.min(distance, dim=0)[0])
        # # print('hausdroff_2', torch.min(distance, dim=1)[0])
        # color = np.random.randint(0, 256, (3), dtype=int)
        # for j in range(18):
        #     cv2.circle(img, (pts_pred[j][0], pts_pred[j][1]), radius=3, color=(int(color[0]),int(color[1]),int(color[2])), thickness=4)
        # img2 = np.zeros([1000,1000,3])
        # color = np.random.randint(0, 256, (3), dtype=int)
        # for j in range(17):
        #     cv2.circle(img2, (pts_target[j][0], pts_target[j][1]), radius=3, color=(int(color[0]),int(color[1]),int(color[2])), thickness=4)

        # cv2.imwrite('./debug_img/convert3.jpg', img)
        # cv2.imwrite('./debug_img/convert2.jpg', img2)
        # if loss_keypoints_refine_>0.5:
        #     print('loss_keypoints_refine_', loss_keypoints_refine_)
        # exit()

        keypoints_eighteen = keypoints_eighteen.reshape(len(keypoints_eighteen), 3*18).astype(int).tolist()
        return keypoints_eighteen

    def _convert2nine(self, keypoints):
        #keypoints - list: [n, 51]
        if len(keypoints) == 0:
            return keypoints
        keypoints = np.array(keypoints)
        keypoints = keypoints.reshape((len(keypoints), -1, 3))
        keypoints_nine = np.zeros((len(keypoints), 9, 3))
        true_index = (keypoints[:,:,2]!=0)

        index0 = np.sum(true_index[:,:5], axis=1)
        index0[index0==0] = 1
        keypoints_nine[:,0] = np.sum(keypoints[:,:5], axis=1) / index0[..., None]
        keypoints_nine[:,1] = keypoints[:,5]
        keypoints_nine[:,2] = keypoints[:,6]   

        index1 = (true_index[:,7].astype(np.uint8) + true_index[:,9].astype(np.uint8))
        index1[index1==0] = 1
        keypoints_nine[:,3] = (keypoints[:,7] + keypoints[:,9])/index1[..., None]
        # print('==============')
        # print(keypoints[:,7])
        # print(keypoints[:,9])
        # print(index1)
        # print(keypoints_nine[:,3])

        index2 = (true_index[:,8].astype(np.uint8) + true_index[:,10].astype(np.uint8))
        index2[index2==0] = 1
        keypoints_nine[:,4] = (keypoints[:,8] + keypoints[:,10])/index2[..., None]
        keypoints_nine[:,5] = keypoints[:,11]
        keypoints_nine[:,6] = keypoints[:,12]

        index3 = (true_index[:,13].astype(np.uint8) + true_index[:,15].astype(np.uint8))
        index3[index3==0] = 1        
        keypoints_nine[:,7] = (keypoints[:,13] + keypoints[:,15])/index3[..., None]

        index4 = (true_index[:,14].astype(np.uint8) + true_index[:,16].astype(np.uint8))
        index4[index4==0] = 1         
        keypoints_nine[:,8] = (keypoints[:,14] + keypoints[:,16])/index4[..., None]

        keypoints_nine[:,:,2] = (keypoints_nine[:,:,2]!=0)

        # img = np.zeros([500,500,3])
        # pts_pred = keypoints_nine[0].astype(np.uint8)
        # pts_target = keypoints[0].astype(np.uint8)
        # # print('pts_pred', pts_pred.shape)
        # # print('pts_target', pts_target.shape)
        # # print('hausdroff_1', torch.min(distance, dim=0)[0])
        # # print('hausdroff_2', torch.min(distance, dim=1)[0])
        # color = np.random.randint(0, 256, (3), dtype=int)
        # for j in range(9):
        #     cv2.circle(img, (pts_pred[j][0], pts_pred[j][1]), radius=3, color=(int(color[0]),int(color[1]),int(color[2])), thickness=4)

        # color = np.random.randint(0, 256, (3), dtype=int)
        # for j in range(17):
        #     cv2.circle(img, (pts_target[j][0], pts_target[j][1]), radius=3, color=(int(color[0]),int(color[1]),int(color[2])), thickness=4)

        # cv2.imwrite('./debug_img/convert3.jpg', img)
        # # if loss_keypoints_refine_>0.5:
        # #     print('loss_keypoints_refine_', loss_keypoints_refine_)
        # # exit()
        keypoints_nine = keypoints_nine.reshape(len(keypoints_nine), 27).astype(int).tolist()

        return keypoints_nine



@PIPELINES.register_module()
class FormatBundleKeypoints(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """


    def __call__(self, results):
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(to_tensor(img), stack=True)
            
        if 'heatmap' in results:
            heatmap = results['heatmap']
            if len(heatmap.shape) < 3:
                heatmap = np.expand_dims(heatmap, -1)
            heatmap = np.ascontiguousarray(heatmap.transpose(2, 0, 1))
            results['heatmap'] = DC(to_tensor(heatmap), stack=True)

        if 'heatmap_weight' in results:
            heatmap_weight = results['heatmap_weight']
            if len(heatmap_weight.shape) < 3:
                heatmap_weight = np.expand_dims(heatmap_weight, -1)
            heatmap_weight = np.ascontiguousarray(heatmap_weight.transpose(2, 0, 1))
            results['heatmap_weight'] = DC(to_tensor(heatmap_weight), stack=True)
            
        for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels', 'gt_keypoints', 'gt_num_keypoints']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))
        if 'gt_masks' in results:
            results['gt_masks'] = DC(results['gt_masks'], cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]), stack=True)
        return results

    def __repr__(self):
        return self.__class__.__name__


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')