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
    def __init__(self, fourteen2fifteen=False):
        self.fourteen2fifteen = fourteen2fifteen

    def __call__(self, results):
        keypoints = results['ann_info']['keypoints']
        num_keypoints = results['ann_info']['num_keypoints']
        if self.fourteen2fifteen:
            keypoints = self._fourteen2fifteen(keypoints)
        results['gt_keypoints'] = keypoints
        results['gt_num_keypoints'] = num_keypoints

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str
    def _fourteen2fifteen(self, keypoints):
        N = len(keypoints)
        # if N==0:
        #     print(keypoints)
        #     exit()
            # return 
        _keypoints = np.array(keypoints).reshape([N, 14, 3])
        l_shoulder = _keypoints[:,0,:]

        r_shoulder = _keypoints[:,1,:]
        l_hip = _keypoints[:,6,:]
        r_hip = _keypoints[:,7,:]
        index = _keypoints[:,:,2]
        index[index==2] = 1
        index = index[:,0] + index[:,1] + index[:,6] + index[:,7]
        index[index==0] = 1
        abdomen = (l_shoulder+r_shoulder+l_hip+r_hip)/index[...,None]

        index[index<3] = 0
        abdomen[:,2] = index

        keypoints_fifteen = np.zeros([N, 15, 3])
        keypoints_fifteen[:,:14,:] = _keypoints
        keypoints_fifteen[:,14,:] = abdomen
        keypoints_fifteen = keypoints_fifteen.reshape((N, 15*3)).tolist()
        return keypoints_fifteen


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
            GaussianMap_list = results['heatmap']
            for i in range(len(GaussianMap_list)):
                GaussianMap_list[i] = DC(to_tensor(GaussianMap_list[i]), stack=True)

        if 'heatmap_weight' in results:
            heatmap_weight_list = results['heatmap_weight']
            for i in range(len(heatmap_weight_list)):
                heatmap_weight_list[i] = DC(to_tensor(heatmap_weight_list[i]), stack=True)
            
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