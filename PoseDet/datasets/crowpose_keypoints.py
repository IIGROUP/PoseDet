from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset

import itertools
import logging
import os.path as osp
import tempfile

import mmcv
import numpy as np
from mmcv.utils import print_log

from .CrowdPose_toolkits.coco import COCO
from .CrowdPose_toolkits.cocoeval import COCOeval

from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from collections import defaultdict
import gc
import os
from PoseDet.utils import add_keypoints_flag
import json

@DATASETS.register_module()
class CrowdPoseKeypoints(CocoDataset):

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_keypoints_ann = []
        gt_keypoints_num_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            # if inter_w * inter_h == 0:
                # continue
            # if ann['area'] <= 0 or w < 1 or h < 1:
                # continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_keypoints_ann.append(ann['keypoints'])
                gt_keypoints_num_ann.append(ann['num_keypoints'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map,
            keypoints=gt_keypoints_ann,
            num_keypoints=gt_keypoints_num_ann)
        return ann

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05),
                 show_dir=None):
        # results_keypoints_output = []
        # for result in results:
        #     result = np.hstack((result[1], result[0][0][:,-1][..., None]))
        #     results_keypoints_output.append(result)

        results_keypoints_output = results
        img_ids_val = self.img_ids
        results_keypoints = []
        for i, output in enumerate(results_keypoints_output):
            image_id = img_ids_val[i]
            category_id = 1
            for output_single in output:
                output_single = output_single.tolist()
                score = output_single[-1]
                keypoints = output_single[:-4] # The last joint is not used in evaluate
                # keypoints = add_keypoints_flag(keypoints)
                results_keypoints.append({
                    'image_id': image_id,
                    'category_id': category_id,
                    'score': score,
                    'keypoints': keypoints,
                    })

        metric_items = ['mAP', 'mAP_50', 'mAP_75', 'mAP_m', 'mAP_l', 'mAR', 'mAR_50', 'mAR_75', 'mAR_m', 'mAR_l']
        results = {}
        if len(results_keypoints)==0:
            for items in metric_items:
                results[items] = 0
            return results

        # ann_filename = './results_keypoints_ForTest.json'
        # ann_filename = './person_keypoints_val2017_PoseDet_results.json'
        ann_filename = './person_keypoints_test-dev2017_PoseDet_results.json'
        # ann_filename = './test.json'
        if os.path.exists(ann_filename):
            os.remove(ann_filename)
        mmcv.dump(results_keypoints, ann_filename)
        anns = json.load(open(ann_filename))

        # os.system('zip -r -P sunsun PoseDet.zip PoseDet')
        # os.system('mv PoseDet.zip person_keypoints_test-dev2017_PoseDet_results.json')
        # zip_filename = ann_filename[:-4] + 'zip'
        # if os.path.exists(zip_filename):
        #     os.remove(zip_filename)
        # os.system('zip %s %s'%(zip_filename, ann_filename))

        cocoGt = COCO(self.ann_file)
        cocoDt = cocoGt.loadRes(ann_filename)
        coco_eval = COCOeval(cocoGt, cocoDt, 'keypoints')

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        for i in range(len(metric_items)):
            key = metric_items[i]
            val = float(f'{coco_eval.stats[i]:.3f}')
            results[key] = val

        if show_dir:
            results_file = os.path.join(show_dir, './results.npy')
        else:
            results_file = './results.npy'
            iou_results = coco_eval.ious
            # results_file = './results.npy'
            np.save(results_file, iou_results) 

        return results