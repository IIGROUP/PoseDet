import torch
import time
import cv2
import numpy as np
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.models.builder import DETECTORS
from PoseDet.keypoints_nms import keypoints_nms
from PoseDet.utils import pose_mapping_back

@DETECTORS.register_module()
class PoseDetDetector(SingleStageDetector):
    def __init__(self,
                 **args):
        super(PoseDetDetector,
              self).__init__(**args)
        self.count = 0
        self.t_total = 0

    #TODO

    def merge_aug_results(self, aug_poses, aug_scores, img_metas):
        recovered_poses = []
        for poses, img_info in zip(aug_poses, img_metas):
            img_shape = img_info[0]['img_shape']
            scale_factor = img_info[0]['scale_factor']
            flip = img_info[0]['flip']
            flip_direction = img_info[0]['flip_direction']

            poses = pose_mapping_back(poses, img_shape, scale_factor, flip,
                                       flip_direction, num_keypoints=self.bbox_head.num_keypoints)
            recovered_poses.append(poses)
        poses = torch.cat(recovered_poses, dim=0)

        scores = torch.cat(aug_scores, dim=0)
        return poses, scores

    def aug_test(self, imgs, img_metas, rescale=False, **args):
        # recompute feats to save memory
        feats = self.extract_feats(imgs)
        aug_poses = []
        aug_scores = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            outs = self.bbox_head(x)
            pose_inputs = outs + (img_meta, self.test_cfg, True, False)
            det_poses, det_scores = self.bbox_head.get_bboxes(*pose_inputs)[0]
            aug_poses.append(det_poses)
            aug_scores.append(det_scores)

        # after merging, bboxes will be rescaled to the original image size
        merged_poses, merged_scores = self.merge_aug_results(
            aug_poses, aug_scores, img_metas)

        pose_results, _ = keypoints_nms(merged_scores,
                                                self.test_cfg.score_thr,
                                                self.test_cfg.nms,
                                                self.test_cfg.max_per_img,
                                                multi_poses=merged_poses,
                                                num_points=self.bbox_head.num_keypoints)
        return pose_results.cpu().numpy()

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_keypoints=None,
                      gt_num_keypoints=None,
                      heatmap=None,
                      heatmap_weight=None):
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, 
                                              img_metas,
                                              gt_bboxes,
                                              gt_labels,
                                              gt_keypoints=gt_keypoints, 
                                              gt_num_keypoints=gt_num_keypoints,
                                              heatmap=heatmap,
                                              heatmap_weight=heatmap_weight)

        #visulize box and weight
        # img_single = img[0].permute(1,2,0).cpu().numpy()  
        # max_, min_ = img_single.max(), img_single.min()
        # img_single = (img_single-min_)/(max_ - min_) * 255
        # weight = heatmap_weight[0][0].cpu().numpy()
        # bboxes = gt_bboxes[0].cpu().numpy() 
        # img_single = cv2.cvtColor(img_single, cv2.COLOR_BGR2RGB)
        # img_single -= img_single*weight[..., None]*0.5
        # for box in bboxes:
        #     start_point = (box[0], box[1])
        #     end_point = (box[2], box[3])
        #     cv2.rectangle(img_single, start_point, end_point, color= (255, 0, 0) , thickness=1)
        # cv2.imwrite('./debug_img/%d.jpg'%(self.count), img_single)
        # self.count += 1
        # if self.count >100:
        #     exit()

        return losses

    def simple_test(self, img, img_metas, rescale=False, **args):

        # torch.cuda.synchronize()
        # t = time.time()

        x = self.extract_feat(img)
        outs = self.bbox_head(x)

        # torch.cuda.synchronize()
        # self.t_total += (time.time()-t)*1000
        # self.count += 1
        # print('average time ms ', self.t_total/self.count)

        # visulize heatmap
        # img_name = img_metas[0]['ori_filename']
        # heatmap_feat_stride = torch.nn.functional.interpolate(outs[-1], img[0].size()[-2:])
        # img_single = img[0].permute(1,2,0).cpu().numpy()  
        # max_, min_ = img_single.max(), img_single.min()
        # img_single = (img_single-min_)/(max_ - min_) * 255
        # img_single = img_single.astype(np.uint8)
        # # cv2.imwrite('./debug_img/%s%d.jpg'%(img_name,self.count), img_single)

        # # for i in range(17):
        # #     g_map = outs[-1][0][i].sigmoid()
        # #     g_map = g_map.cpu().numpy()
        # #     g_map = g_map*250
        # #     g_map = g_map.astype(np.uint8)
        # #     cv2.imwrite('./debug_img/gausssionmap%d%d.jpg'%(self.count, i), g_map)

        # # g_map = outs[-1][0].sigmoid().sum(dim=0)
        # g_map = heatmap_feat_stride[0].sigmoid().sum(dim=0)
        # g_map = g_map.cpu().numpy()
        # g_map = g_map*200

        # cv2.imwrite('./output/PoseDet_dla34_heatmap4/heatmap/%s%d%d.jpg'%(img_name, self.count, -1), g_map)

        # self.count+=1
        # if self.count > 50 :
        #     exit()

        
        poses_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        return poses_list[0].cpu().numpy()
