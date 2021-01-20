import argparse
import os, sys
sys.path.append(os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from tools.fuse_conv_bn import fuse_module

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.core import wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector

from PoseDet import assigners
from PoseDet import backbones
from PoseDet import datasets
from PoseDet import detectors
from PoseDet import losses
from PoseDet import pipelines
from PoseDet import roi_heads
import cv2
from PoseDet.utils import show_pose

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--checkpoint',help='checkpoint file')
    parser.add_argument('--config', help='test config file path')
    # parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument('--image-folder', help='output result file in pickle format')
    args = parser.parse_args()
    return args

def pad(image):
    h, w, c = image.shape
    n = 0 
    while int(w/2) != 0:
        w = w/2
        n+=1
    w_new = 2**n

    n = 0 
    while int(h/2) != 0:
        h = h/2
        n+=1
    h_new = 2**n
    print(image.shape)
    print(w,h, w_new, h_new)
    exit()

def main():    
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.data.samples_per_gpu = 1
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg).cuda()
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    file_names = os.listdir(args.image_folder)
    for file_name in file_names:
        file_path = os.path.join(args.image_folder, file_name)
        image = cv2.imread(file_path)
        image = cv2.resize(image, dsize=(1024, 640))
        # cv2.imwrite('./debug_img/test.jpg', image)
        # exit()
        # print(image.shape)
        # continue
        # image = pad(image)
        image_ = image.transpose((2,0,1))[None,...]
        # print(image.shape)
        image_ = torch.from_numpy(image_).float().cuda()
        image_meta = dict()
        # image_meta['img_shape'] = (1024, 640)
        image_meta['img_shape'] = (640,1024)
        image_meta['scale_factor'] = [1,1]
        # image 
        results = model(return_loss=False, rescale=True, img=[image_], img_metas=[[image_meta]])
        # print(results)
        # print(results.shape)
        # exit()
        show_index = results[:,-1] > 0.3
        image = show_pose(image, results[show_index,:-1])
        # out_file_split = out_file.split('/')
        # out_file_folder = out_file_split[0]
        # for s in out_file_split[1:-1]:
        #     out_file_folder = os.path.join(out_file_folder, s)
        # if not os.path.exists(out_file_folder):
        #     os.makedirs(out_file_folder)
        cv2.imwrite('debug_img/test2.jpg', image)
        # cv2.imwrite(out_file, image)
        exit()


if __name__ == '__main__':
    main()