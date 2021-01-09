_base_ = [
    #data
    './coco_person_keypoints.py',
    
    #model
    './PoseDet_DLA34.py',
    # './PoseDet_HRNetw32.py',
    # './PoseDet_r18.py',
    # './PoseDet_r50.py',

    #shedule
    './schedule.py',
]

exp_name = 'PoseDet_test' # init conv=2, cls conv=3, refine conv=2
# exp_name = 'PoseDet_HRNetw32' #L2
# exp_name = 'PoseDet_test3' # L2, refine (thr0.7)
# exp_name = 'PoseDet_dla34_1' # L2, refine (thr0.6)
# exp_name = 'PoseDet_dla34_2' # L2, refine (thr0.8)
# exp_name = 'PoseDet_dla34_3' # L2, refine (thr0.8) e200
# exp_name = 'PoseDet_dla34_4' # L2, refine (thr0.7), cls weight=2
# exp_name = 'PoseDet_dla34_5' # L2,  refine (thr0.7), cls weight=2, refine weight=1
# exp_name = 'PoseDet_dla34_6' #  --train scale [512, 640]
# exp_name = 'PoseDet_dla34_7' #  --train scale [512, 800]
# exp_name = 'PoseDet_dla34_8' #  type='CenterRandomCropXiao', scale_factor=0.2, rot_factor=0, patch_width=512
# exp_name = 'PoseDet_dla34_9' #  type='CenterRandomCropXiao', scale_factor=0.2, rot_factor=10, patch_width=512

# 改正conv layer的错误, 使用person_keypoints_train2017,  conv layer=3,2,2, train scale=[512;800]
# exp_name = 'PoseDet_dla34' #
# exp_name = 'PoseDet_dla34_heatmap' # -- PoseDet_dla34+heatmap, pos weight=10, concat with 3 brunch
# exp_name = 'PoseDet_dla34_heatmap2' # -- PoseDet_dla34+heatmap, pos weight=10, w/o concat
# exp_name = 'PoseDet_dla34_heatmap3'# # heatmap focal heatmap loss, weigth=0.1 w/o concat, type='CenterRandomCropXiao', scale_factor=0.2, rot_factor=0, patch_width=512
# exp_name = 'PoseDet_dla34_heatmap4' # load from PoseDet_dla34_heatmap3_e100, w/ concat, type='CenterRandomCropXiao', scale_factor=0.2, rot_factor=0, patch_width=512
# exp_name = 'PoseDet_dla34_heatmap5' # heatmap focal heatmap loss, weigth=1, with_sigmas, w/ concat type='CenterRandomCropXiao', scale_factor=0.2, rot_factor=0, patch_width=512
# exp_name = 'PoseDet_dla34_heatmap6' # PoseDet_dla34_heatmap5 + train2017 w/o crowd.json
# exp_name = 'PoseDet_dla34_10' # conv layer=4,2,2
# exp_name = 'PoseDet_dla34_11' #  type='CenterRandomCropXiao', scale_factor=0.2, rot_factor=0, patch_width=512, refine assigner=center point(init)
# exp_name = 'PoseDet_dla34_12' #  type='CenterRandomCropXiao', scale_factor=0.2, rot_factor=0, patch_width=512, refine assigner=center point(init) stage2cls
# exp_name = 'PoseDet_dla34_13' #  type='CenterRandomCropXiao', scale_factor=0.2, rot_factor=0, patch_width=512, adam e210

# 增加type='CenterRandomCropXiao', scale_factor=0.2, rot_factor=0, patch_width=512 , train2017 w/o crowd.json
# exp_name = 'PoseDet_dla34_crop'
# exp_name = 'PoseDet_dla34_crop_2' #init stage使用center point of box
# exp_name = 'PoseDet_dla34_crop_3' # train2017 w crowd.json
# exp_name = 'PoseDet_dla34_crop_4' # init/refine loss加joint weight

# exp_name = 'PoseDet_dla34_crop_5' #init/refine number_keypoints_thr =1
# exp_name = 'PoseDet_dla34_crop_6' #dla34 channel=128
# exp_name = 'PoseDet_dla34_crop_7' #dla34 channel=128, 在init分支增加visible预测, train2017 w crowd.json
# exp_name = 'PoseDet_dla34_crop_8' #dla34 channel=128, type='CenterRandomCropXiao', scale_factor=0.5, rot_factor=0, patch_width=318
# exp_name = 'PoseDet_dla34_crop_9' #load from PoseDet_dla34_heatmap3_e100, dla34, SGD e36
# exp_name = 'PoseDet_dla34_crop_10' # point_loss_single.sum(), init/refine weight=.1/.2, train2017 w crowd.json
# exp_name = 'PoseDet_dla34_crop_11' # point_loss_single.sum(), init/refine weight=.1/.2, train2017 w/o crowd.json, train scale=[512;800]

# exp_name = 'PoseDet_dla34_heatmap_ml' #dla34 channel=128, 给每层FPN输出都加上heatmap监督，w/o concat, 将heatmap gt插值方式改为nearest, with_sigma=False, weight=0.1, train2017 w crowd.json
# exp_name = 'PoseDet_dla34_heatmap_ml2' #给每层FPN输出都加上heatmap监督，w concat, 将heatmap gt插值方式改为nearest

# Data - crop, w/o crow; Init/Refine - point_loss_single.sum(),weight=.1/.2; dla34 channel=128
# exp_name = 'PoseDet_dla34_heatmap_ml3' # heatmap w/o concat, with_sigma=False, 增加sclae gaussian

exp_name = 'PoseDet_dla34c128' #

work_dir = './output/' + exp_name

# load_from = 'pretrained/PoseDet_dla34_heatmap3_e100.pth' 