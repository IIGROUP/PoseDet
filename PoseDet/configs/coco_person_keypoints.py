'''datasets:
    instances_train2017.json
    instances_val2017.json
    instances_val2017_small.json 从val2017里面选1000张出来
    instances_val2017_small100.json 从val2017里面选100张出来
    instances_train2014_for2017.json
    instances_val2014_for2017.json
    使用coco2014的数据集划分，使用coco2017的图片（和2014的图片一样，只是名字少了前缀COCO_train2014）,文件名格式：train2017/000000391895.jpg
    
    person_keypoints_train2017.json
    person_keypoints_val2017.json
    person_keypoints_val2017_small500.json
    image_info_test_dev2017.json
    person_keypoints_wo_crowd_train2017.json # 去掉包含crowd的ann和image
    person_keypoints_lt6_val2017.json # 每张图的平均骨骼点数量<6
    person_keypoints_ht6_lt12_val2017.json# 每张图的平均骨骼点数量>=6, <12
    person_keypoints_ht12_val2017.json# 每张图的平均骨骼点数量>=12
    person_keypoints_plt3_val2017.json# 每张图的含有骨骼点标注的人数<3
    person_keypoints_pht3_lt6_val2017.json #每张图的含有骨骼点标注的人数>6
    person_keypoints_pht6_val2017.json #每张图的含有骨骼点标注的人数>=3,<6
    person_keypoints_person_only_train2017.json #images只包含有人的image
    person_keypoints_person_only_val2017.json #images只包含有人的image
    person_keypoints_wo_small_train2017.json #去掉小目标的ann
    person_keypoints_wo_small_val2017.json #去掉小目标的ann
    person_keypoints_wo_small_train2017_s500.json #去掉小目标的ann, 挑500张有ann的image
    person_keypoints_wo_small_val2017_s500.json #去掉小目标的ann, 挑500张有ann的image
    person_keypoints_wo_small_train2017_s500_9keypoints.json #去掉小目标的ann, 挑500张有ann的image，挑出9个骨骼点
    person_keypoints_wo_small_val2017_s500_9keypoints.json #去掉小目标的ann, 挑500张有ann的image，挑出9个骨骼点

    ochuman_coco_format_test.json
    # ochuman_coco_format_val.json

'''
# dataset_type = 'PorSegInsDataset'
# dataset_type = 'CocoDataset'
# dataset_type = 'CocoSegmentation'
dataset_type = 'CocoKeypoints'
#
data_root = '/data/ly/coco/'

annotation_file_train = 'annotations/person_keypoints_train2017.json'
annotation_file_val = 'annotations/person_keypoints_val2017.json'
annotation_file_test = 'annotations/person_keypoints_val2017_small500.json'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadKeypoints', 
         convert2nine=False,
         convert2eighteen=False,
        ),
    # dict(type='Transform_package'),
    dict(type='LoadAnnotations', with_bbox=True),    
    dict(
        type='CenterRandomCropXiao',
        scale_factor=0.5,
        rot_factor=0,
        patch_width=512,
        patch_height=512),
    dict(type='Resize',         
        img_scale=[
                    # (600,384),
                    (800,512),
                    # (1000,640),
                    # (1333, 800),
                    # (1600,1000),
                    ], 
        multiscale_mode='value', 
        keep_ratio=True,
        with_keypoints=True),
    dict(type='RandomFlip', flip_ratio=0.5, with_keypoints=True, gt_num_keypoints=17),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='GaussianMap', sigma=7.0, num_keypoints=17, pos_weight=1, SCALE=True, scale_factor=100),
    dict(type='FormatBundleKeypoints'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 
                                'gt_keypoints', 'gt_num_keypoints',
                                'heatmap', 'heatmap_weight']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(600,384),
        img_scale=(800,512),
        # img_scale=(1000,640),
        # img_scale=(1200,700),
        # img_scale=(1333, 800),
        # multi-scale test
        # img_scale=[(600,384), (800,512), (1000,640)], #type0
        # img_scale=[(666, 400), (1000, 600), (1333, 800)], #type1
        # img_scale=[(520, 300), (666, 400), (1000, 600), (1333, 800)], #type2
        # img_scale=[(666, 400), (1000, 600), (1333, 800), (1666, 1000)], #type3
        # img_scale=[(666, 400), (1000, 600), (1333, 800), (1666, 1000), (2000, 1200), (2333, 1400)], #type4
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=12,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + annotation_file_train,
        img_prefix=data_root + 'train2017/',
        # img_prefix=data_root + 'val2017/',
        # img_prefix=data_root + 'images/',
        pipeline=train_pipeline
        ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + annotation_file_val,
        # img_prefix=data_root + 'train2017/',
        img_prefix=data_root + 'val2017/',
        # img_prefix=data_root + 'images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + annotation_file_test,
        # img_prefix=data_root + 'train2017/',
        img_prefix=data_root + 'val2017/',
        # img_prefix=data_root + 'test2017/',
        # img_prefix=data_root + 'images/',
        pipeline=test_pipeline))
