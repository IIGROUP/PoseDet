_base_ = [
    '../_base_/dataset_coco.py',
    '../_base_/PoseDet_DLA34.py',
    '../_base_/train_schedule.py',
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadKeypoints'),
    dict(type='LoadAnnotations', with_bbox=True),    
    dict(
        type='CenterRandomCropXiao',
        scale_factor=0.5,
        rot_factor=0,
        patch_width=640,
        patch_height=640),
    dict(type='Resize',         
        img_scale=[
                    # (800,512),
                    (1000,640),
                    ], 
        multiscale_mode='value', 
        keep_ratio=True,
        with_keypoints=True),
    dict(type='RandomFlip', flip_ratio=0.5, with_keypoints=True, gt_num_keypoints=17),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='GaussianMap', strides=[8,16,32,64], num_keypoints=17, sigma=2),
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
        # img_scale=(800,512),
        img_scale=(1000,640),
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

exp_name = 'PoseDet_DLA34_coco640'
work_dir = './output/' + exp_name