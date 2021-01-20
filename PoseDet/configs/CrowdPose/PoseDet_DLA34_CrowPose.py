_base_ = [
    '../_base_/dataset_coco.py',
    '../_base_/PoseDet_DLA34.py',
    '../_base_/train_schedule.py',
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadKeypoints',
        fourteen2fifteen=True,),
    dict(type='LoadAnnotations', with_bbox=True),    
    dict(
        type='CenterRandomCropXiao',
        scale_factor=0.5,
        rot_factor=0,
        patch_width=512,
        patch_height=512),
    dict(type='Resize',         
        img_scale=[
                    (800,512),
                    # (1000,640),
                    ], 
        multiscale_mode='value', 
        keep_ratio=True,
        with_keypoints=True),
    dict(type='RandomFlip', flip_ratio=0.5, with_keypoints=True, gt_num_keypoints=15),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='GaussianMap', strides=[8,16,32,64], num_keypoints=15, sigma=2),
    dict(type='FormatBundleKeypoints'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 
                                'gt_keypoints', 'gt_num_keypoints',
                                'heatmap', 'heatmap_weight']),
]

dataset_type = 'CrowdPoseKeypoints'
data_root = '/mnt/data/tcy/CrowdPose/'
data = dict(
    samples_per_gpu=12,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/crowdpose_trainval.json',
        img_prefix=data_root + 'images/',
        pipeline=train_pipeline
        ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/crowdpose_test.json',
        img_prefix=data_root + 'images/'
        ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/crowdpose_test.json',
        img_prefix=data_root + 'images/'
        )
    )


model = dict( 
    bbox_head=dict(
        dcn_kernel=(1,15),
        num_keypoints=15)
    )

# training and testing settings
train_cfg = dict(
    init=dict(assigner=dict(num_keypoints=15)),
    refine=dict(assigner=dict(num_keypoints=15)),    
    cls=dict(assigner=dict(num_keypoints=15)),
    )
test_cfg = dict(
    nms_pre=500,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='keypoints_nms', iou_thr=0.3),
    max_per_img=100)

exp_name = 'PoseDet_DLA34_CrowdPose'
work_dir = './output/' + exp_name