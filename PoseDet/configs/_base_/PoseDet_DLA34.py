norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict( 
    type='PoseDetDetector',
    pretrained='pretrained/dla34-ba72cf86.pth',
    backbone=dict(
        type='DLA',
        return_levels=True,
        levels=[1, 1, 1, 2, 2, 1],
        channels=[16, 32, 64, 128, 256, 512],
        ouput_indice=[3,4,5,6],
        ),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=128,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=4,
        # num_outs=3,
        norm_cfg=norm_cfg,),
    bbox_head=dict(
        # type='PoseDetHead',
        type='PoseDetHeadHeatMapMl',
        norm_cfg=norm_cfg,
        num_classes=1,
        in_channels=128,
        feat_channels=128,
        embedding_feat_channels=128,
        init_convs=3,
        refine_convs=2,
        cls_convs=2,
        gradient_mul=0.1,
        dcn_kernel=(1,17),
        refine_num=1,
        point_strides=[8, 16, 32, 64],
        point_base_scale=4,
        num_keypoints=17,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_keypoints_init=dict(type='KeypointsLoss', 
                                d_type='L2', 
                                weight=.1, 
                                stage='init',
                                normalize_factor=1,
                                ),
        loss_keypoints_refine=dict(type='KeypointsLoss', 
                                d_type='L2', 
                                weight=.2, 
                                stage='refine',
                                normalize_factor=1,
                                ),
        loss_heatmap=dict(type='HeatmapLoss', weight=.1, with_sigmas=False),
        )
    )

# training and testing settings
train_cfg = dict(
    init=dict(
        assigner=dict(type='KeypointsAssigner', 
                            scale=4, 
                            pos_num=1, 
                            number_keypoints_thr=3, 
                            num_keypoints=17,
                            center_type='keypoints',
                            # center_type='box'
                            ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    refine=dict(
        assigner=dict(
            type='OksAssigner',
            pos_PD_thr=0.7,
            neg_PD_thr=0.7,
            min_pos_iou=0.52,
            ignore_iof_thr=-1,
            match_low_quality=True,
            num_keypoints=17,
            number_keypoints_thr=3, #
            ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False
        ),    
    cls=dict(
        assigner=dict(
            type='OksAssigner',
            pos_PD_thr=0.6,
            neg_PD_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1,
            match_low_quality=False,
            num_keypoints=17,
            number_keypoints_thr=3, 
            ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False
        ),
    )
test_cfg = dict(
    nms_pre=500,
    min_bbox_size=0,
    score_thr=0.05,
    # nms=dict(type='keypoints_nms', iou_thr=0.2),
    nms=dict(type='keypoints_nms', iou_thr=0.3),
    max_per_img=100)