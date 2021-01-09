_base_ = [
    './PoseDet_DLA34.py',
]

# channels = 128
channels = 256
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    pretrained='pretrained/hrnetv2_w32-dc9eeb4f.pth',
    backbone=dict(
        type='HRNet',
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256)))),
    neck=dict(
        type='HRFPN',
        in_channels=[32, 64, 128, 256],
        out_channels=channels,
        stride=2,
        num_outs=3,
        norm_cfg=norm_cfg),
    bbox_head=dict(
        point_strides=[8, 16, 32],
        in_channels=channels,
        feat_channels=channels,
        embedding_feat_channels=channels,
        ),
    ) 