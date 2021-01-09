_base_ = [
    './PoseDet_DLA34.py',
]
channels=128
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    pretrained='pretrained/hrnetv2_w32-dc9eeb4f.pth',
    backbone=dict(
        _delete_=True,
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=channels,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=4,
        norm_cfg=norm_cfg,
        ),
    bbox_head=dict(
        point_strides=[8, 16, 32, 64],
        in_channels=channels,
        feat_channels=channels,
        embedding_feat_channels=channels,
        )
    )