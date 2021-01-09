_base_ = [
    './PoseDet_r18.py',
]
channels=128
# channels=256
model = dict(
    pretrained='pretrained/resnet50-19c8e357.pth',
    backbone=dict(
        depth=50,
        ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=channels,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=4,
        ),
    bbox_head=dict(
        point_strides=[8, 16, 32, 64],
        in_channels=channels,
        feat_channels=channels,
        embedding_feat_channels=channels,
        )
    )