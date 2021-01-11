_base_ = [
    '../_base_/dataset_coco.py',
    '../_base_/PoseDet_DLA34.py',
    '../_base_/train_schedule.py',
]

exp_name = 'PoseDet_DLA34_coco'
work_dir = './output/' + exp_name