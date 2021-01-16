import json
import os

data_root = '/mnt/data/tcy/coco/'
# data_root = '/mnt/data/tcy/CrowdPose/'

# file = 'person_keypoints_val2017.json'
file = 'person_keypoints_train2017.json'
# file = 'crowdpose_train.json'

file_path = os.path.join(data_root, 'annotations', file)

with open(file_path) as f:
    dataset = json.load(f)
print(dataset.keys()) # ['info', 'licenses', 'images', 'annotations', 'categories']

images = dataset['images']
annotations = dataset['annotations'] #['segmentation', 'num_keypoints', 'area', 'iscrowd', 'keypoints', 'image_id', 'bbox', 'category_id', 'id']
# print(annotations[0].keys())
print(dataset['categories'])


for img in images:
    if img['id'] == 416059:
        print('ttt')
        exit()

# dataset['annotations'] = annotations3

# saved_file = 'crowdpose_trainval.json'
# file_path = os.path.join(data_root, 'annotations', saved_file)

# with open(file_path, 'w') as f:
#     json.dump(dataset, f)