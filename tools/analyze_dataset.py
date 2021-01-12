import json
import os

# data_root = '/mnt/data/tcy/coco/'
data_root = '/mnt/data/tcy/CrowdPose/'

# file = 'person_keypoints_val2017.json'
file = 'crowdpose_train.json'

file_path = os.path.join(data_root, 'annotations', file)

with open(file_path) as f:
    dataset = json.load(f)
print(dataset.keys()) # ['info', 'licenses', 'images', 'annotations', 'categories']

images = dataset['images']
annotations = dataset['annotations'] #['segmentation', 'num_keypoints', 'area', 'iscrowd', 'keypoints', 'image_id', 'bbox', 'category_id', 'id']
# print(annotations[0].keys())
print(dataset['categories'])


# saved_file = 'person_keypoints_val2017_small500.json'
# file_path = os.path.join(data_root, 'annotations', saved_file)

# with open(file_path, 'w') as f:
#     json.dump(dataset, f)

data_root = '/mnt/data/tcy/CrowdPose/'
file = 'crowdpose_val.json'
file_path = os.path.join(data_root, 'annotations', file)
with open(file_path) as f:
    dataset2 = json.load(f)

images2 = dataset2['images']
annotations2 = dataset2['annotations'] #['segmentation', 'num_keypoints', 'area', 'iscrowd', 'keypoints', 'image_id', 'bbox', 'category_id', 'id']

images3 = images + images2
annotations3 = annotations+annotations2

dataset['images'] = images3
dataset['annotations'] = annotations3

saved_file = 'crowdpose_trainval.json'
file_path = os.path.join(data_root, 'annotations', saved_file)

with open(file_path, 'w') as f:
    json.dump(dataset, f)