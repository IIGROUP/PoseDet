import json
import os

data_root = '/data/ly/coco/'

file = 'person_keypoints_val2017.json'

file_path = os.path.join(data_root, 'annotations', file)

with open(file_path) as f:
    dataset = json.load(f)
print(dataset.keys()) # ['info', 'licenses', 'images', 'annotations', 'categories']

images = dataset['images']
annotations = dataset['annotations'] #['segmentation', 'num_keypoints', 'area', 'iscrowd', 'keypoints', 'image_id', 'bbox', 'category_id', 'id']
# print(annotations[0].keys())

new_images = []
image_ids = []
small = 500
for img in images[:small]:
    new_images.append(img)
    image_ids.append(img['id'])
new_ann = []
for ann in annotations:
    if ann['image_id'] in image_ids:
        new_ann.append(ann)

dataset['images'] = new_images
dataset['annotations'] = new_ann

saved_file = 'person_keypoints_val2017_small500.json'
file_path = os.path.join(data_root, 'annotations', saved_file)

with open(file_path, 'w') as f:
    json.dump(dataset, f)

