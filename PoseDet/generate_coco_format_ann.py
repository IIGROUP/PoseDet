
import matplotlib.pyplot as plt
import os
import numpy as np
from imantics import Polygons, Mask
import json
from tqdm import tqdm

small_data_num = 500 


def get_file_names(folder, SMALL=False):
    all_file_names = os.listdir(folder)
    file_names = []

    for file_name in all_file_names:

        if file_name.endswith('jpg'):
            file_names.append(file_name[:-4])
    if SMALL:
        if not small_data_num < len(file_names):
            print('small_data_num is more than true data number')
            pass
        file_names = file_names[:small_data_num]
    return file_names

def get_img_mask_path(folder, file_names):
    img_pathes = []
    mask_pathes = []
    for file_name in file_names:
        img_path = os.path.join(folder, file_name+'.jpg')
        img_pathes.append(img_path)
        mask_path = os.path.join(folder, file_name+'.png')
        mask_pathes.append(mask_path)
    return img_pathes, mask_pathes

def get_img_shape(img_path):
    pic = plt.imread(img_path)
    h, w, c = pic.shape
    return h, w

def get_mask_cate(mask_path):
    mask = plt.imread(mask_path)*255
    mask = mask.astype(np.uint8)
    cat_value = []
    for i in range(256):
        table = [mask==i]
        if not np.sum(table) == 0:
            cat_value.append(i)
    if len(cat_value) > 9:
        print(cat_value)

#从png的mask里提取mask per object, 
#input: mask(0-1)
#output:masks: list, each of them is numpy array(bool)
def extract_mask(mask):
    mask = mask.astype(np.uint8)
    masks = []
    masks_cat = []
    for i in range(1, 256):
        sub_mask = (mask==i)
        if np.sum(sub_mask) !=0:
            masks.append(sub_mask)
            masks_cat.append(i)
    return masks, masks_cat


#extract bbox from mask of one object
def bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    width = cmax - cmin + 1
    height = rmax - rmin + 1
    maskInt = mask.astype(int)
    area = np.sum(maskInt)
    return area, [int(cmin), int(rmin), int(width), int(height)]

 
def mask_to_polygons(mask):
    polygons = Mask(mask).polygons().points

    # filter out invalid polygons (< 3 points)
    polygons_filtered = []
    for polygon in polygons:
        polygon = polygon.reshape(-1)
        polygon = polygon.tolist()
        if  len(polygon) % 2 == 0 and len(polygon) >= 6:
            polygons_filtered.append(polygon)
    return polygons_filtered

def devide_set(file_names, mode, val_split=0.2, seed=1234):
    file_names = np.array(file_names)
    np.random.seed(seed)
    num = len(file_names)
    index = np.zeros(num)
    val_num = int(num*val_split)
    index[:val_num] = 1
    np.random.shuffle(index)
    train_index = [index==0]
    val_index = [index==1]
    if mode == 'train':
        return file_names[train_index]
    elif mode =='val':
        return file_names[val_index]
    elif mode =='test':
        return file_names

def generate_standert_dataset_dict(folder, mode='train', SMALL=False):
    # folder = '/data/public/Transfer/dat, aset/Seg/self_labelled/PorSegIns/imgs/train'
    file_names = get_file_names(folder, SMALL)
    file_names_set = devide_set(file_names, mode)
    img_pathes, mask_pathes = get_img_mask_path(folder, file_names_set)

    standert_dataset_dicts = {'categories':[], 'images':[], 'annotations':[]}

    img_id = 0
    ann_id = 0

    standert_dataset_dicts['categories'].append({'id':1, 'name':'person'})

    for img_path, mask_path in tqdm(zip(img_pathes, mask_pathes)):

        img_file_name = img_path.split('/')[-1]
        # print(img_file_name)
        # img_path_saved = img_path #旧版本
        img_path_saved = 'self_labelled/single_person/' + img_file_name #新版本
        image = {'file_name':img_path_saved,
                    'id':img_id,
                    'width': 512,
                    'height': 512}

        standert_dataset_dicts['images'].append(image)

        mask = plt.imread(mask_path)
        if len(mask.shape)==3:
            assert np.sum(mask)==np.sum(mask[:,:,0])*3
            mask = mask[:,:,0]
            # print('convet rgb to gray ', mask_path)
        assert len(mask.shape)==2
        mask = mask*255
        mask = mask.astype(np.uint8)
        masks, masks_cat = extract_mask(mask)
        annotation = []
        for sub_mask, sub_mask_cat in zip(masks, masks_cat):

            obj = {}
            area, box = bbox(sub_mask)
            segmentation = mask_to_polygons(sub_mask)
            if len(segmentation)==0:
                continue
            segmentation_np = np.array(segmentation)
            # print(segmentation_np.shape)
            # assert len(segmentation_np.shape)==2
            assert len(segmentation)!=0
            annotation = {'area': int(area),
                        'bbox': box,
                        'category_id': 1,
                        'id': ann_id,
                        'image_id': img_id,
                        'iscrowd': 0,
                        # mask, 矩形是从左上角点按顺时针的四个顶点
                        'segmentation': segmentation,
                        'sub_mask_cat': sub_mask_cat
                        } 
            standert_dataset_dicts['annotations'].append(annotation)
            ann_id += 1

        img_id += 1
    print('Found %d imgs in %s set'%(len(standert_dataset_dicts['images']), mode))
    return standert_dataset_dicts

def load_save_dict(ann_path, img_folder, mode='train', SMALL=False):

    print('start generate...')
    standert_dataset_dicts = generate_standert_dataset_dict(img_folder, mode, SMALL)

    with open(ann_path, 'w') as f:
        print(type(standert_dataset_dicts))
        json.dump(standert_dataset_dicts, f)

    print('save data file in ', ann_path)


def main():
    '''
    ann_path：要保存的ann file路径
    img_folder：要读取的img folder，里面同时包含.jpg的image和.png的标注
    mode: 
        'train' - 从img folder取80%生成训练集
        'val' - 从img folder取20%生成验证集
        'test' - 从img folder取所有图片生成测试集
    '''
    ann_path = '/data/public/Transfer/dataset/Seg/public_dataset/imgs/annotations/train.json'
    img_folder = '/data/public/Transfer/dataset/Seg/self_labelled/PorSegIns/imgs/train'
    mode = 'train'

    load_save_dict(ann_path, img_folder, mode=mode)

if __name__=='__main__':
    main()

