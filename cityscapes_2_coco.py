import cv2
import numpy as np
import os, glob
import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools
from data.local import CITY_VAL_IMAGE, CITY_VAL_LABEL, CITY_ROOT, CITY_COCO_VAL_LABEL, CITY_INSTANCE_ROOT, CITY_TRAIN_IMAGE, CITY_TRAIN_LABEL, CITY_INSTANCE_TRAIN_ROOT
from tqdm import tqdm

# ROOT_DIR = '/data/cityscapes/val'
# IMAGE_DIR = os.path.join(ROOT_DIR, "images/frankfurt")
# ANNOTATION_DIR = os.path.join(ROOT_DIR, "gt/frankfurt")
# INSTANCE_DIR = os.path.join(ROOT_DIR, "instances")

# ROOT_DIR = '/data/cityscapes/val'
# IMAGE_DIR = os.path.join(CITY_VAL_IMAGE, "frankfurt")
# ANNOTATION_DIR = os.path.join(CITY_VAL_LABEL, "frankfurt")
# INSTANCE_DIR = os.path.join(CITY_COCO_VAL_LABEL, "instances")

INFO = {
    "description": "Cityscapes_Instance Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": "2019",
    "contributor": "Kevin_Jia",
    "date_created": "2019-12-30 16:16:16.123456"
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]


CATEGORIES = [
    {
        'id': 1,
        'name': 'person',
        'supercategory': 'cityscapes',
    }
]

# 除了行人都设置为背景类
background_label = list(range(-1, 24, 1)) + list(range(25, 34, 1))


def masks_generator(imges, annotations_dir, instance_dir):
    idx = 0
    for pic_name in tqdm(imges):
        annotation_name = pic_name.split('_')[0] + '_' + pic_name.split('_')[1] + '_' + pic_name.split('_')[
            2] + '_gtFine_instanceIds.png'
        # print(annotation_name)
        annotation = cv2.imread(os.path.join(annotations_dir, annotation_name), -1)
        name = pic_name.split('.')[0]
        h, w = annotation.shape[:2]
        ids = np.unique(annotation)
        for id in ids:
            if id in background_label:
                continue
            instance_id = id
            class_id = instance_id // 1000
            if class_id == 24:
                instance_class = 'person'
            else:
                continue
            # print(instance_id)
            instance_mask = np.zeros((h, w, 3), dtype=np.uint8)
            mask = annotation == instance_id
            instance_mask[mask] = 255
            mask_name = name + '_' + instance_class + '_' + str(idx) + '.png'
            cv2.imwrite(os.path.join(instance_dir, mask_name), instance_mask)
            idx += 1


def filter_for_pic(files):
    file_types = ['*.jpeg', '*.jpg', '*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [f for f in files if re.match(file_types, f)]
    # files = [os.path.join(root, f) for f in files]
    return files


def filter_for_instances(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [f for f in files if re.match(file_types, f)]
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    # files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]
    return files

def turn(image_dir, annotation_dir, instance_dir, coco_output):
    """
    :param image_dir: 某个city的image目录
    :param annotation_dir: 某个city的ground truth目录
    :param instance_dir: 用于保存该city的每张image的每个instance的二值Mask图像
    :param coco_output:最终解析出来的coco json文件
    :return:
    """

    # for root, _, files in os.walk(ANNOTATION_DIR):
    background_label = list(range(-1, 24, 1)) + list(range(29, 34, 1))

    # 打开IMAGE_DIR下所有的文件 并过滤出其中的图片 并为每张图片生成每个实例的二值mask图
    files = os.listdir(image_dir)
    image_files = filter_for_pic(files)
    masks_generator(image_files, annotation_dir, instance_dir)

    image_id = 1
    segmentation_id = 1

    files = os.listdir(instance_dir)
    instance_files = filter_for_pic(files)

    # go through each image
    for image_filename in tqdm(image_files):
        image_path = os.path.join(image_dir, image_filename)
        image = Image.open(image_path)
        image_info = pycococreatortools.create_image_info(
            image_id, os.path.basename(image_filename), image.size)
        coco_output["images"].append(image_info)

        # filter for associated png annotations
        # for root, _, files in os.walk(INSTANCE_DIR):
        annotation_files = filter_for_instances(instance_dir, instance_files, image_filename)

        # go through each associated annotation
        for annotation_filename in annotation_files:
            annotation_path = os.path.join(instance_dir, annotation_filename)
            # print(annotation_path)
            class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename][0]

            category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
            binary_mask = np.asarray(Image.open(annotation_path)
                                     .convert('1')).astype(np.uint8)

            annotation_info = pycococreatortools.create_annotation_info(
                segmentation_id, image_id, category_info, binary_mask,
                image.size, tolerance=2)

            if annotation_info is not None:
                coco_output["annotations"].append(annotation_info)

                segmentation_id = segmentation_id + 1

        image_id = image_id + 1

def remove_dir(dir):
    dir = dir.replace('\\', '/')
    if(os.path.isdir(dir)):
        for p in os.listdir(dir):
            remove_dir(os.path.join(dir,p))
        if(os.path.exists(dir)):
            os.rmdir(dir)
    else:
        if(os.path.exists(dir)):
            os.remove(dir)

def main():

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }
    CITY_TRAIN_LABEL
    save_dir = CITY_ROOT
    for city_name in os.listdir(CITY_TRAIN_IMAGE):
        image_dir = os.path.join(CITY_TRAIN_IMAGE, city_name)
        annotation_dir = os.path.join(CITY_TRAIN_LABEL, city_name)
        instance_dir = os.path.join(CITY_INSTANCE_TRAIN_ROOT, city_name)
        if not os.path.exists(instance_dir):
            os.makedirs(instance_dir)
        turn(image_dir, annotation_dir, instance_dir, coco_output)

    with open('{}/{}.json'.format(save_dir, 'train'), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)

if __name__ == "__main__":
    main()



