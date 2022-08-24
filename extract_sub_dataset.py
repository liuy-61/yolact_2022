"""
    提取coco格式数据集中一部分数据
"""
import math
import os

from tqdm import tqdm
import random
from data.local import coco_sub_person_train, coco_sub_person_val, person_train_info, person_val_info, coco_mini_person_train, coco_mini_person_val

import json

#  多增了一个返回值
def extract_sub_dataset(ann_file, extract_num, new_ann_file):
    """
    :param ann_file: coco原来的annotation文件路径
    :param extract_num: 需要提取的数据数目
    :param new_ann_file: 提取后的文件保存路径
    :return: 提取的image list
    """
    with open(ann_file, 'r', encoding='utf8') as fp:
        info_json = json.load(fp)
        #  先shuffle info_json['images'] list, 保留前extract_num个元素， 生成一个 image_id_list。
        random.shuffle(info_json['images'])
        if extract_num < 1:
            extract_num = min(len(info_json['images']),  math.ceil(len(info_json['images']) * extract_num))
        if extract_num > len(info_json['images']):
            print("extract_num参数大于数据集样本总数")
            return
        info_json['images'] = info_json['images'][:extract_num]
        image_id_list = [image['id'] for image in info_json['images']]
        #  再修改info_json['annotations'] list， list每个元素都是一个dict, 将dict中image_id不在image_id_list的删除掉。
        print("开始提取info_json['annotations']")
        for i in tqdm(range(len(info_json['annotations']) - 1, -1, -1)):
            if info_json['annotations'][i]['image_id'] not in image_id_list:
                info_json['annotations'].pop(i)
        print("提取info_json['annotations']已经完成")
        #  保存info_json 到 new_ann_file
        ret = [x['file_name'] for x in info_json['images']]
        json.dump(info_json, open(new_ann_file, 'w'))
        print("保存" + new_ann_file + "已完成")
        return ret

# TODO 将训练集随机划分成两份
def split_dataset(ann_file, new_ann_file_A, new_ann_file_B) :
    pass

if __name__ == '__main__':
    ann_file_list = [person_train_info, person_val_info]
    new_ann_file_list = [coco_mini_person_train, coco_mini_person_val]
    extract_num_list = [250, 50]

    for ann_file, extract_num, new_ann_file in zip(ann_file_list, extract_num_list, new_ann_file_list):
        extract_sub_dataset(ann_file, extract_num, new_ann_file)