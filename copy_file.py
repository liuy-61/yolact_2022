import os
from data.local import city_coco_format_image

import shutil
from tqdm import tqdm

def copy_file(sorce_file_folder, new_file_folder):
    if not os.path.exists(new_file_folder):
        os.makedirs(new_file_folder)

    for sub_dir in os.listdir(sorce_file_folder):
        dir = os.path.join(sorce_file_folder, sub_dir)
        for file in tqdm(os.listdir(dir)):
            shutil.copy(os.path.join(dir, file), new_file_folder)
            debug = 0

def move_file(sorce_file_folder, new_file_folder):
    if not os.path.exists(new_file_folder):
        os.makedirs(new_file_folder)

    for sub_dir in os.listdir(sorce_file_folder):
        sub_dir = os.path.join(sorce_file_folder, sub_dir)
        for dir in os.listdir(sub_dir):
            print("开始移动"+dir+"目录下文件")
            dir = os.path.join(sub_dir, dir)
            for file in tqdm(os.listdir(dir)):
                debug = 0
                shutil.move(os.path.join(dir, file), new_file_folder)
                debug = 0


if __name__ == '__main__':

   move_file('/home/algroup/liuy/cityscapes/leftImg8bit', city_coco_format_image)