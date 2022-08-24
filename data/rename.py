import os
import re
from data.local import weight_root
# TODO 修改uncertainty 采样下的权重文件名和 tensorboard文件名修改的更清楚
def rename(src_folder):
    # TODO 找出src_folder下的所有子文件夹，如：train_coco_sample_uncertainty_drop_0.0
    for sub_folder in os.listdir(src_folder):
        if re.match("train_coco_sample_uncertainty_drop", sub_folder, flags=0) and 'small' not in sub_folder:

            pos = re.search('train_coco_sample_uncertainty_drop', sub_folder).span()
            pre = sub_folder[pos[0]: pos[1]]
            post = sub_folder[pos[1]: len(sub_folder)]
            list = [pre, post]
            new_name = os.path.join("_large_loss".join(list))
            new_name = os.path.join(src_folder, new_name)
            old_name = os.path.join(src_folder, sub_folder)
            os.rename(old_name, new_name)
            dbeug = 0



if __name__ == '__main__':
    rename('/home/algroup/liuy/yolact/tensorboard')