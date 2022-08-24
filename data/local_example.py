import os
"""
    只需要按实际情况配置
    1、coco_person_train
    2、coco_person_val
    3、tb_dir
    4、loss_root
    5、confidence_root
    6、score_root
    7、tmp_json_root
    8、weight_root
    
    

"""
# coco数据地址
COCO_DIR = '/home/algroup/data/coco'
train_image = f'{COCO_DIR}/train2014'
val_image = f'{COCO_DIR}/val2014'
test_image = f'{COCO_DIR}/test2014'
train_info = f'{COCO_DIR}/annotations/instances_train2014.json'
val_info = f'{COCO_DIR}/annotations/instances_val2014.json'

person_train_info = f'{COCO_DIR}/annotations/person_train2014.json'
person_val_info = f'{COCO_DIR}/annotations/person_val2014.json'


PROJ_DIR = '/home/algroup/liuy/yolact'
# tensorborad记录目录
tb_dir = f'{PROJ_DIR}/tensorboard'

# 保存loss的地点
loss_root = f'{PROJ_DIR}/loss'

# 保存confidence的地点
confidence_root = f'{PROJ_DIR}/confidence'

# 保存score的地点
score_root = f'{PROJ_DIR}/score'

# 一个缓存临时json的地址
tmp_json_root = f'{PROJ_DIR}/tmp_json'

# 保存weight的地址
weight_root = f'{PROJ_DIR}/weights'
