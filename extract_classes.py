from pycocotools.coco import COCO
import os
import shutil
import json

'''
  提取coco格式数据集某些类的数据
'''
#原coco数据集的路径
dataDir = '/home/algroup/liuy/COCO'
#用于保存新生成的数据的路径
savepath = "/home/algroup/liuy/COCO/person"
#最后生产的json文件的保存路径
anno_save = savepath + 'annotations'
'''
数据集参数
'''
#coco有80类,这里写自己需要提取的类的名称
classes_names = ['person']
datasets_list = ['val2014', 'train2014']

#生成保存路径
def mkr(path):
    if os.path.exists(path):
         shutil.rmtree(path)
         os.makedirs(path)
    else:
        os.makedirs(path)

#获取并处理所有需要的json数据
def process_json_data(annFile):
    #获取COCO_json的数据
    coco = COCO(annFile)
    #拿到所有需要的图片数据的id
    classes_ids = coco.getCatIds(catNms = classes_names)
    #加载所有需要的类别信息
    classes_list = coco.loadCats(classes_ids)
    #取所有类别的并集的所有图片id
    #如果想要交集，不需要循环，直接把所有类别作为参数输入，即可得到所有类别都包含的图片
    imgIds_list = []
    for idx in classes_ids:
        imgidx = coco.getImgIds(catIds=idx)
        imgIds_list += imgidx
    #去除重复的图片
    imgIds_list = list(set(imgIds_list))
    #一次性获取所有图像的信息
    image_info_list = coco.loadImgs(imgIds_list)
    #获取图像中对应类别的分割信息,由catIds来指定
    annIds = coco.getAnnIds(imgIds=[], catIds=classes_ids, iscrowd=None)
    anns_list = coco.loadAnns(annIds)
    return classes_list, image_info_list, anns_list

#保存数据到json
def save_json_data(json_file,classes_list,image_info_list,anns_list):
    coco_sub = dict()
    coco_sub['info'] = dict()
    coco_sub['licenses'] = []
    coco_sub['images'] = []
    coco_sub['type'] = 'instances'
    coco_sub['annotations'] = []
    coco_sub['categories'] = []
    #以下非必须,为coco数据集的前缀信息
    coco_sub['info']['description'] = 'COCO 2017 sub Dataset'
    coco_sub['info']['url'] = 'https://www.cnblogs.com/lhdb/'
    coco_sub['info']['version'] = '1.0'
    coco_sub['info']['year'] = 2020
    coco_sub['info']['contributor'] = 'smh'
    coco_sub['info']['date_created'] = '2020-7-1 10:06'
    sub_license = dict()
    sub_license['url'] =  'https://www.cnblogs.com/lhdb/'
    sub_license['id'] = 1
    sub_license['name'] = 'Attribution-NonCommercial-ShareAlike License'
    coco_sub['licenses'].append(sub_license)
    #以下为必须插入信息,包括image、 annotations、 categories三个字段
    #插入image信息
    coco_sub['images'].extend(image_info_list)
    #插入annotation信息
    coco_sub['annotations'].extend(anns_list)
    #插入categories信息
    coco_sub['categories'].extend(classes_list)
    #自此所有该插入的数据就已经插入完毕啦٩( ๑╹ ꇴ╹)۶
    #最后一步，保存数据
    json.dump(coco_sub, open(json_file, 'w'))


if __name__ == '__main__':
    mkr(anno_save)
    #按单个数据集进行处理
    for dataset in datasets_list:
        #获取要处理的json文件路径
        annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataset)
        #存储处理完成的json文件路径
        json_file = '{}/instances_{}_sub.json'.format(anno_save, dataset)
        #处理数据
        classes_list, image_info_list, anns_list = process_json_data(annFile)
        #保存数据
        save_json_data(json_file, classes_list, image_info_list, anns_list)
        print('instances_{}_sub.json'.format(dataset))

