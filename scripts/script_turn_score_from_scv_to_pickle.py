import pandas as pd
import os
import pickle
from pycocotools.coco import COCO
from data.local import score_root, person_train_info
from tqdm import tqdm
if __name__ == '__main__':
    csv_name = 'average_weight.csv'
    pickle_name = 'yolact_score.pickle'
    score_csv = pd.read_csv(os.path.join(score_root, csv_name))
    coco = COCO(person_train_info)
    img_ids = []
    img_scores = []
    for img_id, mask_score, mask_conf, boundary_conf, boundary_score, bbox_score, bbox_conf in tqdm (zip(score_csv.image_id,
                    score_csv.mask_score,
                    score_csv.mask_conf,
                    score_csv.boundary_conf,
                    score_csv.boundary_score,
                    score_csv.bbox_score,
                    score_csv.bbox_conf)):

        img_ids.append(img_id)
        img_scores.append(mask_score +
                    mask_conf +
                    boundary_conf +
                    boundary_score +
                    bbox_score +
                    bbox_conf)

    imgs = coco.loadImgs(img_ids)
    file_names = [x['file_name'] for x in imgs]

    score_list = []
    for name , score in zip(file_names, img_scores):
        dict = {}
        dict['file_name'] = name
        dict['score'] = score
        score_list.append(dict)

    with open(os.path.join(score_root, pickle_name), 'wb') as f:
        pickle.dump(score_list, f)

    with open(os.path.join(score_root,  pickle_name), 'rb') as f:
        l = pickle.load(f)


    debug = 0