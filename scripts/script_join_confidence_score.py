from data.local import score_root, confidence_root
import os
import pickle
score_name = 'yolact_score.pickle'
confidence_name = 'confidence_COCO_Person_threshold_0.3.pkl'
confidence_add_score_name = 'confidence_threshold_0.3_add_yolact_score.pkl'
confidence_with_score_name = 'confidence_threshold_0.3_with_yolact_score.pkl'
with open((os.path.join(score_root, score_name)), 'rb') as f:
    score_list = pickle.load(f)

with open((os.path.join(confidence_root, confidence_name)), 'rb') as f:
    confidence_list = pickle.load(f)

score_list.sort(key=lambda x: x['file_name'])
confidence_list.sort(key=lambda x: x['file_name'])

for i in range(len(score_list)):
    if score_list[i]['file_name'] != confidence_list[i]['file_name']:
        score_list.pop(i)
        break

confidence_add_score_list =[{'file_name': x['file_name'], 'value': y['confidence'] + x['score']} for x, y in zip(score_list, confidence_list)]
confidence_with_score_list =[{'file_name': x['file_name'], 'confidence': y['confidence'], 'score': x['score']} for x, y in zip(score_list, confidence_list)]

with open((os.path.join(score_root, confidence_add_score_name)), 'wb') as f:
    pickle.dump(confidence_add_score_list, f)

with open((os.path.join(score_root, confidence_with_score_name)), 'wb') as f:
    pickle.dump(confidence_with_score_list, f)

debug = 0