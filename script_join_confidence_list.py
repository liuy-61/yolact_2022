from data.local import confidence_root
import os
import pickle

confidence_list_path = os.path.join(confidence_root, 'confidence_COCO_Person_threshold_0.pkl')
confidence_list_A_path = os.path.join(confidence_root, 'confidence_COCO_Person_A_threshold_0.0.pkl')
confidence_list_B_path = os.path.join(confidence_root, 'confidence_COCO_Person_B_threshold_0.0.pkl')

with open(confidence_list_A_path, 'rb') as f:
    confidence_list_A = pickle.load(f)

with open(confidence_list_B_path, 'rb') as f:
    confidence_list_B = pickle.load(f)

confidence_list = [x for x in confidence_list_A]

for x in confidence_list_B:
    confidence_list.append(x)

with open(confidence_list_path, 'wb') as f:
    pickle.dump(confidence_list, f)

with open(confidence_list_path, 'rb') as f:
    l = pickle.load(f)
set = set()
for x in l:
    set.add(x['file_name'])
debug = 0

