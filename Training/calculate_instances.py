import os
from tqdm import tqdm
import json

path = 'Own_data/Data_val'


def get_json(path):
    selected_labels = {}
    for roots, dirs, files in os.walk(path):
        for jsn in tqdm(files, total=len(files)):
            if jsn.endswith('.json'):
                with open(os.path.join(roots, jsn), 'r') as j:
                    contents = json.loads(j.read())
                    for lab in contents['shapes']:
                        if not lab['label'] in selected_labels.keys():
                            selected_labels[lab['label']] = 1
                        else:
                            selected_labels[lab['label']]+= 1
    return selected_labels, list(selected_labels.keys()).__len__()


print(get_json(path))
