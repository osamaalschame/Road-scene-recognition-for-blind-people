import json
import os
import numpy as np
from tqdm import tqdm


classes={'Road': 0, 'Lane Marking - Crosswalk': 1, 'Sidewalk': 2, 'Obstcale': 3, 'Car': 4,
'Person': 5, 'Traffic Light-Street': 6, 'Bike Lane': 7, 'Bicycle': 8, 'Traffic Light-Sidewalk': 9,
'Pedestrian Area': 10}  # you should specify your classes here as this dictionary

path_for_labelme_files = '/Osama/validation/v2.0/LabelMe'  # specify you path to the labelme files
output_path = '/Osama/validation/labels'  # specify your path for output conversion


def repairMins(pnts, w, h):
    for i, pnt in enumerate(pnts):
        if pnt[0] < 0:
            pnts[i][0] = 0
        if pnt[0] > w:
            pnts[i][0] = w
        if pnt[1] < 0:
            pnts[i][1] = 0
        if pnt[1] > h:
            pnts[i][1] = h
    return pnts


def prepare_txt():
    for roots, dirs, files in os.walk(path_for_labelme_files):
        for file in tqdm(files, total=len(files)):
            if file.endswith('.json') and not file.startswith(('.', '_')):
                with open(os.path.join(roots, file), "r") as fp, open(os.path.join(output_path, file.replace('.json', '.txt')), 'w') as out_txt:
                    data = json.load(fp)
                    height = data["imageHeight"]
                    width = data["imageWidth"]

                    for shapes in data["shapes"]:
                        label = shapes["label"]
                        points = shapes["points"]
                        points = repairMins(points, width, height)

                        points = list(np.asarray(points).astype(np.int).flatten())
                        for i in range(0, len(points), 2):
                            points[i] = round(points[i] / width, 6)
                            points[i + 1] = round(points[i + 1] / height, 6)
                        points.insert(0, classes[label])
                        points=" ".join([str(i) for i in points])
                        out_txt.write(points+ '\n')
                    out_txt.close()


def remove_txt():
    for roots, dirs, files in os.walk(output_path):
        for file in tqdm(files, total=len(files)):
            if file.endswith('.txt') and not file.startswith(('.', '_')):
                os.remove(os.path.join(roots, file))


prepare_txt()
