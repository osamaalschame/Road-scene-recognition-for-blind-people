import cv2
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
import random

register_coco_instances("data_val", {}, "/home/chiron/Desktop/Pavement/val.json", "/home/chiron/Desktop/Pavement/val_data")

cfg=get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "output/model_0109999.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50
cfg.DATASETS.TEST = ("data_val",)
my_dataset_train_metadata = MetadataCatalog.get("data_val")
dataset_dicts = DatasetCatalog.get("data_val")

predictor = DefaultPredictor(cfg)


for d in random.sample(dataset_dicts, 6):
    img = cv2.imread(d["file_name"])
    outputs = predictor(img)
    v = Visualizer(img[:, :, ::-1],
                   metadata=my_dataset_train_metadata,
                   scale=0.8,
                   instance_mode=ColorMode.SEGMENTATION  # removes the colors of unsegmented pixels
                   )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    cv2.imshow('osama', cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    cv2.waitKey()
    cv2.destroyAllWindows()
