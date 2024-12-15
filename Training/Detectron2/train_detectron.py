import torch
assert torch.__version__.startswith("1.8")
import os
from detectron2.data.datasets import register_coco_instances
from Detectron2.evaluator import CocoTrainer, cfg

register_coco_instances("data_train", {}, "/home/chiron/Desktop/Osama/Own_data/train.json", "/home/chiron/Desktop/Osama/Own_data/Data_train")  # specify the train.json directory and train images directroy
register_coco_instances("data_val", {}, "/home/chiron/Desktop/Osama/Own_data/val.json", "/home/chiron/Desktop/Osama/Own_data/Data_val")  # specify the val.json directory and val images directroy

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CocoTrainer(cfg)  # without data augmentation
trainer.resume_or_load(resume=False)
trainer.train()
