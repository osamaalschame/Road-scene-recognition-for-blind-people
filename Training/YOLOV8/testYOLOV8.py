from ultralytics import YOLO
import torch
torch.cuda.empty_cache()
# Load the YOLOv8 model
model = YOLO('./weights/best.pt')  # trained weights
model.val()
