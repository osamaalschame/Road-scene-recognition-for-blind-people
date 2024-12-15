from ultralytics import YOLO

# Load a model
model = YOLO("./weights/bestMapV2.pt")  # load pretrained model
# model = YOLO("yolov8x-seg.pt")  # load a pretrained model (recommended for training)
model.train(data="data.yaml", pretrained=True,
            epochs=600, imgsz=640, batch=-1, overlap_mask=True,
            patience=50, cache=True, verbose=True, optimizer='auto', project='Mapiliary_11cls')
