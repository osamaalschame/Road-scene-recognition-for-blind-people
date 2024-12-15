# Road Scene Recognition for Blind People

A computer vision system that combines YOLO segmentation with Monodepth2 distance estimation to assist visually impaired individuals in understanding road scenes. This project runs on Python 3.7 and above.

## Demo
[![Demo Video](output/IMG_1508.jpg)](output/IMG_1546.MOV)

## ⚙️ Setup

Install the required dependencies using pip:

```shell
pip install torch==1.8.0
pip install torchvision==0.9.0
pip install ultralytics
pip install opencv-contrib-python
pip install numpy
pip install Pillow
```

Alternatively, install all dependencies at once using:

```shell
pip install -r requirements.txt
```

## 🖼️ Prediction

The system can process a single image, multiple images from a directory, or video input. Use the following command:

```shell
python test.py --input_path <path> --output_path <path> --segmentation_data <dataset>
```

### Arguments

| Parameter | Description | Type | Options |
|-----------|-------------|------|----------|
| `--input_path` | Input file or directory path | string | - Single image: `image.jpg`<br>- Image directory: `./images_folder/images`<br>- Video file: `video.mp4` |
| `--output_path` | Output directory for results | string | e.g., `./save_out/output` |
| `--segmentation_data` | Dataset used for segmentation | string | `our` or `mapillary` |

## Segmentation Datasets

The project supports two different segmentation datasets with distinct class sets:

### 1. Mapillary Dataset Classes
- Road
- Lane Marking - Crosswalk
- Sidewalk
- Obstacle
- Car
- Person
- Traffic Light-Street
- Bike Lane
- Bicycle
- Traffic Light-Sidewalk
- Pedestrian Area

### 2. Our Dataset Classes
- Bike
- Bikelane
- Car
- Crosswalk
- E-Scooter
- Obstacle
- Person
- Road
- Sidewalk
- Stairs
- Traffic Light
