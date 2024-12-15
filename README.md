# Road scene segmentation and Distance estimation

This project depends on YOLO model for segementation and monodepth2 model for distance estimation.This code work on python >= 3.7X. before run the code, you should install the dependencies libraries.

# Demo
[![Watch the video]](output/IMG_1546.MOV)
[(output/IMG_1508.jpg)]

## ⚙️ Setup

```shell
pip install torch==1.8.0
pip install torchvision==0.9.0
pip install ultralytics
pip install opencv-contrib-python   
pip install numpy 
pip install Pillow

```
or run the command to download the required libraries: 

```shell
 pip install -r  requirements.txt                               
```


<!-- We recommend using a [conda environment](https://conda.io/docs/user-guide/tasks/manage-environments.html) to avoid dependency conflicts.

We also recommend using `pillow-simd` instead of `pillow` for faster image preprocessing in the dataloaders. -->


## 🖼️ Prediction for  a single image or directory path or video

You can predict using this command:

```shell
python test.py --input_path assets/IMG_1533.jpg --output_path ./output --segmentation_data our
                            *./images_folder/images 
                            * assets/video.mp4                                
```



## Arguments
| source              | value                                          | type                   |
|---------------------|----------------------------------------------- |------------------------|
|  --input_path       | image.jpg , ./images_folder/images , video.mp4 | string                 |
| --output_path       | ./save_out/output                              | string                 |
| --segmentation_data | options=['our','mapillary']                    |     -                  |

## Segementation Data
for this project, segementation model is trained on different dataset and different classes.

1. Mapillary dataset classes: 
    'Road', 
    'Lane Marking - Crosswalk', 
    'Sidewalk',
    'Obstacle',
    'Car',
    'Person',
    'Traffic Light-Street', 
    'Bike Lane',
    'Bicycle',
    'Traffic Light-Sidewalk',
    'Pedestrian Area'

2. our dataset: 
    'Bike',
    'Bikelane',
    'Car',  
    'Crosswalk', 
    'E-Scooter',
    'Obstacle', 
    'Person', 
    'Road', 
    'Sidewalk', 
    'Stairs', 
    'Traffic Light'



