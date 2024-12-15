# Training Segmentation Models (YOLOv8 - Detectron2)


## YOLOv8
in order to train the yolov8, you should follow the steps:
- library requirements:
```shell
    pip install ultralytics
    pip install tqdm                              
```
- prepare data label as txt file as this format
    <class id><polygon coordinates>
    class id start form 0 and polygon coordinates should be normalized (0 to 0)  and flaten as (px0 py0 ..... pxn pyn)
    for example:

    0 0.2121 0.432 ......

if you label your data using labelme, you can use changeLabelMeToYOLO.py to convert labelme files to YOLOv8. 

- prepare your dataset as following:
    -train
        -images
        -labels
    -val
        -images
        -labels

- prepare data.yaml and include :
        train: path to train images

        val: path to val images

        nc: num of classes

        classes: [
            include the classes names in this list
        ]
- use the trainYOLOV8.py file to train you data. for more train parameters you can refer to [Train arguments](https://docs.ultralytics.com/modes/train/#arguments)

- testYOLOV8.py to evaluate the trained model, just specify you path for best trained weight and update it in line 5.



## Detectron2
in order to train the Detectron2, you should have python>=3.7 and torch>=1.8 and openCVfollow the steps:
* gcc & g++ ≥ 5.4 are required to build the Detectron2

- library requirements:
```shell
    pip install torch==1.8.0

    pip install torchvision==0.9.0

    python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

    pip install matplotlib

    pip install numpy   

    pip install pillow                           
```
for Installation information, you may refer this page [detectron2 installation](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

- you should prepare your labels for train and val as COCO format.if you have your labels as LabelMe format thus you can use labelme2coco.py file to change your labels to COCO format.
    - train.json
    - val.json
if you will use labelme2coco.py you have install labelme library
```shell
pip install labelme
```
and use this command to create train.json and val.json:
```shell
python labelme2coco.py --labelme_images Directory to labelme images and annotation json files --output Output json file path as train.json 
```
for more information on coco format [coco foramt](https://haobin-tan.netlify.app/ai/computer-vision/object-detection/coco-dataset-format/)

- to run the training for detectron2, you should edit train_detectron.py line 24 and 25, specifying the directory for your train.json and val.json with their corresponding images
