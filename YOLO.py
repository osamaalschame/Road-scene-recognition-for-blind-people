import cv2
import numpy as np
from ultralytics import YOLO
import os
from random import randrange

import torch
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class YOLOImageProcessor:
    """
    A class for processing images using YOLOv8 object detection model.
    """

    def __init__(self, model_path, class_names, include_class):
        """
        Initialize the YOLOv8 object detection model and class names.

        Args:
            model_path (str): The path to the trained YOLOv8 model weights file.
            class_names (List[str]): A list of all class names in the image.
            include_class (List[str]): A list of class names to detect in the image.
        """
        self.model = YOLO(model_path)
        self.class_names = class_names
        self.include_class = include_class

    
    def getMask(self,masks,img):
        num_instances = masks.shape[0]
        h = img.shape[0]
        w = img.shape[1]
        # print(f' difference : {abs(masks.shape[1]-img.shape[0])} / {abs(masks.shape[2]-img.shape[1])}')
        img_mask = np.zeros([h, w, 3], np.uint8)
        #
        for i in range(num_instances):
            rand_color = (randrange(255), randrange(255), randrange(255))
            #
            array_img = np.zeros([h, w, 3], np.uint8)
            singleChannelMask=cv2.resize(masks[i],(w,h),interpolation=cv2.INTER_NEAREST)
            # print(singleChannelMask.shape,img.shape)
            array_img[:,:,0],array_img[:,:,1],array_img[:,:,2]=singleChannelMask,singleChannelMask,singleChannelMask
            img_mask[np.where((array_img==1).all(axis=2))]=rand_color
        #
        img_mask = np.asarray(img_mask)
       
        
        return img_mask

    def process_detect_image(self, image_path,mode, confidence_threshold= 0.65):
        """
        Process an image to detect objects using YOLOv8 model and extract regions of interest (ROIs).

        Args:
            image_path (str): The path to the input image file.
            confidence_threshold (float): The confidence threshold for object detection (default: 0.5).

        Returns:
            A list of tuples containing the ROI image array, bounding box coordinates, and center pixel value.
        """

        img = cv2.imread(image_path) if mode !='video' else image_path
        # img=image_path.copy()
        
        results = self.model.predict(img, conf=confidence_threshold)
        # print('Result',results)
        rois = []
        # print(results[0].masks.masks[0].unique())
        # masks=self.getMask(results[0].masks.data.numpy(),img)
        
        for i in range(len(results[0])):
            
            class_id = int(results[0].boxes.cls.tolist()[i])
            class_name = self.class_names[class_id]
            
            if class_name in self.include_class:
                # xywh = results[0].boxes.xywh[i].tolist() # box boundary [xCenter, yCenter, width, height]
                xyxy = results[0].boxes.xyxy[i].tolist() # box boundary [xCenter, yCenter, width, height]
                rois.append({class_name:xyxy})

        return rois


# def detect(image_path):
    
#     # Set up the YOLO image processor
#     model_path = 'bestV2.pt'
#     class_names = {0: 'Bike', 1: 'Bikelane', 2: 'Car', 3: 'Crosswalk', 4: 'E-Scooter', 5: 'Obstacle', 6: 'Person', 7: 'Road', 8: 'Sidewalk', 9: 'Stairs', 10: 'Traffic Light'}
#     include_class = ["E-Scooter","Obstacle","Person","Bike"]
#     processor = YOLOImageProcessor(model_path, class_names, include_class)

#     rois ,masks= processor.process_detect_image(image_path)
    
#     return rois,masks
    
    

# if __name__ == '__main__':
#     detect('0242.jpg')