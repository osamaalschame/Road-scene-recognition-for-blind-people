import numpy as np
from  PIL import Image
import cv2 
from Monodepth2.test_simple import test_simple
from YOLO import YOLOImageProcessor
import os
import glob
import argparse
#
class Distance_estimation:
    def __init__(self,image,output,model=None,dataset='mapillary'):
        self.output=output
        self.model=model
        self.imgPath=image
        self.mode=None
        # self.img=cv2.imread(image)
        self.scaleY=1
        self.scaleX=1
        self.video_size=None
        self.Yolo_path='models/bestMapV4.pt'\
            if dataset=='mapillary'\
                else\
                    'models/bestOwnDataV2.pt'
        # self.classesNames = {0: 'Bike', 1: 'Bikelane', 2: 'Car', 3: 'Crosswalk', 4: 'E-Scooter', 5: 'Obstacle', 6: 'Person', 7: 'Road', 8: 'Sidewalk', 9: 'Stairs', 10: 'Traffic Light'}
        self.classesNames={0:'Road', 1:'Lane Marking - Crosswalk', 
                            2:'Sidewalk', 3:'Obstacle',4:'Car',
                            5:'Person',6:'Traffic Light-Street', 
                            7:'Bike Lane',
                            8:'Bicycle', 9:'Traffic Light-Sidewalk',
                            10:'Pedestrian Area'} if dataset=='mapillary'\
                                else {0: 'Bike', 1: 'Bikelane', 2: 'Car', 3: 'Crosswalk', 4: 'E-Scooter', 5: 'Obstacle', 6: 'Person', 7: 'Road', 8: 'Sidewalk', 9: 'Stairs', 10: 'Traffic Light'}
        self.selected_classes=["Obstacle","Person","Bicycle","Traffic Light-Sidewalk"]\
                    if dataset=='mapillary' else \
                        ["E-Scooter","Obstacle","Person","Bike"]
        # self.selected_classes = ["E-Scooter","Obstacle","Person","Bike"]
    #
    def getScale(self,image):
        img=cv2.imread(image) if self.mode !='video' else image
        self.scaleY=320/img.shape[0] if self.model=='mono' else 352/img.shape[0]
        self.scaleX=1024/img.shape[1] if self.model=='mono' else 704/img.shape[1]

    def plot_bbox(self,image,XYXY,label,distance):
        # Define the text and font properties
        text = f"{label} : {round(distance/1,2)} m"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 5

        # Get the text size
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

        # Calculate the font scale to fit the text inside the rectangle
        font_scale = min((int(XYXY[2])-int(XYXY[0])) / text_width, (int(XYXY[3])-int(XYXY[1])) / text_height)

        # Draw the rectangle
        cv2.rectangle(image, (int(XYXY[0]), int(XYXY[1])), (int(XYXY[2]), int(XYXY[3])), (0, 255, 0), thickness)
        # cv2.circle(image, (x,y), radius=10, color=(0, 0, 255), thickness=-1)
        # Draw the text
        cv2.putText(image, text, (int(XYXY[0])+5, int(XYXY[1])-20), font, font_scale, (255, 0, 0), thickness)
        
        return image
    
    def get_depth_mono(self,img):
        depth=test_simple(
            model_name='mono_1024x320',
            no_cuda=False,
            image_path=img,
            mode=self.mode,
            pred_metric_depth=True
            )
        return depth.squeeze()
    # 
    # 
    def createOutputPath(self):
        if not os.path.isdir(self.output):
            os.makedirs(self.output)
    #
    def checkPath(self):
        if os.path.isdir(self.imgPath):
            return 'dir'
        elif os.path.isfile(self.imgPath) and (self.imgPath).endswith(('.jpg','.png','.jpeg','.JPG','.PNG','.JPEG')):
            return 'singleImg'
        elif (self.imgPath).endswith(('.mp4','.avi','.mov','.MOV','.MP4','.AVI')):
            return 'video'
        else:
            raise TypeError
    #    
    def get_predictions(self,img):
        bbox=YOLOImageProcessor(self.Yolo_path,self.classesNames,
                                   self.selected_classes).process_detect_image(img,self.mode)
        
        return bbox    
    # 
    def read_video(self,video):
        vidcap = cv2.VideoCapture(video)
        output_path = os.path.join(self.output,os.path.basename(video)) 
        output_codec = cv2.VideoWriter_fourcc('m','p','4','v') 
        output_fps = 30.0
        height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
        output_size = (width,height)  

        out = cv2.VideoWriter(output_path, output_codec, output_fps, output_size)

        while vidcap.isOpened():
            ret, frame = vidcap.read()
            if not ret:
                break

            # Write the processed frame to the output video
            # frame = cv2.flip(frame,-1)
            frame=self.process(frame)
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        out.release()
        vidcap.release()
        cv2.destroyAllWindows()
    #
    def process(self,img):
        self.getScale(img)
        image=cv2.imread(img) if self.mode !='video' else None
        bbox=self.get_predictions(img)
        depth=self.get_depth_mono(img)
        if bbox:
            for i in bbox:
                xyxy=list(i.values())[-1]
                className=list(i.keys())[-1]
                centerBottomX=int(((xyxy[2]+xyxy[0])/2)*self.scaleX)
                centerBottomY=int(xyxy[3]*self.scaleY)
                
                #
                value=round(depth[centerBottomY,centerBottomX],2)
                
                plotImg=self.plot_bbox(image if self.mode !='video' else img,xyxy,className,value)
            #
            if self.mode !='video':
                cv2.imwrite(os.path.join(self.output,os.path.basename(img)),plotImg) 
            else:
                return plotImg
        else:
            if self.mode !='video':
                cv2.imwrite(os.path.join(self.output,os.path.basename(img)),cv2.imread(img)) 
            else:
                return img
    #
    def prediction(self):
        self.mode=self.checkPath()
        self.createOutputPath()
        if self.mode=='singleImg':
            self.process(self.imgPath)
        elif self.mode=='dir':
            paths = glob.glob(os.path.join(self.imgPath, '*.jpg'))\
                    +glob.glob(os.path.join(self.imgPath, '*.jpeg'))\
                    +glob.glob(os.path.join(self.imgPath, '*.png'))
            for _img in paths:
                self.process(_img)
        elif self.mode=='video':
            self.read_video(self.imgPath)


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument( "--input_path",type=str, required=True,
	                        help="input path which can be either single image or directory or movie")
    ap.add_argument( "--segmentation_data", type=str, default='mapillary',
                                help="trained data")
    ap.add_argument( "--output_path",type=str,
                                required=True,help="output path ")
    
    args = vars(ap.parse_args())
    
    Distance_estimation(args['input_path'],args['output_path'],
                    model='mono',dataset=args['segmentation_data']).prediction()
    print('-> Done')