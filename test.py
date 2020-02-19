##处理视频相关
import cv2
import os
import numpy
import dlib
import sys
# sys.path.append("/home/tangxi.zq/code+data/detecting_deepfake/")
# Networks
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet
# from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
# from keras.applications.nasnet import NASNetLarge, NASNetMobile
from keras.preprocessing.image import ImageDataGenerator
# from SimpleModel import *
# Layers
from keras.layers import Dense, Activation, Flatten, Dropout,Input,Conv2D,BatchNormalization
from keras import backend as K
import tensorflow as tf
import keras
#config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 1} ) 
#sess = tf.Session(config=config)
# from util import save_class_list,get_num_files
# from util import *
# from config import *
from utils import *
# import util
import utils
# from double_stream_model import *
from keras.applications.mobilenet import preprocess_input
preprocessing_function = preprocess_input
HEIGHT = 224
WIDTH  = 224
import pandas as pd
from PIL import Image
def getvideos(path):
    return [os.path.join(path,each) for each in os.listdir(path) if each.endswith(".mp4")]

def predict_single_image(model,face):
    image = Image.fromarray(cv2.cvtColor(face,cv2.COLOR_BGR2RGB))
    image = np.array(image)
    image = np.float32(cv2.resize(image,(HEIGHT, WIDTH)))
    #image_noise = preprocessing_function_noise(image)
    image_median = preprocessing_function(image)    
    #print(image.shape)
    #image_noise = image_noise.reshape(1, img_rows, img_cols, 3)
    image_median = image_median.reshape(1,HEIGHT,WIDTH,3)
    #double_input = [image_noise,image_median]
    #print(type(double_input))
    out = model.predict(image_median)

    confidence = out[0]
    class_prediction = list(out[0]).index(max(out[0]))
    #class_name = class_list[class_prediction]
    #print("Predicted class = ", class_name)
    print("Confidence = ", confidence)
    #print("Run time = ", run_time)
    return confidence[0]

def main():
    #model = double_stream_model_12()
    #model.load_weights("checkpoints/"+"doublestream"+"_model_weights.h5")
    from keras.applications.mobilenet import preprocess_input
    base_model = MobileNet(weights='imagenet',include_top=False,input_shape=(HEIGHT,WIDTH,3))
    preprocessing_function = preprocess_input
    class_list_file = "checkpoints/MobileNet_class_list.txt"
    class_list = utils.load_class_list(class_list_file)
    model = utils.build_finetune_model(base_model,dropout=1e-3,num_classes=len(class_list),fc_layers=[1024,1024])
    model.load_weights("checkpoints/MobileNet_model_weights.h5")
    path = "/Users/tangxi/Downloads/Compressed/deepfake_baselinev1_1/test_videos"
    videos = getvideos(path)
    predictor = dlib.shape_predictor("/Users/tangxi/Downloads/Compressed/deepfake_baselinev1_1/shape_predictor_68_face_landmarks.dat")
    detector = dlib.get_frontal_face_detector()
    predictions=[]
    for each in videos:
        p_each_video=0.0
        vc = cv2.VideoCapture(each)
        rval, frame = vc.read()
        # 获取视频fps
        fps = vc.get(cv2.CAP_PROP_FPS)
        # 获取视频总帧数
        frame_all = vc.get(cv2.CAP_PROP_FRAME_COUNT)
        print("[INFO] 视频FPS: {}".format(fps))
        print("[INFO] 视频总帧数: {}".format(frame_all))
        print("[INFO] 视频时长: {}s".format(frame_all/fps))
        fake_count=0
        total_count=0
        while True:
            ret,frame = vc.read()
            if ret is False:
                break
            total_count+=1
            img = frame.copy()
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            dets = detector(gray, 0)
            if len(dets)!=1:continue
            d = dets[0]
            x1 = d.top() if d.top() > 0 else 0
            y1 = d.bottom() if d.bottom() > 0 else 0
            x2 = d.left() if d.left() > 0 else 0
            y2 = d.right() if d.right() > 0 else 0
            face = img[x1-25:y1+10,x2-8:y2+8]
            print(face.shape)
            p_fake = predict_single_image(model,face)
            if p_fake>0.5:
                fake_count+=1
        if fake_count/float(total_count)>0.5:
            p_each_video = fake_count/float(total_count)
        else:
            p_each_video = 0.5
    predictions.append(p_each_video)
    submission_df = pd.DataFrame({"filename": test_videos, "label": predictions})
    submission_df.to_csv("submission.csv", index=False)    
if __name__=="__main__":
    main()   
        
