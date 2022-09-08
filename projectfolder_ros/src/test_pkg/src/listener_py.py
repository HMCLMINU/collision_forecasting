#!/usr/bin/env python3
#-*- coding:utf-8 -*- 

import rospy
import sys,os,random
import tensorflow as tf
import cv2
import numpy as np
import json
import shutil
from collections import deque
from cv_bridge import CvBridge, CvBridgeError
# from test_pkg.msg import test_msg

from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
# from model_architecture import build_tools


tf.compat.v1.reset_default_graph()
tf.random.set_seed(0)
random.seed(0)
np.random.seed(0)

sys.path.append("/home/hmcl/Vehicle_Collision_Prediction_Using_convLSTM")
from model_architecture import build_tools
from config import *

base_folder_ = os.path.abspath(os.curdir)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

class drivabilityPredictor():
    def __init__(self):
        rospy.init_node('drivability_predicter', anonymous=True)
        rospy.Subscriber("/image0", Image, self.videoCallback)
        self.bridge = CvBridge()

        self._frame = None

    def videoCallback(self, msg):
        # Used to convert between ROS and OpenCV images
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)

        # im = np.frombuffer(msg.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        self._frame = cv2.resize(cv_image, (width*3, height*3))
        cv2.imshow("camera", self._frame)

        cv2.waitKey(1)

    def inference(self, network):
        print("hgcahjbc")
        image_seq = deque([],8) # 양방향 큐
        counter = 0 
        stat = 'safe'

        # if os.path.isfile(self._frame):
        cap = cv2.VideoCapture(self._frame) # video capture object
        width_org  = int(cap.get(3))  # float `width`
        height_org = int(cap.get(4))  # float `height`
        out = cv2.VideoWriter(os.path.join(base_folder_,'files','RideFlux.mp4'),fourcc, 12.0, (width_org, height_org), True)

            # print("Success generation of the vedio capture object")
        # else:
            # print("No Video file or Wrong directory")
            
        while (cap.isOpened()):
            
            ret, frame = cap.read()
            if ret:
                _frame = cv2.resize(frame,(width, height))
                image_seq.append(_frame)
                if counter%(batch_size/8) == 0:
                    if len(image_seq)==8:
                        # np.reshape(변경할 배열, 차원)
                        np_image_seqs = np.reshape(np.array(image_seq)/255,(1,time,height,width,color_channels))
                        r = network.predict(np_image_seqs) # 두 클래스 확률 행렬
                        stat = ['safe', 'collision'][np.argmax(r,1)[0]] # 8*2 행렬에서 행 축방향으로 max index 
                
                cv2.putText(frame, stat, (230,230), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0),3)
                cv2.imshow('frame', frame)
                cv2.waitKey(12)
                out.write(frame)
                counter+=1
            else:
                cap.release()
                out.release()
                cv2.destroyAllWindows()

    def main(self):
        # gpus = tf.config.experimental.list_physical_devices('GPU')
        # if gpus:
        #     try:
        #         # Currently, memory growth needs to be the same across GPUs
        #         for gpu in gpus:
        #             tf.config.experimental.set_memory_growth(gpu, True)
        #         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        #     except RuntimeError as e:
        #         # Memory growth must be set before GPUs have been initialized
        #         print(e)
        # model_tools = build_tools()
        # strategy = tf.distribute.MirroredStrategy()
        # with strategy.scope():
            # network = model_tools.create_network(model_name)
            # network.load_weights(os.path.join(model_save_folder,'model_weights_048.ckpt'))
            # self.inference(network)
            
        rospy.spin()

if __name__ == '__main__':
    try:
        dap = drivabilityPredictor()
        dap.main()
    except KeyboardInterrupt:
        pass
    finally:
        print('Finish')