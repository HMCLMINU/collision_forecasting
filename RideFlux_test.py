import os,random
import tensorflow as tf
import cv2
import numpy as np
import json
import shutil
from collections import deque
from model_architecture import build_tools
from utils import data_tools
from config import *
from cv_bridge import CvBridge
import rospy

# tf.reset_default_graph()
tf.compat.v1.reset_default_graph()
# tf.set_random_seed(0)
tf.random.set_seed(0)
random.seed(0)
np.random.seed(0)


fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# out = cv2.VideoWriter('inferencee.mp4',fourcc, 12.0, (width*4,height*4))

def inference(network,video_file):
    print("hgcahjbc")
    image_seq = deque([],8) # 양방향 큐
    counter = 0 
    stat = 'safe'

    if os.path.isfile(video_file):
        cap = cv2.VideoCapture(video_file) # video capture object
        width_org  = int(cap.get(3))  # float `width`
        height_org = int(cap.get(4))  # float `height`
        out = cv2.VideoWriter(os.path.join(base_folder,'files','RideFlux.mp4'),fourcc, 12.0, (width_org, height_org), True)

        print("Success generation of the vedio capture object")
    else:
        print("No Video file or Wrong directory")
        
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


if  __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    model_tools = build_tools()
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        network = model_tools.create_network(model_name)

        if mode == 'train':
            train_generator = data_tools(train_folder,'train')
            valid_generator = data_tools(valid_folder,'valid')
            _trainer(network,train_generator.batch_dispatch(),valid_generator.batch_dispatch())

        if mode == 'test':
            network.load_weights(os.path.join(model_save_folder,'model_weights_500.ckpt'))
            inference(network,os.path.join(base_folder,'files','test.mp4'))
            # inference(network,os.path.join(base_folder,'files','inference_video.mp4'))

            #testing from batch
            # test_generator = data_tools(valid_folder,'test')
            # # test_generator = get_valid_data(test_folder)
            # for img_seq,labels in test_generator.batch_dispatch():
            #     r = network.predict(img_seq)
            #     print ('accuracy',np.count_nonzero(np.argmax(r,1)==np.argmax(labels,1))/8)