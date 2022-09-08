#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hickle as hkl
import numpy as np
import random 
import json
np.random.seed(9 ** 10)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from config import * 

from sys import stdout
from model_architecture import build_tools
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

import argparse
import math
import cv2 as cv
import os, glob
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam

tf.random.set_seed(0)

_data = os.listdir(data_save_path)
random.shuffle(_data)
num_of_train = int(len(_data) * 0.7) 
num_of_val = int(len(_data) * 0.2) 
num_of_test = int(len(_data) - num_of_train - num_of_val) 
it = int(10/10)

tensorboard_save_folder = '/home/hmcl/Vehicle_Collision_Prediction_Using_convLSTM/project_folder/tensorboard'
checkpoint_path = '/home/hmcl/Vehicle_Collision_Prediction_Using_convLSTM/project_folder/checkpoint/'
model_save_folder = '/home/hmcl/Vehicle_Collision_Prediction_Using_convLSTM/project_folder/model_save/'

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=True,
                                                 save_weights_only=False,period=100)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_save_folder, histogram_freq=0, write_graph=True,
                                                      write_images=False)
early_stopping = tf.keras.callbacks.EarlyStopping()
        
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

def batch_dispatch(mode):
    train_data = _data[:num_of_train]
    val_data = _data[num_of_train:num_of_train+num_of_val]
    random.shuffle(train_data)
    random.shuffle(val_data)
    if mode == "train":
        counter = 0
        while counter<=num_of_train:
            image_seqs = np.empty((0,time,height,width,color_channels))
            labels = np.empty((0, 2))
            for i in range(it):          
                np_data = np.load(os.path.join(data_save_path, train_data[counter]), allow_pickle=True)
                label = np_data['arr_1']
                for i in range(len(label)):
                    if label[i] > 0:
                       label[i] = 1
                encoding = to_categorical(label, 2)
                # encoding = np.eye(2)[np_data['arr_1']]
                if len(np_data['arr_0']) == 0:
                    continue

                for j in range(np_data['arr_0'].shape[0] - time):
                    t = np_data['arr_0'][j:j+time, :, :].reshape(1, time,height,width,color_channels)
                    image_seqs = np.vstack((image_seqs, t/255))
                    labels = np.vstack((labels, encoding[j+time].reshape(1,2)))

                counter += 1
                if counter>=num_of_train:
                    counter = 0
                    random.shuffle(train_data)

            yield image_seqs, labels

    elif mode == "val":
        counter = 0
        while counter<=num_of_val:
            val_image_seqs = np.empty((0,time,height,width,color_channels))
            val_labels = np.empty((0, 2))
            for i in range(it):
                np_data = np.load(os.path.join(data_save_path, train_data[counter]), allow_pickle=True)
                label = np_data['arr_1']
                for i in range(len(label)):
                    if label[i] > 0:
                       label[i] = 1
                encoding = to_categorical(label, 2)
                # encoding = np.eye(2)[np_data['arr_1']]
                if len(np_data['arr_0']) == 0:
                    continue

                for j in range(np_data['arr_0'].shape[0] - time):
                    t = np_data['arr_0'][j:j+time, :, :].reshape(1, time,height,width,color_channels)
                    val_image_seqs = np.vstack((val_image_seqs, t/255))
                    val_labels = np.vstack((val_labels, encoding[j+time].reshape(1,2)))
                            
                counter += 1
                if counter>=num_of_val:
                    counter = 0
                    random.shuffle(val_data)

            yield val_image_seqs, val_labels

def test_batch(): 
    test_data = _data[num_of_train + num_of_val:]
    random.shuffle(test_data)
    counter = 0
    image_seqs = np.empty((0,time,height,width,color_channels))
    labels = np.empty((0, 2))

    while counter<num_of_test:
        # print("1")    
        np_data = np.load(os.path.join(data_save_path, test_data[counter]), allow_pickle=True)
        encoding = np.eye(2)[np_data['arr_1']]
        if len(np_data['arr_0']) == 0:
            continue

        for j in range(np_data['arr_0'].shape[0] - time):
            # print("2")
            t = np_data['arr_0'][j:j+time, :, :].reshape(1, time,height,width,color_channels)
            image_seqs = np.vstack((image_seqs, t/255.))
            labels = np.vstack((labels, encoding[j+time].reshape(1,2)))
            # print(image_seqs.shape)
        counter += 1
    
        # if counter>=num_of_test:
        #     counter = 0
        #     random.shuffle(test_data)
    print(image_seqs.shape)
    return image_seqs, labels
    
def test_batch_gen(): 
    test_data = _data[num_of_train + num_of_val:]
    random.shuffle(test_data)
    counter = 0
    while counter<num_of_test:

        image_seqs = np.empty((0,time,height,width,color_channels))
        labels = np.empty((0, 2))
        np_data = np.load(os.path.join(data_save_path, test_data[counter]), allow_pickle=True)
        encoding = np.eye(2)[np_data['arr_1']]
        if len(np_data['arr_0']) == 0:
            continue

        for j in range(np_data['arr_0'].shape[0] - time):
            # print("2")
            t = np_data['arr_0'][j:j+time, :, :].reshape(1, time,height,width,color_channels)
            image_seqs = np.vstack((image_seqs, t/255.))
            labels = np.vstack((labels, encoding[j+time].reshape(1,2)))
            # print(image_seqs.shape)
        counter += 1
    
        # if counter>=num_of_test:
        #     counter = 0
        #     random.shuffle(test_data)
        yield image_seqs, labels

def _trainer(network):
    network.compile(optimizer = 'adam', loss= 'binary_crossentropy',metrics = ['accuracy', 'precision'])
    network.save_weights(checkpoint_path.format(epoch=0))
    history = network.fit(batch_dispatch(mode = "train"),
                    epochs=epochs,
                    steps_per_epoch = num_of_train*6//batch_size,
                    batch_size= batch_size,
                    validation_data = batch_dispatch(mode = "val"),
                    validation_steps=1,
                    callbacks=[cp_callback, tensorboard_callback])

    network.save(model_save_folder + 'collision_v1.h5')

def predict():
    # # # Load Model
    network = tf.keras.models.load_model(model_save_folder + 'collision_v1.h5')
    test_x, test_y = test_batch()
    test_x = test_x.astype('float32')
    test_y = test_y.astype('float32')
    # Prediction ...
    predicted_ = np.empty((0, 2))
    y_pred = []
    y_true = []
    sim = 0
    n = 10
    while (sim < test_x.shape[0]):
        predicted_labels = network.predict(test_x[sim].reshape(1, time, height, width, color_channels)) # (1, 4, h, w, c)        
        stat = np.argmax(predicted_labels)
        y_stat = np.argmax(test_y[sim])
        predicted_ = np.vstack((predicted_, predicted_labels))
        y_pred.append(stat)
        y_true.append(y_stat)
        sim += 1 
    
    cm = tf.math.confusion_matrix(
        y_true,
        y_pred,
        num_classes=2,
        weights=None,
        dtype=tf.dtypes.int32,
        name=None
    )

    labels = ["Safe", "Collision"]
    cm = confusion_matrix(y_true, y_pred)
    # tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    # disp.plot(cmap=plt.cm.Blues)
    plt.show()

    # network.evaluate_generator(test_batch_gen())

    # plt.imshow(predicted_frames[0]
    # plot original label
    # for i in range(int(test_y.shape[0])):
    #     if i % 7 == 6:
    #         img = test_y[i] # all image at last batch
    #         test_image_seqs = np.vstack((test_image_seqs, img))
    #     else:
    #         img = test_y[i][0].reshape(1, height, width, color_channels) # first image at each batch
    #         test_image_seqs = np.vstack((test_image_seqs, img))

    # for sim in range(int(predicted_.shape[0]/n)):
    #     plt.figure(figsize=(n*2, 5))
    #     plt.rcParams["figure.figsize"] = (5, 5)
    #     start = sim*10
    #     for i in range(1, n + 1):
    #         # Display original
    #         plt.suptitle('Batch' + str(sim) + 'True label vs Predicted label',fontweight="bold")
    #         ax = plt.subplot(2, n, i)
    #         ax.set_title("Time at :" + str(i+4))
    #         plt.imshow(test_image_seqs[start+i-1])
    #         plt.gray()
    #         ax.get_xaxis().set_visible(False)
    #         ax.get_yaxis().set_visible(False)

    #         # Display reconstruction
    #         ax = plt.subplot(2, n, i + n)
    #         ax.set_title("Time at :" + str(i+4))
    #         plt.imshow(predicted_[start+i-1])
    #         plt.gray()
    #         ax.get_xaxis().set_visible(False)
    #         ax.get_yaxis().set_visible(False)
    #     # plt.show()
    #     my_file = 'batch_'+str(sim) + 'th_result.png'
    #     plt.savefig(os.path.join(img_save_folder, my_file))
    
    # print("~~~~~~~~PREDICTION DONE~~~~~~~~~~~")


if  __name__ == "__main__":
    # batch_dispatch("train")
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
    # Use Multi GPU
    # strategy = tf.distribute.MirroredStrategy()
    communication_options = tf.distribute.experimental.CommunicationOptions(
    implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)
    strategy = tf.distribute.MultiWorkerMirroredStrategy(communication_options=communication_options)
    # use gpu
    with strategy.scope():
        network = model_tools.create_network(model_name)
        mode = "train"

        if mode == 'train':
            _trainer(network)
        else:
            predict()

        # if mode == 'test':
        #     network.load_weights(os.path.join(model_save_folder,'model_weights_048.ckpt'))
        #     inference(network,os.path.join(base_folder,'files','test.mp4'))
            # inference(network,os.path.join(base_folder,'files','inference_video.mp4'))

            #testing from batch
            # test_generator = data_tools(valid_folder,'test')
            # # test_generator = get_valid_data(test_folder)
            # for img_seq,labels in test_generator.batch_dispatch():
            #     r = network.predict(img_seq)
            #     print ('accuracy',np.count_nonzero(np.argmax(r,1)==np.argmax(labels,1))/8)