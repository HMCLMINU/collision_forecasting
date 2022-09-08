import os

epochs = 3
time = 8
n_classes = 2
width,height,color_channels = 320, 180, 1
number_of_hiddenunits = 32
batch_size = 6

model_name = 'inception'
mode = 'train'

#config
data_save_path = '/home/hmcl/Vehicle_Collision_Prediction_Using_convLSTM/project_folder/dataset/'
# base_folder = os.path.abspath(os.curdir)
# data_path = os.path.join(base_folder,'datasets_h5')
# train_folder = os.path.join(data_path,'train_set')
# test_folder = os.path.join(data_path,'test_set')
# valid_folder = os.path.join(data_path,'valid_set')
# model_save_folder = os.path.join(base_folder,'files',model_name,'model_folder')
# tensorboard_save_folder = os.path.join(base_folder,'files',model_name,'tensorboard_folder')
# checkpoint_path = os.path.join(model_save_folder,"model_weights_{epoch:03d}.ckpt")
