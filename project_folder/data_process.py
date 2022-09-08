import os, glob, numpy as np
import cv2 as cv

data_folder_path = '/home/hmcl/Vehicle_Collision_Prediction_Using_convLSTM/for_project_data'
data_save_path = '/home/hmcl/Vehicle_Collision_Prediction_Using_convLSTM/project_folder/dataset/'

files = glob.glob(data_folder_path)
cnt = 0
print("Start Data Processing...")
for folder in glob.glob(f'{data_folder_path}/*'):
    n_sim = folder[len(data_folder_path + '/'):len(folder)]
    X = []
    Y = []
    Z = []
    # frames_per_sim[cnt].append(int(n_sim))
    # frames_per_sim[cnt].append(len(glob.glob(f'{folder}/*.png')))
    # for num in range(len(glob.glob(f'{folder}/*.png'))):
    #     filename = folder + '/' + str((num+1)*2-1) + '.png'
    #     img = cv.imread(filename, cv.IMREAD_COLOR)
    #     data = np.asarray(img)
    #     X.append(data)
    # 14 장 기준으로 npz 저장
    if len(glob.glob(f'{folder}/*.npz')) == 0:
        pass
    folders = glob.glob(f'{folder}/*.npz')
    folders.sort()
    for num in range(len(folders)):
        # filename = folder + '/' + str((num+1)*2-1) + '.npz'
        filename = folders[num]
        data = np.load(filename, allow_pickle=True)
        x = data['arr_0']
        y = data['arr_1']
        ego_stat = data['arr_2']
        # data = np.asarray(img)
        X.append(x)
        Y.append(y)
        Z.append(ego_stat)
        cnt += 1    
    # if n_sim == 63:
    #     print("here")
    if X:
        np.savez(data_save_path + str(n_sim) + '.npz', X, Y, Z)
    else:
        print("pass empty folder : {}".format(n_sim))
        pass
        



print("END Data Processing...")

            

