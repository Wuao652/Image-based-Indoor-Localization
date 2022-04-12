'''
reads the vector of size 2048 which is given by the the posenet layers of posnet model and combines the training,validation
and testing files and saves them to h5py.

As the dataset is a vector it does not require downsampling and cropping routine

'''

import h5py
import numpy as np
from tqdm import tqdm
import os
import random
import sys
import argparse
import time
import tensorflow as tf
# import geometric_transformations as gt
# from scipy.spatial import distance
import math
import shutil
from scipy.spatial.transform import Rotation as R
#####################
## read images
# image = cv2.imread(data.images[i])
# image = cv2.resize(image, (network_input_image_width, network_input_image_height), cv2.INTER_CUBIC)

## read .txt file
def parse_data_folder_index(train_txt, test_txt):
    '''Get training and testing data folder index.'''
    train_folder_index = []
    test_folder_index  = []

    f = open(train_txt, 'r')
    for line in f:
        seq_index  = line.split()
        train_folder_index.append(int(seq_index[0][-1]))
    f.close()
    f = open(test_txt, 'r')
    for line in f:
        seq_index  = line.split()
        test_folder_index.append(int(seq_index[0][-1]))
    f.close()

    return train_folder_index, test_folder_index

def write_img_newfolder(dataset_dir, train_folder_index, test_folder_index, training_dir, testing_dir):
    #   I. Write the training image
    if os.path.exists(training_dir):
        shutil.rmtree(training_dir)
    os.makedirs(training_dir)

    ind = -1

    for i in range(len(train_folder_index)):
        img_dir =  dataset_dir+ '/seq-' + ('%02d/'%train_folder_index[i])
        j = 0
        image_j = img_dir+'frame-'+('%06d'%j)+'.color.png'

        while os.path.exists(image_j):
            ind += 1
            image_new = training_dir + 'image_' + ('%010d.png' % ind)
            tmp = 'cp ' + image_j + ' ' + image_new
            os.popen(tmp)
            j = j + 1		#	j = j + 1 change this back
            image_j = img_dir + 'frame-' + ('%06d' % j) + '.color.png'

    #   II. Write the testing image
    if os.path.exists(testing_dir):
        shutil.rmtree(testing_dir)
    os.makedirs(testing_dir)

    ind = -1

    for i in range(len(test_folder_index)):
        img_dir = dataset_dir + '/seq-' + ('%02d/' % test_folder_index[i])
        j = 0
        image_j = img_dir + 'frame-' + ('%06d' % j) + '.color.png'
        while os.path.exists(image_j):
            ind = ind + 1
            image_new = testing_dir + 'image_' + ('%010d.png' % ind)
            tmp = 'cp ' + image_j + ' ' + image_new
            os.popen(tmp)
            j = j + 1
            image_j = img_dir + 'frame-' + ('%06d' % j) + '.color.png'
    return 0


def write_pose_newfile_rotation(text_file_train, text_file_test, train_folder_index, test_folder_index):
    #   I. Write the pose info
    if os.path.exists(text_file_train):
        os.remove(text_file_train)
    file = open(text_file_train, 'w')
    if os.path.exists(text_file_test):
        os.remove(text_file_test)

    #   Write the training trajectory
    ind = -1
    for i in range(len(train_folder_index)):
        img_dir = dataset_dir + '/seq-' + ('%02d/' % train_folder_index[i])
        j = 0
        pose_j = img_dir + 'frame-' + ('%06d' % j) + '.pose.txt'
        while os.path.exists(pose_j):
            ind = ind + 1
            p = np.empty((16,), dtype=np.float32)
            k = 0

            f = open(pose_j, 'r')
            for line in f:
                p[4*k], p[4*k+1], p[4*k+2], p[4*k+3] = line.split()
                k = k + 1
            f.close()

            tmp = 'image_' + ('%010d' % ind) + '.png' + ' ' + str(p[3]) + ' ' + str(p[7]) + ' ' + str(p[11]) + ' ' + str(p[0])\
                  + ' ' + str(p[1]) + ' ' + str(p[2]) + ' ' + str(p[4]) + ' ' + str(p[5]) + ' ' + str(p[6]) + ' ' + str(p[8])\
                  + ' ' + str(p[9]) + ' ' + str(p[10]) + '\n'
            file.write(tmp)
            j = j + 1
            pose_j = img_dir + 'frame-' + ('%06d' % j) + '.pose.txt'

    file.close()

    file = open(text_file_test, 'w')
    #  Write the testing trajectory
    ind = -1

    for i in range(len(test_folder_index)):
        img_dir = dataset_dir + '/seq-' + ('%02d/' % test_folder_index[i])
        j = 0
        pose_j = img_dir + 'frame-' + ('%06d' % j) + '.pose.txt'

        while os.path.exists(pose_j):
            ind = ind + 1
            p = np.empty((16,), dtype=np.float32)
            k = 0

            f = open(pose_j, 'r')
            for line in f:
                p[4 * k], p[4 * k + 1], p[4 * k + 2], p[4*k+3] = line.split()
                k = k + 1
            f.close()

            tmp = 'image_' + ('%010d' % ind) + '.png' + ' ' + str(p[3]) + ' ' + str(p[7]) + ' ' + str(
                p[11]) + ' ' + str(p[0]) \
                  + ' ' + str(p[1]) + ' ' + str(p[2]) + ' ' + str(p[4]) + ' ' + str(p[5]) + ' ' + str(p[6]) + ' ' + str(
                p[8]) \
                  + ' ' + str(p[9]) + ' ' + str(p[10]) + '\n'
            file.write(tmp)
            j = j + 1
            pose_j = img_dir + 'frame-' + ('%06d' % j) + '.pose.txt'

    file.close()


def write_pose_newfile_quaternion(text_file_train, text_file_test, train_folder_index, test_folder_index):
    #   I. Write the pose info
    if os.path.exists(text_file_train):
        os.remove(text_file_train)
    file = open(text_file_train, 'w')
    if os.path.exists(text_file_test):
        os.remove(text_file_test)

    #   Write the training trajectory
    ind = -1
    for i in range(len(train_folder_index)):
        img_dir = dataset_dir + '/seq-' + ('%02d/' % train_folder_index[i])
        j = 0
        pose_j = img_dir + 'frame-' + ('%06d' % j) + '.pose.txt'
        while os.path.exists(pose_j):
            ind = ind + 1
            p = np.empty((16,), dtype=np.float32)
            k = 0

            f = open(pose_j, 'r')
            for line in f:
                p[4*k], p[4*k+1], p[4*k+2], p[4*k+3] = line.split()
                k = k + 1
            f.close()

            rotation = R.from_matrix([[p[0], p[4], p[8]],
                                      [p[1], p[5], p[9]],
                                      [p[2], p[6], p[10]]])
            quaternion = rotation.as_quat()

            tmp = 'image_' + ('%010d' % ind) + '.png' + ' ' + str(p[3]) + ' ' + str(p[7]) + ' ' + str(p[11]) + ' ' + str(quaternion[0])\
                  + ' ' + str(quaternion[1]) + ' ' + str(quaternion[2]) + ' ' + str(quaternion[3]) + '\n'

            file.write(tmp)
            j = j + 1		#	j = j + 1 change this back
            pose_j = img_dir + 'frame-' + ('%06d' % j) + '.pose.txt'

    file.close()

    file = open(text_file_test, 'w')
     #  Write the testing trajectory
    ind = -1
    for i in range(len(test_folder_index)):
        img_dir = dataset_dir + '/seq-' + ('%02d/' % test_folder_index[i])
        j = 0
        pose_j = img_dir + 'frame-' + ('%06d' % j) + '.pose.txt'
        while os.path.exists(pose_j):
            ind = ind + 1
            p = np.empty((16,), dtype=np.float32)
            k = 0

            f = open(pose_j, 'r')
            for line in f:
                p[4 * k], p[4 * k + 1], p[4 * k + 2], p[4*k+3] = line.split()
                k = k + 1
            f.close()

            rotation = R.from_matrix([[p[0], p[4], p[8]],
                                      [p[1], p[5], p[9]],
                                      [p[2], p[6], p[10]]])
            quaternion = rotation.as_quat()

            tmp = 'image_' + ('%010d' % ind) + '.png' + ' ' + str(p[3]) + ' ' + str(p[7]) + ' ' + str(
                p[11]) + ' ' + str(quaternion[0]) \
                  + ' ' + str(quaternion[1]) + ' ' + str(quaternion[2]) + ' ' + str(quaternion[3]) + '\n'

            file.write(tmp)
            j = j + 1
            pose_j = img_dir + 'frame-' + ('%06d' % j) + '.pose.txt'

    file.close()


if __name__ == '__main__':

    dataset_dir = '/media/jingwei/Lenovo/dataset/FXPAL/7scenes/chess'
    train_txt = dataset_dir + '/TrainSplit.txt'
    test_txt  = dataset_dir + '/TestSplit.txt'
    index_train = 0
    index_test  = 1

    #   I. Read training and testing folder index
    train_folder_index, test_folder_index = parse_data_folder_index(train_txt, test_txt)

    #   II. Put training data in 00 and testing data in 01 folder
    training_dir = dataset_dir+ '/sequences/' + ('%02d/'%index_train)
    testing_dir  = dataset_dir + '/sequences/' + ('%02d/' % index_test)
    write_img_newfolder(dataset_dir, train_folder_index, test_folder_index, training_dir, testing_dir)

    #   III. Put trajectory in the folder
    text_file_train = os.path.join(dataset_dir, ('%02d_inrotation.txt' % index_train))
    text_file_test  = os.path.join(dataset_dir, ('%02d_inrotation.txt' % index_test))
    write_pose_newfile_rotation(text_file_train, text_file_test, train_folder_index, test_folder_index)
    text_file_train = os.path.join(dataset_dir, ('%02d.txt' % index_train))
    text_file_test = os.path.join(dataset_dir, ('%02d.txt' % index_test))
    write_pose_newfile_quaternion(text_file_train, text_file_test, train_folder_index, test_folder_index)

    print('All the process is finished')
