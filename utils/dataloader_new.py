import numpy as np
import h5py
import glob
import os
from scipy.spatial.transform import Rotation as R
def load_TUM_data(scene):
    """
    data loader for TUM data-set
    :param scene: the sub-name of the TUM data-set, for example "1_disk2".
    :return: data_dict: a dictionary containing "train_images", "train_position",\
            "train_orientation", "test_images", "test_position","test_orientation".
            posenet_x_predicted: the index of the most similar images in\
            the train data-set for each given test image.
    """
    print('=>>>>>>>>>>>>>>>>>>>>>>Loading TUM dataset', scene)
    path = '../data/TUM/' + scene
    if not os.path.exists(path):
        path = './data/TUM/' + scene
    data_dict = {}

    print('=>>>>>>>>>>>>>>>>>>>>>>Loading training data')
    dset_train_data_path = path + '/sequences/00/'
    dset_train_images = glob.glob(dset_train_data_path + 'image_*.png')
    dset_train_images.sort()
    dset_train_label_path = path + '/00.txt'
    data = np.loadtxt(dset_train_label_path, usecols=range(1, 8))
    dset_train_num = data.shape[0]
    position = data[:, 0:3]
    orientation = np.zeros((dset_train_num, 3, 3))
    for i in range(dset_train_num):
        r = R.from_quat(data[i, 3:])
        orientation[i, :, :] = r.as_matrix()

    data_dict['train_images'] = dset_train_images
    data_dict['train_position'] = position
    data_dict['train_orientation'] = orientation

    print('=>>>>>>>>>>>>>>>>>>>>>>Loading testing data')
    dset_test_data_path = path + '/sequences/01/'
    dset_test_images = glob.glob(dset_test_data_path + 'image_*.png')
    dset_test_images.sort()
    dset_test_label_path = path + '/01.txt'
    data = np.loadtxt(dset_test_label_path, usecols=range(1, 8))
    dset_test_num = data.shape[0]
    position = data[:, 0:3]
    orientation = np.zeros((dset_test_num, 3, 3))
    for i in range(dset_test_num):
        r = R.from_quat(data[i, 3:])
        orientation[i, :, :] = r.as_matrix()

    data_dict['test_images'] = dset_test_images
    data_dict['test_position'] = position
    data_dict['test_orientation'] = orientation

    print('=>>>>>>>>>>>>>>>>>>>>>>Loading descriptor result')
    filename = path + "/posenet_training_output/cdf/chess_train__siamese_FXPAL_output_1.h5py"
    with h5py.File(filename, "r") as f:
        file_list = list(f.keys())
        posenet_x_predicted = np.array(f[file_list[1]]).reshape(-1)
    print('=>>>>>>>>>>>>>>>>>>>>>>Loading dataset finished')
    return data_dict, posenet_x_predicted
if __name__ == '__main__':
    subdataset = '1_desk2'
    print('hello world from data loader!')
    dset, posenet_x_predicted = load_TUM_data(subdataset)

