import numpy as np
import h5py
import glob
from scipy.spatial.transform import Rotation as R
def load_7scenes_data(scene):

    print('=>>>>>>>>>>>>>>>>>>>>>>Loading 7Scenes dataset', scene)
    print('=>>>>>>>>>>>>>>>>>>>>>>Loading testing data')
    dset_test_data_path = '/home/dennis/indoor_localization/data/7Scenes/' + scene + '/sequences/01/'
    dset_test_names = glob.glob(dset_test_data_path + 'image_*.png')
    dset_test_names.sort()
    dset_test_label_path = '/home/dennis/indoor_localization/data/7Scenes/' + scene + '/01.txt'
    data = np.loadtxt(dset_test_label_path, usecols=range(1, 8))
    dset_test_num = data.shape[0]
    position = data[:, 0:3]
    orientation = np.zeros((dset_test_num, 3, 3))
    for i in range(dset_test_num):
        r = R.from_quat(data[i, 3:])
        orientation[i, :, :] = r.as_matrix().T
    test = {}
    test['names'] = dset_test_names
    test['position'] = position
    test['orientation'] = orientation
    print('=>>>>>>>>>>>>>>>>>>>>>>Loading training data')
    dset_train_data_path = '/home/dennis/indoor_localization/data/7Scenes/' + scene + '/sequences/00/'
    dset_train_names = glob.glob(dset_train_data_path + 'image_*.png')
    dset_train_names.sort()
    dset_train_label_path = '/home/dennis/indoor_localization/data/7Scenes/' + scene + '/00.txt'
    data = np.loadtxt(dset_train_label_path, usecols=range(1, 8))
    dset_train_num = data.shape[0]
    position = data[:, 0:3]
    orientation = np.zeros((dset_train_num, 3, 3))
    for i in range(dset_train_num):
        r = R.from_quat(data[i, 3:])
        orientation[i, :, :] = r.as_matrix().T
    train = {}
    train['names'] = dset_train_names
    train['position'] = position
    train['orientation'] = orientation

    dset = {}
    dset['train'] = train
    dset['test'] = test

    print('=>>>>>>>>>>>>>>>>>>>>>>Loading 7Scenes descriptor result')
    filename = "/home/dennis/indoor_localization/data/7Scenes/chess/posenet_training_output/cdf/chess_train__siamese_FXPAL_output_1.h5py"
    with h5py.File(filename, "r") as f:
        file_list = list(f.keys())
        posenet_x_predicted = np.array(f[file_list[1]])
    print('=>>>>>>>>>>>>>>>>>>>>>>Loading 7Scenes data finished')
    return dset, posenet_x_predicted
if __name__ == '__main__':
    subdataset = 'chess'
    print('hello world from data loader!')
    sevenscenes, predicted = load_7scenes_data(subdataset)

