import numpy as np
import h5py
import glob
from scipy.spatial.transform import Rotation as R
def load_7scenes_data(scene):
    driverpath = '/home/dennis/indoor_localization/data/7Scenes/';
    print('=>>>>>>>>>>>>>>>>>>>>>>Loading 7Scenes dataset', scene)
    # load('7scenes/cameraParams_kinect_7scenes.mat');
    dset_test_data_path = '/home/dennis/indoor_localization/data/7Scenes/' + scene + '/sequences/01/'
    # test_data_path = '/home/dennis/indoor_localization/data/7Scenes/' + subdataset + '/sequences/01/'
    dset_test_names = glob.glob(dset_test_data_path + 'image_*.png')
    dset_test_names.sort()
    dset_test_label_path = '/home/dennis/indoor_localization/data/7Scenes/' + scene + '/01.txt'
    data = np.loadtxt(dset_test_label_path, usecols=range(1, 8))
    print(data.shape)
    dset_test_num = data.shape[0]
    position = data[:, 0:3]
    orientation = np.zeros((dset_test_num, 3, 3))
    for i in range(dset_test_num):
        r = R.from_quat(data[i, 3:])
        orientation[i, :, :] = r.as_matrix().T

    return None
    # filepath = [driverpath subdataset '/sequences/01/image_'];
    # posepath = [driverpath subdataset '/01.txt'];
    # filepath_history = [driverpath subdataset '/sequences/00/image_'];
    # posepath_history = [driverpath subdataset '/00.txt'];
    #
    # FILENAME_BASE = [driverpath subdataset '/posenet_training_output/cdf/chess_train__siamese_FXPAL_output'];
    #
    # for i = 1: 2
    # filename = [FILENAME_BASE '_' int2str(i - 1) '.h5py'];
    # pose_ID_groudtruth
    # {i} = h5read(filename, '/posenet_x_label');
    # pose_ID_predict
    # {i} = h5read(filename, '/posenet_x_predicted');
    # return X_train, y_train, X_test, y_test



pass
if __name__ == '__main__':
    subdataset = 'chess'
    # load_7scenes_data()
    print('hello world from data loader!')
    test_data_path = '/home/dennis/indoor_localization/data/7Scenes/' + subdataset + '/sequences/01/'
    mylist = glob.glob(test_data_path + 'image_*.png')
    mylist.sort()

    load_7scenes_data(subdataset)