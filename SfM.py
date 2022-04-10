import numpy as np
import matplotlib.pyplot as plt
from utils.cameraParams import generateIntrinsics
from utils.dataloader_new import load_TUM_data
from scripts.forward_process import process_7scene_SIFT
from scripts.optimization_GL import optimizationLS

def SfM(dataset='TUM', subdataset='1_desk2', plot_on=False):
    # print("hello world from SfM!")
    ### Load dataset
    if dataset == 'TUM':
        data_dict, posenet_x_predicted = load_TUM_data(subdataset)
    gap = 2
    num_images = 7

    ### parameters dictionary
    params = {}
    params['bm']        = 0.1
    params['sigma']     = 0.2
    params['alpha_m']   = 3
    params['max_range'] = 100

    ### results dictionary
    results = dict()
    results['orient']          = list()
    results['pose']            = list()
    results['orient_error']    = list()
    results['pose_error']      = list()
    results['reproject_error'] = list()

    index_list = list()
    # for i in range(0, min(posenet_x_predicted.shape[0], len(data_dict['train_images'])), 100):
    for i in range(0, 100, 10):
        idx = int(posenet_x_predicted[i] - 1)
        camParams = generateIntrinsics()

        ### Forward process for retrieves 3D points
        orientation, robotpose, pts2D, pts3D, K = process_7scene_SIFT(data_dict, i, idx,
                                                                      camParams, params,
                                                                      num_images=num_images, gap=gap)

        ### Need enough of 3D points for backward intersection
        if pts3D.shape[0] < 3:
            continue

        ### Backward intersection and optimization
        ### the output estimation is a tuple contains (orientation, robotpose, reproerror2, angle_var, position_var)
        estimation = optimizationLS(orientation, robotpose, pts2D, pts3D, pts3D.shape[0], K)

        if estimation[0] is None:
            continue
        else:
            index_list.append(i)
            for i, key in enumerate(results):
                results[key].append(estimation[i])

    ### plot the final results
    if plot_on:
        xyz_gt  = np.array([data_dict['test_position'][i] for i in index_list])
        xyz_est = np.array(results['pose'])
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(xyz_gt[:, 0], xyz_gt[:, 1], xyz_gt[:, 2], label='Ground Truth')
        ax.scatter3D(xyz_est[:, 0], xyz_est[:, 1], xyz_est[:, 2], label='Estimation')
        ax.set_zlim3d(0, 2)
        ax.set_xlim3d(-4, 4)
        ax.set_ylim3d(-4, 4)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    SfM(dataset='TUM', subdataset='1_desk2', plot_on=True)
