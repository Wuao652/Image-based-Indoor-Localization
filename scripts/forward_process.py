import numpy as np
# import scipy.io as sio # need scipy 1.8.0
import cv2
import sys
sys.path.append('.')
from utils.dataloader_new import load_TUM_data
from utils.keyframes_selection import selectimages
from utils.triangulateMultiView import triangulateMultiView
from utils.cameraParams import generateIntrinsics
from utils.vl_func import vl_ubcmatch
from optimization_GL import optimizationLS
import matplotlib.pyplot as plt


def plot_imgs(nImgList):
    """
    :param nImgList:
    :return:
    """
    fig, ax = plt.subplots(2, 4)
    for a in ax.reshape((-1)):
        a.set_axis_off()

    ax[0][0].imshow(cv2.imread(nImgList[0]))
    ax[0][1].imshow(cv2.imread(nImgList[1]))
    ax[0][2].imshow(cv2.imread(nImgList[2]))
    ax[0][3].imshow(cv2.imread(nImgList[3]))
    ax[1][0].imshow(cv2.imread(nImgList[4]))
    ax[1][1].imshow(cv2.imread(nImgList[5]))
    ax[1][2].imshow(cv2.imread(nImgList[6]))
    ax[1][3].imshow(cv2.imread(nImgList[7]))

    plt.show()



def process_7scene_SIFT(data_dict, i, idx,
                        camParams, params,
                        num_images=7, gap=2):
    """
    This function is used to triangulate sparse
    3D map points from the 2D tracking in the training set.
    Firstly, N images are selected from the training set according
    to the predict id from the posenet and gap, num_images.
    Then, sift feature extractor is used to detect the key points
    and the features in the selected images. Vl_UBC algorithm is used
    to match the keypoints from the consecutive two images. A track of
    a 3D point is a list containing the id and its (u, v) projection
    on the training images. Based on the given 2D tracks, we can recover
    the position of the key points in the world frame.
    :param data_dict: the dataset containing images, position, orientation
    :param i: test image index
    :param idx: predict result of the posenet
    :param camParams: camera intrinsic matrix, focal length, distortion
    :param params: bm, sigma, alpha_m, max_range
    :param num_images: the number of selected images in the train set.
    :param gap: gap to select image
    :return:
    """
    # keyframes selection
    idx_neighbor = selectimages(data_dict['train_position'], data_dict['train_orientation'],
                                idx, params, num_images)
    print(idx_neighbor)

    num_img_left = (num_images - 1) / 2    #TODO: can we remove this?
    nImgIndList = idx_neighbor
    nImgList = [data_dict['test_images'][i]] + [data_dict['train_images'][i] for i in nImgIndList]

    # init cv2.sift feature extractor
    sift = cv2.SIFT_create(nOctaveLayers=6, edgeThreshold=20)

    # load the test image
    Iprev = cv2.imread(nImgList[0])
    Iprev = cv2.cvtColor(Iprev, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(Iprev, None)

    # create tracks to record the total matched pairs
    tracks = dict()
    for i in range(len(kp1)):
        tracks[i] = list()

    for i in range(1, len(nImgList)):
        Ipost = cv2.imread(nImgList[i])
        Ipost = cv2.cvtColor(Ipost, cv2.COLOR_BGR2GRAY)
        kp2, des2 = sift.detectAndCompute(Ipost, None)
        matches = vl_ubcmatch(des1, des2)

        
        # TODO: add RANSAC to eliminate the outliers
        for j in range(matches.shape[0]):
            queryIdx, trainIdx = matches[j]
            pt = kp2[trainIdx].pt
            # tracks[queryIdx].append((nImgIndList[i-1], pt))
            tracks[queryIdx].append((i - 1, pt))

    # filter the tracks
    tracks = {k: v for k, v in tracks.items() if len(v) >= 2}

    # add the camera pose
    camPoses = list()
    for c in range(len(nImgIndList)):
        camPoses.append({'ViewId': c,
                         # Transpose here
                         'Orientation': data_dict['train_orientation'][nImgIndList[c]].T,
                         'Location': data_dict['train_position'][nImgIndList[c]]})

    ### Test the data from matlab
    # test_tracks = dict()
    # mat_data = sio.loadmat('./matlab/BA_matlab/output.mat')['output']
    # pts2d = list()
    # for i in range((mat_data.shape[0])//2):
    #     ind = 2*i
    #     test_tracks[i] = list()
    #     for j in range(mat_data[ind][0].shape[1]):
    #         viewIds = mat_data[ind][0].reshape(-1)
    #         points = mat_data[ind+1][0]
    #         if viewIds[j] == 1:
    #             pts2d.append((points[j][1], points[j][0]))
    #         else:
    #             test_tracks[i].append((viewIds[j] - 2, tuple(points[j])))
    # xyz, errors = triangulateMultiView(test_tracks, camPoses, camParams)

    # triangulate to get the 3d points in the world
    xyz, errors = triangulateMultiView(tracks, camPoses, camParams)

    pts2d = []
    for k in tracks.keys():
        # Exchange the x and y point, why?
        pts2d.append((kp1[k].pt[1], kp1[k].pt[0]))  
    pts2d = np.array(pts2d)

    # error cut
    xyz = xyz[(errors < 5).reshape(-1)]
    pts2d = pts2d[(errors < 5).reshape(-1)]

    robotpose = data_dict['train_position'][idx]
    orientation = data_dict['train_orientation'][idx] #TODO: check transpose
    K = np.array(camParams['IntrinsicMatrix']).T

    return orientation, robotpose, pts2d, xyz, K



if __name__ == "__main__":
    data_dict, posenet_x_predicted = load_TUM_data('1_desk2')
    gap = 2
    num_images = 7

    params = {}
    params['bm']        = 0.1
    params['sigma']     = 0.2
    params['alpha_m']   = 3
    params['max_range'] = 100

    results = dict()
    results['orient']          = list()
    results['pose']            = list()
    results['orient_error']    = list()
    results['pose_error']      = list()
    results['reproject_error'] = list()

    index_list = list()
    # for i in range(0, min(posenet_x_predicted.shape[0], len(data_dict['train_images'])), 100):
    for i in range(0, 100, 10):
        index_list.append(i)
        idx = int(posenet_x_predicted[i] - 1)
        camParams = generateIntrinsics()

        orientation, robotpose, pts2D, pts3D, K = process_7scene_SIFT(data_dict, i, idx,
                                                                      camParams, params,
                                                                      num_images=7, gap=2)

        ### the output estimation is a tuple contains (orientation, robotpose, reproerror2, angle_var, position_var)
        estimation = optimizationLS(orientation, robotpose, pts2D, pts3D, pts3D.shape[0], K)

        for i, key in enumerate(results):
            results[key].append(estimation[i])


    ### plot the final result
    if True:
        xyz_gt  = np.array([data_dict['test_position'][i] for i in index_list])
        xyz_est = np.array(results['pose'])
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(xyz_gt[:, 0], xyz_gt[:, 1], xyz_gt[:, 2], label='Ground Truth')
        ax.scatter3D(xyz_est[:, 0], xyz_est[:, 1], xyz_est[:, 2], label='Estimation')
        ax.set_zlim3d(-2, 2)
        ax.set_xlim3d(-8, 4)
        ax.set_ylim3d(-8, 4)
        plt.legend()
        plt.show()


    # # keyframes selection
    # idx_neighbor = selectimages(data_dict['train_position'], data_dict['train_orientation'], idx, params, num_images)
    #
    # num_img_left = (num_images - 1) / 2
    # # train images only
    # nImgIndList = idx_neighbor
    # # The first one is the test image
    # nImgList = [data_dict['test_images'][i]] + [data_dict['train_images'][i] for i in nImgIndList]
    #
    # # init cv2.sift feature extractor
    # sift = cv2.SIFT_create(nOctaveLayers=6, edgeThreshold=20)
    #
    # # create tracks to record the total matched pairs
    # tracks = dict()
    #
    # # load the test image
    # Iprev = cv2.imread(nImgList[0])
    # Iprev = cv2.cvtColor(Iprev, cv2.COLOR_BGR2GRAY)
    # kp1, des1 = sift.detectAndCompute(Iprev, None)
    #
    # # Loop all the key points in test image
    # for i in range(len(kp1)):
    #     tracks[i] = list()
    #
    # for i in range(1, len(nImgList)):
    #     Ipost = cv2.imread(nImgList[i])
    #     Ipost = cv2.cvtColor(Ipost, cv2.COLOR_BGR2GRAY)
    #     kp2, des2 = sift.detectAndCompute(Ipost, None)
    #     matches = vl_ubcmatch(des1, des2)
    #
    #     # TODO: add RANSAC to eliminate the outliers
    #     for j in range(matches.shape[0]):
    #         queryIdx, trainIdx = matches[j]
    #         pt = kp2[trainIdx].pt
    #         # tracks[queryIdx].append((nImgIndList[i-1], pt))
    #         tracks[queryIdx].append((i - 1, pt))
    #
    #
    # # filter the tracks
    # tracks = {k: v for k, v in tracks.items() if len(v) >= 2}
    #
    # # add the camera pose
    # camParams = generateIntrinsics()
    # camPoses = list()
    # for c in range(len(nImgIndList)):
    #     camPoses.append({'ViewId': c,
    #                      # Transpose here
    #                      'Orientation': data_dict['train_orientation'][nImgIndList[c]].T,
    #                      'Location': data_dict['train_position'][nImgIndList[c]]})
    #
    #
    # xyz, errors = triangulateMultiView(tracks, camPoses, camParams)
    #
    # # error cut
    # xyz = xyz[(errors < 5).reshape(-1)]
    #
    # plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    # ax.set_zlim3d(-2, 2)
    # ax.set_xlim3d(-8, 4)
    # ax.set_ylim3d(-8, 4)
    # plt.show()


