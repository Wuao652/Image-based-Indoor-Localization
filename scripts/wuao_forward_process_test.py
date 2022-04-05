import numpy as np
import cv2
from utils.dataloader_new import load_TUM_data
from utils.keyframes_selection import selectimages
from utils.triangulateMultiView import triangulateMultiView
from utils.cameraParams import generateIntrinsics
from utils.vl_func import vl_ubcmatch
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

def plot_imgs(images):
    """
    :param images:
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
def process_7scene_SIFT():
    pass
if __name__ == "__main__":
    data_dict, posenet_x_predicted = load_TUM_data('1_desk2')
    gap = 2
    num_images = 7
    params = {}
    params['bm'] = 0.1
    params['sigma'] = 0.2
    params['alpha_m'] = 3
    params['max_range'] = 100

    observe_ith_position = np.array([1.34420, 0.26860, 1.72490])
    observe_ith_orientation = np.array([[0.45171, 0.57196, -0.68471],
                                        [0.89165, -0.26329, 0.36829],
                                        [0.03037, -0.77688, -0.62892]])
    observe_init_position = np.array([1.34420, 0.26860, 1.72490])
    observe_init_orientation = np.array([[0.45171, 0.57196, -0.68471],
                                        [0.89165, -0.26329, 0.36829],
                                        [0.03037, -0.77688, -0.62892]])
    i = 0
    idx = 535
    # keyframes selection
    idx_neighbor = selectimages(data_dict['train_position'], data_dict['train_orientation'], idx, params, num_images)
    print(idx_neighbor)
    num_img_left = (num_images - 1) / 2
    # train images only
    nImgIndList = idx_neighbor
    # The first one is the test image
    nImgList = [data_dict['test_images'][i]] + [data_dict['train_images'][i] for i in nImgIndList]
    print("Start to load img")
    # plot_imgs(nImgList)

    # init cv2.sift feature extractor
    sift = cv2.SIFT_create(nOctaveLayers=6, edgeThreshold=20)

    # create tracks to record the total matched pairs
    tracks = dict()

    # load the test image
    Iprev = cv2.imread(nImgList[0])
    Iprev = cv2.cvtColor(Iprev, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(Iprev, None)

    # Loop all the key points in test image
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
            tracks[queryIdx].append((i-1, pt))

    print("The total matching results:")
    print(len(tracks))
    # filter out the tracks with less than 2 points
    # for k in list(tracks.keys()):
    #     if len(tracks[k]) < 2:
    #         del tracks[k]

    tracks = {k: v for k, v in tracks.items() if len(v) >= 2}
    print("The tracks after the filter")
    print(len(tracks))
    # add the camera pose
    camParams = generateIntrinsics()
    camPoses = list()
    for c in range(len(nImgIndList)):
        # camPoses.append({'ViewId': nImgIndList[c],
        camPoses.append({'ViewId': c,
                         # Transpose here
                         'Orientation': data_dict['train_orientation'][nImgIndList[c]].T,
                         'Location': data_dict['train_position'][nImgIndList[c]]})
    print(camPoses)
    
    # print(list(tracks.keys())[0])
    # print(tracks[1])
    test_tracks = {0: [(0, (393.976501464844, 334.523040771484)),
                       (1, (387.846374511719, 335.493988037109)),
                       (3, (369.973510742188, 342.951934814453)),
                       (6, (429.238983154297, 274.907409667969))],
                   1: [(3, (62.4908638000488, 352.728759765625)),
                       (6, (134.445571899414, 278.729431152344))],
                   2: [(2, (119.998207092285, 467.221679687500)),
                       (5, (116.117393493652, 444.217376708984)),
                       (6, (214.065246582031, 385.179901123047))]}

    xyz, errors = triangulateMultiView(tracks, camPoses, camParams)

    # error cut
    xyz = xyz[(errors<5).reshape(-1)]

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    ax.set_zlim3d(-2, 2)
    ax.set_xlim3d(-8, 4)
    ax.set_ylim3d(-8, 4)
    plt.show()

    # plt.axes(projection='3d')
    # plt.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2])
