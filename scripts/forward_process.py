import numpy as np
import cv2
from utils.dataloader_new import load_TUM_data
from utils.keyframes_selection import selectimages
from utils.triangulateMultiView import triangulateMultiView
from utils.cameraParams import generateIntrinsics
import matplotlib.pyplot as plt

# from cyvlfeat.sift import sift as vlsift


def vl_ubcmatch(template_descriptors, current_img_descriptors, match_thresh=1.5):
    """

    :param template_descriptors:
    :param current_img_descriptors:
    :param match_thresh:
    :return:
    """
    flann_params = dict(algorithm=1, trees=4)
    flann = cv2.flann_Index(current_img_descriptors, flann_params)
    idx, dist = flann.knnSearch(template_descriptors, 2, params={})
    del flann
    matches = np.c_[np.arange(len(idx)), idx[:, 0]]
    pass_filter = dist[:, 0]*match_thresh < dist[:, 1]
    matches = matches[pass_filter]

    return matches

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
    Iprev = cv2.imread(nImgList[0])
    Iprev = cv2.cvtColor(Iprev, cv2.COLOR_BGR2GRAY)
    Ipost = cv2.imread(nImgList[1])
    Ipost = cv2.cvtColor(Ipost, cv2.COLOR_BGR2GRAY)
    Ipost1 = cv2.imread(nImgList[2])
    Ipost1 = cv2.cvtColor(Ipost1, cv2.COLOR_BGR2GRAY)
# cyvlfeat
    # kp1, des1 = vlsift(Iprev, edge_thresh=20, n_octaves=4, n_levels=6, compute_descriptor=True, float_descriptors=True)
    # kp2, des2 = vlsift(Ipost, edge_thresh=20, n_octaves=4, n_levels=6, compute_descriptor=True, float_descriptors=True)
# opencv
    sift = cv2.SIFT_create(nOctaveLayers=6, edgeThreshold=20)
    kp1, des1 = sift.detectAndCompute(Iprev, None)
    kp2, des2 = sift.detectAndCompute(Ipost, None)
    kp3, des3 = sift.detectAndCompute(Ipost1, None)

    print(len(kp1))
    print(len(kp2))
    print(len(kp3))


    bf = cv2.BFMatcher()
    # Use these two matches to get a track
    matches = bf.match(des1, des2)
    matches1 = bf.match(des1, des3)

    # Loop all the key points in test image
    tracks = dict()

    # for j in range(10):
    #     print(matches[j].queryIdx)
    #     print(matches1[j].queryIdx)

    # for i in range(len(kp1)):
    for i in range(3): # test 3 points only
        tracks[i] = list()

        id = matches[i].trainIdx
        pt = kp2[id].pt
        tracks[i].append((nImgIndList[0], pt))

        id = matches1[i].trainIdx
        pt = kp3[id].pt
        tracks[i].append((nImgIndList[1], pt))
    print(tracks)

    camParams = generateIntrinsics()
    camPoses = list()
    camPoses.append({'ViewId': nImgIndList[0],
                     'Orientation': data_dict['train_orientation'][nImgIndList[0]],
                     'Location': data_dict['train_position'][nImgIndList[0]]})

    camPoses.append({'ViewId': nImgIndList[1],
                     'Orientation': data_dict['train_orientation'][nImgIndList[1]],
                     'Location': data_dict['train_position'][nImgIndList[1]]})

    print(camPoses)

    xyz = triangulateMultiView(tracks, camPoses, camParams)
    print(xyz)

    # pts_1, pts_2 = np.array(pts_1), np.array(pts_2)
    # match_plot = cv2.drawMatches(Iprev, kp1, Ipost, kp2, matches, None, flags=2)
    # plt.imshow(match_plot), plt.show()