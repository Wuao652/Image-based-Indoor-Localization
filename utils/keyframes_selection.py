import numpy as np
import cv2
from utils.dataloader_new import load_TUM_data
from utils.mathfunc import angleDifference
def selectimages(position, orientation, idx, params, num_images):
    """
    This function is used to select the most similar images given the
    prediction of the neural networks according to the s(T1, T2) scores.
    Note that the output of the NN is not included
    :param position: train_position
    :param orientation: train_orientation
    :param idx: the output of the NN
    :param params: dm, sigma, alpha_m use in the formula
    :param num_images: default to 7
    :return:
    """
    idx_neighbor_l = []
    idx_neighbor_r = []
    seed_l = idx
    seed_r = idx

    while len(idx_neighbor_l + idx_neighbor_r) < num_images:
        # right search
        ind = selectimages_onedirection(position, orientation, seed_r, params, 1)
        if ind > 0:
            idx_neighbor_r.append(ind - 1)
            seed_r = ind - 1
        if len(idx_neighbor_l + idx_neighbor_r) == num_images:
            break
        # left search
        ind = selectimages_onedirection(position, orientation, seed_l, params, -1)
        if ind > 0:
            idx_neighbor_l.append(ind - 1)
            seed_l = ind - 1
        if len(idx_neighbor_l + idx_neighbor_r) == num_images:
            break
    idx_neighbor_l.reverse()
    idx_neighbor = idx_neighbor_l + idx_neighbor_r
    return idx_neighbor

def selectimages_onedirection(position, orientation, idx, params, sign):
    """

    :param position:
    :param orientation:
    :param idx:
    :param params:
    :param sign: 1 means right-hand searching and -1 means left-hand searching
    :return: the correct index to update + 1
    """
    num_train = position.shape[0]
    T_l = np.zeros((4, 4)) # Camera pose left in SE(3).Body to world
    T_r = np.zeros((4, 4))  # Camera pose right in SE(3).Body to world

    T_l[0:3, 3] = position[idx, :]
    T_l[0:3, 0:3] = orientation[idx, :]
    score = np.zeros((params['max_range']))
    for i in range(1, params['max_range'] + 1):
        idx_next = idx + i * sign
        if idx_next < 0 or idx_next > num_train - 1:
            break
        T_r[0:3, 3] = position[idx_next, :]
        T_r[0:3, 0:3] = orientation[idx_next, :]
        dist_position = np.linalg.norm(T_l[0:3, 3] - T_r[0:3, 3])
        dist_angle = angleDifference(T_l[0:3, 0:3], T_r[0:3, 0:3])
        wb = np.exp(-np.square(dist_position - params['bm']) / np.square(params['sigma']))
        wv = np.min((params['alpha_m'] / dist_angle, 1.0))
        score[i-1] = wb * wv
    sorted_idx = np.argsort(-score)
    if score[sorted_idx[0]] == 0.0:
        return 0
    idx_next = idx + (sorted_idx[0] + 1) * sign
    return idx_next + 1

if __name__ == "__main__":
    data_dict, posenet_x_predicted = load_TUM_data('1_desk2')
    num_images = 7
    params = {}
    params['bm'] = 0.1
    params['sigma'] = 0.2
    params['alpha_m'] = 3.0
    params['max_range'] = 100
    position = data_dict['train_position']
    orientation = data_dict['train_orientation']
    idx = 535
    # keyframes selection
    idx_neighbor_l = []
    idx_neighbor_r = []
    seed_l = idx
    seed_r = idx

    while len(idx_neighbor_l + idx_neighbor_r) < num_images:
        # right search
        ind = selectimages_onedirection(position, orientation, seed_r, params, 1)
        if ind > 0:
            idx_neighbor_r.append(ind - 1)
            seed_r = ind - 1
        if len(idx_neighbor_l + idx_neighbor_r) == num_images:
            break
        # left search
        ind = selectimages_onedirection(position, orientation, seed_l, params, -1)
        if ind > 0:
            idx_neighbor_l.append(ind - 1)
            seed_l = ind - 1
        if len(idx_neighbor_l + idx_neighbor_r) == num_images:
            break
    idx_neighbor_l.reverse()
    idx_neighbor = idx_neighbor_l + idx_neighbor_r
    print(idx_neighbor)


