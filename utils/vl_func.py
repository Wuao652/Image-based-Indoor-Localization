import cv2
import numpy as np
def vl_ubcmatch(descr1, descr2, match_thresh=1.5):
    """
    uses the specified threshold match_thresh. A descriptor D1 is matched to a
    descriptor D2 only if the distance d(D1,D2) multiplied by match_thresh is
    not greater than the distance of D1 to all other descriptors.
    :param descr1: feature descriptors of the first image to match
    :param descr2: feature descriptors of the second image to match
    :param match_thresh: a default value of match_thresh is 1.5.
    :return: a numpy matrix of shape (N, 2). The first element is the query Id of
    the descriptor D1, the second element is the train Id of the descriptor D2.
    """
    flann_params = dict(algorithm=1, trees=4)
    flann = cv2.flann_Index(descr2, flann_params)
    idx, dist = flann.knnSearch(descr1, 2, params={})
    del flann
    matches = np.c_[np.arange(len(idx)), idx[:, 0]]
    pass_filter = dist[:, 0]*match_thresh < dist[:, 1]
    matches = matches[pass_filter]

    return matches