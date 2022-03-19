import numpy as np
import cv2
from utils.dataloader_new import load_TUM_data
from utils.keyframes_selection import selectimages
def process_7scene_SIFT():
    pass
if __name__ == "__main__":
    data_dict, posenet_x_predicted = load_TUM_data('1_desk2')
    gap = 2
    num_image = 7
    param = {}
    param['bm'] = 0.1
    param['sigma'] = 0.2
    param['alpha_m'] = 3
    param['max_range'] = 100

    observe_ith_position = np.array([1.34420, 0.26860, 1.72490])
    observe_ith_orientation = np.array([[0.45171, 0.57196, -0.68471],
                                        [0.89165, -0.26329, 0.36829],
                                        [0.03037, -0.77688, -0.62892]])
    observe_init_position = np.array([1.34420, 0.26860, 1.72490])
    observe_init_orientation = np.array([[0.45171, 0.57196, -0.68471],
                                        [0.89165, -0.26329, 0.36829],
                                        [0.03037, -0.77688, -0.62892]])
    idx = 535
    # keyframes selection

