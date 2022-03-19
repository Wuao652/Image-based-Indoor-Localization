import numpy as np
import cv2
from utils.dataloader_new import load_TUM_data


class camera():
    def __init__(self, intrinsics, position, orientation):
        self.position = position
        self.orientation = orientation
        self.intrinsics = intrinsics


if __name__ == '__main__':
    print('hello world from main_gl')
    num_image = 7
    gap = 2
    # Keyframe triangulation parameters
    param = {}
    param['bm'] = 0.1
    param['sigma'] = 0.2
    param['alpha_m'] = 3
    param['max_range'] = 100
    data_dict, posenet_x_predicted = load_TUM_data('1_desk2')
    posenet_x_predicted = posenet_x_predicted.astype(int)
    for i in range(0, 100, 10):
        idx = posenet_x_predicted[i]-1
        observe_ith_position = data_dict['train_position'][idx]
        observe_ith_orientation = data_dict['train_orientation'][idx]
        observe_init_position = data_dict['train_position'][idx]
        observe_init_orientation = data_dict['train_orientation'][idx]

        pass


