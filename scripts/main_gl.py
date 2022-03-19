import numpy as np
import cv2
from utils.dataloader_new import load_TUM_data
from utils.keyframes_selection import selectimages

class camera():
    def __init__(self, intrinsics, position, orientation):
        self.position = position
        self.orientation = orientation
        self.intrinsics = intrinsics


if __name__ == '__main__':
    print('hello world from main_gl')
    num_images = 7
    gap = 2
    # Keyframe triangulation parameters
    params = {}
    params['bm'] = 0.1
    params['sigma'] = 0.2
    params['alpha_m'] = 3.0
    params['max_range'] = 100
    data_dict, posenet_x_predicted = load_TUM_data('1_desk2')
    posenet_x_predicted = posenet_x_predicted.astype(int)
    for i in range(0, 100, 10):
        print("step : ", i + 1)
        idx = posenet_x_predicted[i]-1
        observe_ith_position = data_dict['train_position'][idx]
        observe_ith_orientation = data_dict['train_orientation'][idx]
        observe_init_position = data_dict['train_position'][idx]
        observe_init_orientation = data_dict['train_orientation'][idx]

        # idx_neighbor = selectimages(data_dict['train_position'], data_dict['train_orientation'], idx, params, num_images)

        # Start to process 7Scenes

        pass


