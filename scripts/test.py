# import h5py
# filename = "chess_train__siamese_FXPAL_output_1.h5py"
#
# with h5py.File(filename, "r") as f:
#     # List all groups
#     print("Keys: %s" % f.keys())
#     file_list = list(f.keys())
#     # Get the data
#     posenet_x_label = list(f[file_list[0]])
#     posenet_x_predicted = np.array(f[file_list[1]])

import numpy as np
from utils.dataloader_new import load_TUM_data
def main():
    data_dict, posenet_x_predicted = load_TUM_data('1_desk2')
    pass
if __name__ == '__main__':
    # data_dict, posenet_x_predicted = load_TUM_data('1_desk2')
    main()