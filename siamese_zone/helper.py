from tqdm import tqdm
import numpy as np
import os.path
import sys
import random
import math
import cv2
import h5py
import tensorflow as tf
from tensorflow import keras
from custom_callbacks import BatchCSVLogger, TensorBoard
# from keras import callbacks
# from keras import backend as K
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
import time
import progressbar
from scipy.spatial.transform import Rotation as R

from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections

plt.switch_backend('agg')

# We make the learning_rate and epochs global so that they can be set once in
# the class constructor and then used in the learningRateschedule callback.
learning_rate_global = None
epochs_global = None


class print_current_learning_rate(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = float(K.get_value(self.model.optimizer.lr))
        decay = float(K.get_value(self.model.optimizer.decay))
        print('===> Current learning rate at end of epoch', epoch, '= ', lr)
        print('===> Current decay rate at end of epoch', epoch, '= ', decay)
        if decay > 0:
            lr *= (1. / (1. + decay * epoch))
            print('===> Current effective learning rate at end of epoch', epoch, '= ', lr)

    def on_epoch_begin(self, epoch, logs=None):
        lr = float(K.get_value(self.model.optimizer.lr))
        decay = float(K.get_value(self.model.optimizer.decay))
        print('===> Current learning rate at begin of epoch', epoch, '= ', lr)
        print('===> Current decay rate at begin of epoch', epoch, '= ', decay)
        if decay > 0:
            lr *= (1. / (1. + decay * epoch))
            print('===> Current effective learning rate at begin of epoch', epoch, '= ', lr)


def learningRateschedule(epoch):
    global learning_rate_global, epochs_global

    epochs = epochs_global
    if epoch >= 2 * epochs // 3:
        learning_rate = float(learning_rate_global * 0.01)
        print('Learning rate 3. third = ', learning_rate)
    elif epoch >= epochs // 3:
        learning_rate = float(learning_rate_global * 0.1)
        print('Learning rate 2. third = ', learning_rate)
    else:
        learning_rate = float(learning_rate_global)
        print('Learning rate 1. third = ', learning_rate)
    return float(learning_rate)


def extract_fn(example_proto):
    features = {"height": tf.io.FixedLenFeature((), tf.int64, default_value=0),
                "width": tf.io.FixedLenFeature((), tf.int64, default_value=0),
                "depth": tf.io.FixedLenFeature((), tf.int64, default_value=0),
                "poses": tf.io.FixedLenFeature((), tf.string, default_value=""),
                "compressed_image": tf.io.FixedLenFeature([], tf.string, default_value="")
                }
    parsed_features = tf.io.parse_single_example(serialized=example_proto, features=features)
    return parsed_features


def parse_function(example_proto):
    features = {"height": tf.io.FixedLenFeature((), tf.int64, default_value=0),
                "width": tf.io.FixedLenFeature((), tf.int64, default_value=0),
                "depth": tf.io.FixedLenFeature((), tf.int64, default_value=0),
                "poses": tf.io.FixedLenFeature((), tf.string, default_value=""),
                "compressed_image": tf.io.FixedLenFeature([], tf.string, default_value="")
                }
    parsed_features = tf.io.parse_single_example(serialized=example_proto, features=features)
    # image
    height = tf.cast(parsed_features["height"], tf.int32)
    width = tf.cast(parsed_features["width"], tf.int32)
    depth = tf.cast(parsed_features["depth"], tf.int32)

    image = tf.io.decode_raw(parsed_features["compressed_image"], tf.uint8)
    image = tf.reshape(image, [height, width, depth])
    image = tf.cast(image, tf.float32) / 255.0

    label = tf.io.decode_raw(parsed_features["poses"], tf.float32)
    # label = tf.cast(label, tf.int64)
    # print('label str: {}'.format(np.`array(label)))
    # label_pos, label_ori = tf.split(label, [3, 4])
    #     label_pos = tf.reshape(label, [7])

    return image, label  # (label_pos, label_ori)#label#label_pos, label_ori


# def initialise_callbacks(io_files, batch_size, learning_rate, epochs):
def initialise_callbacks(checkpoint_weights_file, tensor_log_file, csv_log_file, batch_size, learning_rate, epochs):
    global learning_rate_global, epochs_global
    learning_rate_global = learning_rate
    epochs_global = epochs

    checkpointer = callbacks.ModelCheckpoint(
        filepath=checkpoint_weights_file,
        verbose=1, save_best_only=True,
        save_weights_only=True)
    version = tf.__version__
    if (int(version.split('.')[0]) == 1):
        tensor_board = TensorBoard(log_dir=tensor_log_file,
                                   histogram_freq=0, batch_size=batch_size,
                                   write_graph=False, write_grads=False,
                                   write_images=False, embeddings_freq=0,
                                   write_batch_performance=False,
                                   embeddings_layer_names=None,
                                   embeddings_metadata=None)

    else:
        tensor_board = callbacks.TensorBoard(log_dir=tensor_log_file,
                                             histogram_freq=0,
                                             write_graph=True,
                                             write_images=True,
                                             update_freq='epoch',
                                             embeddings_freq=0,
                                             profile_batch=1000,
                                             embeddings_metadata=None)

    csv_logger = callbacks.CSVLogger(csv_log_file)

    early_stopping = callbacks.EarlyStopping('loss', min_delta=0.5,
                                             patience=5, verbose=1,
                                             mode='min')

    # lrate = callbacks.LearningRateScheduler(step_decay(learning_rate, epoch_drop, lr_step))

    callback_list = []
    # 	callback_list.append(checkpointer)
    callback_list.append(tensor_board)
    callback_list.append(csv_logger)
    # callback_list.append(early_stopping)
    # append learning rate callback
    callback_list.append(callbacks.LearningRateScheduler(learningRateschedule, 1))
    callback_list.append(print_current_learning_rate())

    return callback_list


def custom_generator(input_data_file, batch_size, image_mean, shuffle, flag2D):
    '''Custom generator to generator batches of data. It produces a tuple
	   containing the input data (batch_size, 224, 224, 3) and a 6 dimensional
	   array containing 3 copies of the position and orientation labels
	   [(b_s, 3), (b_s, 4), (b_s, 3), (b_s, 4), (b_s, 3), (b_s, 4)], where b_s
	   is the batch_size. (3 copies since there are 6 outputs of posenet.py
	   The data is cropped to be the correct size but there are no other
	   pre processing steps applied to the data.'''
    f = h5py.File(input_data_file, "r")
    number_of_data = len(f['images'])

    count = 0

    while True:
        # Generate a list of indices for each image in the dataset and shuffle
        # the list.
        image_idx_list = list(range(number_of_data))

        if shuffle:
            random.shuffle(image_idx_list)

        # Yields a single, randomly selected batch.
        for idx in range(0, number_of_data, batch_size):
            if (idx <= number_of_data - batch_size):
                image_batch = np.empty((batch_size, 224, 224, 3), dtype=float)
                if flag2D:
                    output_xy_batch = np.empty((batch_size, 2), dtype=float)
                    output_yaw_batch = np.empty((batch_size, 1), dtype=float)
                else:
                    output_xyz_batch = np.empty((batch_size, 3), dtype=float)
                    output_wpqr_batch = np.empty((batch_size, 4), dtype=float)

                image_idx = image_idx_list[idx:idx + batch_size]

                images_out = []

                i = 0
                for val in image_idx:

                    if flag2D:
                        output_xy_batch[i] = f['poses'][val][0:2]
                        quaternion = f['poses'][val][3:7]
                        euler_output = convert_quaternion_to_euler(quaternion, True)
                        output_yaw_batch[i] = euler_output[0]

                    else:
                        output_xyz_batch[i] = f['poses'][val][0:3]
                        output_wpqr_batch[i] = f['poses'][val][3:7]

                    flat_image = f['images'][val]

                    augmented_image = np.reshape(flat_image, (224, 224, 3))
                    augmented_image = np.transpose(augmented_image, (2, 0, 1))
                    image_mean = image_mean.reshape((3, 224, 224))
                    augmented_image = augmented_image - image_mean.astype(float)
                    augmented_image = np.squeeze(augmented_image)
                    augmented_image = np.transpose(augmented_image, (1, 2, 0))

                    image_batch[i] = augmented_image.astype(float)
                    i += 1
                if flag2D:
                    output_pose_batch = [output_xy_batch, output_yaw_batch]
                else:
                    output_pose_batch = [output_xyz_batch, output_wpqr_batch]

                yield image_batch, output_pose_batch


def convert_quaternion_to_euler(quat, DEGREE_FLAG=False):
    #     print('quaternion: {}'.format(quat))
    rotation_matrix = gt.quaternion_matrix(quat)
    #     print('rotation matrix test: {}'.format(posetest))
    euler_angles = gt.euler_from_matrix(rotation_matrix, 'sxyz')
    euler_angles = np.array(euler_angles)
    #     print('euler angles radian: {}'.format(euler_angles))
    #     print('euler_angles type after: {}'.format(type(euler_angles)))
    if DEGREE_FLAG:
        euler_angles_degree = np.empty((3,), dtype=float)
        euler_angles[0] = euler_angles[0] * (180 / np.pi)
        euler_angles[1] = euler_angles[1] * 180 / np.pi
        euler_angles[2] = euler_angles[2] * 180 / np.pi
    #     print('euler angles degrees: {}'.format(euler_angles))
    return euler_angles


## legacy code.. delete in future
def custom_generator_2d(input_data_file, batch_size, image_mean, shuffle):
    '''Custom generator to generator batches of data. It produces a tuple
	   containing the input data (batch_size, 224, 224, 3) and a 3 dimensional
	   array containing 2 copies of the position (x,y) and 1 copy of orientation labels
	   [(b_s, 3), (b_s, 4), (b_s, 3)], where b_s
	   is the batch_size. (3 copies since there are 3 outputs of posenet.py
	   The data is cropped to be the correct size but there are no other
	   pre processing steps applied to the data.'''
    f = h5py.File(input_data_file, "r")
    number_of_data = len(f['images'])

    count = 0

    while True:
        # Generate a list of indices for each image in the dataset and shuffle
        # the list.
        image_idx_list = list(range(number_of_data))

        if shuffle:
            random.shuffle(image_idx_list)

        # Yields a single, randomly selected batch.
        for idx in range(0, number_of_data, batch_size):
            if (idx <= number_of_data - batch_size):
                image_batch = np.empty((batch_size, 224, 224, 3), dtype=float)
                output_xy_batch = np.empty((batch_size, 2), dtype=float)
                output_yaw_batch = np.empty((batch_size, 1), dtype=float)
                image_idx = image_idx_list[idx:idx + batch_size]

                images_out = []

                i = 0
                for val in image_idx:
                    output_xy_batch[i] = f['poses'][val][0:2]
                    quaternion = f['poses'][val][3:7]
                    euler_output = convert_quaternion_to_euler(quaternion, True)
                    output_yaw_batch[i] = euler_output[0]

                    flat_image = f['images'][val]

                    augmented_image = np.reshape(flat_image, (224, 224, 3))
                    augmented_image = np.transpose(augmented_image, (2, 0, 1))
                    image_mean = image_mean.reshape((3, 224, 224))
                    augmented_image = augmented_image - image_mean.astype(float)
                    augmented_image = np.squeeze(augmented_image)
                    augmented_image = np.transpose(augmented_image, (1, 2, 0))

                    image_batch[i] = augmented_image.astype(float)
                    i += 1

                output_pose_batch = [output_xy_batch, output_yaw_batch]

                yield image_batch, output_pose_batch


## legacy code.. delete in future
def custom_generator_sample(input_data_file, batch_size, image_mean, shuffle):
    '''Custom generator to generator batches of data. It produces a tuple
	   containing the input data (batch_size, 224, 224, 3) and a 6 dimensional
	   array containing 3 copies of the position and orientation labels
	   [(b_s, 3), (b_s, 4), (b_s, 3), (b_s, 4), (b_s, 3), (b_s, 4)], where b_s
	   is the batch_size. (3 copies since there are 6 outputs of posenet.py
	   The data is cropped to be the correct size but there are no other
	   pre processing steps applied to the data.'''
    f = h5py.File(input_data_file, "r")
    number_of_data = len(f['images'])

    count = 0

    while True:
        # Generate a list of indices for each image in the dataset and shuffle
        # the list.
        image_idx_list = list(range(number_of_data))

        if shuffle:
            random.shuffle(image_idx_list)

        # Yields a single, randomly selected batch.
        for idx in range(0, number_of_data, batch_size):
            if (idx <= number_of_data - batch_size):
                image_batch = np.empty((batch_size, 224, 224, 3), dtype=float)
                # output_xyz_batch = np.empty((batch_size, 3), dtype=float)
                # output_wpqr_batch = np.empty((batch_size, 4), dtype=float)
                output_label_batch = np.empty((batch_size, 7), dtype=float)

                image_idx = image_idx_list[idx:idx + batch_size]

                images_out = []

                i = 0
                for val in image_idx:
                    # output_xyz_batch[i] = f['poses'][val][0:3]
                    # output_wpqr_batch[i] = f['poses'][val][3:7]
                    output_label_batch[i] = f['poses'][val][:]
                    flat_image = f['images'][val]

                    augmented_image = np.reshape(flat_image, (224, 224, 3))
                    augmented_image = np.transpose(augmented_image, (2, 0, 1))
                    image_mean = image_mean.reshape((3, 224, 224))
                    augmented_image = augmented_image - image_mean.astype(float)
                    augmented_image = np.squeeze(augmented_image)
                    augmented_image = np.transpose(augmented_image, (1, 2, 0))

                    image_batch[i] = augmented_image.astype(float)
                    i += 1

                output_pose_batch = output_label_batch

                yield image_batch, output_pose_batch


class PreprocessingData(object):
    def __init__(self):
        self.image_mean = None
        self.sample_images = None

    def calculate_sample_mean(self, sample_size, input_data_file):
        '''Calculates the mean of a sample of images'''
        # compute images mean
        self.getRepresentativeImageSample(sample_size, input_data_file)
        self.sample_images = self.sample_images.astype(float)

        # compute images mean
        images_cropped = self.sample_images
        N = 0
        mean = np.zeros((1, 3, 224, 224), dtype=float)
        for X in tqdm(images_cropped):
            mean[0][0] += X[:, :, 0]
            mean[0][1] += X[:, :, 1]
            mean[0][2] += X[:, :, 2]
            N += 1
        mean[0] /= N

        image_mean = mean.reshape((3, 224, 224))

        return image_mean

    def writeDataToFile(self, output_h5py_file, data_dict):
        '''Write preprocessing data (such as image mean) to a file to be used
		   during evaluation.'''
        with h5py.File(output_h5py_file, 'w') as f:
            grp = f.create_group("posenet")

            # 			image_mean_set = grp.create_dataset("image_mean",
            # 											  (1, 224*224*3),
            # 											  dtype=np.float)
            learning_rate = grp.create_dataset('posenet_learning_rate', (1,),
                                               dtype=float)
            batch_size = grp.create_dataset('posenet_batch_size', (1,),
                                            dtype=int)
            decay = grp.create_dataset('posenet_decay', (1,), dtype=float)
            epochs = grp.create_dataset('posenet_epochs', (1,), dtype=int)
            beta = grp.create_dataset('posenet_beta', (1,), dtype=float)

            dt = h5py.special_dtype(vlen=bytes)
            dataset_train = grp.create_dataset('posenet_dataset_train', (1,), dtype=dt)
            dataset_test = grp.create_dataset('posenet_dataset_test', (1,), dtype=dt)
            checkpoint_weight_file = grp.create_dataset('posenet_checkpoint_weight_file',
                                                        (1,), dtype=dt)
            weight_file = grp.create_dataset('posenet_weight_file', (1,), dtype=dt)

            # Serialize image mean
            # 			flat_np_image = data_dict['image_mean'].flatten()
            # 			flat_np_image = np.squeeze(flat_np_image)

            # 			image_mean_set[0] = flat_np_image
            learning_rate[0] = data_dict['posenet_learning_rate']
            batch_size[0] = data_dict['posenet_batch_size']
            beta[0] = data_dict['posenet_beta']
            decay[0] = data_dict['posenet_decay']
            epochs[0] = data_dict['posenet_epochs']
            # 			dataset_train[0] = data_dict['posenet_dataset_train']
            # 			dataset_test[0] = data_dict['posenet_dataset_test']
            checkpoint_weight_file[0] = data_dict['posenet_checkpoint_weight_file']
            weight_file[0] = data_dict['posenet_weight_file']

    # 			grp2 = f.create_group("general")

    # 			data_size = grp2.create_dataset("general_data_size", (1,), dtype=int)
    # 			validation_data_size = grp2.create_dataset("general_validation_data_size",
    # 													  (1,), dtype=int)

    # 			data_size[0] = data_dict['general_data_size']
    # 			validation_data_size[0] = data_dict['general_validation_data_size']

    def getRepresentativeImageSample(self, sample_size, input_data_file):
        '''Gets a representative sample of the entire dataset. This is used to
		   calculate the mean and variance of the dataset used in normalisation'''
        f = h5py.File(input_data_file, "r")

        # Generate a list of indices for each image in the dataset and shuffle
        # the list. Then take a the first $sample_size indices and return a list of
        # their corresponding images.
        image_idx_list = list(range(len(f['images'])))
        # print('before image list: {}'.format(image_idx_list))
        random.shuffle(image_idx_list)
        # print('after image list: {}'.format(image_idx_list))
        image_idx_sample_list = image_idx_list[0:sample_size]

        image_list = np.empty((sample_size, 224, 224, 3), dtype=np.float)
        i = 0
        for image_idx in image_idx_sample_list:
            reshaped_image = np.reshape(f['images'][image_idx], (224, 224, 3))
            image_list[i] = reshaped_image
            i += 1

        self.sample_images = image_list
        f.close()


def euc_loss_testing(y_true, y_pred):
    '''This loss function does not scale the output loss (which is done in the
	   loss functions used for testing). This results in the loss being in real
	   world units.'''
    lx = K.sqrt(K.sum(K.square(y_true[:, :] - y_pred[:, :]), axis=1, keepdims=True))
    return lx


def readDataFromFile(input_h5py_file):
    with h5py.File(input_h5py_file, "r") as f:
        data_dict = {}

        for group in f.keys():
            for key in f[group].keys():
                if key == 'image_mean':
                    image_mean = f['posenet'][key]
                    image_mean = np.reshape(image_mean, (224, 224, 3))
                    data_dict['image_mean'] = image_mean
                else:
                    data_dict[key] = f[group][key][0]
        for key, val in data_dict.items():
            if isinstance(val, bytes):
                # 				print(str(val,'utf-8'))
                data_dict[key] = str(val, 'utf-8')
        # 				print(key, type(val) )
        return data_dict


def angleDifference( q1, q2):
    angle = 0
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    r1 = r1.as_matrix()
    r2 = r2.as_matrix()
    if abs(np.trace(np.dot(r1.transpose(), r2)) - 3) < 0.0000000001:
        angle = 0
    else:
        angle = math.acos((np.trace(np.dot(r1.transpose(), r2)) - 1) / 2) * 180 / np.pi
    return angle


def distDifference( p1, p2):
    tmp = p1 - p2
    return np.linalg.norm(tmp)


def getImageSimilarity(pose1, pose2, threshold_anlge, threshold_dist):
    distDif = distDifference(pose1[0:3], pose2[0:3])
    angleDif = angleDifference(pose1[3:], pose2[3:])
    if (angleDif < threshold_anlge) and (distDif < threshold_dist):
        return True, distDif, angleDif
    else:
        return False, distDif, angleDif
def customPredictSiamese_tf2(based_model,
                             dataset_test,
                             train_results, #   Imgae descriptors for training images
                             batch_size,
                             pose_train,
                             pose_test,
                             threshold_anlge,
                             threshold_dist,
                             num_candidate,
                               save_output_data_file=None):
    #   Generate result
    # zone_groudtrtruth = []
    # for features in dataset_test:
    #     zone_groudtrtruth   += [int(features[1].numpy()[0]) + 1]


    #   Modified by Jingwei: Predict
    dataset_test = dataset_test.batch(batch_size)
    dataset_test = dataset_test.prefetch(5)
    test_results = based_model.predict(dataset_test)
    prediction = []  

    p = progressbar.ProgressBar()
    results = np.zeros((len(test_results), 1), dtype=float)
    results_dist  = np.zeros((len(test_results), 1), dtype=float)
    results_angle = np.zeros((len(test_results), 1), dtype=float)


    #   Use nearpy
    dimension = train_results.shape[1]
    # Create a random binary hash with 10 bits
    rbp = RandomBinaryProjections('rbp', 10)
    # Create engine with pipeline configuration
    engine = Engine(dimension, lshashes=[rbp])
    for index in range(train_results.shape[0]):
        engine.store_vector(train_results[index,:],index)


    scores = np.zeros(len(train_results), dtype=float)
    for i in p(range(len(test_results))):   #   For every image
        # #   Lookup the dictionary
        # #   Find the top 'num_candidate' scores in zone
        # #   1. Initialize the candidate
        # score = np.linalg.norm(test_results[i] - zone_dict[0][0]) * np.ones((num_candidate,1),float)
        # for k in range(1,num_img_siamese):
        #     tmp = np.linalg.norm(test_results[i] - zone_dict[0][k])
        #     # for m in range(1, num_candidate):
        #     #     if (tmp < score[m]):
        #     #         score[m] = tmp
        #     #         break
        #     if (tmp < score[0]):
        #         score[0] = tmp
        # score[:] = score[0]

        # # ind_zone = np.ones((num_candidate,1),dtype=int)
        # scores = np.zeros(len(zone_dict), dtype=float)
        # for j in range(0,len(zone_dict)):
        #     temp = np.linalg.norm(test_results[i] - zone_dict[j][0])
        #     for k in range(1, num_img_siamese):
        #         tmp = np.linalg.norm(test_results[i] - zone_dict[j][k])
        #         if (tmp < temp):
        #             temp = tmp
        #     scores[j] = temp
            # # temp = np.linalg.norm(test_results[i] - zone_dict[j])
            # for m in range(0,num_candidate):
            #     if(temp < score[m]):
            #         ind_zone[m] = j+1
            #         score[m] = temp
            #         break
        # #   Version I: Search the optimal by brutal force
        for k in range(len(train_results)):
            scores[k] = np.linalg.norm(test_results[i] - train_results[k])
        ind_image = np.argsort(scores) + 1
        #   Select top num_candidate images:
        #   Add ind_image[0] as the candidate image
        candidate = [ind_image[0]]

        #   Version II: Approximate hash searching.
        #N = engine.neighbours(test_results[i] )
        #candidate = [N[0][1]]
        #num_candidate = 1


        #   Select ind_image[k] as cadndiate. Make sure it is not close to other candaite images
        if num_candidate > 1:
            for k in range(1,len(train_results)):
                ind = k
                flag_add = True
                for q in range(len(candidate)):
                    [bool_similar, distDif, angleDif] = getImageSimilarity(pose_train[ind_image[k]-1], pose_train[candidate[q]-1], 2*threshold_anlge,
                                                      4*threshold_dist)
                    if bool_similar==True:
                        flag_add = False
                        break
                if flag_add == True:
                    candidate += [ind_image[k]]
                if len(candidate)==num_candidate:
                    break
            while(len(candidate) < num_candidate):
                candidate += [-1]
            prediction += [candidate]
            time.sleep(0.01)
            error = 1
        else:
            prediction += [candidate]
        #   Compare with the ground truth
        for m in range(num_candidate):
            [bool_similar, distDif, angleDif] = getImageSimilarity(pose_train[candidate[m]-1], pose_test[i], 2*threshold_anlge, 4*threshold_dist)
            if m == 0:
                results_dist[i]  = distDif
                results_angle[i] = angleDif
                error = 0
            if bool_similar==True:
                error = 0
                results_dist[i] = distDif
                results_angle[i] = angleDif
        # error_x = np.linalg.norm(ind_image[0] - zone_groudtrtruth[i])
        results[i] = error
    median_result = np.mean(results, axis=0)  


    median_result = np.mean(results, axis=0)
    median_result_str = '(Median error: {} zones)'.format(median_result)
    print(median_result_str)
    median_result = np.mean(results_dist, axis=0)
    median_result_str = '(Median distance error: {} meter)'.format(median_result)
    print(median_result_str)
    median_result = np.mean(results_angle, axis=0)
    median_result_str = '(Median angular error: {} degree)'.format(median_result)
    print(median_result_str)


    #   Calculate groundtruth (Not unique!!)
    groundtruth = []
    for i in range(len(pose_test)):
        bool_find = False
        for k in range(len(pose_train)):
            [bool_similar, distDif, angleDif] = getImageSimilarity(pose_train[k], pose_test[i], threshold_anlge,
                                              threshold_dist)
            if bool_similar == True:
                groundtruth += [k]
                bool_find = True
                break
        if bool_find == False:
            groundtruth += [-1]

    save_siamese_data(prediction, groundtruth, save_output_data_file)

    return median_result
def customPredictGenerator_tf2(model, data, batch_size, training_name,
                               should_plot=False, should_plot_cdf=False, should_save_output_data=True,
                               save_cdf_position_file=None,
                               save_cdf_orientation_file=None,
                               save_trajectory_file=None,
                               save_output_data_file=None,
                               flag2D=False):
    for idx, data_val in enumerate(data):
        # print(data_val)
        number_of_samples = idx
        if idx == 0:
            xyz_label = np.array(data_val[1])
        # 	# wpqr_label = np.array(data_val[1][1])
        else:
            xyz_label = np.concatenate((xyz_label, np.array(data_val[1])), axis=0)
    # 	# wpqr_label = np.concatenate( (wpqr_label, np.array(data_val[1][1])), axis=0)

    xyz_prediction = model.predict(data, verbose=1)

    theta_error = []
    time_vector = []
    results = np.zeros((len(xyz_prediction), 1), dtype=float)
    for idx, (xyz_l, xyz_p) in enumerate(zip(xyz_label, xyz_prediction)):
        time_vector.append(idx)
        if flag2D:
            error_x = np.linalg.norm(xyz_l - xyz_p)  # for 2D case it will be xy only
            theta = np.linalg.norm(ori_l - ori_p)  # for 2D case it will only be yaw
            theta_error.append(theta)
            results[idx, :] = [error_x, theta]
        else:
            # q1 = ori_l / np.linalg.norm(ori_l)
            # q2 = ori_p / np.linalg.norm(ori_p)
            # d = abs(np.sum(np.multiply(q1,q2)))
            # d = min(d, 1)
            error_x = np.linalg.norm(xyz_l - xyz_p)
            # theta = 2 * np.arccos(d) * 180/math.pi
            # theta_error.append(theta)
            results[idx] = error_x

    pos_error = results[:, 0]

    median_result = np.median(results, axis=0)
    median_result_str = '(Median error: {} zones)'.format(median_result)
    median_result_return = '{}'.format(median_result)
    print(median_result_str)

    if flag2D:
        x_predicted = xyz_prediction[:, 0]
        y_predicted = xyz_prediction[:, 1]
        x_label = xyz_label[:, 0]
        y_label = xyz_label[:, 1]
    else:
        x_predicted = xyz_prediction[:, 0]
        y_predicted = xyz_prediction[:, 1]
        z_predicted = xyz_prediction[:, 2]
        x_label = xyz_label[:, 0]
        y_label = xyz_label[:, 1]
        z_label = xyz_label[:, 2]

    if should_save_output_data:
        save_output_data(x_predicted, y_predicted, z_predicted, x_label, y_label, z_label, theta_error, training_name,
                         flag2D, save_output_data_file)

    if should_plot_cdf:
        if flag2D:
            plot_cdf_2d(x_predicted, y_predicted, x_label, y_label, theta_error, training_name, save_cdf_position_file,
                        save_cdf_orientation_file)
        else:
            plot_cdf(x_predicted, y_predicted, z_predicted, x_label, y_label, z_label, theta_error, training_name,
                     save_cdf_position_file, save_cdf_orientation_file)
    #### NOTE: for both 3d and 2d case we are currently only interested in 2d trajectory plot. Hence plot_trajectory should plot for both cases
    if should_plot:
        plot_trajectory(x_predicted, y_predicted, x_label, y_label, pos_error, time_vector, training_name,
                        save_trajectory_file)

    return median_result_return


def customPredictGenerator(model, generator, batch_size, steps, training_name, training_gen,
                           should_plot=False, should_plot_cdf=False, should_save_output_data=False,
                           save_cdf_position_file=None,
                           save_cdf_orientation_file=None,
                           save_trajectory_file=None,
                           save_output_data_file=None,
                           flag2D=False):
    # print('batchsize: {} and steps: {}'.format(batch_size,steps))
    # print('generator: {} and training_gen: {}'.format(generator, training_gen))

    results = np.zeros((batch_size * steps, 2), dtype=float)
    if flag2D:

        x_predicted = []
        y_predicted = []
        z_predicted = []  ## just kept to pass as empty list to the fuction.. ITS A HACK
        yaw_predicted = []

        x_label = []
        y_label = []
        z_label = []  ## just kept to pass as empty list to the fuction.. ITS A HACK
        yaw_label = []

        xy_predicted = []

        xy_label = []
        theta_error = []

    else:

        x_predicted = []
        y_predicted = []
        z_predicted = []

        w_predicted = []
        p_predicted = []
        q_predicted = []
        r_predicted = []

        xyz_predicted = []

        x_label = []
        y_label = []
        z_label = []

        w_label = []
        p_label = []
        q_label = []
        r_label = []

        # 	x_label_train = []
        # 	y_label_train = []
        # 	z_label_train = []

        xyz_label = []
        theta_error = []

    for step in range(steps):
        batch = generator.__next__()
        batch_training = training_gen.__next__()

        prediction_batch = model.predict(batch[0], batch_size=batch_size, verbose=1)

        for i in range(batch_size):
            if flag2D:
                x_predicted.append(prediction_batch[0][i][0])
                y_predicted.append(prediction_batch[0][i][1])
                yaw_predicted.append(prediction_batch[1][i][0])

                x_label.append(batch[1][0][i][0])
                y_label.append(batch[1][0][i][1])
                yaw_label.append(batch[1][1][i][0])

                xy_predicted.append(prediction_batch[0][i])
                xy_label.append(batch[1][0][i])

                theta = np.linalg.norm(yaw_predicted[i] - yaw_label[i])
                theta_error.append(theta)

            else:
                x_predicted.append(prediction_batch[0][i][0])
                y_predicted.append(prediction_batch[0][i][1])
                z_predicted.append(prediction_batch[0][i][2])

                w_predicted.append(prediction_batch[1][i][0])
                p_predicted.append(prediction_batch[1][i][1])
                q_predicted.append(prediction_batch[1][i][2])
                r_predicted.append(prediction_batch[1][i][3])

                x_label.append(batch[1][0][i][0])
                y_label.append(batch[1][0][i][1])
                z_label.append(batch[1][0][i][2])

                w_label.append(batch[1][1][i][0])
                p_label.append(batch[1][1][i][1])
                q_label.append(batch[1][1][i][2])
                r_label.append(batch[1][1][i][3])

                # 			x_label_train.append(batch_training[1][0][i][0])
                # 			y_label_train.append(batch_training[1][0][i][1])
                # 			z_label_train.append(batch_training[1][0][i][2])

                xyz_predicted.append(prediction_batch[0][i])
                xyz_label.append(batch[1][0][i])

                q1 = batch[1][1][i] / np.linalg.norm(batch[1][1][i])
                q2 = prediction_batch[1][i] / np.linalg.norm(prediction_batch[1][i])
                d = abs(np.sum(np.multiply(q1, q2)))
                d = min(d, 1)
                if d > 1:
                    print('value of d: {} at interation number: {}'.format(d, i))
                    theta = 0.0
                else:
                    theta = 2 * np.arccos(d) * 180 / math.pi
                    theta_error.append(theta)

    time_vector = []

    for i in range(steps * batch_size):
        time_vector.append(i)

        if flag2D:
            # Compute Individual Sample Error
            error_x = np.linalg.norm(xy_label[i] - xy_predicted[i])
            results[i, :] = [error_x, theta_error[i]]
        else:
            # Compute Individual Sample Error
            error_x = np.linalg.norm(xyz_label[i] - xyz_predicted[i])
            results[i, :] = [error_x, theta_error[i]]
    # if i % 100 == 0:
    #     print('Iteration:  ', i, '  Error XYZ (m):  ', error_x, '  Error Q (degrees):  ', theta_error[i])

    pos_error = results[:, 0]

    median_result = np.median(results, axis=0)
    median_result_str = '(Median error: {} m and {} degrees.)'.format(median_result[0],
                                                                      median_result[1])
    median_result_return = '{},{}'.format(median_result[0], median_result[1])
    print(median_result_str)

    if should_save_output_data:
        save_output_data(x_predicted, y_predicted, z_predicted, x_label, y_label, z_label, theta_error, training_name,
                         flag2D, save_output_data_file)

    if should_plot_cdf:
        if flag2D:
            plot_cdf_2d(x_predicted, y_predicted, x_label, y_label, theta_error, training_name, save_cdf_position_file,
                        save_cdf_orientation_file)
        else:
            plot_cdf(x_predicted, y_predicted, z_predicted, x_label, y_label, z_label, theta_error, training_name,
                     save_cdf_position_file, save_cdf_orientation_file)
    #### NOTE: for both 3d and 2d case we are currently only interested in 2d trajectory plot. Hence plot_trajectory should plot for both cases
    if should_plot:
        plot_trajectory(x_predicted, y_predicted, x_label, y_label, pos_error, time_vector, training_name,
                        save_trajectory_file)

    return median_result_return


def plot_trajectory(x_predicted, y_predicted, x_label, y_label, pos_error, time_vector, training_name,
                    save_trajectory_file):
    fig = plt.figure(figsize=(7, 7))
    # 	ax3 = plt.scatter(x_predicted[np.argmax(pos_error)], y_predicted[np.argmax(pos_error)], c='r', s=30)
    ax1 = plt.scatter(x_label, y_label, marker="^", c='r')
    ax3 = plt.scatter(x_predicted, y_predicted, marker='x', c='g')
    fig.suptitle('Predicted Pose vs Ground Truth Pose')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')

    plt.legend((ax1, ax3), ('GT testing pose', 'Predicted pose'), loc=2)
    # 	plt.savefig('/ssd/data/fxpal_dataset/posenetDataset/posenet_brendan/posenet_data/results/{}_trajectory.png'.format(training_name), bbox_inches='tight')
    plt.savefig(save_trajectory_file, bbox_inches='tight')


# plt.show()


def save_siamese_data(prediction, x_label, save_output_data_file):
    # 	output_file_plot = '/ssd/data/fxpal_dataset/posenetDataset/posenet_brendan/posenet_data/results/{}_plot_data.h5py'.format(training_name)
    # 	output_file_cdf = '/ssd/data/fxpal_dataset/posenetDataset/posenet_brendan/posenet_data/results/{}_cdf_data.h5py'.format(training_name)

    x_estimate = np.asarray(prediction)

    # x_actual = np.asarray(x_label)

    x_estimate,x_actual = np.squeeze(x_estimate),np.squeeze(x_label)
    #
    # rmse_xyz = x_estimate - x_actual

    with h5py.File(save_output_data_file, 'w') as f:
        x_label_set = f.create_dataset('posenet_x_label', (len(x_label), 1), dtype=np.float)

        x_prediced_set = f.create_dataset('posenet_x_predicted', (len(prediction), len(prediction[0])), dtype=np.float)

        # rmse_xyz_set = f.create_dataset('posenet_rmse_xyz', (len(rmse_xyz), 1), dtype=np.float)

        for i in range(len(x_label)):
            x_label_set[i] = x_label[i]
            x_prediced_set[i] = prediction[i]
            # rmse_xyz_set[i] = rmse_xyz[i]

    f.close()

def save_output_data(x_predicted, y_predicted, z_predicted, x_label, y_label, z_label, theta_error, training_name,
                     flag2D, save_output_data_file):
    # 	output_file_plot = '/ssd/data/fxpal_dataset/posenetDataset/posenet_brendan/posenet_data/results/{}_plot_data.h5py'.format(training_name)
    # 	output_file_cdf = '/ssd/data/fxpal_dataset/posenetDataset/posenet_brendan/posenet_data/results/{}_cdf_data.h5py'.format(training_name)

    if flag2D:
        x_estimate = np.asarray(x_predicted)
        y_estimate = np.asarray(y_predicted)

        x_actual = np.asarray(x_label)
        y_actual = np.asarray(y_label)

        rmse_xy = np.sqrt((x_estimate - x_actual) ** 2 + (y_estimate - y_actual) ** 2)
        rmse_yaw = theta_error

        with h5py.File(save_output_data_file, 'w') as f:
            x_label_set = f.create_dataset('posenet_x_label', (len(x_label), 1), dtype=np.float)
            y_label_set = f.create_dataset('posenet_y_label', (len(y_label), 1), dtype=np.float)

            x_prediced_set = f.create_dataset('posenet_x_predicted', (len(x_predicted), 1), dtype=np.float)
            y_prediced_set = f.create_dataset('posenet_y_predicted', (len(y_predicted), 1), dtype=np.float)

            rmse_xy_set = f.create_dataset('posenet_rmse_xy', (len(rmse_xy), 1), dtype=np.float)
            rmse_yaw_set = f.create_dataset('posenet_rmse_yaw', (len(rmse_yaw), 1), dtype=np.float)

            for i in range(len(x_label)):
                x_label_set[i] = x_label[i]
                y_label_set[i] = y_label[i]
                x_prediced_set[i] = x_predicted[i]
                y_prediced_set[i] = y_predicted[i]
                rmse_xy_set[i] = rmse_xy[i]
                rmse_yaw_set[i] = rmse_yaw[i]

        f.close()


    else:
        x_estimate = np.asarray(x_predicted)
        y_estimate = np.asarray(y_predicted)
        z_estimate = np.asarray(z_predicted)

        x_actual = np.asarray(x_label)
        y_actual = np.asarray(y_label)
        z_actual = np.asarray(z_label)

        rmse_xyz = np.sqrt((x_estimate - x_actual) ** 2 + (y_estimate - y_actual) ** 2 + (z_estimate - z_actual) ** 2)
        rmse_wpqr = theta_error

        with h5py.File(save_output_data_file, 'w') as f:
            x_label_set = f.create_dataset('posenet_x_label', (len(x_label), 1), dtype=np.float)
            y_label_set = f.create_dataset('posenet_y_label', (len(y_label), 1), dtype=np.float)
            z_label_set = f.create_dataset('posenet_z_label', (len(z_label), 1), dtype=np.float)

            x_prediced_set = f.create_dataset('posenet_x_predicted', (len(x_predicted), 1), dtype=np.float)
            y_prediced_set = f.create_dataset('posenet_y_predicted', (len(y_predicted), 1), dtype=np.float)
            z_prediced_set = f.create_dataset('posenet_z_predicted', (len(z_predicted), 1), dtype=np.float)

            rmse_xyz_set = f.create_dataset('posenet_rmse_xyz', (len(rmse_xyz), 1), dtype=np.float)
            rmse_wpqr_set = f.create_dataset('posenet_rmse_wpqr', (len(rmse_wpqr), 1), dtype=np.float)

            for i in range(len(x_label)):
                x_label_set[i] = x_label[i]
                y_label_set[i] = y_label[i]
                z_label_set[i] = z_label[i]
                x_prediced_set[i] = x_predicted[i]
                y_prediced_set[i] = y_predicted[i]
                z_prediced_set[i] = z_predicted[i]
                rmse_xyz_set[i] = rmse_xyz[i]
        #				rmse_wpqr_set[i] = rmse_wpqr[i]

        # 			for i in range(len(x_predicted)):
        # 				x_prediced_set[i] = x_predicted[i]
        # 				y_prediced_set[i] = y_predicted[i]
        # 				z_prediced_set[i] = z_predicted[i]

        f.close()


# 		x_estimate = np.asarray(x_predicted)
# 		y_estimate = np.asarray(y_predicted)
# 		z_estimate = np.asarray(z_predicted)

# 		x_actual = np.asarray(x_label)
# 		y_actual = np.asarray(y_label)
# 		z_actual = np.asarray(z_label)

# 		rmse_xyz = np.sqrt((x_estimate-x_actual)**2+(y_estimate-y_actual)**2 +(z_estimate-z_actual)**2)
# 		rmse_wpqr = theta_error

# 		with h5py.File(output_file_cdf, 'w') as f:
# 			rmse_xyz_set = f.create_dataset('posenet_rmse_xyz', (len(rmse_xyz),1), dtype=np.float)
# 			rmse_wpqr_set = f.create_dataset('posenet_rmse_wpqr', (len(rmse_wpqr),1), dtype=np.float)

# 			for i, val in enumerate(rmse_xyz):
# 				rmse_xyz_set[i] = val

# 			for i, val in enumerate(rmse_wpqr):
# 				rmse_wpqr_set[i] = val
# 		f.close()


def plot_cdf_2d(x_predicted, y_predicted, x_label, y_label, theta_error, training_nam, save_cdf_position_file,
                save_cdf_orientation_file):
    SMALL_SIZE = 20
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 20

    font = {'family': 'serif',
            'weight': 'bold'}

    x_estimate = np.asarray(x_predicted)
    y_estimate = np.asarray(y_predicted)

    x_actual = np.asarray(x_label)
    y_actual = np.asarray(y_label)

    rmse_xy = np.sqrt((x_estimate - x_actual) ** 2 + (y_estimate - y_actual) ** 2)
    rmse_yaw = theta_error

    cdf_xy = []
    cdf_yaw = []

    for i in np.linspace(0, max(rmse_xy), 100):
        cdf_xy.append(float(np.sum(rmse_xy < i)) / float(len(rmse_xy)))

    for i in np.linspace(0, max(rmse_yaw), 100):
        cdf_yaw.append(float(np.sum(rmse_yaw < i)) / float(len(rmse_yaw)))

    ## Position error plot
    fig_xy = plt.figure(figsize=(7, 7))
    plt.plot(np.linspace(0, max(rmse_xy), 100), cdf_xy)

    plt.xlabel("Position Error (m)")
    plt.ylim([0, 1])
    plt.xlim([0, max(rmse_xy)])

    plt.tight_layout()
    plt.savefig(save_cdf_position_file, bbox_inches='tight')
    # 	plt.savefig('/ssd/data/fxpal_dataset/posenetDataset/posenet_brendan/posenet_data/results/{}_position_cdf.png'.format(training_name), bbox_inches='tight')
    # plt.show()

    ## Orientation error plot
    fig_yaw = plt.figure(figsize=(7, 7))
    plt.plot(np.linspace(0, max(rmse_yaw), 100), cdf_yaw)

    plt.xlabel("Orientation Error Yaw (deg)")
    plt.ylim([0, 1])
    plt.xlim([0, max(rmse_yaw)])

    plt.tight_layout()
    plt.savefig(save_cdf_orientation_file, bbox_inches='tight')


# 	plt.savefig('/ssd/data/fxpal_dataset/posenetDataset/posenet_brendan/posenet_data/results/{}_orientation_cdf.png'.format(training_name), bbox_inches='tight')
# plt.show()

def plot_cdf(x_predicted, y_predicted, z_predicted, x_label, y_label, z_label, theta_error, training_name,
             save_cdf_position_file, save_cdf_orientation_file):
    SMALL_SIZE = 20
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 20

    font = {'family': 'serif',
            'weight': 'bold'}
    x_estimate = np.asarray(x_predicted)
    y_estimate = np.asarray(y_predicted)
    z_estimate = np.asarray(z_predicted)

    x_actual = np.asarray(x_label)
    y_actual = np.asarray(y_label)
    z_actual = np.asarray(z_label)

    rmse_xyz = np.sqrt((x_estimate - x_actual) ** 2 + (y_estimate - y_actual) ** 2 + (z_estimate - z_actual) ** 2)
    rmse_wpqr = theta_error

    cdf_xyz = []
    cdf_wpqr = []

    for i in np.linspace(0, max(rmse_xyz), 100):
        cdf_xyz.append(float(np.sum(rmse_xyz < i)) / float(len(rmse_xyz)))

    # for i in np.linspace(0, max(rmse_wpqr), 100):
    # 	cdf_wpqr.append(float(np.sum(rmse_wpqr < i)) / float(len(rmse_wpqr)))

    ## Position error plot
    fig_xyz = plt.figure(figsize=(7, 7))
    plt.plot(np.linspace(0, max(rmse_xyz), 100), cdf_xyz)

    plt.xlabel("Position Error (m)")

    plt.ylim([0, 1])
    plt.xlim([0, max(rmse_xyz)])

    plt.tight_layout()
    plt.savefig(save_cdf_position_file, bbox_inches='tight')
# 	plt.savefig('/ssd/data/fxpal_dataset/posenetDataset/posenet_brendan/posenet_data/results/{}_position_cdf.png'.format(training_name), bbox_inches='tight')

# plt.show()


# ## Orientation error plot
# fig_wpqr = plt.figure(figsize=(7,7))
# plt.plot(np.linspace(0, max(rmse_wpqr), 100), cdf_wpqr)
#
# plt.xlabel("Orientation Error (deg)")
#
# plt.ylim([0, 1])
# plt.xlim([0, max(rmse_wpqr)])
#
# plt.tight_layout()
# plt.savefig(save_cdf_orientation_file, bbox_inches='tight')
# 	plt.savefig('/ssd/data/fxpal_dataset/posenetDataset/posenet_brendan/posenet_data/results/{}_orientation_cdf.png'.format(training_name), bbox_inches='tight')

# plt.show()
