import helper
import posenet
import tensorflow as tf
import numpy as np
import sys
from tensorflow import keras
from helper import parse_function
# from keras.optimizers import Adam
# from keras.preprocessing.image import ImageDataGenerator
# from keras import callbacks
import random
from PIL import Image
import cv2
import h5py
import os.path
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from helper import initialise_callbacks, custom_generator, customPredictGenerator_tf2
from helper import PreprocessingData, readDataFromFile
import math
from random import uniform
from matplotlib import pyplot as plt
import shutil
from tensorflow.keras.utils import multi_gpu_model
import pydot
import graphviz
# from keras import backend as K
# from vis.visualization import saliency
# import vis
# from vis.utils import utils
import argparse
import time
import progressbar
from scipy.spatial.transform import Rotation as R

# from tensorflow.keras.backend import set_session
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
# from keras.backend.tensorflow_backend import set_session
import re

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 1.0
# set_session(tf.compat.v1.Session(config=config))

random.seed(datetime.now())

import time
import gc
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
# tf.enable_eager_execution()

class Train():

    def __init__(self, args):
        
        # data paths
        self.input_data_dir = args.input_data_dir
        self.output_data_dir = args.output_data_dir
        self.training_data = args.training_data
        self.validation_data = args.validation_data
        self.testing_data = args.testing_data
        self.pretrained_model_path = args.pretrained_model
        
        # Hyper paramters 
        self.batch_size = args.batch_size
        self.valid_size = args.valid_size
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.beta = args.beta
        self.decay = uniform(-5, -6) # not used as of now
        
        # training name 
        self.training_name_suffix = args.training_name_suffix
        self.training_name        = args.training_name
        
        # gpu machine to use
        self.GPU_FLAG = args.GPU_FLAG
        
        # multiple GPU flag
        self.MULTIPLE_GPU_GLAG = args.MULTIPLE_GPU_GLAG
        
        #tensorboard and model checkpoint saving paths
        self.tensorboard_event_logs = args.tensorboard_event_logs
        self.csv_logs = args.csv_logs
        self.checkpoint_weights = args.checkpoint_weights
        self.weights = args.weights #location where final weights are stored
        self.training_data_info = args.training_data_info
        
        #loading data from different location num_img_siamese
        self.diff_location = args.diff_location
        # self.num_img_siamese = args.num_img_siamese
        self.count = args.count
        self.date_time = None
        
        # retraing from checkpoint
        self.retrain = args.retrain
        self.train_2d = args.train_2d
        self.train2dflag = False
        
        #function calls on initialization
        self.setupIOFiles()
        self.setTrainingParams()
        
        if self.retrain:
            print('model will be retrained from the following model name: {}'. format(output_preprocess_file))
            file_path_valid = Path(self.csv_log_file)
            assert not csv_log_path.is_file(), \
            'there is no pretrained model info from which model can be loaded... please check the path'
            
    def contrastive_loss(self,y_true, y_pred):
        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
        margin = 1
        # y_pred = tf.cast(y_pred, dtype=tf.int64)
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
    def accuracy(self,y_true, y_pred):
        '''Compute classification accuracy with a fixed threshold on distances.
        '''
        return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))	

    def angleDifference(self, q1, q2):

        angle = 0
        r1 = R.from_quat(q1)
        r2 = R.from_quat(q2)
        r1 = r1.as_matrix()
        r2 = r2.as_matrix()
        if abs(np.trace(np.dot(r1.transpose(),r2))-3) < 0.0000000001:
            angle = 0
        else:
            angle = math.acos((np.trace(np.dot(r1.transpose(),r2))-1)/2) * 180 / np.pi
        return angle
    def distDifference(self, p1, p2):
        tmp = p1 - p2
        return np.linalg.norm(tmp)
    def getImageSimilarity(self, pose1, pose2, threshold_anlge, threshold_dist):

        distDif = self.distDifference(pose1[0:3], pose2[0:3])
        angleDif  = self.angleDifference(pose1[3:], pose2[3:])
        if (angleDif < threshold_anlge) and (distDif < threshold_dist):
            return True
        else:
            return False
    def train_pairs_generator(self,x_train,pose,threshold_angle,threshold_dist, num_data,batchsize): #num_size: training data size
        while 1:
            pairs_1 = []
            pairs_2 = []
            labels = []

            k = 0
            while k<batchsize:
                i = random.randrange(0, num_data-1)

                while True:
                    delta = random.randrange(-30, 30)
                    while (i + delta < 0 or i + delta >= num_data):
                        delta = random.randrange(-num_data, num_data)

                    ind = i + delta
                    bool_similar = self.getImageSimilarity(pose[i], pose[ind], threshold_angle, threshold_dist)
                    if bool_similar==True:
                        pairs_1 += [x_train[i]]
                        pairs_2 += [x_train[ind]]
                        labels += [1.0]
                        break

                while True:
                    ind = random.randrange(0, num_data - 1)
                    bool_similar = self.getImageSimilarity(pose[i], pose[ind], threshold_angle, threshold_dist)
                    if bool_similar==False:
                        pairs_1 += [x_train[i]]
                        pairs_2 += [x_train[ind]]
                        labels += [0.0]
                        break

                k = k+1
            pairs_1 = np.array(pairs_1)
            pairs_2 = np.array(pairs_2)
            labels = np.array(labels)
            # labels  = np.array(labels,dtype=np.float)
            yield [pairs_1,pairs_2], labels
    def valid_pairs_generator(self,x_train,pose,threshold_angle,threshold_dist, num_data,batchsize): #num_size: training data size
        pairs_1 = []
        pairs_2 = []
        labels = []

        k = 0
        while k < batchsize:
            i = random.randrange(0, num_data - 1)

            while True:
                delta = random.randrange(-30, 30)
                while (i + delta < 0 or i + delta >= num_data):
                    delta = random.randrange(-num_data, num_data)

                ind = i + delta
                bool_similar = self.getImageSimilarity(pose[i], pose[ind], threshold_angle, threshold_dist)
                if bool_similar == True:
                    pairs_1 += [x_train[i]]
                    pairs_2 += [x_train[ind]]
                    labels += [1.0]
                    break

            while True:
                ind = random.randrange(0, num_data - 1)
                bool_similar = self.getImageSimilarity(pose[i], pose[ind], threshold_angle, threshold_dist)
                if bool_similar == False:
                    pairs_1 += [x_train[i]]
                    pairs_2 += [x_train[ind]]
                    labels += [0.0]
                    break

            k = k + 1
        pairs_1 = np.array(pairs_1)
        pairs_2 = np.array(pairs_2)
        labels = np.array(labels)
        # labels  = np.array(labels,dtype=np.float)
        return [pairs_1,pairs_2], labels

    def euc_loss3x(self, y_true, y_pred):		
# 		lx = K.sqrt(K.sum(K.square(y_true[:, :] - y_pred[:, :]), axis=1, keepdims=True))
        lx = K.sum(K.square(y_true[:, :] - y_pred[:, :]), axis=None)
        return 1 * lx

    def euc_loss3q(self, y_true, y_pred):
# 		lq = K.sqrt(K.sum(K.square(y_true[:, :] - y_pred[:, :]), axis=1, keepdims=True))
        lq = K.sum(K.square(y_true[:, :] - y_pred[:, :]), axis=None)
        return self.beta * lq
    
    def setupIOFiles(self):
        self.date_time = datetime.now().strftime('%Y_%m_%d_%H_%M')

# 		################################
# 		# Inputs: create input paths
# 		#################################
        self.input_train_tfrecord = []
        self.input_validation_tfrecord = []
        self.input_test_tfrecord = []
        for file_name in self.training_data:
            self.input_train_tfrecord.append(self.input_data_dir + str(file_name) + '.tfrecords')
        for file_name in self.validation_data:
            self.input_validation_tfrecord.append(self.input_data_dir + str(file_name) + '.tfrecords')
        for file_name in self.testing_data:
            self.input_test_tfrecord.append(self.input_data_dir + str(file_name) + '.tfrecords')
        
        print('training data list: {}'.format(self.input_train_tfrecord))
        print('validation data list: {}'.format(self.input_validation_tfrecord))
        print('testing data list: {}'.format(self.input_test_tfrecord))
        
        
    def setTrainingParams(self):
        
        #################################
        # Output: Do not edit
        #################################
        self.training_name = '{}_train_{}'.format(self.training_name,
                                                  self.training_name_suffix)
        # self.training_name = '{}_train_{}'.format(self.date_time,
        #                                           self.training_name_suffix)

        self.training_info = '{} (posenet): Learning rate = {}. Batch size = {}. Beta = {}' \
            .format(self.training_name, self.lr, self.batch_size,self.valid_size, self.beta)

        self.tensor_log_file = self.output_data_dir + self.tensorboard_event_logs + '{}_posenet_tb_logs'.format(self.training_name)
        self.csv_log_file = self.output_data_dir + self.csv_logs + '{}_posenet_csv.log'.format(self.training_name)
        self.weights_info_file = self.output_data_dir + self.weights + 'weights_info_posenet.csv'
        self.checkpoint_weights_file = self.output_data_dir +  self.checkpoint_weights + '{}_posenet.h5'.format(self.training_name)
        self.weights_out_file = self.output_data_dir + self.weights + '{}_posenet_weights.h5'.format(self.training_name)
        self.output_preprocess_file = self.output_data_dir  + self.training_data_info + '{}_posenet_data.h5py'.format(self.training_name)
        self.model_json_file = self.output_data_dir + self.weights + '{}_model.json'.format(self.training_name)
        self.threshold_angle = args.threshold_angle
        self.threshold_dist = args.threshold_dist
        csv_log_path = Path(self.csv_log_file)
        weights_out_path = Path(self.weights_out_file)


        assert not os.path.isdir(self.tensor_log_file), \
            "Log file already exist. Ensure" \
            " that the name you have entered" \
            " is unique. Names set in " \
            " TensorBoard callback."

        assert not csv_log_path.is_file(), \
            "Log file already exist. Ensure" \
            " that the name you have entered" \
            " is unique. Names set in " \
            " CSVLogger callback."

        if os.path.exists(self.weights_out_file):
            os.remove(self.weights_out_file)
        assert not weights_out_path.is_file(), \
            "Weight file already exists. Ensure" \
            " that the name you have entered" \
            " is unique. Name set in " \
            " model.save_weights()."
    

    def write_training_info_to_csv():
        with open(self.weights_info_file, 'a') as f:
            f.write(self.training_info)
            f.write('\n')
        f.close()
        
    def train(self):

# 			#################################
# 			# Inputs: Edit these
# 			#################################
            # Reading data file
            dataset_tr = tf.data.TFRecordDataset(self.input_train_tfrecord,compression_type='GZIP')
            dataset_val = tf.data.TFRecordDataset(self.input_validation_tfrecord,compression_type='GZIP')
                
            
            if self.retrain:  ### only works for 3D can be made to work for 2D but needs refactoring the code
                
                # Get data from the previous training run
# 				data_dict = readDataFromFile(self.output_data_dir \
# 											 + 'training_data_info/{}_posenet_data.h5py'.format(previous_training_name))
                #if we are retraining from a certain check point then default values loaded using tf Flags will be overwritten by the data from 
                # output_preprocess_file
                data_dict = readDataFromFile(self.output_preprocess_file)
    
                self.lr = data_dict['posenet_learning_rate'] * 0.1
                self.batch_size = data_dict['posenet_batch_size']
                self.valid_size = data_dict['posenet_valid_size']
                self.decay = data_dict['posenet_decay']
                self.beta = data_dict['posenet_beta']
                data_size = data_dict['general_data_size']
                validation_data_size = data_dict['general_validation_data_size']
                sample_size = min(4000, data_dict['general_data_size']) # to get mean image info


                model = posenet.create_posenet_keras(self.pretrained_model_path, True)  # GoogLeNet (Trained on Places)
                adam = Adam(lr=self.lr, clipvalue=1.5)  # not sure if clipvalue is even considered
                model.compile(optimizer=adam,
                              loss={'cls3_fc_pose_xyz': self.euc_loss3x,
                                    'cls3_fc_pose_wpqr': self.euc_loss3q})
                model.summary()
                
                try:
                    model.load_weights(data_dict['posenet_checkpoint_weight_file'])
                except:
                    model.load_weights(data_dict['posenet_weight_file'])

            else:
            
                model = posenet.create_siamese_keras(self.pretrained_model_path, True)  # GoogLeNet (Trained on Places)
                

                ################################################################################
                # MITESH:
                # decay should be used rather then clipvalue. Decay progressively reduces the learning rate
                # http://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/
                # https://keras.io/optimizers/#adam
                ################################################################################
                adam = keras.optimizers.Adam(lr=self.lr, clipvalue=1.5)  # not sure if clipvalue is even considered
                model.compile(loss=self.contrastive_loss, optimizer=adam, metrics=[self.accuracy])
            model.summary()
            # save model graph 
            model_json = model.to_json()
            if os.path.exists(os.path.dirname(self.model_json_file)):
                shutil.rmtree(os.path.dirname(self.model_json_file))
                os.makedirs(os.path.dirname(self.model_json_file))
            with open(self.model_json_file, 'w') as f:
                    f.write(model_json)
            f.close()
            # Setup callbacks
            callback_list = initialise_callbacks(self.checkpoint_weights_file, self.tensor_log_file, self.csv_log_file, self.batch_size, self.lr, self.num_epochs)
            
            ### Dataset API
            dataset_train = dataset_tr.map(parse_function)
            dataset_validation = dataset_val.map(parse_function)
                
            
            #calculate image mean
            preprocessing_data = PreprocessingData()
# 			image_mean = preprocessing_data.calculate_sample_mean(sample_size, self.input_train_h5py)
            
            # num_trainimg = 0
            # xtrain = []
            # zone   = []         #   Label zone for each training data. E.g. [1 1 1 2 2 3 3]
            # zone_ind = []       #   Each zone covers from. E.g. [0 2;3 4;5 6]
            # ind,tmp,i = 0,1,0
            # for features in dataset_train:
            #     if(i == 0):
            #         zone_ind = [[i,0]]
            #         if(features[1].numpy()[0] != tmp):
            #             print('Error: the index for class should start from 1')
            #             sys.exit()
            #     num_trainimg = num_trainimg + 1
            #     xtrain += [features[0].numpy()]
            #     zone   += [int(features[1].numpy()[0])]
            #     if(features[1].numpy()[0] != tmp):
            #         tmp = features[1].numpy()[0]
            #         zone_ind[-1][1] = int(i-1)
            #         zone_ind += [[i,0]]
            #     i = i + 1
            # zone_ind[-1][1] = i-1
            num_trainimg = 0
            xtrain = []
            pose   = []         #   Label zone for each training data. E.g. [1 1 1 2 2 3 3]
            for features in dataset_train:
                num_trainimg = num_trainimg + 1
                xtrain += [features[0].numpy()]
                pose   += [features[1].numpy()]
            te_pairs, te_y = self.valid_pairs_generator(xtrain,pose,self.threshold_angle,self.threshold_dist, num_trainimg,self.valid_size)

            # create training info dict which will be saved for either retraining or testing
            data_dict = {'image_mean': self.lr,#image_mean,
                         'posenet_learning_rate': self.lr,
                         'posenet_batch_size': self.batch_size,
                         'posenet_valid_size': self.valid_size,
                         'posenet_decay': self.decay,
                         'posenet_epochs': self.num_epochs,
                         'posenet_beta': self.beta,
                         'posenet_dataset_train': self.input_train_tfrecord,
                         'posenet_dataset_validation': self.input_validation_tfrecord,
                         'posenet_dataset_test': self.input_test_tfrecord,
                         'posenet_checkpoint_weight_file': self.checkpoint_weights_file,
                         'posenet_weight_file': self.weights_out_file}#,
                        #  'general_data_size': data_size,
                        #  'general_validation_data_size': validation_data_size}
            
            print('data dict: {}'.format(data_dict))
            
            preprocessing_data.writeDataToFile(self.output_preprocess_file,
                                               data_dict)
            
            print("#############Training details###############:\n Training name = {} and hyper parameters are: batch_size: {},valid_size: {}, lr: {}, beta: {} ".format(self.training_name, self.batch_size,self.valid_size, self.lr, self.beta))
            
            ## write training name to csv file where results can be stored in future
            with open(self.weights_info_file, 'a') as f:
                f.write(self.training_info)
                f.write('\n')
            f.close()
            # Fits the model. steps_per_epoch is the total number of batches to
            # yield from the generator before declaring one epoch finished and
            # starting the next. This is important since batches will be yielded
            # infinitely.
            t_start = time.time()
            print(t_start)

            # Training
            version = tf.__version__
            print('version: {} and type: {}: {}'.format(version, type(version), int(version.split('.')[0])))
            epoch_per_each_training = 2
            cycles = math.floor(self.num_epochs/ epoch_per_each_training)
            previous_loss = 99999999999999999999999

            if (int(version.split('.')[0]) == 1 ):
                print('!!  Please use TENSORFLOW 2.0 !!')
            else:
                print('TENSORFLOW 2.0.0 SELECTED')

            model.fit(self.train_pairs_generator(xtrain,pose,self.threshold_angle,self.threshold_dist, num_trainimg,self.batch_size),
            # model.fit(self.train_pairs_generator(xtrain,zone,zone_ind, num_trainimg,self.batch_size),
                                  epochs=self.num_epochs,
                                  steps_per_epoch=10,
                                  validation_data=(te_pairs, te_y),
                                  verbose=1,
                                  callbacks=ClearMemory())
                                  #validation_data=([te_pairs[0, :,:,:,:], te_pairs[1, :,:,:,:]], te_y))
            model.save_weights(self.weights_out_file, overwrite=True)
        
            print("Training name = {} and hyper parameters are: batch_size: {}, lr: {}, beta: {} ".format(self.training_name, self.batch_size, self.lr, self.beta))
            self.count += 1

            t_end = time.time()
            print(t_end)
            print('time taken: {}'.format(t_end-t_start))

class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        K.clear_session()
class Test():
    def __init__(self, args):
        
        # data paths
        self.input_data_dir = args.input_data_dir
        self.output_data_dir = args.output_data_dir
        self.training_data = args.training_data
        self.validation_data = args.validation_data
        self.testing_data = args.testing_data
        
        # training, validation and testing data
        self.input_train_tfrecord = []
        for file_name in self.training_data:
            self.input_train_tfrecord.append(self.input_data_dir + str(file_name) + '.tfrecords')
# 		self.input_validation_tfrecord = self.input_data_dir + self.validation_data
# 		testing_data = []
# 		for file_name in self.testing_data:
# 			testing_data.append(self.input_data_dir + str(file_name) + '.tfrecords')
# 		self.testing_data = testing_data
        self.testing_data = [self.input_data_dir + str(file_name) + '.tfrecords' for file_name in self.testing_data]
        print('testing data: {}'.format(self.testing_data))
        self.testing_name = args.testing_name_suffix
        
        self.should_plot_cdf = True
        self.should_plot = True
        self.should_save_output_data = True
        
        self.training_info_file = self.output_data_dir  + 'training_data_info/{}_posenet_data.h5py'.format(self.testing_name)
        self.weights_info_file = self.output_data_dir + 'weights/weights_info_posenet.csv'
        self.save_cdf_position_file = self.output_data_dir + 'cdf/{}_position_cdf.png'.format(self.testing_name)
        self.save_cdf_orientation_file = self.output_data_dir + 'cdf/{}_orientation_cdf.png'.format(self.testing_name)
        self.save_trajectory_file = self.output_data_dir + 'cdf/{}_trajectory.png'.format(self.testing_name)
        self.save_output_data_file = self.output_data_dir + 'cdf/{}_output.h5py'.format(self.testing_name)
        self.save_op_per_seq = self.output_data_dir + 'cdf/{}_each_sequence.csv'.format(self.testing_name)
        
        self.diff_location = args.diff_location
        # self.num_img_siamese = args.num_img_siamese
        self.num_candidate = args.num_candidate
        self.test_2d = args.test_2d
        self.testing_2d_flag = False

        self.threshold_angle = args.threshold_angle
        self.threshold_dist = args.threshold_dist

    def test(self):
        
        
        print('###################### IN TESTING')
        
        # Get data from training run
        data_dict = readDataFromFile(self.training_info_file)
        
        # Test model
        if self.test_2d:
            self.testing_2d_flag = True
            model = posenet.create_posenet_keras_2d()
        else:
            model = posenet.create_posenet_keras()
        
        if self.diff_location:
            batch_size = 64
            model_file_path = self.output_data_dir  + 'weights/{}.h5'.format(self.testing_name)
            
            # Reading data file
            self.input_test_tfrecord = []
            
            dataset = tf.data.TFRecordDataset(self.testing_data,compression_type='GZIP')
            
            print('model file loading: {}'.format(self.testing_name))
            model.load_weights(model_file_path)
            
            
        else:
            try:
                print('model file loading: {}'.format(data_dict['posenet_checkpoint_weight_file']))
                model.load_weights(data_dict['posenet_checkpoint_weight_file'])
            except:
                model.load_weights(data_dict['posenet_weight_file'])
        
            
            # Reading data file
            dataset = tf.data.TFRecordDataset(data_dict['posenet_dataset_test'],compression_type='GZIP')
            batch_size=data_dict['posenet_batch_size']
            valid_size=data_dict['posenet_valid_size']

        
        ### Dataset API
        dataset_test = dataset.map(parse_function)
        dataset_test = dataset_test.batch(batch_size)
        dataset_test = dataset_test.batch(valid_size)
        dataset_test = dataset_test.prefetch(5)

        
        median_result = customPredictGenerator_tf2(model, dataset_test,
                                                      batch_size,
                                                      training_name=self.testing_name,
                                                      should_plot=self.should_plot,
                                                      should_plot_cdf=self.should_plot_cdf,
                                                      should_save_output_data=self.should_save_output_data,
                                                      save_cdf_position_file=self.save_cdf_position_file,
                                                      save_cdf_orientation_file=self.save_cdf_orientation_file,
                                                      save_trajectory_file=self.save_trajectory_file,
                                                      save_output_data_file=self.save_output_data_file,
                                                      flag2D = self.testing_2d_flag
                                                    )

        shutil.copyfile(self.weights_info_file, self.weights_info_file + '.tmp')

        with open(self.weights_info_file, 'r') as f:
            with open(self.weights_info_file + '.tmp', 'w') as g:
                for line in f:
                    if (self.testing_name in line) and ('Median error' not in line):
                        g.write('{}. {}\n'.format(line.rstrip('\n'), median_result))
                    else:
                        g.write(line)

        shutil.copyfile(self.weights_info_file + '.tmp',self.weights_info_file)
    
    def test_all_sequences(self):
        # Get data from training run
        data_dict = readDataFromFile(self.training_info_file)

        # Test model
        if self.test_2d:
            self.testing_2d_flag = True
            model = posenet.create_posenet_keras_2d()
        else:
            model =  posenet.create_siamese_keras()
        
        try:
            print('model file loading: {}'.format(data_dict['posenet_weight_file']))
            model.load_weights(data_dict['posenet_weight_file'])
        except:
            model.load_weights(data_dict['posenet_checkpoint_weight_file'])



        # #   Modified by Jingwei, build a searching dict
        # dataset_tr = tf.data.TFRecordDataset(self.input_train_tfrecord,compression_type='GZIP')
        # dataset_train = dataset_tr.map(parse_function)
        # num_trainimg = 0
        # xtrain = []
        # zone   = []         #   Label zone for each training data. E.g. [1 1 1 2 2 3 3]
        # zone_ind = []       #   Each zone covers from. E.g. [0 2;3 4;5 6]
        # ind,tmp,i = 0,1,0
        # for features in dataset_train:
        #     if(i == 0):
        #         zone_ind = [[i,0]]
        #         if(features[1].numpy()[0] != tmp):
        #             print('Error: the index for class should start from 1')
        #             sys.exit()
        #     num_trainimg = num_trainimg + 1
        #     xtrain += [features[0].numpy()]
        #     zone   += [int(features[1].numpy()[0])]
        #     if(features[1].numpy()[0] != tmp):
        #         tmp = features[1].numpy()[0]
        #         zone_ind[-1][1] = int(i-1)
        #         zone_ind += [[i,0]]
        #     i = i + 1
        #tf.keras.utils.plot_model(model, to_file='multilayer_perceptron_graph.png')
        pose_train = []
        dataset_tr = tf.data.TFRecordDataset(self.input_train_tfrecord, compression_type='GZIP')
        dataset_train = dataset_tr.map(parse_function)
        for features in dataset_train:
            pose_train += [features[1].numpy()]

        based_model = tf.keras.Model(inputs=model.layers[2].input,
                                    outputs=model.layers[2].get_layer('cls3_fc_pose_xyz').output)
        # zone_dict = []
        # for i in range(len(zone_ind)):
        #     descriptor_tmp = []
        #     for j in range(self.num_img_siamese):
        #         ind = zone_ind[i][0] + j*round((zone_ind[i][1]-zone_ind[i][0])/(self.num_img_siamese-1))
        #         input_tmp = np.expand_dims(xtrain[ind],axis=0)
        #         descriptor = based_model.predict(input_tmp)
        #         descriptor_tmp += [descriptor]
        #     zone_dict += [descriptor_tmp]




        # interate over each sequence Modified by Jingwei 3 Aug
        number_of_sequence = 2
        aggregated_results = []
        batch_size=data_dict['posenet_batch_size']

        #   Get training data descriptions
        dataset_train = dataset_train.batch(batch_size)
        dataset_train = dataset_train.prefetch(5)
        train_results = based_model.predict(dataset_train)
        for idx in range(number_of_sequence):
            temp = []
            input_data_file = self.input_data_dir  + str(idx) + '.tfrecords'
            print('##########testing file########### : {}'.format(input_data_file))
            dataset = tf.data.TFRecordDataset(input_data_file,compression_type='GZIP')

            pose_test = []
            dataset_test = dataset.map(parse_function)
            for features in dataset_test:
                pose_test += [features[1].numpy()]
            
            
            # break sequence for plotting
            temp = self.save_cdf_position_file.split('.')
            save_cdf_position_file = temp[0] + '_' + str(idx) + '.png'
            temp = self.save_cdf_orientation_file.split('.')
            save_cdf_orientation_file = temp[0] + '_' + str(idx) + '.png'
            temp = self.save_trajectory_file.split('.')
            save_trajectory_file = temp[0] + '_' + str(idx) + '.png'
            # Modified by jingwei 32 July 2019
            temp = self.save_output_data_file.split('.')
            save_output_data_file = temp[0] + '_' + str(idx) + '.h5py'

            # Training
            version = tf.__version__
            print('version: {} and type: {}: {}'.format(version, type(version), int(version.split('.')[0])))
            if (int(version.split('.')[0]) == 1):
                print('TENSORFLOW 1.0.0 SELECTED')
            else:
                print('TENSORFLOW 2.0.0 SELECTED')

            #  #   Generate result
            # zone_groudtrtruth = []
            # for features in dataset_test:
            #     zone_groudtrtruth   += [int(features[1].numpy()[0])] 


            # #   Modified by Jingwei: Predict
            # dataset_test = dataset_test.batch(batch_size)
            # dataset_test = dataset_test.prefetch(5)
            # test_results = based_model.predict(dataset_test)
            # prediction = []  

            # p = progressbar.ProgressBar()
            # error = 0
            # results = np.zeros((len(test_results), 1), dtype=float)
            # for i in p(range(len(test_results))):   #   For every image
            #     #   Lookup the dictionary
            #     ind_zone = 1
            #     score = np.linalg.norm(test_results[i] - zone_dict[0])
            #     for j in range(1,len(zone_dict)):
            #         temp = np.linalg.norm(test_results[i] - zone_dict[j]) 
            #         if(temp < score):
            #             ind_zone = j + 1
            #             score = temp
            #     prediction += [ind_zone]
            #     time.sleep(0.01)
            #     error_x = np.linalg.norm(ind_zone - zone_groudtrtruth[i])
            #     results[i] = error_x
            # median_result = np.median(results, axis=0)  
            
            median_result = helper.customPredictSiamese_tf2(based_model, dataset_test,
                                                            train_results,
                                                          batch_size,pose_train,pose_test,
                                                          self.threshold_angle,
                                                          self.threshold_dist,
                                                          self.num_candidate,
                                                          save_output_data_file=save_output_data_file
                                                        )

            

            # median_result = helper.customPredictGenerator_tf2(model, dataset_test,
            #                                               batch_size,
            #                                               training_name=self.testing_name,
            #                                               should_plot=self.should_plot,
            #                                               should_plot_cdf=self.should_plot_cdf,
            #                                               should_save_output_data=self.should_save_output_data,
            #                                               save_cdf_position_file=save_cdf_position_file,
            #                                               save_cdf_orientation_file=save_cdf_orientation_file,
            #                                               save_trajectory_file=save_trajectory_file,
            #                                               save_output_data_file=save_output_data_file,
            #                                               flag2D = self.testing_2d_flag
            #                                             )
            temp.append(input_data_file)
            temp.append(median_result)
            aggregated_results.append(temp)
        ## write training name to csv file where results can be stored in future
        with open(self.save_op_per_seq, 'w') as f:
            for value in aggregated_results:
                f.write('%s\n'%value)
        
    


def main_routine(args):
    print("flags argument: {}".format(args))
    ###run training
    if (args.train  and args.test_all_sequence):
        print('Doing both training and testing')
        train = Train(args)
        train.train()
        train.weights_out_file
        test = Test(args)
        test.test_all_sequences()
        exit(0)		
    elif (args.train):
        if (args.train_2d):
            print('training for 2d settings')
        else:
            print('training for 3d settings')
        train = Train(args)
        train.train()
    elif (args.test):
        if (args.test_2d):
            print('testing for 2d settings')
        else:
            print('testing for 3d settings')
        test = Test(args)
        test.test()
    # elif (args.test_all):
    # 	test = Test(args)
    # 	test.testAll()
    elif(args.test_all_sequence):
        if (args.test_2d):
            print('testing each sequence for 2d settings')
        else:
            print('testing each sequence for 3d settings')
        test = Test(args)
        test.test_all_sequences()

    else:
        print('No flags selected for training or testing')
        

# This function is needed to read bool values from the command line
# using type=bool is not sufficient!!
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

        
if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser()

    # Flags needed for Training or retraining
    parser.add_argument('--retrain', default=False, help='if retraining from a check point', action='store')
    parser.add_argument('--train', default=True, help='train normal to estimate 7 paramters', action='store')
    parser.add_argument('--sample_train', default=False, help='only use to check the data flow and tensor graph', action='store')
    parser.add_argument('--train_2d', default=False, help='make the pose problem a 2d problem, so only train on x,y, yaw', action='store')

    #IO file locations Modify here dataset_name
    parser.add_argument('--output_data_dir', default='/media/jingwei/Lenovo/dataset/NVIDIA/indoor_dataset/warehouse_dark/posenet_training_output/', help='location of outputs', action='store')
    # Modified by Jingwei 19 July
    parser.add_argument('--input_data_dir', default='/media/jingwei/Lenovo/dataset/NVIDIA/indoor_dataset/warehouse_dark/posenetDataset/tfrecord2/', help='location of training tfrecords file', action='store')
    parser.add_argument('--training_data', default=[0],
                        help=' list fo training tfrecord file', action='store')
    parser.add_argument('--validation_data', default=[0], help='location of validation h5py file',
                        action='store')
    parser.add_argument('--testing_data', default=[1], help='location of testing h5py file', action='store')
    #parser.add_argument('--input_data_dir', default='/ssd/data/fxpal_dataset/posenetDataset/tfrecord2/', help='location of training tfrecords file', action='store')
#	parser.add_argument('--training_data', default=[4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], help=' list fo training tfrecord file', action='store')
#	parser.add_argument('--validation_data', default=[0, 1, 2, 5, 13], help='location of validation h5py file', action='store')
#	parser.add_argument('--testing_data', default=[3, 11, 4, 12], help='location of testing h5py file', action='store')
# 	parser.add_argument('--training_data', default=[10], help=' list fo training tfrecord file', action='store')
# 	parser.add_argument('--validation_data', default=[11], help='location of validation h5py file', action='store')
# 	parser.add_argument('--testing_data', default=[11], help='location of testing h5py file', action='store')
    parser.add_argument('--pretrained_model', default='./posenet.npy', help='pretrained googlenet model on places dataset', action='store')

    ## training and testing names Modify here dataset_name
    parser.add_argument('--training_name', default='chess', help='name of training', action='store')
    parser.add_argument('--training_name_suffix', default='_siamese_FXPAL', help='name of training', action='store')
    parser.add_argument('--training_name_for_retraining', default='data_training_w_halloweendecor_tf_final', help='name of trained model that needs to be loaded for retraining', action='store')

    #default hyper paramters  Jingwei: batch_size:64 num_epochs 180/300  lr 0.001/0.0001 beta 120/1
    parser.add_argument('--batch_size', default=16, help='num of samples in a batch', type=int, action='store')
    parser.add_argument('--valid_size', default=512, help='num of samples in a batch', type=int, action='store')
    parser.add_argument('--num_epochs', default=300, help='num of epochs to train', type=int, action='store')
    parser.add_argument('--lr', default=0.0001, help='initial learning rate', type=float, action='store')
    parser.add_argument('--beta', default=1, help='beta value', type=int, action='store')

    #machine on which training is running
    # Modified by Jingwei 17 Sep
    parser.add_argument('--GPU_FLAG', default='gpu01', help='which GPU machine to use', action='store' )
    ## multiple GPU training
    parser.add_argument('--MULTIPLE_GPU_GLAG', default=False, help='which GPU machine to use', action='store' )

    # traning info file, check point and model saving location
    parser.add_argument('--tensorboard_event_logs', default='logs/tensorboard_logs/', help='tensorboard event logs', action='store')
    parser.add_argument('--csv_logs', default='logs/csv_logs/', help='csv logs', action='store')
    parser.add_argument('--checkpoint_weights', default='weights/checkpoint_weights/', help='model checkpoint storage location', action='store')
    parser.add_argument('--weights', default='weights/', help='final weight location storage', action='store')
    parser.add_argument('--training_data_info', default='training_data_info/', help='stores information such as hyper parameter, model and data location etc', action='store')



    #Other flags
    ### by default regular training and testing will run which is estimate 7 parameters (xyz, wpqr)
    parser.add_argument('--diff_location', default=False, help='test model with location specified and not from the location in training_data_info.h5py', action='store')
    # run more then one training with different hyper paramters
    parser.add_argument('--count', default=0, help='number of training to run with different hyper parameters', action='store')
    #   Add by Jingwei. Number of images for Siamese comparison
    # parser.add_argument('--num_img_siamese', default=5, help='Number of images for Siamese comparison',
    #                     action='store')
    #   Modified by Jingwei 14 March Jingwei
    parser.add_argument('--num_candidate', default=5, help='Number of candidate zones from the Siamese comparison',
                        action='store')
    parser.add_argument('--threshold_angle', default=20, help='Threshold of angle to segment image pairs',
                        action='store')
    parser.add_argument('--threshold_dist', default=0.4, help='Threshold of angle to segment image pairs',
                        action='store')

    ##################
    #Flags needed for testing
    parser.add_argument('--test', default=False, help='test normal to estimate 7 paramters', action='store')
    parser.add_argument('--test_2d', default=False, help='test specified model by testing_name string', action='store')
    parser.add_argument('--test_all', default=False, help='test all models listed in the CSV file', action='store')
    parser.add_argument('--test_all_sequence', default=True, help='test over all sequences on the database', action='store')
#2D model
# 	parser.add_argument('--testing_name_suffix', default='2018_11_09_23_58_train_train_data_training_largedataset_w_halloweendecor_rotated_2d_BN_more_oct2018', help='name of model to be tested', action='store')
#3D model

    if(parser.get_default('train')  and parser.get_default('test_all_sequence')):
        parser.add_argument('--testing_name_suffix', default='{}_train_{}'.format(parser.get_default('training_name'),parser.get_default('training_name_suffix')), action='store')
    else:
        #   Modified by Jingwei 14 March Jingwei  Modify here dataset_name
        parser.add_argument('--testing_name_suffix', default='chess_siamese_FXPAL', help='name of model to be tested', action='store')
    args = parser.parse_args()
    
    main_routine(args)



