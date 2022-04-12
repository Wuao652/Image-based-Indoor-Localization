import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, concatenate, Flatten, Dropout, BatchNormalization, Input
import numpy as np
import h5py
import math
import run_siamese
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda


beta = None

def euc_loss3x(y_true, y_pred):
	lx = keras.backend.sum(tf.square(y_true[:, :] - y_pred[:, :]), axis=1, keepdims=True)
	return 1 * lx


def euc_loss3q(y_true, y_pred):
	lq = keras.backend.sum(tf.square(y_true[:, :] - y_pred[:, :]), axis=1, keepdims=True)
	return beta * lq


def mseLossKeras(y_true, y_pred):
	global beta
	diff = tf.square(tf.subtract(y_pred, y_true))
	pos, ori = tf.split(diff, [3, 4], 2)
	#ori *= beta
	mzsum = tf.reduce_sum(pos, axis=2) + beta * tf.reduce_sum(ori, axis=2)
	loss = tf.reduce_mean(mzsum, axis=None) # reduce over all dimensions

	return loss

#	Added by Jingwei
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def create_base_network(input_shape):
	input = Input(shape=(224, 224, 3))

	conv = Sequential([input, Conv2D(64, (7,7), strides=2, padding='same', activation='relu'),
						MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
						BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9),
						Conv2D(64, 1, padding='same', activation='relu'),
						Conv2D(192, 3, padding='same', activation='relu'),
						BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9),
						MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')]);

	#Filter sizes for all 9 ic
	o0 = [64,128,192,160,128,112,256,256,384]
	o1 = {0:[96,128],1:[128,192],2:[96,208],3:[112,224],4:[128,256],5:[144,288],6:[160,320],7:[160,320],8:[192:384]}
	o2 = {0:[16,32],1:[32,96],2:[16,48],3:[24,64],4:[24,64],5:[32,64],6:[32,128],7:[32,128],8:[48,128]}
	o3 = [32,64,64,64,64,64,128,128,128]

	for i in range(8):
		if i==2 or i==7:
			conv = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv)
		out0 = Conv2D(o0[i], (1, 1), padding='same', activation='relu')(conv)
		temp = o1[i]
		out1 = Conv2D(temp[0], (1, 1), padding='same', activation='relu')(conv)
		out1 = Conv2D(temp[1], (3, 3), padding='same', activation='relu')(out1)
		temp = o2[i]
		out2 = Conv2D(temp[0], (1, 1), padding='same', activation='relu')(conv)
		out2 = Conv2D(temp[1], (5, 5), padding='same', activation='relu')(out2)
		out3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(conv)
		out3 = Conv2D(o3[i], (1, 1), padding='same', activation='relu')(icp1_pool)
		conv = concatenate(inputs=[out0, out1, out2, out3], axis=3)

	conv = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='valid')(conv)
	conv = Flatten()(conv)
	conv = Dense(2048, activation='relu')(conv)
	conv = Dropout(0.5)(conv)
	descriptor = Dense(128)(conv)

	return keras.Model(inputs=input, outputs=[descriptor])

def create_siamese_keras(weights_path=None, tune=False):
	input_a = Input(shape=(224, 224, 3))
	input_b = Input(shape=(224, 224, 3))

	# network definition
	base_network = create_base_network(input_shape=(224, 224, 3))

	processed_a = base_network(input_a)
	processed_b = base_network(input_b)
	distance = Lambda(euclidean_distance,
					  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

	siamesenet = keras.Model(inputs=[input_a, input_b], outputs=distance)

	return siamesenet


def create_posenet_keras_2d(weights_path=None, tune=False):
	input = Input(shape=(224, 224, 3))

	conv = Sequential([input, Conv2D(64, 7, strides=2, padding='same', activation='relu'),
						MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
						BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9),
						Conv2D(64, 1, padding='same', activation='relu'),
						Conv2D(192, 3, padding='same', activation='relu'),
						BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9),
						MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')]);

	#Filter sizes for all 9 ic
	o0 = [64,128,192,160,128,112,256,256,384]
	o1 = {0:[96,128],1:[128,192],2:[96,208],3:[112,224],4:[128,256],5:[144,288],6:[160,320],7:[160,320],8:[192:384]}
	o2 = {0:[16,32],1:[32,96],2:[16,48],3:[24,64],4:[24,64],5:[32,64],6:[32,128],7:[32,128],8:[48,128]}
	o3 = [32,64,64,64,64,64,128,128,128]

	for i in range(8):
		if i==2 or i==7:
			conv = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv)
		out0 = Conv2D(o0[i], (1, 1), padding='same', activation='relu')(conv)
		temp = o1[i]
		out1 = Conv2D(temp[0], (1, 1), padding='same', activation='relu')(conv)
		out1 = Conv2D(temp[1], (3, 3), padding='same', activation='relu')(out1)
		temp = o2[i]
		out2 = Conv2D(temp[0], (1, 1), padding='same', activation='relu')(conv)
		out2 = Conv2D(temp[1], (5, 5), padding='same', activation='relu')(out2)
		out3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(conv)
		out3 = Conv2D(o3[i], (1, 1), padding='same', activation='relu')(icp1_pool)
		conv = concatenate(inputs=[out0, out1, out2, out3], axis=3)

	conv = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='valid')(conv)
	conv = Flatten()(conv)
	conv = Dense(2048, activation='relu')(conv)
	conv = Dropout(0.5)(conv)

	cls3_fc_pose_xy = Dense(3)(conv)
	cls3_fc_pose_yaw = Dense(4)(conv)

	distance = Lambda(euclidean_distance,
					  output_shape=eucl_dist_output_shape)([processed_a, processed_b])
	posenet = keras.Model(inputs=input, outputs=[cls3_fc_pose_xy, cls3_fc_pose_yaw])

	if tune:
		if weights_path:
			weights_data = np.load(weights_path, encoding='latin1', allow_pickle=True).item()
			for layer in posenet.layers:
				if layer.name in weights_data.keys():
					layer_weights = weights_data[layer.name]
					layer.set_weights((layer_weights['weights'], layer_weights['biases']))
	return posenet


if __name__ == "__main__":
	print("Please run either test.py or train.py to evaluate or fine-tune the network!")
