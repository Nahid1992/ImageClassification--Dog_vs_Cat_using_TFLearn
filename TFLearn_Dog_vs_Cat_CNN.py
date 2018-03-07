from __future__ import division, print_function, absolute_import
import matplotlib.pyplot as plt
import numpy as np
import h5py
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.data_utils import build_hdf5_image_dataset
from tflearn.layers.normalization import local_response_normalization

img_height = 150
img_width = 150

def Data_Generate(dataset_file,type,task):
	output_path = 'data/'+type+'_dataset.h5'
	
	if task == 'all':
		build_hdf5_image_dataset(dataset_file, image_shape=(img_width,img_height), mode='file', output_path=output_path, categorical_labels=True, normalize=True)
		h5f = h5py.File(output_path, 'r')
		X = h5f['X']
		Y = h5f['Y']
	if task == 'load':
		h5f = h5py.File(output_path, 'r')
		X = h5f['X']
		Y = h5f['Y']
	
	#shuffle
	X,Y = tflearn.data_utils.shuffle(X,Y)
		
	return X,Y
	
	
def create_model():
	# Building 'AlexNet'
	
	network = input_data(shape=[None, img_width, img_height, 3])
	
	network = conv_2d(network, 96, 11, strides=4, activation='relu')
	network = max_pool_2d(network, 3, strides=2)
	network = local_response_normalization(network)
	
	network = conv_2d(network, 256, 5, activation='relu')
	network = max_pool_2d(network, 3, strides=2)
	network = local_response_normalization(network)
	
	network = conv_2d(network, 384, 3, activation='relu')
	network = conv_2d(network, 384, 3, activation='relu')
	network = conv_2d(network, 256, 3, activation='relu')
	network = max_pool_2d(network, 3, strides=2)
	network = local_response_normalization(network)
	
	network = fully_connected(network, 4096, activation='tanh')
	network = dropout(network, 0.5)
	
	network = fully_connected(network, 4096, activation='tanh')
	network = dropout(network, 0.5)
	
	network = fully_connected(network, 2, activation='softmax')
	
	network = regression(network, optimizer='momentum',
						 loss='categorical_crossentropy',
						 learning_rate=0.001)
	
	return network
						 
def train_model(network,x_train,y_train,x_val,y_val):
	model = tflearn.DNN(network, tensorboard_verbose=2)
					
	model.fit(x_train, y_train, n_epoch=10, validation_set=(x_val, y_val), shuffle=True,
			show_metric=True, batch_size=64, snapshot_step=200,
			snapshot_epoch=False, run_id='dogvscat_tflearn_run01')
		
	#save Model
	model.save('models/tflearn_cnn_dogcat_model_alexnet.model')

def load_model():
	model.load('models/tflearn_cnn_dogcat_model_alexnet.model')
	
def main():
	TRAIN_dataset_file = 'data/train_data.txt'
	VALIDATION_dataset_file = 'data/validation_data.txt'
	task = 'all'
	#task = 'load'
	x_train,y_train = Data_Generate(TRAIN_dataset_file,'train',task)
	print('Training Data Ready...')
	x_val,y_val = Data_Generate(VALIDATION_dataset_file,'validation',task)
	print('Validation Data Ready...')
	
	network = create_model()
	print('AlexNet Created...')
	
	print('Training Samples = ' + str(x_train.shape[0]))
	print('Validation Samples = ' + str(x_val.shape[0]))
	
	print('')
	print('Training Started...')
	train_model(network,x_train,y_train,x_val,y_val)
	print('Model Saved...')	
	print('')
	
	breakPOINT = 1
	
if __name__== "__main__":
  main()
