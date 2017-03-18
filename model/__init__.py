import matplotlib.pyplot as plt
import tensorflow as tf
import modules as model
import pandas as pd
import numpy as np
import datetime
import time
import os

class CNN:

	def __init__(self, alpha):
		"""
		defines the architecture of the model
		"""

		"""
		Initialize variables related to training the model
		"""
		# alpha used for leaky relu
		self.alpha = alpha


		# Input to the model
		self.x = tf.placeholder(tf.float32, shape=[None, 448, 448, 3])

		# Stack the layers of the network
		print "    Stacking layers of the network"
		self.conv_01 = model.conv2d(1, self.x, kernel=[7,7,3,64], stride=2, name='conv_01', alpha=self.alpha)
		self.pool_02 = model.max_pool(2, self.conv_01, name='pool_02')

		self.conv_03 = model.conv2d(3, self.pool_02, kernel=[3,3,64,192], stride=1, name='conv_03', alpha=self.alpha)
		self.pool_04 = model.max_pool(4, self.conv_03, name='pool_04')

		self.conv_05 = model.conv2d(5, self.pool_04, kernel=[1,1,192,128], stride=1, name='conv_05', alpha=self.alpha)
		self.conv_06 = model.conv2d(6, self.conv_05, kernel=[3,3,128,256], stride=1, name='conv_06', alpha=self.alpha)
		self.conv_07 = model.conv2d(7, self.conv_06, kernel=[1,1,256,256], stride=1, name='conv_07', alpha=self.alpha)
		self.conv_08 = model.conv2d(8, self.conv_07, kernel=[3,3,256,512], stride=1, name='conv_08', alpha=self.alpha)
		self.pool_09 = model.max_pool(9, self.conv_08, name='pool_09')

		self.conv_10 = model.conv2d(10, self.pool_09, kernel=[1,1,512,256], stride=1, name='conv_10', alpha=self.alpha)
		self.conv_11 = model.conv2d(11, self.conv_10, kernel=[3,3,256,512], stride=1, name='conv_11', alpha=self.alpha)
		self.conv_12 = model.conv2d(12, self.conv_11, kernel=[1,1,512,256], stride=1, name='conv_12', alpha=self.alpha)
		self.conv_13 = model.conv2d(13, self.conv_12, kernel=[3,3,256,512], stride=1, name='conv_13', alpha=self.alpha)
		self.conv_14 = model.conv2d(14, self.conv_13, kernel=[1,1,512,256], stride=1, name='conv_14', alpha=self.alpha)
		self.conv_15 = model.conv2d(15, self.conv_14, kernel=[3,3,256,512], stride=1, name='conv_15', alpha=self.alpha)
		self.conv_16 = model.conv2d(16, self.conv_15, kernel=[1,1,512,256], stride=1, name='conv_16', alpha=self.alpha)
		self.conv_17 = model.conv2d(17, self.conv_16, kernel=[3,3,256,512], stride=1, name='conv_17', alpha=self.alpha)
		self.conv_18 = model.conv2d(18, self.conv_17, kernel=[1,1,512,512], stride=1, name='conv_18', alpha=self.alpha)
		self.conv_19 = model.conv2d(19, self.conv_18, kernel=[3,3,512,1024],stride=1, name='conv_19', alpha=self.alpha)
		self.pool_20 = model.max_pool(20, self.conv_19, name='pool_20')

		self.conv_21 = model.conv2d(21, self.pool_20, kernel=[1,1,1024,512],  stride=1, name='conv_21', alpha=self.alpha)
		self.conv_22 = model.conv2d(22, self.conv_21, kernel=[3,3,512,1024],  stride=1, name='conv_22', alpha=self.alpha)
		self.conv_23 = model.conv2d(23, self.conv_22, kernel=[1,1,1024,512],  stride=1, name='conv_23', alpha=self.alpha)
		self.conv_24 = model.conv2d(24, self.conv_23, kernel=[3,3,512,1024],  stride=1, name='conv_24', alpha=self.alpha)
		self.conv_25 = model.conv2d(25, self.conv_24, kernel=[3,3,1024,1024], stride=1, name='conv_25', alpha=self.alpha)
		self.conv_26 = model.conv2d(26, self.conv_25, kernel=[3,3,1024,1024], stride=2, name='conv_26', alpha=self.alpha)
		self.conv_27 = model.conv2d(27, self.conv_26, kernel=[3,3,1024,1024], stride=1, name='conv_27', alpha=self.alpha)
		self.conv_28 = model.conv2d(28, self.conv_27, kernel=[3,3,1024,1024], stride=1, name='conv_28', alpha=self.alpha)

		# Reshape 'self.conv_28' from 4D to 2D
		shape = self.conv_28.get_shape().as_list()
		flat_shape = int(shape[1])*int(shape[2])*int(shape[3])
		fully_flat = tf.reshape(self.conv_28, [-1, flat_shape])		
		self.fc_29 = model.fully_connected(29, fully_flat, 4096, name='fc_29', alpha=self.alpha)
		# skip the dropout layer
		self.fc_31 = model.fully_connected(31, self.fc_29, 1470, name='fc_31', alpha=self.alpha, activation=None)
 
	def model_architecture(self):
		architecture = ''
		for variable in tf.trainable_variables():
			architecture += str(variable.name) + str(variable.get_shape())
			architecture += '\n'
		return architecture


	def train(self):
		"""
		train the model
		"""
		pass

	def validate(self):
		"""
		validate the model
		"""
		pass

	def test(self):
		"""
		test the model
		"""
		pass
	