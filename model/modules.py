import tensorflow as tf
import numpy as np

"""
This file contains helper functions for building the model
"""

def weight_init(shape, name=None):
	"""
	Initialize the random weight matrix from truncated normal distribution
	"""
	W = tf.get_variable(name, initializer=tf.truncated_normal(shape=shape, stddev=0.1))
	return W	


def bias_init(shape, name=None):
	"""
	Initialize the biases, setting it to 0.1 float32
	"""
	b = tf.get_variable(name, shape=shape, initializer=tf.constant_initializer(0.1))
	return b


def conv2d(idx, input_volume, kernel, name, alpha, stride=1, is_training=True):
	"""
	Perform 2D convolutions

	Input:
		input_volume : [a, b, c, d], 4D tensor
			a: batch_size
			b: input dimension along x
			c: input dimension along y
			d: input depth (number of input channels)
		kernel : [a, b, c, d], 4D tensor
			a: kernel dimension along x
			b: kernel dimenstion along y
			c: input depth
			d: output depth
		stride : stride 
		name : name of the layer

	Output:
		Returns the volume after 2D convolution operation
	"""
	print '    Layer  %2d : Type = Conv, Size = %d * %d, Stride = %d, Filters = %d, Input channels = %d' % (idx, kernel[0], kernel[1], stride, int(kernel[3]), int(kernel[2]))
	# with tf.variable_scope(name):
	# W = weight_init(kernel, 'W')
	# b = bias_init(kernel[3], 'b')
	W = tf.Variable(tf.truncated_normal(kernel, stddev=0.1))
	b = tf.Variable(tf.constant(0.1, shape=[kernel[3]]))	
	strides = [1,stride, stride, 1]

	conv = tf.nn.conv2d(input=input_volume, filter=W, strides=strides, padding='SAME')
	final = tf.add(conv, b)

	# Apply leaky relu activation function with alpha as alpha
	return tf.maximum(alpha*final, final)	


def max_pool(idx, input_volume, kernel=2, stride=2, name=None):
	"""	
	Perform max-pool operation

	Input:
		conv_activations : result of convolution operation
		kernel : kernel
		stride : stride

	Output:
		Returns volume after max pool operation
	"""	
	print '    Layer  %2d : Type = Pool, Size = %d * %d, Stride = %d' % (idx, kernel, kernel, stride)
	ksize = [1, kernel, kernel, 1]
	strides = [1, stride, stride, 1]
	max_pool = tf.nn.max_pool(input_volume, ksize=ksize, strides=strides, padding='SAME')
	return max_pool


def fully_connected_linear(_input, _output):
	"""
	Input:
		_input : 
					- input to fully connected layer 
					- this is of shape [batch_size, no_of_input_units]
					- in case of transition from conv/max_pool layer to this fully connected, 
					  the shape of input is chaned accordingly
		_output :
					- output shape of fully connected layer
					- this is a single integer representing the number of output units of the layer

	Output:
		Returns the linear activations of the layer
	"""

	shape = _input.get_shape()
	input_units = int(shape[1])

	# W = weight_init([input_units, _output], 'W')
	# b = bias_init([_output], 'b')			
	W = tf.Variable(tf.truncated_normal([input_units, _output], stddev=0.1))
	b = tf.Variable(tf.constant(0.1, shape=[_output]))

	output = tf.add(tf.matmul(_input, W), b)
	return output	


def fully_connected(idx, _input, _output, name, alpha, activation=None, is_training=True):
	
	"""
	Input:
		_input : 
					- input to fully connected layer 
					- this is of shape [batch_size, no_of_input_units]
					- in case of transition from conv/max_pool layer to this fully connected, 
					  the shape of input is chaned accordingly
		_output :
					- output shape of fully connected layer
					- this is a single integer representing the number of output units of the layer

	Output:
		Returns the non-linear activations of the layer
	"""
	print '    Layer  %2d : Type = Full, Input dimension = %d, Output dimension = %d ' % (idx, int(_input.get_shape()[1]), _output)
	# with tf.variable_scope(name):
	linear_output = fully_connected_linear(_input=_input, _output=_output)

	if activation is None:
		return linear_output
	else:
		# Apply leaky relu activation function with alpha as alpha
		return tf.maximum(alpha*linear_output, linear_output)	
	