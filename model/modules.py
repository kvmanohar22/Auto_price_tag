import tensorflow as tf

"""
This file contains helper functions for building the model
"""

def weight_init(shape, name):
	"""
	Initialize the random weight matrix from truncated normal distribution
	"""
	W = tf.get_variable(name=name, tf.truncated_normal(shape, stddev=0.1))
	return W	


def bias_init(shape, name):
	"""
	Initialize the biases, setting it to 0.1 float32
	"""
	b = tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(0.1))
	return b


def conv2d(input_volume, kernel, stride=1, name, alpha):
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

	with tf.variable_scope(name):
		W = weight_init(kernel, 'W')
		b = bias_init(kernel[3], 'b')
		strides = [1,stride, stride, 1]

		conv = tf.nn.conv2d(input=input_volume, filter=W, strides=strides, padding='SAME')
		final = conv+b

		# Apply leaky relu activation function with alpha as alpha
		return tf.maximum(alpha*final, final)	


def max_pool(input_volume, kernel=3, stride=2, name):
	"""	
	Perform max-pool operation

	Input:
		conv_activations : result of convolution operation
		kernel : kernel
		stride : stride

	Output:
		Returns volume after max pool operation
	"""	
	ksize = [1, kernel, kernel, 1]
	strides = [1, stride, stride, 1]
	max_pool = tf.nn.max_pool(input_volume, ksize=ksize, strides=strides, padding='SAME')
	return max_pool


def fully_connected_linear(_input, _output, name):
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

	W = weight_init([input_units, _output], 'W')
	b = bias_init([_output], 'b')			

	output = tf.add(tf.matmul(_input, W), b)
	return output	


def fully_connected(_input, _output, activation=tf.nn.relu, name, alpha):
	
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

	with tf.variable_scope(name):
		linear_output = fully_connected_linear(_input=_input, _output=_output)

		if activation is None:
			return linear_output
		else:
			# Apply leaky relu activation function with alpha as alpha
			return tf.maximum(alpha*linear_output, linear_output)	
	