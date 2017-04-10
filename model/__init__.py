import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
import time
import cv2
import os

from options import Options
import modules as model
import utils as util

np.set_printoptions(threshold=np.nan)

class CNN:

	def __init__(self, args):
		"""
		defines the architecture of the model
		"""
		self.options = Options()
		self.alpha = args.a
		self.threshold = args.tr
		self.iou_threshold = args.itr
		self.classes = self.options.custom_labels
		self.image_file = self.options.image_file

		# Input to the model
		self.x = tf.placeholder(tf.float32, shape=[None, self.options.img_x*self.options.img_y*3])
		input_data = tf.reshape(self.x, [-1, self.options.img_x, self.options.img_y, 3])
		self.utils   = util.Utilities(self.options.annotations_dir, self.classes, self.options)

		# Stack the layers of the network
		print "    Stacking layers of the network"
		self.conv_01 = model.conv2d(1, input_data, kernel=[7,7,3,64], stride=2, name='conv_01', alpha=self.alpha, is_training=False)
		self.pool_02 = model.max_pool(2, self.conv_01, name='pool_02')

		self.conv_03 = model.conv2d(3, self.pool_02, kernel=[3,3,64,192], stride=1, name='conv_03', alpha=self.alpha, is_training=False)
		self.pool_04 = model.max_pool(4, self.conv_03, name='pool_04')

		self.conv_05 = model.conv2d(5, self.pool_04, kernel=[1,1,192,128], stride=1, name='conv_05', alpha=self.alpha, is_training=False)
		self.conv_06 = model.conv2d(6, self.conv_05, kernel=[3,3,128,256], stride=1, name='conv_06', alpha=self.alpha, is_training=False)
		self.conv_07 = model.conv2d(7, self.conv_06, kernel=[1,1,256,256], stride=1, name='conv_07', alpha=self.alpha, is_training=False)
		self.conv_08 = model.conv2d(8, self.conv_07, kernel=[3,3,256,512], stride=1, name='conv_08', alpha=self.alpha, is_training=False)
		self.pool_09 = model.max_pool(9, self.conv_08, name='pool_09')

		self.conv_10 = model.conv2d(10, self.pool_09, kernel=[1,1,512,256], stride=1, name='conv_10', alpha=self.alpha, is_training=False)
		self.conv_11 = model.conv2d(11, self.conv_10, kernel=[3,3,256,512], stride=1, name='conv_11', alpha=self.alpha, is_training=False)
		self.conv_12 = model.conv2d(12, self.conv_11, kernel=[1,1,512,256], stride=1, name='conv_12', alpha=self.alpha, is_training=False)
		self.conv_13 = model.conv2d(13, self.conv_12, kernel=[3,3,256,512], stride=1, name='conv_13', alpha=self.alpha, is_training=False)
		self.conv_14 = model.conv2d(14, self.conv_13, kernel=[1,1,512,256], stride=1, name='conv_14', alpha=self.alpha, is_training=False)
		self.conv_15 = model.conv2d(15, self.conv_14, kernel=[3,3,256,512], stride=1, name='conv_15', alpha=self.alpha, is_training=False)
		self.conv_16 = model.conv2d(16, self.conv_15, kernel=[1,1,512,256], stride=1, name='conv_16', alpha=self.alpha, is_training=False)
		self.conv_17 = model.conv2d(17, self.conv_16, kernel=[3,3,256,512], stride=1, name='conv_17', alpha=self.alpha, is_training=False)
		self.conv_18 = model.conv2d(18, self.conv_17, kernel=[1,1,512,512], stride=1, name='conv_18', alpha=self.alpha, is_training=False)
		self.conv_19 = model.conv2d(19, self.conv_18, kernel=[3,3,512,1024],stride=1, name='conv_19', alpha=self.alpha, is_training=False)
		self.pool_20 = model.max_pool(20, self.conv_19, name='pool_20')

		self.conv_21 = model.conv2d(21, self.pool_20, kernel=[1,1,1024,512],  stride=1, name='conv_21', alpha=self.alpha, is_training=False)
		self.conv_22 = model.conv2d(22, self.conv_21, kernel=[3,3,512,1024],  stride=1, name='conv_22', alpha=self.alpha, is_training=False)
		self.conv_23 = model.conv2d(23, self.conv_22, kernel=[1,1,1024,512],  stride=1, name='conv_23', alpha=self.alpha, is_training=False)
		self.conv_24 = model.conv2d(24, self.conv_23, kernel=[3,3,512,1024],  stride=1, name='conv_24', alpha=self.alpha, is_training=False)
		self.conv_25 = model.conv2d(25, self.conv_24, kernel=[3,3,1024,1024], stride=1, name='conv_25', alpha=self.alpha, is_training=False)
		self.conv_26 = model.conv2d(26, self.conv_25, kernel=[3,3,1024,1024], stride=2, name='conv_26', alpha=self.alpha, is_training=False)
		self.conv_27 = model.conv2d(27, self.conv_26, kernel=[3,3,1024,1024], stride=1, name='conv_27', alpha=self.alpha, is_training=False)
		self.conv_28 = model.conv2d(28, self.conv_27, kernel=[3,3,1024,1024], stride=1, name='conv_28', alpha=self.alpha, is_training=False)

		# Reshape 'self.conv_28' from 4D to 2D
		shape = self.conv_28.get_shape().as_list()
		flat_shape = int(shape[1])*int(shape[2])*int(shape[3])
		inputs_transposed = tf.transpose(self.conv_28, (0,3,1,2))
		fully_flat = tf.reshape(inputs_transposed, [-1, flat_shape])

		self.fc_29 = model.fully_connected(29, fully_flat, 512, name='fc_29', alpha=self.alpha, is_training=True, activation=tf.nn.relu)
		self.fc_30 = model.fully_connected(30, self.fc_29, 4096, name='fc_30', alpha=self.alpha, is_training=True, activation=tf.nn.relu)
		self.fc_31 = model.fully_connected(31, self.fc_30, self.options.O, name='fc_31', alpha=self.alpha, is_training=True, activation=None)
 		
 		self.predictions = self.fc_31

 		self.init_operation = tf.global_variables_initializer()
 		self.saver = tf.train.Saver()
 		self.sess = tf.Session()

 		# Build the loss operation
 		self.loss(self.predictions)
 		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.options.learning_rate).minimize(-self._loss)

	def model_variables(self):
		architecture = ''
		for variable in tf.trainable_variables():
			architecture += str(variable.name) 
			architecture += '\n'
		return architecture

	def count_params(self):
		total_parameters = 0
		for variable in tf.trainable_variables():
			count = 1
			for dimension in variable.get_shape().as_list():
				count *= dimension
			total_parameters += count

		return total_parameters


	def train(self):
		"""
		train the model
		"""
		# analyze some data
		self.total_batches = self.utils.size // self.options.batch_size

		# Restore the pre-trained model here
		checkpoint = os.path.join(self.options.checkpoint_dir, 'YOLO_small.ckpt')
		self.saver.restore(self.sess, checkpoint)
		print 'Successfully restored the saved model !'
		moving_loss=-1.0

		for epoch in xrange(self.options.epochs):
			epoch_loss    	  = 0.0
			batch_number  	  = 0
			epoch_begin_time = time.time()

			for batch_begin, batch_end in zip(xrange(0, self.utils.size+1, self.options.batch_size), 
						xrange(self.options.batch_size, self.utils.size+1, self.options.batch_size)):
				
				# Load chunk of data here
				# This includes both the image data and their corresponding annotations
				images, loss_feed = self.utils.load_data(self.options.dataset_dir, self.options.ann_parsed_file, batch_begin, batch_end, batch_begin==0)

				# Evaluate loss here and do back-prop
				feed_dict = {self.loss_feed_dict[key]: loss_feed[key] for key in self.loss_feed_dict}
				feed_dict[self.x] = images

				_, _loss = self.sess.run([self.predictions, self._loss], feed_dict=feed_dict)

				if moving_loss == -1.0:
					moving_loss = _loss
				moving_loss = 0.9 * moving_loss + 0.1 * _loss

			
			print 'Epoch: %3d\tMoving Loss: %3f\tTime: %3f' % (epoch+1, moving_loss, time.time()-epoch_begin_time)
			if epoch % self.options.save_ckpt_after == 0:
				saved_dir = self.saver.save(self.sess, self.options.new_ckpt_dir+'model_{}.ckpt'.format(epoch))
			

	def test(self, test_image):
		"""
		test the model
		"""
		with tf.Session() as sess:
			checkpoint = self.options.checkpoint_dir+'YOLO_small.ckpt'																						
			self.saver.restore(sess, checkpoint)
			print 'Restored the model successfully from "{}"!!'.format(checkpoint)

			print '\nFollowing are the detected objects in the image "{}"'.format(test_image)
			img = cv2.imread(test_image)
			s = time.time()
			
			# print 'Infering shape of the image'
			self.h_img,self.w_img,_ = img.shape
			# print 'Height : {}\tWidth : {}'.format(self.h_img, self.w_img)

			# print '\nReshaping the image'
			img_resized = cv2.resize(img, (448, 448))
			img_RGB = cv2.cvtColor(img_resized,cv2.COLOR_BGR2RGB)
			img_resized_np = np.asarray( img_RGB )
			img_resized_np = img_resized_np.reshape(1, 448*448*3)
			inputs = np.zeros((1, 448*448*3), dtype='float32')
			inputs[0] = (img_resized_np/255.0)*2.0-1.0


			net_output = sess.run(self.fc_31, feed_dict={self.x : inputs})
			self.result = self.interpret_output(net_output[0])
			self.show_results(img, self.result)
			print '\nTotal time taken : {}'.format(time.time()-s)

	def interpret_output(self, output):
		
		# these are the final class specific probability scores for each of the box - 
		probs = np.zeros((7,7,2,20))

		# [980] class specific probability for each grid cell
		class_probs = np.reshape(output[0:980],(7,7,20))
		
		# [98] 
		scales = np.reshape(output[980:1078],(7,7,2))
		
		# [392] 
		boxes = np.reshape(output[1078:],(7,7,2,4))
		
		# 
		offset = np.transpose(np.reshape(np.array([np.arange(7)]*14),(2,7,7)),(1,2,0))

		# 
		boxes[:,:,:,0] += offset												# x-center of the box
		boxes[:,:,:,1] += np.transpose(offset,(1,0,2))					# y-center of the box
		boxes[:,:,:,0:2] = boxes[:,:,:,0:2] / 7.0                   #
		boxes[:,:,:,2] = np.multiply(boxes[:,:,:,2],boxes[:,:,:,2]) # height of the box
		boxes[:,:,:,3] = np.multiply(boxes[:,:,:,3],boxes[:,:,:,3]) #  width of the box
		
		boxes[:,:,:,0] *= self.w_img
		boxes[:,:,:,1] *= self.h_img
		boxes[:,:,:,2] *= self.w_img
		boxes[:,:,:,3] *= self.h_img

		# Generate the class specific probability scores for each of the bounding box in each of the grid cell
		for i in range(2):
			for j in range(20):
				probs[:,:,i,j] = np.multiply(class_probs[:,:,j],scales[:,:,i])

		# Threshold the probability values for each bounding box
		filter_mat_probs = np.array(probs>=self.threshold,dtype='bool')
		filter_mat_boxes = np.nonzero(filter_mat_probs) # 4D tensor

		boxes_filtered = boxes[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]
		probs_filtered = probs[filter_mat_probs]
		classes_num_filtered = np.argmax(filter_mat_probs,axis=3)[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]] 

		# print boxes_filtered 		# (x,y,w,h) for each of the fitered box
		# print probs_filtered 		# probability for each of the predicted class
		# print classes_num_filtered # class index of the predicted probability


		argsort = np.array(np.argsort(probs_filtered))[::-1]
		boxes_filtered = boxes_filtered[argsort]
		probs_filtered = probs_filtered[argsort]
		classes_num_filtered = classes_num_filtered[argsort]

		# print 'After sorting'
		# print boxes_filtered 		# (x,y,w,h) for each of the fitered box
		# print probs_filtered 		# probability for each of the predicted class

		# Loop over each of the predicted bounding box
		for i in range(len(boxes_filtered)):
			if probs_filtered[i] == 0 : 
				continue
			for j in range(i+1,len(boxes_filtered)):
				if self.iou(boxes_filtered[i],boxes_filtered[j]) > self.iou_threshold : 
					probs_filtered[j] = 0.0
		
		filter_iou = np.array(probs_filtered>0.0,dtype='bool')
		boxes_filtered = boxes_filtered[filter_iou]
		probs_filtered = probs_filtered[filter_iou]
		classes_num_filtered = classes_num_filtered[filter_iou]

		result = []
		for i in range(len(boxes_filtered)):
			result.append([self.classes[classes_num_filtered[i]],boxes_filtered[i][0],boxes_filtered[i][1],boxes_filtered[i][2],boxes_filtered[i][3],probs_filtered[i]])

		# Each of the result : ['person', 248.64821, 279.18292, 352.34753, 488.08188, 0.61149513721466064]
		return result

	def show_results(self, img, results):
		img_cp = img.copy()
		
		for i in range(len(results)):
			x = int(results[i][1])
			y = int(results[i][2])
			w = int(results[i][3])//2
			h = int(results[i][4])//2
			
			print '    class : ' + results[i][0] + ' , [x,y,w,h]=[' + str(x) + ',' + str(y) + ',' + str(int(results[i][3])) + ',' + str(int(results[i][4]))+'], Confidence = ' + str(results[i][5])
			cv2.rectangle(img_cp, (x-w,y-h), (x+w,y+h), (0,255,0), 8)
			# cv2.putText(img_cp, results[i][0] + ' : %.2f' % results[i][5], (x-w+5,y-h-7), cv2.FONT_ITALIC, 0.5, (0,0,0), 1)
			
		cv2.imwrite(self.image_file, img_cp)			
		# cv2.imshow('Price Prediction', img_cp)
		cv2.waitKey(5000)


	def iou(self, box1, box2):
		tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
		lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
		if tb < 0 or lr < 0 : 
			intersection = 0
		else : 
			intersection =  tb*lr

		return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)

	
	def loss(self, net_out):
		"""
			Calculate loss given predicted values and ground truth
		"""

		sprob = 1.
		sconf = 1.
		snoob = 0.5
		scoor = 5.

		# grid details
		S = self.options.S
		C = self.options.C
		B = self.options.B

		# number of grid cells
		SS = S * S 

		# placeholders for loss operation
		_probs = tf.placeholder(tf.float32, [None, SS, C])
		_confs = tf.placeholder(tf.float32, [None, SS, B])	
		_coord = tf.placeholder(tf.float32, [None, SS, B, 4])

		# L2 Loss
		_proid  = tf.placeholder(tf.float32, [None, SS, C])

		# IOU
		_areas  	 = tf.placeholder(tf.float32, [None, SS, B])
		_upleft 	 = tf.placeholder(tf.float32, [None, SS, B, 2])
		_botright = tf.placeholder(tf.float32, [None, SS, B, 2])


		self.loss_feed_dict = {'probs':_probs, 'confs': _confs, 'coord':_coord, 'proid':_proid, 
									  'areas':_areas, 'upleft':_upleft, 'botright':_botright}

		coords = net_out[:, SS * (C + B):]
		coords = tf.reshape(coords, [-1, SS, B, 4])
		wh = tf.pow(coords[:,:,:,2:4], 2) * S # unit: grid cell
		area_pred = wh[:,:,:,0] * wh[:,:,:,1] # unit: grid cell^2
		centers = coords[:,:,:,0:2] # [batch, SS, B, 2]
		floor = centers - (wh * .5) # [batch, SS, B, 2]
		ceil  = centers + (wh * .5) # [batch, SS, B, 2]

		intersect_upleft   = tf.maximum(floor, _upleft)
		intersect_botright = tf.minimum(ceil , _botright)
		intersect_wh = intersect_botright - intersect_upleft
		intersect_wh = tf.maximum(intersect_wh, 0.0)
		intersect = tf.multiply(intersect_wh[:,:,:,0], intersect_wh[:,:,:,1])

		iou = tf.truediv(intersect, _areas + area_pred - intersect)
		best_box = tf.equal(iou, tf.reduce_max(iou, [2], True))
		best_box = tf.to_float(best_box)
		confs = tf.multiply(best_box, _confs)

		conid = snoob * (1. - confs) + sconf * confs
		weight_coo = tf.concat(4 * [tf.expand_dims(confs, -1)], 3)
		cooid = scoor * weight_coo
		proid = sprob * _proid

		probs = slim.flatten(_probs)
		proid = slim.flatten(proid)
		confs = slim.flatten(confs)
		conid = slim.flatten(conid)
		coord = slim.flatten(_coord)
		cooid = slim.flatten(cooid)

		true = tf.concat([probs, confs, coord], 1)
		wght = tf.concat([proid, conid, cooid], 1)

		loss = tf.pow(net_out - true, 2)
		loss = tf.multiply(loss, wght)
		loss = tf.reduce_sum(loss, 1)
		self._loss = .5 * tf.reduce_mean(loss)
