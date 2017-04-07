from copy import deepcopy
import pandas as pd
import numpy as np
import scipy.misc
import pickle
import time
import os

import options as _opt
from utils import xml_parser as parser

class Utilities(object):

	def __init__(self, annotations_dir, labels_list, opt):
		"""
		Parse the annotations if not done and dump it into a file
		"""
		self.opt = opt
		self.labels = self.opt.labels
		if os.path.isfile(opt.ann_parsed_file):
			print 'Annotations are already parsed...'
			unpickled_data = []
			with open(opt.ann_parsed_file, 'rb') as parsed_file:
				while True:
					try:
						unpickled_data.append(pickle.load(parsed_file))
					except EOFError:
						break
			# Set some variables
			self.dumps = unpickled_data[0][0]
			self.size  = len(self.dumps)
			return

		print 'Parsing the annotations'
		self.dumps = parser.pascal_voc_clean_xml(annotations_dir, labels_list)
		self.size = len(self.dumps)
		with open(opt.ann_parsed_file, 'wb') as file:
			pickle.dump([self.dumps], file)

	def pre_process(self):
		"""
		Data Preprocessing
		"""
		pass

	def load_data(self, Images_dir, Ann_dir, batch_begin, batch_end, shuffle=False):
		"""
		Load chunk of data 
		
		Input:
			Images_dir  : directory containing images
			Ann_dir     : Parsed annotations file
			batch_begin : start index of loading data
			batch_end   : end index of loading data 
			shuffle     : shuffle entire dataset after each epoch

		Output:
			batch of images and their annotations
		"""
		begin_time = time.time()
		if batch_end > self.size:
			print 'Out of bounds index'

		image_idx = []
		image_h   = []
		image_w   = []
		_allobjs   = []
		for idx in xrange(batch_begin, batch_end):
			image_idx.append(self.dumps[idx][0])
			image_w.append(self.dumps[idx][1][0])
			image_h.append(self.dumps[idx][1][1])
			_allobjs.append(self.dumps[idx][1][2][0])

		image_h = np.array(image_h)
		image_w = np.array(image_w)
		allobjs = deepcopy(_allobjs)

		# Read the corresponding images
		final_image_set = np.zeros((self.opt.batch_size, self.opt.img_x*self.opt.img_y*3))
		for idx, img in enumerate(image_idx):
			image = scipy.misc.imread(os.path.join(Images_dir, img))
			resized_img = scipy.misc.imresize(image, (self.opt.img_x, self.opt.img_y))
			resized_img = resized_img[:, :, ::-1]
			reshape_img = resized_img.reshape((1, self.opt.img_x*self.opt.img_y*3))
			reshape_img = reshape_img / 255.0
			final_image_set[idx] = reshape_img

		# print 'Loaded {} image(s) in {} sec'.format(len(image_idx), time.time()-begin_time)

		# Parse the annotations here to feed them to the loss function's operation
		S = self.opt.S
		C = self.opt.C
		B = self.opt.B
		labels = self.opt.labels
		w = image_w[0]
		h = image_h[0]

		cellx = 1. * image_w / S
		celly = 1. * image_h / S

		final_obj_grid_idx = []

		for obj in allobjs:			

			# TODO : Handle the case for multiple objects in the image here
			centerx = 0.5 * (obj[1]+obj[3])
			centery = 0.5 * (obj[2]+obj[4])
			cx = centerx / cellx
			cy = centery / celly

			obj[3] = float(obj[3]-obj[1]) / w
			obj[4] = float(obj[4]-obj[2]) / h
			obj[3] = np.sqrt(obj[3])
			obj[4] = np.sqrt(obj[4])
			obj[1] = cx - np.floor(cx)
			obj[2] = cy - np.floor(cy)
			obj += [int(np.floor(cy) * S + np.floor(cx))]
			final_obj_grid_idx.append(int(np.floor(cy) * S + np.floor(cx)))

		probs = np.zeros([S*S,C])
		confs = np.zeros([S*S,B])
		coord = np.zeros([S*S,B,4])
		proid = np.zeros([S*S,C])
		prear = np.zeros([S*S,4])

		for obj in allobjs:
			probs[obj[5], :] = [0.] * C
			probs[obj[5], labels.index(obj[0])] = 1.

			proid[obj[5], :] = [1] * C
			coord[obj[5], :, :] = [obj[1:5]] * B

			prear[obj[5],0] = obj[1] - obj[3]**2 * .5 * S
			prear[obj[5],1] = obj[2] - obj[4]**2 * .5 * S
			prear[obj[5],2] = obj[1] + obj[3]**2 * .5 * S
			prear[obj[5],3] = obj[2] + obj[4]**2 * .5 * S

			confs[obj[5], :] = [1.] * B

		upleft   = np.expand_dims(prear[:,0:2], 1)
		botright = np.expand_dims(prear[:,2:4], 1)
		wh = botright - upleft
		area = wh[:,:,0] * wh[:,:,1]
		upleft   = np.concatenate([upleft] * B, 1)
		botright = np.concatenate([botright] * B, 1)
		areas = np.concatenate([area] * B, 1)

		upleft 	= np.expand_dims(upleft, 0)
		botright = np.expand_dims(botright, 0)
		probs    = np.expand_dims(probs, 0)
		proid    = np.expand_dims(proid, 0)
		areas  	= np.expand_dims(areas, 0)
		confs  	= np.expand_dims(confs, 0)
		coord 	= np.expand_dims(coord, 0)

		# Set the final placeholder's values
		loss_feed_vals = {'probs': probs, 'confs': confs, 'coord': coord, 'proid': proid,
								'areas': areas, 'upleft': upleft, 'botright': botright}

		return final_image_set, loss_feed_vals
