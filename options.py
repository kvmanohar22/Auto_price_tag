import os
"""
This class contains all the options
"""

class Options:

	def __init__(self):
		self.checkpoint_dir 	= '/home/kv/Git/Auto_price_tag/'
		self.new_ckpt_dir 	= '/home/kv/Git/Auto_price_tag/ckpt/'
		self.image_file 		= '/home/kv/Git/Auto_price_tag/pred/predictions.png'
		self.annotations_dir = '/home/kv/Git/Auto_price_tag/Dataset/Annotations/'
		self.dataset_dir     = '/home/kv/Git/Auto_price_tag/Dataset/Images/'
		self.ann_parsed_file = '/home/kv/Git/Auto_price_tag/Dataset/Ann.parsed'

		# hyper-parameters
		self.alpha 			   = 0.1		# alpha for leaky relu activation
		self.batch_size      = 1		# batch size
		self.epochs 		   = 200		# number of epochs
		self.learning_rate   = 1e-5	# learning rate
		self.ckpt_after      = 99		# checkpoint after these many epochs 
		self.iou_threshold   = 0.5		# iou threshold
		self.det_threshold   = 0.15   # detection threshold 

		# Loss function's target values
		self.H = 7		 	# output shape of the grid along x
		self.W = 7		 	# output shape of the grid along y
		self.C = 20		 	# number of classes (actual YOLO has 20 classes)
		self.B = 2 		 	# number of bounding boxes generated per grid cell
		self.S = 7 			# SxS grid cell is generated for YOLO-v1 but is different in case of
								# YOLO-v2

		# Misc
		self.img_x = 448
		self.img_y = 448

		# Labels
		self.labels = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
