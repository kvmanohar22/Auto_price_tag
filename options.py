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