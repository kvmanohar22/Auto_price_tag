import pandas as pd
import numpy as np
import scipy.misc
import pickle
import os

import options as _opt
from utils import xml_parser as parser

class Utilities(object):

	def __init__(self, annotations_dir, labels_list, opt):
		"""
		Parse the annotations if not done and dump it into a file
		"""
		if os.path.isfile(opt.ann_parsed_file):
			print 'Annotatoins are already parsed...'
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
		print self.size
		with open(opt.ann_parsed_file, 'wb') as file:
			pickle.dump([self.dumps], file)

	def pre_process(self):
		"""
		Data Preprocessing
		"""
		pass

	def load_data(self):
		"""
		Load chunk of data 
		"""
		pass
