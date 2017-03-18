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
	