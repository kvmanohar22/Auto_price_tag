import argparse
import model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def main(args):
	print '    Building the model'
	yolo = model.CNN(args)
	
	print '\nTotal parameters of the model are : ', yolo.count_params()
	
	if args.t:
		print '\nTesting the model on test image(s)...'
		yolo.test(args.t)
		print '\nDone testing the model...'

	if args.T:
		print '\nBeginning training...'
		yolo.train()
		print '\nDone training the model...'

def parser():
	"""
	Parse the arguements
	"""
	parser = argparse.ArgumentParser(description="Automatic Bill Calculator")
	parser.add_argument("-a", help="Alpha for Leaky ReLU activation", default=0.1)
	parser.add_argument("-t", help="Test the model by providing an image")
	parser.add_argument("-T", help="Train the model", action="store_true")
	parser.add_argument("-tr", help="Threshold value for object detection", default=0.15)
	parser.add_argument("-itr", help="IoU Threshold value for object detection", default=0.5)

	args = parser.parse_args()
	return args

if __name__ == "__main__":
	try:
		args = parser()
		main(args)
	except Exception as E:
		print E
