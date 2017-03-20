import argparse
import model

def main(args):
	print '    Building the model'
	yolo = model.CNN(args.alpha)
	
	print 'Total parameters of the model are : ', yolo.count_params()
	if args.t:
		print 'Testing the model'
		yolo.test(args.t)



def parser():
	"""
	Parse the arguements
	"""
	parser = argparse.ArgumentParser(description="Yolo Description")
	parser.add_argument("alpha", type=float, help="Alpha for Leaky ReLU activation")
	parser.add_argument("-t", "-Test", help="Test the model by providing an image")
	parser.add_argument("-T", "-Train", help="Train the model by providing training directory")

	args = parser.parse_args()
	return args

if __name__ == "__main__":
	try:
		args = parser()
		main(args)
	except Exception as E:
		print E
