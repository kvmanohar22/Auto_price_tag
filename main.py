import argparse
import model

def main(args):
	print 'Building the model'
	yolo = model.CNN(args.alpha)

def parser():
	"""
	Parse the arguements
	"""
	parser = argparse.ArgumentParser(description="Yolo Description")
	parser.add_argument("alpha", type=float, help="Alpha for Leaky ReLU activation")

	args = parser.parse_args()
	return args

if __name__ == "__main__":
	try:
		args = parser()
		main(args)
	except Exception as E:
		print E
