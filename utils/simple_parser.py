import os
import sys

def simple_parser(ann_dir, labels):

	h = 448
	w = 448
	all_annotations = os.listdir(ann_dir)
	dumps = list()
	for annotation in all_annotations:
		file_name = annotation.split('.')[0]+'.jpeg'
		current = list()
		current = [None, None, None, None, None]
		class_label = annotation.split('_')[0]
		current[0] = labels[int(class_label)]
		with open(os.path.join(ann_dir, annotation)) as file:
			line = file.readline()
			line = file.readline()
			current[1:] = [int(val) for val in line.split(' ')]
		dumps += [[file_name, [w, h, current]]]

	# display some statistics of the dataset
	print 'Dataset statistics are as follows:'
	stats = dict()
	for dump in dumps:
		current = dump[1][2]
		if current[0] in stats:
			stats[current[0]] += 1
		else:
			stats[current[0]] = 1
	for key in stats:
		print '{}: {}'.format(key, stats[key])
	print 'Total dataset size: ', len(dumps)
	return dumps