import os
import sys


def pascal_voc_clean_xml(ANN, pick):
	print('Parsing for {}'.format(pick))
	def pp(l): # pretty printing 
		for i in l: print('{}: {}'.format(i,l[i]))

	def parse(line): # exclude the xml tag
		x = line.split('>')[1].split('<')[0]
		try: r = int(x)
		except: r = x
		return r

	def _int(literal): # for literals supposed to be int 
		return int(float(literal))

	dumps = list()
	annotations = os.listdir(ANN)
	annotations = [file for file in annotations if '.xml' in file]
	size = len(os.listdir(ANN))

	for i, file in enumerate(annotations):

		with open(os.path.join(ANN, file), 'r') as f:
			lines = f.readlines()
		w = h = int()
		all = current = list()
		name = str()
		obj = False
		flag = False
		for i in range(len(lines)):
			line = lines[i]
			if '<filename>' in line:
				jpg = str(parse(line))
			if '<width>' in line:
				w = _int(parse(line))
			if '<height>' in line:
				h = _int(parse(line))
			if '<object>' in line:
				obj = True
			if '</object>' in line:
				obj = False
			if '<part>' in line:
				obj = False
			if '</part>' in line:
				obj = True
			if not obj: continue
			if '<name>' in line:
				if current != list():
					if current[0] in pick:
						all += [current]
				current = list()
				name = str(parse(line))
				if name not in pick: 
					obj = False
					continue
				current = [name,None,None,None,None]
			if len(current) != 5: 
				continue
			xn = '<xmin>' in line
			xx = '<xmax>' in line
			yn = '<ymin>' in line
			yx = '<ymax>' in line
			if xn: current[1] = _int(parse(line))
			if xx: current[3] = _int(parse(line))
			if yn: current[2] = _int(parse(line))
			if yx: current[4] = _int(parse(line))

		if flag: 
			continue
		if current != list() and current[0] in pick:
			all += [current]

		add = [[jpg, [w, h, all]]]
		dumps += add

	# gather all stats
	stat = dict()
	for dump in dumps:
		all = dump[1][2]
		for current in all:
			if current[0] in pick:
				if current[0] in stat:
					stat[current[0]]+=1
				else:
					stat[current[0]] =1

	print() 
	print('Statistics:')
	pp(stat)
	print('Dataset size: {}'.format(len(dumps)))
	return dumps	
