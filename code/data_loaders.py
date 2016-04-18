import numpy as np
from sklearn.utils import shuffle

"""
This file loads the data from the different data sets offered
Because every data set has an almost unique format,
they all have a separate function. They all turn the data
into numpy matrixes and arrays that are ready to be used
by the algorithms in the program.
"""
def load_iris():
	file = open("./data/iris.data.txt",'r').readlines()
	labels={'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}
	x=[]
	y=[]
	for line in file:
		l=line.rstrip()
		l=l.split(',')
		if(len(l)>1):
			xi=[]
			for i in range(0,len(l)-1):
				xi.append(float(l[i]))
			yi=labels[l[-1]]
			y.append(yi)
			x.append(xi)

	x, y = np.array(x), np.array(y)
	x, y = shuffle(x, y, random_state=0)
	return x, y


def load_vehicle():
	letters = ['a','b','c','d','e','f','g','h','i']
	x=[]
	y=[]
	for letter in letters:
		file = open("./data/vehicle/xa%s.dat.txt"%letter,'r').readlines()
		labels={'opel':0,'saab':1,'bus':2,'van':3}
		for line in file:
			l=line.rstrip()
			l=l.split(' ')
			if(len(l)>1):
				xi=[]
				for i in range(0,len(l)-1):
					xi.append(float(l[i]))
				yi=labels[l[-1]]
				y.append(yi)
				x.append(xi)

	x, y = np.array(x), np.array(y)
	x, y = shuffle(x, y, random_state=0)
	return x, y

def load_sonar():
	file = open("./data/sonar.all-data.txt",'r').readlines()
	labels={'R':0,'M':1}
	x=[]
	y=[]
	for line in file:
		l=line.rstrip()
		l=l.split(',')
		if(len(l)>1):
			xi=[]
			for i in range(0,len(l)-1):
				xi.append(float(l[i]))
			yi=labels[l[-1]]
			y.append(yi)
			x.append(xi)

	x, y = np.array(x), np.array(y)
	x, y = shuffle(x, y, random_state=0)
	return x, y

def load_imgseg():
	file = open("./data/segmentation.data.txt",'r').readlines()
	labels={'BRICKFACE':0, 'SKY':1, 'FOLIAGE':2, 'CEMENT':3, 'WINDOW':4, 'PATH':5, 'GRASS':6}
	x=[]
	y=[]
	for line in file:
		l=line.rstrip()
		l=l.split(',')
		if(len(l)>1):
			xi=[]
			for i in range(1,len(l)):
				xi.append(float(l[i]))
			yi=labels[l[0]]
			y.append(yi)
			x.append(xi)

	x, y = np.array(x), np.array(y)
	x, y = shuffle(x, y, random_state=0)
	return x, y

def load_bcw():
	file = open("./data/breast-cancer-wisconsin.data.txt",'r').readlines()
	labels={'2':0, '4':1}  #2 is benign, 4 is malign
	x=[]
	y=[]
	for line in file:
		l=line.rstrip()
		l=l.split(',')
		if(len(l)>1):
			xi=[]
			for i in range(1,len(l)-1):
				if l[i] is '?':
					l[i]='1'	#Possible change. Missing value replaced by 1, avg could be better choice.
				xi.append(float(l[i]))
			yi=labels[l[len(l)-1]]
			y.append(yi)
			x.append(xi)

	x, y = np.array(x), np.array(y)
	x, y = shuffle(x, y, random_state=0)
	return x, y

def load_pendig():
	letters = ['tes','tra']
	x=[]
	y=[]
	for letter in letters:
		file = open("./data/pen-dig/pendigits.%s.txt"%letter,'r').readlines()
		for line in file:
			l="".join(line.split())
			l=l.split(',')
			if(len(l)>1):
				xi=[]
				for i in range(0,len(l)-1):
					xi.append(int(l[i]))
				yi=int(l[len(l)-1])
				y.append(yi)
				x.append(xi)

	x, y = np.array(x), np.array(y)
	x, y = shuffle(x, y, random_state=0)
	return x, y
