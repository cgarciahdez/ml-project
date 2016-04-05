
import numpy as np
import os
from pylab import *
import pdb
import random
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import shuffle
from sklearn.cross_validation import KFold
from collections import defaultdict
from collections import Counter



class Project():

	def __init__(self):
		pass

	def main_menu(self):
		while(True):
			print("Welcome to our program")
			print("To continue, please select which dataset you would like to load:")
			print("1. Iris")
			print("2. Vehicle")
			print("3. Sonar")
			print("4. Image Segmentation")
			print("5. Breast Cancer Wisconsin")
			print("6. Pen-digits")

			var = input("Input number selection:")
			if var is 'h':
				self.show_help()
			elif var is '1':
				self.load_iris()
			elif var is '2':
				self.load_vehicle()
			elif var is '3':
				self.load_sonar()
			elif var is '4':
				self.load_imgseg()
			elif var is '5':
				self.load_bcw()
			elif var is '6':
				self.load_pendig()

			print("Data has been loaded succesfully.")
			c = input("Press 1 to choose a different data set or any other key to terminate: ")
			if c is not '1':
				break


		print (self.x,self.y)

	def show_help(self):
		print("This program info info info")

	def load_iris(self):
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

		x, y= np.array(x), np.array(y)
		x, y = shuffle(x, y, random_state=0)
		self.x, self.y= x,y


	def load_vehicle(self):
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

		x, y= np.array(x), np.array(y)
		x, y = shuffle(x, y, random_state=0)
		self.x, self.y= x,y

	def load_sonar(self):
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

		x, y= np.array(x), np.array(y)
		x, y = shuffle(x, y, random_state=0)
		self.x, self.y= x,y

	def load_imgseg(self):
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

		x, y= np.array(x), np.array(y)
		x, y = shuffle(x, y, random_state=0)
		self.x, self.y= x,y

	def load_bcw(self):
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
						l[i]='1'    #Possible change. Missing value replaced by 1, avg could be better choice.
					xi.append(float(l[i]))
				yi=labels[l[len(l)-1]]
				y.append(yi)
				x.append(xi)

		x, y= np.array(x), np.array(y)
		x, y = shuffle(x, y, random_state=0)
		self.x, self.y= x,y

	def load_pendig(self):
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
						if(l[i] is '?'):
							l[i]='1'    #Possible change. Missing value replaced by 1, avg could be better choice.
						xi.append(int(l[i]))
					yi=int(l[len(l)-1])
					y.append(yi)
					x.append(xi)

		x, y= np.array(x), np.array(y)
		x, y = shuffle(x, y, random_state=0)
		self.x, self.y= x,y

proj = Project()
proj.main_menu()
