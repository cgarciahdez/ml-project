
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
from data_loaders import load_iris, load_vehicle, load_sonar, \
			load_imgseg, load_bcw, load_pendig
from GCNN import GCNN

class Project():

	def __init__(self):
                self.classifier = GCNN()

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
				self.x, self.y = load_iris()
			elif var is '2':
				self.x, self.y = load_vehicle()
			elif var is '3':
				self.x, self.y = load_sonar()
			elif var is '4':
				self.x, self.y = load_imgseg()
			elif var is '5':
				self.x, self.y = load_bcw()
			elif var is '6':
				self.x, self.y = load_pendig()

			print("Data has been loaded succesfully.")
			c = input("Press 1 to choose a different data set or any other key to terminate: ")
			if c is not '1':
				break

		print (self.x,self.y)

	def show_help(self):
		print("This program info info info")


proj = Project()
#proj.main_menu()
proj.x, proj.y = load_iris()
print(len(proj.x))
proj.classifier.fit(proj.x, proj.y)
for test in range(40,80):
    print(proj.classifier.classify(proj.x, proj.x[test]))
