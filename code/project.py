
import numpy as np
import time
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
from performance import get_results
from GCNN import GCNN
from OGCNN import OGCNN


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

	def cross_validation(self,k_f=10):
		labels=list(set(self.y))
		train_e={'p':0,'r':0,'f_m':0,'a':0}
		test_e={'p':0,'r':0,'f_m':0,'a':0}
		
		kf = KFold(len(self.x),k_f)
		for train, test in kf:
			self.classifier.fit(self.x[train],self.y[train])
			y_label_train = self.classifier.classify_batch( self.x[train] )
			y_label_test = self.classifier.classify_batch( self.x[test] )
			train_r = get_results( self.y[train], y_label_train,labels )
			test_r = get_results( self.y[test], y_label_test,labels )

			train_e["p"] += train_r[1]
			train_e["r"] += train_r[2]
			train_e["f_m"] += (train_r[3])
			train_e["a"] += train_r[4]

			test_e["p"] += (test_r[1])
			test_e["r"] += (test_r[2])
			test_e["f_m"] += (test_r[3])
			test_e["a"] += test_r[4]


		for key in train_e:
			train_e[key]/=k_f
			test_e[key]/=k_f


		return train_e, test_e


proj = Project()
#proj.main_menu()
proj.x, proj.y = load_iris()
# print(proj.classifier.classify(proj.x[137]))
# print(proj.y[137])




start_time = time.time()
print(proj.cross_validation())
print("Time for GCNN was: %f"%(time.time() - start_time))

proj.classifier=OGCNN()


start_time = time.time()
print(proj.cross_validation())
print("Time for OGCNN was: %f"%(time.time() - start_time))


