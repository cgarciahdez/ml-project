
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
				self.gcnn = GCNN()
				self.ogcnn = OGCNN()
				self.classifier = self.gcnn
				self.x=None
				self.y=None

	def main_menu(self):
		while(True):
			print("-----------------------------------------------------")
			print("Welcome to our program\n"
				"Please be aware that should you need help at any point, you need only type h and hit enter")
			print("To continue, please select an action:")
			print("1. Load a dataset")
			print("2. Run GCNN on dataset")
			print("3. Run OGCNN on dataset")
			print("4. Run both algorithms and compare")
			print("5. Change program inputs")
			print("6. See static results (gotten during trials)")
			print("7. Exit")

			var = input("Input number selection:")
			if var is 'h':
				self.show_help()
			elif var is '1':
				self.load_data()
			elif var is '2':
				self.run_GCNN()
			elif var is '3':
				self.run_OGCNN()
			elif var is '4':
				self.run_both()
			elif var is '5':
				self.change_params()
			elif var is '6':
				self.see_static()
			elif var is '7':
				break
			else:
				print("Please choose a valid number.")

	def run_GCNN(self):
		if self.x is None:
			print("You have to load a dataset before running any algorithm")
			var=input("Press enter to return")
			return

		self.classifier=self.gcnn
		start_time = time.time()
		train_e, test_e = proj.cross_validation()
		end_time = time.time()

		print("Average train scores")
		for k, v in train_e.items():
			print(k, " : ", v)
		print("Average test scores")
		for k, v in test_e.items():
			print(k, " : ", v)

		print("Time elapsed: %f"%(end_time - start_time))

	def run_OGCNN(self):
		if self.x is None:
			print("You have to load a dataset before running any algorithm")
			var=input("Press enter to return")
			return

		self.classifier=self.ogcnn
		start_time = time.time()
		train_e, test_e = proj.cross_validation()
		end_time = time.time()

		print("Average train scores")
		for k, v in train_e.items():
			print(k, " : ", v)
		print("Average test scores")
		for k, v in test_e.items():
			print(k, " : ", v)

		print("Time elapsed: %f"%(end_time - start_time))
		var=input("Press enter to return")
		return

	def run_both(self):
		if self.x is None:
			print("You have to load a dataset before running any algorithm")
			var=input("Press enter to return")
			return

		self.classifier=self.gcnn
		Gstart_time = time.time()
		Gtrain_e, Gtest_e = proj.cross_validation()
		Gend_time = time.time()

		self.classifier=self.ogcnn
		Ostart_time = time.time()
		Otrain_e, Otest_e = proj.cross_validation()
		Oend_time = time.time()

		print("Average train scores")
		for k, v in Gtrain_e.items():
			print(k, " for GCNN: ", v)
			print(k, " for OGCNN: ", Otrain_e[k])
		print("Average test scores")
		for k, v in Gtest_e.items():
			print(k, " for GCNN: ", v)
			print(k, " for OGCNN: ", Otest_e[k])

		print("Time elapsed for GCNN: %f"%(Gend_time - Gstart_time))
		print("Time elapsed for OGCNN: %f"%(Oend_time - Ostart_time))
		var=input("Press enter to return")
		return

	def see_static(self):
		print("resultados")
		var = input("Press enter to return")
		return
		


	def load_data(self):
		while(True):
			print("-----------------------------------------------------")
			print("In this section, you can choose which dataset to load\n"
				"Please be aware that big datasets might take a long time\n"
				"running the GNCC algorithm. Some of them might not run it \n"
				"at all. Check the static results page to know the times\n"
				"We calculated.")
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
			c = input("Press 1 to choose a different data set or any other key to go back: ")
			if c is not '1':
				break

	def change_params(self):
		while(True):
			print("-----------------------------------------------------")
			print("In this section, you can change the parameters of the algorithms")
			print("Please be aware that putting a big number for max_iterations")
			print("and a very small one for the acceptable error might make the")
			print("program run for a very long time.\n")
			print("To continue, please select which parameter you would like to change:")
			print("1. GCNN's Learning rate")
			print("2. GCNN's Acceptable error")
			print("3. GCNN's Maximum iterations")
			print("4. Return")

			var = input("Input number selection:")
			if var is 'h':
				self.show_help()
			elif var is '1':
				print("The learning rate modifies how big are the steps in the gradient descent.")
				l_r = input("Choose the new learning rate: ")
				try:
					l_r = float(l_r)
					self.gcnn.l_r=l_r
					print("Value succesfully changed")
					
				except ValueError:
					print("Please choose a real number")
					
			elif var is '2':
				print("The acceptable error gives the minimum error that needs to be achived to stop iterating the gradien descent.")
				th = input("Choose the new acceptable error: ")
				try:
					th = float(th)
					self.gcnn.th=th
					print("Value succesfully changed")
					
				except ValueError:
					print("Please choose a real number")
					
			elif var is '3':
				print("The maximum iteration number dictates the maximum number of times that the gradient descent will iterate (for every example) before stopping.")
				m_i = input("Choose the new number of maximum iterations rate: ")
				try:
					m_i = float(m_i)
					self.gcnn.max_iter=m_i
					print("Value succesfully changed")
					
				except ValueError:
					print("Please choose a real number")
					
			elif var is '4':
				break
			else:
				print("Please choose a valid number.")

			input("Press enter to continue")



	def show_help(self):
		print("This program info info info")
		var = input("Pres enter to return")
		return

	def cross_validation(self,k_f=10):
		labels=list(set(self.y))
		train_e={'Precision':0,'Recall':0,'F-measure':0,'Accuracy':0}
		test_e={'Precision':0,'Recall':0,'F-measure':0,'Accuracy':0}
		
		kf = KFold(len(self.x),k_f)
		for train, test in kf:
			self.classifier.fit(self.x[train],self.y[train])
			y_label_train = self.classifier.classify_batch( self.x[train] )
			y_label_test = self.classifier.classify_batch( self.x[test] )
			train_r = get_results( self.y[train], y_label_train,labels )
			test_r = get_results( self.y[test], y_label_test,labels )

			train_e['Precision'] += train_r[1]
			train_e['Recall'] += train_r[2]
			train_e['F-measure'] += (train_r[3])
			train_e['Accuracy'] += train_r[4]

			test_e['Precision'] += (test_r[1])
			test_e['Recall'] += (test_r[2])
			test_e['F-measure'] += (test_r[3])
			test_e['Accuracy'] += test_r[4]


		for key in train_e:
			train_e[key]/=k_f
			test_e[key]/=k_f


		return train_e, test_e


proj = Project()
proj.main_menu()
#proj.x, proj.y = load_iris()
# print(proj.classifier.classify(proj.x[137]))
# print(proj.y[137])




# start_time = time.time()
# train_e, test_e = proj.cross_validation()
# end_time = time.time()

# print("Average train scores")
# for k, v in train_e.items():
# 	print(k, " : ", v)
# print("Average test scores")
# for k, v in test_e.items():
# 	print(k, " : ", v)
	
# print("Time for GCNN was: %f"%(end_time - start_time))

# proj.classifier=OGCNN()

# start_time = time.time()
# train_e, test_e = proj.cross_validation()
# end_time = time.time()

# print("Average train scores")
# for k, v in train_e.items():
# 	print(k, " : ", v)
# print("Average test scores")
# for k, v in test_e.items():
# 	print(k, " : ", v)
	
# print("Time for OGCNN was: %f"%(end_time - start_time))
