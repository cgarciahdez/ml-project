
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
			print("Welcome to the One Pass Generalized Neural Network Program\n"
				"Here, you can compare the performance and time requirements of\n"
				"both the traditional Generalized Neural Network and our proposed\n"
				"improvement, the One Pass Generalized Neural Network, in different datasets.\n"
				"Please be aware that should you need help at any point, you need only type h"
				"and hit enter. You can also find a README with the same information in the folder.")
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
		print("""
|                          | Running Time(s)      | Difference (s) | Difference (min) |
|--------------------------|----------------------|----------------|------------------|
| Dataset                  | GCNN      |   OGCNN  |                |                  |
| Iris                     | 97.22     |   5.73   | 91.49          | 1.5248           |
| Vehicle                  | 4462.85   |   170.52 | 4292.33        | 71.5388          |
| Sonar                    | 162.26    |   8.01   | 154.25         | 2.5708           |
| Image Segmentation       | 230.03    |   13.73  | 216.3          | 3.6050           |
| Breast Cancer Winsconsin | 2601.94   |   85.52  | 2516.42        | 41.9403          |

| TRAINING PERFORMANNCE    |     Accuracy      |     F-Measure      |
|--------------------------|-------------------|--------------------|
| Dataset                  | GCNN     | OGCNN  | GCNN      | OGCNN  |
| Iris                     | 0.2874   | 0.3274 | 0.26      | 0.3533 |
| Vehicle                  | 0.2269   | 0.2175 | 0.1961    | 0.1852 |
| Sonar                    | 0.5145   | 0.4952 | 0.4723    | 0.5039 |
| Image Segmentation       | 0.1434   | 0.1286 | 0.1435    | 0.1474 |
| Breast Cancer Winsconsin | 0.5474   | 0.4883 | 0.5161    | 0.3483 |

| TESTING PERFORMANNCE     |     Accuracy      |     F-Measure      |
|--------------------------|-------------------|--------------------|
| Dataset                  | GCNN     | OGCNN  | GCNN      | OGCNN  |
| Iris                     | 0.26     | 0.353  | 0.3589    | 0.382  |
| Vehicle                  | 0.2269   | 0.2186 | 0.2929    | 0.2316 |
| Sonar                    | 0.509    | 0.5238 | 0.578     | 0.5377 |
| Image Segmentation       | 0.1619   | 0.1429 | 0.2445    | 0.1828 |
| Breast Cancer Winsconsin | 0.5638   | 0.4937 | 0.5189    | 0.3506 |
						""")
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
			print("6. Go back")

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
				break
			else:
				print("Please choose a valid number.")

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
					if l_r<0:
						float("a")
					self.gcnn.l_r=l_r
					print("Value succesfully changed")
					
				except ValueError:
					print("Please choose a positive real number")
					
			elif var is '2':
				print("The acceptable error gives the minimum error that needs to be achived to stop iterating the gradien descent.")
				th = input("Choose the new acceptable error: ")
				try:
					th = float(th)
					if th<0:
						float("a")
					self.gcnn.th=th
					print("Value succesfully changed")
					
				except ValueError:
					print("Please choose a positive real number")
					
			elif var is '3':
				print("The maximum iteration number dictates the maximum number of times that the gradient descent will iterate (for every example) before stopping.")
				m_i = input("Choose the new number of maximum iterations rate: ")
				try:
					m_i = int(m_i)
					if l_r<0:
						float("a")
					self.gcnn.max_iter=m_i
					print("Value succesfully changed")
					
				except ValueError:
					print("Please choose a positive integer")
					
			elif var is '4':
				break
			else:
				print("Please choose a valid number.")

			input("Press enter to continue")



	def show_help(self):
		print("""
Before attempting to run either algorithm, it's necessary that you choose one of the
datasets to load it. To do so, please choose the load a dataset option in the main menu and 
choose whichever dataset you want. To see the results gotten from this datasets before
choosing one, please go to the static results section, where you will be able to see the
time they took and their performance. It is recommended that you do this so as not to 
run one of the long lasting algorithms by accident. If you do, and want to force end the 
program, you will have to press ctr+c.

You also have the option to change the inputs for the GNCC algorithm. Please be aware
that changing them drastically might make the program run for a much longer time. We 
do not recommend choosing a very small step, a very big number of maximum iterations, or 
a very small acceptable error.

Finally, you have the option to run the algorithms separately or both of them to copare.
Be aware that the GNCC algorith takes a longer time than the OGCNN. To see the times
gotten during the making of the report, please refer to the static results.
			""")
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
