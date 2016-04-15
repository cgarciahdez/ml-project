
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



	def find_o(self):
		o=np.random.uniform(0.5,1.0,(len(self.y),))
		for i in range(len(self.y)):
			o = self.GNCC(o,self.x[i],self.y[i])
		return o

	def GNCC(self,o_,x_,y_original,max_iter=10,l_r=0.3,th=0.01):
		classes = set(self.y)
		y_max=0.9
		o=o_#Its gonna be len(y)
		it=0
		pattern = np.copy(self.x)
		while it<max_iter:
			u = defaultdict(float)
			r_ = defaultdict(float)
			dist_ = defaultdict(float)
			D = 0
			for j in range(len(pattern)):
				t_j=pattern[j]
				dist_[j]=self.dist(t_j,x_)
				r_[j] = self.r(dist_[j],o[j])
				D+=r_[j]
				for i in classes:
					d_=self.d(j,i,y_max)
					u[i]+=d_*r_[j]

			c = defaultdict(float)
			for i in classes:
				c[i]=u[i]/D
			c=dict(c)

			sort = sorted(c.items(), key=lambda x: x[1], reverse=True)

			winner = sort[0]
			e = (self.y_(y_original,winner[0])-winner[1])**2
			y_max= winner[1] if winner[1]>y_max else y_max
			it+=1
			if abs(math.sqrt(e))<=th: #acceptable error was reached
				break
			else:
				o = self.update_o(o,r_,dist_,D,winner,e,y_max,l_r)
		return o


	def update_o(self,o_old,r_,dist_,D,winner,e,y_max,l_r):
		sum_b = 0
		for j in range(len(r_)):
			sum_b+=self.d(j,winner[0],y_max)*r_[j]*dist_[j]/o_old**3
		b_id = 2*sum_b

		sum_l = 0
		for j in range(len(r_)):
			sum_l+=r_[j]*dist_[j]/o_old**3
		l_id = 2*sum_l

		cid_o = (b_id-l_id*winner[1])/D

		e_o = 2*math.sqrt(e)*cid_o

		return o_old+l_r*e_o

	def classify(self,o_,x_):
		classes = set(self.y)
		y_max=0.9
		o=o_#Its gonna be len(y)
		it=0
		pattern = np.copy(self.x)
		u = defaultdict(float)
		r_ = defaultdict(float)
		dist_ = defaultdict(float)
		D = 0
		for j in range(len(pattern)):
			t_j=pattern[j]
			dist_[j]=self.dist(t_j,x_)
			r_[j] = self.r(dist_[j],o[j])
			D+=r_[j]
			for i in classes:
				d_=self.d(j,i,y_max)
				u[i]+=d_*r_[j]

		c = defaultdict(float)
		for i in classes:
			c[i]=u[i]/D
		c=dict(c)

		sort = sorted(c.items(), key=lambda x: x[1], reverse=True)

		winner = sort[0]
		return winner[0]



	def dist(self,t_j,x):
		return np.sum(x-t_j)**2

	def r(self,dist_,o_):
		ret = dist_/(2*o_**2)
		ret = (-1)*ret
		ret = np.exp(ret)

		return ret

	def y_(self,i,j):
		return 0.9 if j==i else 0.1

	def d(self,j,i,y_max):
		y_ij=self.y_(i,j)
		ret=y_ij-y_max
		ret=np.exp(ret)
		ret = ret * y_ij

		return ret



proj = Project()
#proj.main_menu()
proj.x, proj.y = load_iris()
print(len(proj.x))
o= (proj.find_o())
print(proj.classify(o,proj.x[76]))
