import numpy as np
import math
from collections import defaultdict

class GCNN(object):
	def __init__(self, max_iter=10, l_r=0.3, th=0.01):
		self.max_iter = max_iter
		self.l_r = l_r
		self.th = th
	"""
	This function calculates each step of the gradient descent, taking the function through
	each of the network's layers and producing an update for the o in the end. The rest of the
	functions in this file serve mainly to do this steps, with the exception of the classify
	functions, which serve to classify a particular input, and the fit function, which is in charge
	of iterating over every training data.
	"""
	def calc_o_step(self, o, x_i, y_i):
		it = 0
		y_max = 0.9
		while it < self.max_iter:
			u = defaultdict(float)
			r_ = defaultdict(float)
			dist_ = defaultdict(float)
			D = 0
			for j in range(len(self.pattern)):
				t_j = self.pattern[j]
				dist_[j] = self.dist(t_j, x_i)
				r_[j] = self.r(dist_[j], o[j])
				D += r_[j]
				for i in self.classes:
					d_ = self.d(j, i, y_max)
					u[i] += d_*r_[j]

			c = dict()
			for i in self.classes:
				c[i] = u[i]/D

			sort = sorted(c.items(), key=lambda x: x[1], reverse=True)

			winner = sort[0]
			e = (self.y_(y_i, winner[0]) - winner[1])**2
			y_max = winner[1] if winner[1]>y_max else y_max
			it += 1
			if abs(math.sqrt(e)) <= self.th: #acceptable error was reached
				break
			else:
				o = self.update_o(o, r_ ,dist_, D, \
						  winner, e, y_max)
		return o

	def update_o(self, o_old, r_, dist_, D, winner, e, y_max):
		sum_b = 0
		for j in range(len(r_)):
			sum_b += self.d(j, winner[0], y_max)*r_[j]*dist_[j]/o_old**3
		b_id = 2*sum_b

		sum_l = 0
		for j in range(len(r_)):
			sum_l += r_[j]*dist_[j]/o_old**3
		l_id = 2*sum_l

		cid_o = (b_id - l_id*winner[1])/D

		e_o = 2*math.sqrt(e)*cid_o

		return o_old + self.l_r*e_o

	"""
	This function trains the neural network to the data it gets as a parameter.
	It calculates the aproppiate o and defines the network's pattern layer.
	"""
	def fit(self, x, y):
		self.classes = set(y)
		self.pattern=(np.copy(x))
		self.y_max=0.9
		it=0
		o = np.random.uniform(0.5,1.0,(len(y),))

		for i in range(len(y)):
			o = self.calc_o_step(o, x[i], y[i])

		self.o = o
	"""
	This function classifies one example according to its features and
	the previously trained smoothing parameter.
	"""
	def classify(self, x_test):
		y_max=0.9
		it=0
		u = defaultdict(float)
		r_ = defaultdict(float)
		dist_ = defaultdict(float)
		D = 0
		for j in range(len(self.pattern)):
			t_j = self.pattern[j]
			dist_[j] = self.dist(t_j, x_test)
			r_[j] = self.r(dist_[j], self.o[j])
			D += r_[j]
			for i in self.classes:
				d_ = self.d(j, i, y_max)
				u[i] += d_*r_[j]

		c = dict()
		for i in self.classes:
			c[i]=u[i]/D

		sort = sorted(c.items(), key=lambda x: x[1], reverse=True)

		winner = sort[0]
		return winner[0]

	"""
	This function classifies multiple examples.
	"""
	def classify_batch(self,x_test):
		y_pred=[]
		for x_ in x_test:
			y_pred.append(self.classify(x_))
		return y_pred

	def dist(self,t_j,x):
		return np.sum(x-t_j)**2

	def r(self, dist_, o_):
		ret = dist_/(2*o_**2)
		ret = (-1)*ret
		ret = np.exp(ret)

		return ret

	def y_(self,i,j):
		return 0.9 if j==i else 0.1

	def d(self, j, i, y_max):
		y_ij = self.y_(i,j)
		ret = y_ij - y_max
		ret = np.exp(ret)
		ret = ret * y_ij

		return ret
