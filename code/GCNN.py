import numpy as np
import math
from collections import defaultdict

class GCNN(object):
	def __init__(self, max_iter=10, l_r=0.3, th=0.01):
		self.max_iter = max_iter
		self.l_r = l_r
		self.th = th

	def calc_o_step(self, o, x_i, y_i):
		it = 0
		y_max = 0.9
		pattern = np.copy(x_i)
		while it < self.max_iter:
			u = defaultdict(float)
			r_ = defaultdict(float)
			dist_ = defaultdict(float)
			D = 0
			for j in range(len(pattern)):
				t_j = pattern[j]
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


	def fit(self, x, y):
		self.classes = set(y)
		self.y_max=0.9
		it=0
		o = np.random.uniform(0.5,1.0,(len(y),))

		for i in range(len(y)):
			o = self.calc_o_step(o, x[i], y[i])

		self.o = o

	def classify(self, x, x_test):
		y_max=0.9
		it=0
		pattern = np.copy(x)
		u = defaultdict(float)
		r_ = defaultdict(float)
		dist_ = defaultdict(float)
		D = 0
		for j in range(len(pattern)):
			t_j = pattern[j]
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
