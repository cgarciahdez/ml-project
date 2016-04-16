import numpy as np
from collections import defaultdict
from math import log, sqrt

class OGCNN(object):

    def fit(self, x, y):
        self.pattern = np.copy(x)
        self.classes = set(y)
        self.n_feats = x.shape[1]
        mu_sigma = self.compute_mu_sigma(x, y)

        temp_o = self.calc_smoothing_parameters(mu_sigma)
        self.o = np.array([temp_o[j] for j in y])

    def compute_mu_sigma(self, X, y):
        # dict mapping class to [mu, sigma, #] trios
        mu_sigma = defaultdict(lambda: [np.zeros(self.n_feats), \
                                        np.zeros(self.n_feats), \
                                        0.0])

        # compute mean
        for x_i, y_i in zip(X, y):
            # add all x_i that belong to class j
            mu_sigma[y_i][0] = mu_sigma[y_i][0] + x_i
            # total number of class j
            mu_sigma[y_i][2] += 1
        # finish computing mean
        for key in mu_sigma.keys():
            mu_sigma[key][0] /= mu_sigma[key][2]

        # compute sigma
        for x_i, y_i in zip(X, y):
            # sum MSE
            mu_sigma[y_i][1] += (mu_sigma[y_i][0] - x_i) ** 2
        # average MSE and square root
        for key in mu_sigma.keys():
            mu_sigma[key][1] /= mu_sigma[key][2] - 1
            mu_sigma[key][1] = np.array(list(map(sqrt, mu_sigma[key][1])))

        return dict(mu_sigma)
        
    def calc_class_priors(self, y):
        priors = defaultdict(lambda: 0.0)
        for l in y:
            priors[l] += 1
        for key in priors.keys():
            priors[key] /= len(y)

        return dict(priors)

    def above_threshold(self, V_i, M_i):
        return abs(log(sum(abs(V_i - M_i))))

    def below_threshold(self, ma, mi):
        return 0.5 * (ma - mi)

    def max_min_V(self, st_devs):
        ma = max(st_devs)
        mi = min(st_devs)
        ma_ft = 0
        for s in range(len(st_devs)):
            if ma == st_devs[s]:
                ma_ft = s
                break
        return ma, mi, ma_ft

    def calc_smoothing_parameters(self, mu_sigma):
        # iterate over the classes
        o = {}
        for cl in mu_sigma.keys():
            """
            Vmax and Vmin standard deviations for current class
            Vmax and Vmin are the SDs for the most dispersed
            and least dispersed classes
            """
            m_v = mu_sigma[cl]
            M_i = m_v[0]
            V_i = m_v[1]
            Vmax_i, Vmin_i, ma_ft = self.max_min_V(V_i)
            if Vmax_i > 1 and (Vmax_i / M_i[ma_ft]) > 0.1:
                o[cl] = self.above_threshold(V_i, M_i)
            else:
                o[cl] = self.below_threshold(Vmax_i, Vmin_i)

        return o

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

#X = np.arange(20).reshape(10,2)
#y = np.array([1,1,1,1,0,0,0,0,0,0])
#mv = compute_mu_sigma(X,y)
#sigma = calc_smoothing_parameters(mv, X.shape[1])
#print(sigma)
