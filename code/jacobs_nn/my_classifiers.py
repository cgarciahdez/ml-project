import numpy as np
from collections import defaultdict
from math import sqrt, log, e
from sklearn.preprocessing import PolynomialFeatures

def sigmoid(X):
    # X must be np.ndarray
    return 1.0 / (1.0 + e ** (-1.0 * X))

class Classifier(object):
    # base classifier object
    def fit(self):
        return "base"

    def predict(self):
        return "base"

class MyLogisticRegression(Classifier):
    def __init__(self, iterations = 200, alpha = 0.01, \
                    threshold = 0.01, degree = 1):
        self.iterations = iterations
        self.min_iterations = 5
        self.threshold = threshold
        #I've tried different values for alpha for higher degrees,
        #but I always have either underflowed and gotten linear answers,
        #or exploded and crashed
        #I didn't have time to create a dynamic alpha
        self.balpha = alpha
        self.degree = degree

    def fit(self, train, labels):
        #preprocess data
        self.classes = sorted(list(set(labels)))
        y = np.array([self.classes.index(l) for l in labels])
        m = len(train)

        arr_train = np.array(train)
        trans = PolynomialFeatures( degree = self.degree, \
                                    interaction_only = True,
                                    include_bias = True)
        X = trans.fit_transform(arr_train)
        self.n_feats = len(X[0])
        #one vs many classifier
        if len(y) > 2:
            self.alpha = self.balpha * np.ones([len(self.classes), \
                                                self.n_feats])
        else: #if len(y) < 2:
            self.alpha = self.balpha * np.ones(self.n_feats)
#        self.J_history = []
        
        self.theta = self.gradient_descent(X, y)
    
    def gradient_descent(self, X, y):
        #set up variables
        m = float(len(X))
        theta = np.ones(self.alpha.shape)
        old_theta = None

        #iterate self.iterations number of times
        for i in range(self.iterations):
            old_theta = theta
            theta = self.gd_step(m, theta, X, y)
        return theta

    def gd_step(self, m, theta, X, y):
        if theta.ndim == 1:
            return self.class_compare(m, theta, X, y)
        else: # if theta.ndim > 1
            for i in range(len(self.classes)):
                c_y = np.array([1 if j == i else 0 for j in y])
                theta[i] = self.class_compare(m, theta[i], X, c_y)
            return theta

#            print(theta)

    def class_compare(self, m, theta, X, y):
        #predict value of each y with current theta
        #calculate h for each feature vector
        pred = np.dot(X, theta.T)
        # difference from true y
        # compute output with sigmoid function
        h_theta = sigmoid(pred) 
        h_theta.flatten()
#        J_theta = self.cost_func(h_theta, y)
        y_err = h_theta - y
        #calcualte the gradient upon which to descend
        grads = (y_err.T).dot(X) / m
#        self.J_history.append(J_theta)
        return theta - (grads * theta)

    # J(theta)
    def cost_func(self, h_theta, y):
        m = len(y)
        return (1.0/m) * sum(-1*(y.T.dot(h_theta)) - \
                ((1-y).T.dot(1 - h_theta)))

    def predict(self, test):
        trans = PolynomialFeatures( degree = self.degree, \
                                    interaction_only = True,
                                    include_bias = True)
        x = trans.fit_transform(np.array(test))
        res = sigmoid(x.dot(self.theta.T)).flatten()
        if len(self.classes) == 2:
            if res < 0.5:
               return classes[0]
            else:
               return classes[1]
        maxima = [None, -1]
        for i in range(len(self.classes)):
            if res[i] > maxima[1]:
                maxima = [i, res[i]]
        return self.classes[maxima[0]]

class MyFeedForwardMLP(Classifier):
    def __init__(self, iterations = 400, alpha = 0.002, \
                    threshold = 0.01, degree = 1):
        self.iterations = iterations
        self.min_iterations = 5
        self.threshold = threshold
        #I've tried different values for alpha for higher degrees,
        #but I always have either underflowed and gotten linear answers,
        #or exploded and crashed
        #I didn't have time to create a dynamic alpha
        self.balpha = alpha
        self.degree = degree

    def fit(self, train, labels):
        #preprocess data
        self.classes = sorted(list(set(labels)))
        y = np.array([self.classes.index(l) for l in labels])
        m = len(train)

        arr_train = np.array(train)
        trans = PolynomialFeatures( degree = self.degree, \
                                    interaction_only = True,
                                    include_bias = True)
        X = trans.fit_transform(arr_train)
        self.n_feats = len(X[0])
        #one vs many classifier
        self.alpha = self.balpha * np.ones([len(self.classes), \
                                            self.n_feats])
        self.beta = self.balpha * np.ones(len(self.classes))
#        self.J_history = []
        
        self.w, self.v = self.gradient_descent(X, y)
    
    def gradient_descent(self, X, y):
        #set up variables
        m = float(len(X))
        w = np.ones(self.alpha.shape)
        v = np.ones(self.beta.shape)
        old_w = None
        old_v = None

        #iterate self.iterations number of times
        for i in range(self.iterations):
            old_w = w
            w = self.gd_step(m, w, X, y)
            wX = X.dot(w.T)
            v = self.gd_step(m, v, wX, y)
        return w, v

    def gd_step(self, m, theta, X, y):
        for i in range(len(self.classes)):
            c_y = np.array([1 if j == i else 0 for j in y])
            theta[i] = self.class_compare(m, theta[i], X, c_y)
        return theta

    def class_compare(self, m, theta, X, y):
        #predict value of each y with current theta
        #calculate h for each feature vector
        pred = np.dot(X, theta.T)
        # difference from true y
        # compute output with sigmoid function
        h_theta = sigmoid(pred) 
        h_theta.flatten()
#        J_theta = self.cost_func(h_theta, y)
        try:
            y_err = h_theta - y
        except ValueError:
            y_err = h_theta - np.array([list(y),]*h_theta.shape[1]).T
        #calcualte the gradient upon which to descend
        grads = (y_err.T).dot(X) / m
#        self.J_history.append(J_theta)
        return (theta - (grads * theta))[0]

    # J(theta)
    def cost_func(self, h_theta, y):
        m = len(y)
        return (1.0/m) * sum(-1*(y.T.dot(h_theta)) - \
                ((1-y).T.dot(1 - h_theta)))

    def predict(self, test):
        trans = PolynomialFeatures( degree = self.degree, \
                                    interaction_only = True,
                                    include_bias = True)
        x = trans.fit_transform(np.array(test))

        pred = np.dot(x, self.w.T)
        h_theta = sigmoid(pred) 
        h_theta.flatten()
        
        res = self.v.T.dot(h_theta)

        maxima = [None, -1]
        for i in range(len(self.classes)):
            if res[i] > maxima[1]:
                maxima = [i, res[i]]
        return self.classes[maxima[0]]
