#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class MyLinearRegressor():

    def __init__(self, kappa=0.002, lamb=10, max_iter=200, opt='sgd'):
        self._kappa = kappa
        self._lamb = lamb
        self._opt = opt
        self._max_iter = max_iter

    def fit(self, X, y):
        X = self.__feature_rescale(X)
        X = self.__feature_prepare(X)
        error = []
        if self._opt == 'sgd':
            error = self.__stochastic_gradient_descent(X, y)
        elif self._opt == 'batch':
            error = self.__batch_gradient_descent(X, y)
        elif self._opt == 'isgd':
            error = self.__improved_stochastic(X, y)
        else:
            print 'unknow opt'
        return error

    def predict(self, X):
        pass

    def __batch_gradient_descent(self, X, y):
        N, M = X.shape                                                  #N = 506 (rows/instances), M = 14(columns/features)
        niter = 0                                                       #Iteration counter
        error = []                                                      #list of running errors from total_error()
        self._w = np.ones(X.shape[1])                                   #[1. 1. 1. 1. 1. ...] 14 items b/c X.shape[1] = 14
        
        #Initial error with unoptimized W's
        error.append(self.__total_error(X, y, self._w))
        
        #Loop the predefined number of times (= max_iter)
        for j in range(self._max_iter):

            #update all W's simultaneously to the result of the gradient decent equation
            self._w += self._kappa * (1.0/N) * sum([(y[i] - np.dot(self._w.transpose(), X[i]))*X[i] for i in range(N)])
            
            #Add running errors to the final error list
            error.append(self.__total_error(X, y, self._w))
        
        #debugging print statements
        print 'BATCH'
        print 'INITIAL ERROR: ', error[0]
        print 'LOWEST ERROR: ', min(error)
        print 'FINAL ERROR: ', error[-1]

        return error

    def __stochastic_gradient_descent(self, X, y):
        N, M = X.shape
        niter = 0
        error = []
        self._w = np.ones(X.shape[1])

        #Initial error with unoptimized W's
        error.append(self.__total_error(X, y, self._w))
        
        #Loop all the training instances an arbitrary amount of times (5x)
        for i in range(N):

            #update all W's simultaneously to the result of the gradient decent equation
            self._w += self._kappa * ((y[i%N] - np.dot(self._w,X[i%N]))*X[i%N])
            
            #Add running errors to the final error list
            error.append(self.__total_error(X, y, self._w))


            #----L2 REGULARIZATION--------

            #update all W's simultaneously to the result of the gradient decent equation *WITH L2 REGULARIZATION*
            #self._w += self._kappa * (np.sum(y[i%N] - (np.dot(X[i%N], self._w)))   +
            #                         (self._lamb * np.sum(self._w[j] for j in range(1, M))))

            #Add running errors *WITH L2 REGULARIZATION* to the final error list 
            #error.append(   (1.0/(2*len(y)) * np.sum(y[i%N] - np.dot(X[i%N], self._w))**2)   +
            #                (self._lamb * np.sum(self._w[j]**2 for j in range(1,M)))     )

            #-----------------------------

        #debugging print statements
        print 'SGD'
        print 'INITIAL ERROR: ', error[0]
        print 'LOWEST ERROR:  ', min(error)
        print 'FINAL ERROR:   ', error[-1]
        #print 'self._w:       ', self._w
        #print error

        return error

    def __improved_stochastic(self, X, y):
        N, M = X.shape
        niter = 0
        error = []
        self._w = np.ones(X.shape[1])
        G = np.zeros((X.shape[1], X.shape[1]))
        return error

    def __total_error(self, X, y, w):
        tl = 0.5 * np.sum((np.dot(X, w) - y)**2)/len(y)
        return tl

    # add a column of 1s to X
    def __feature_prepare(self, X_):
        M, N = X_.shape
        X = np.ones((M, N+1))
        X[:, 1:] = X_
        return X

    # rescale features to mean=0 and std=1
    def __feature_rescale(self, X):
        self._mu = X.mean(axis=0)
        self._sigma = X.std(axis=0)
        return (X - self._mu)/self._sigma


if __name__ == '__main__':
    from sklearn.datasets import load_boston

    data = load_boston()
    X, y = data['data'], data['target']
    mylinreg = MyLinearRegressor()
    mylinreg.fit(X, y)
