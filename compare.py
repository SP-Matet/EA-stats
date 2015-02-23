# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 16:11:12 2015

@author: user
"""

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


N = 1000000
turns = 10

def compare (phi1, phi2, pi1, pi2, rho1, rho2):
    p = phi1.shape[1]
    estimated_quadratic_distance = 0
    h0 = 0
    
    print "Cross-Validating..."
    for i in range (turns):
        print str(i) + " of " + str(turns)
        X = np.random.uniform (low = -1, high = 1, size = N * (p+1))
        X = np.reshape(X, (N, p+1))
        X = np.matrix(X)
        density1 = compute_density (X, phi1, pi1, rho1)
        density2 = compute_density (X, phi2, pi2, rho2)
        estimated_quadratic_distance = estimated_quadratic_distance + np.dot(np.transpose(density1 - density2), (density1 - density2) )
        h0 = h0 + np.sum(np.dot(np.transpose(density1), density1))
        
    print "Estimated quadratic error : " + str(np.sum(estimated_quadratic_distance)/(N*turns))
    print "Null hypothesis error : " + str(h0/(N*turns))
    print "Score : " + str(np.sum(estimated_quadratic_distance / h0))
    
    if (p == 1):
        n = 1000
        k = pi1.shape[1]
        Y = np.random.normal(size = n)
        X = np.random.normal(size = n * p)
        X = np.matrix(np.reshape(X, (n, p)))
        
        random_gen = scipy.stats.rv_discrete (values = (range(k), pi1[0,:]))
        indexes = random_gen.rvs(size = n)
        for i in range (n):
            Y[i] = 1/rho1[0,indexes[i]] * Y[i] +  np.sum(np.dot(np.matrix((X[i,:])), np.transpose(np.matrix(phi1[indexes[i], :])))) / rho1[0,indexes[i]]
        
        X = np.array(X)
        
        plt.plot(X,Y, 'o')
        plt.show
        
        Y = np.random.normal(size = n)
        X = np.random.normal(size = n * p)
        X = np.matrix(np.reshape(X, (n, p)))
        
        random_gen = scipy.stats.rv_discrete (values = (range(k), pi2[0,:]))
        indexes = random_gen.rvs(size = n)
        for i in range (n):
            Y[i] = 1/rho2[0,indexes[i]] * Y[i] +  np.sum(np.dot(np.matrix((X[i,:])), np.transpose(np.matrix(phi2[indexes[i], :])))) / rho2[0,indexes[i]]
        
        X = np.array(X)
        
        plt.plot(X,Y, 'ro')
        
        plt.show
    
    
    
def compute_density (X, phi, pi, rho):
    n = X.shape[0]
    p = X.shape[1] -1
    Y = X[:,p]
    X = X[:,:p]
    A = np.dot(Y, rho)
    B = np.dot(X, np.transpose(phi))
    C = -1/2 * np.square(A-B)
    D = np.exp(C)
    product = np.multiply(pi, rho)
    product = np.dot(np.ones((n, 1)), product)
    E = np.multiply(product, D)
    E = np.matrix(np.sum(E, axis = 1))
    return E