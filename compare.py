# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 16:11:12 2015

@author: user
"""

import numpy as np

N = 10000

def compare (phi1, phi2, pi1, pi2, rho1, rho2):
    p = rho1.shape[1]
    X = np.random.uniform (low = -2, high = 2, size = N * (p+1))
    X = np.reshape(X, (N, p+1))
    X = np.matrix(X)
    density1 = compute_density (X, phi1, pi1, rho1)
    density2 = compute_density (X, phi2, pi2, rho2)
    estimated_quadratic_distance = np.dot(np.transpose(density1 - density2), (density1 - density2) )
    print "Estimated quadratic error : " + str(np.sum(estimated_quadratic_distance)/N)
    print "Null hypothesis error : " + str(np.sum(np.dot(np.transpose(density1), density1))/N)
    
    
def compute_density (X, phi, pi, rho):
    n = X.shape[0]
    p = X.shape[1] -1
    k = pi.shape[1]
    Y = X[:,p]
    X = X[:,:p]
    A = np.dot(Y, rho)
    B = np.dot(X, phi)
    C = -1/2 * np.square(A-B)
    D = np.exp(C)
    product = np.multiply(pi, rho)
    product = np.dot(np.ones((n, 1)), product)
    E = np.multiply(product, D)
    E = np.matrix(np.sum(E, axis = 1))
    return E