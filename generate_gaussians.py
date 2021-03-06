# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 13:41:21 2015

@author: user
"""

import numpy as np
import scipy.stats
import time
import matplotlib.pyplot as plt


scale_sigma = 0
scale_beta = 10

def generate_gaussian_mix (n,p,k):
    print 'Simulation launched - n = ' + str(n) + ' - p = ' + str(p) + ' - k = ' + str(k)
    t = int(time.time() * 1000)
    
    # Generates Normal law and random data points X
    Y = np.random.normal(size = n)
    X = np.random.uniform(low=0.0, high=5, size = n * p)
    X = np.reshape(X, (n, p))
    print 'X and Y generated in ' + str(int(time.time() * 1000) - t) + ' ms.'
    t = int(time.time() * 1000)
    
    
    # Generates random parameters
    #sigma = np.random.normal(loc = 1.0, scale = scale_sigma, size = k)
    sigma = np.ones(k)
    #sigma = np.abs(sigma)
    beta = np.random.normal(loc = 10.0, scale = scale_beta, size = k*p)
    beta = np.reshape(beta, (k,p))
    beta[0,:] = 0
    pi = np.random.uniform(size = k)
    pi = pi / sum(pi)
    print 'Parameters generated in ' + str(int(time.time() * 1000) - t) + ' ms.'
    t = int(time.time() * 1000)
    
    
    # Picks the gaussian according to which which each data point is drawn
    random_gen = scipy.stats.rv_discrete (values = (range(k), pi))
    indexes = random_gen.rvs(size = n)
    print 'Gaussians chosen in ' + str(int(time.time() * 1000) - t) + ' ms.'
    t = int(time.time() * 1000)
    
    # Multiplies everything
    for i in range (n):
        Y[i] = Y[i] * sigma[indexes[i]] + np.sum(np.dot(X[i,:], np.transpose(beta[indexes[i], :])))
    print 'Multiplication done in ' + str(int(time.time() * 1000) - t) + ' ms.'
    
    rho = 1/sigma
    phi = np.divide(beta, np.dot(np.transpose(np.matrix(sigma)), np.ones((1,p))))
    
    rho = np.matrix(rho)
    pi = np.matrix(pi)  
    
    return np.matrix(X), np.transpose(np.matrix(Y)), np.matrix(phi), rho, pi
    