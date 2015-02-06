# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 13:41:21 2015

@author: user
"""

import numpy as np
import scipy.stats
import time


def generate_gaussian_mix (n,p,k):
    print 'Simulation launched - n = ' + str(n) + ' - p = ' + str(p) + ' - k = ' + str(k)
    t = int(time.time() * 1000)
    
    # Generates Normal law and random data points X
    Y = np.random.normal(size = n)
    X = np.random.uniform(size = n * p)
    X = np.reshape(X, (n, p))
    print 'X and Y generated in ' + str(int(time.time() * 1000) - t) + ' ms.'
    t = int(time.time() * 1000)
    
    
    # Generates random parameters
    sigma = np.random.normal(size = k)
    sigma = np.abs(sigma)
    beta = np.random.normal(size = k*p)
    beta = np.reshape(beta, (k,p))
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
    Y = np.einsum('ij,ij->i', X, beta[indexes,:]) + np.multiply(Y,sigma[indexes])
    print 'Multiplication done in ' + str(int(time.time() * 1000) - t) + ' ms.'
    
    rho = 1/sigma
    phi = np.divide(beta, np.dot(np.transpose(np.matrix(sigma)), np.ones((1,p))))
    
    return X, Y, rho, phi, pi
    
    
X, Y, rhoTh, phiTh, piTh = generate_gaussian_mix (100, 10, 5)