# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 11:04:03 2015

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

# X : chaque ligne correspond à une réalisation
# Y : chaque ligne correspond à la variable observée
# Les lignes de X correspondent à celles de Y

   
# Parameters
delta = 0.1
lautreGamma = 0
petitSeuil = 0
seuilConvergence = 0.0001
TURNS = 100
myLambda = 0


# Performs GEM algorithm
# Returns estimated phi, rho, pi
def BCD_GEM (X, Y, k, l):
    X = np.matrix(X)
    Y = np.matrix(Y)
    n = X.shape[0]
    p = X.shape[1]
    
    
    print "delta=" +str(delta)
    print "lambda=" +str(myLambda)
    print "gamma=" +str(lautreGamma)
    
    
    # Initialisation (cf. article)
    # Picks the gaussian according to which which each data point is drawn
    indexes = np.random.randint(k, size = n)
    gamma = np.matrix(np.zeros((n,k)))
    for i in range (n):
        gamma[i,indexes[i]] = 9
    gamma[:,:] = gamma / (k + 8)
    
    pi = np.matrix(np.ones(k) / k)
    phi = np.matrix(np.zeros((k,p)))
    rho = np.matrix(np.ones(k) * 2)
        
    for turn in range(TURNS):
        # M Step
        phi, rho, pi = M_Step (phi, rho, pi, X, Y, gamma, n, p, k)   
        
        if (np.isnan(np.sum(phi) + np.sum(rho) + np.sum(pi))):
            print 'Problème avant gamma'
            sys.exit()
        
        # E Step
        gamma = E_Step(phi, rho, pi, X, Y, n, p, k)
        
    return phi, rho, pi
    

# Compute a partial log likelihood for pi        
# CHECKED
def log_likelihood_pi (gamma, pi, phi):
    resultat = np.sum(np.dot(gamma, np.transpose(np.log(pi))))
    resultat = -1 * resultat / gamma.shape[0]
    resultat = resultat + myLambda * np.dot(np.power(pi,lautreGamma), np.sum(np.abs(phi), axis = 1))
    return resultat
    
    
# Performs E-Step, reurns gamma
# CHECKED
def E_Step (phi, rho, pi, X, Y, n, p, k):
    A = np.dot(Y, rho)
    B = np.dot(X, np.transpose(phi))
    C = -1/2 * np.square(A-B)
    
    for i in range (n):
        C[i,:] = C[i,:] - np.max(C[i,:]) # Avoid problems with very low exponents        
        
    D = np.exp(C)
    product = np.multiply(pi, rho)
    product = np.dot(np.ones((n, 1)), product)
    E = np.multiply(product, D)    
    gamma = np.matrix(np.zeros((n,k)))
    # gamma = np.divide(E, np.dot(np.sum(E, axis = 1), np.ones((1,k))))
    
    for i in range (n):
        gamma [i,np.argmax(E[i,:])] = 1
    
    if (np.isnan(np.sum(gamma))):
        print 'problème gamma'
        sys.exit()
            
    return gamma


# Generalized M Step, returns phi, rho, pi
def M_Step (phi, rho, pi, X, Y, gamma, n, p, k):
    # Adjust pi
    # À peu près checked
    pi_barre = np.sum(gamma, axis = 0) / n
    
    if (lautreGamma != 0 and myLambda != 0) :
        valeur_initiale = log_likelihood_pi (gamma, pi, phi)
        t=1
        valeur = log_likelihood_pi(gamma, pi_barre - pi, phi)
        while valeur > valeur_initiale:
            t = t * delta
            valeur = log_likelihood_pi(gamma, pi + t*(pi_barre - pi), phi)
        
        pi = pi + t*(pi_barre - pi)
    
    else :
        pi = pi_barre
        
    # Compute rho and phi
    for r in range (k):
        buff = np.dot(np.sqrt(gamma[:,r]), np.matrix(np.ones(p)) )
        Xtilde = np.multiply(buff, X)
        Ytilde = np.multiply(np.sqrt(gamma[:,r]), Y)
        prod = np.dot(np.transpose(Ytilde), np.dot(Xtilde, np.transpose(phi[r,:])))
        norme2Y = np.dot(np.transpose(Ytilde),Ytilde)
        nr = np.sum(gamma[:,r])
        
        if (norme2Y != 0):
            rho[0,r] = (prod + np.sqrt(prod**2 + 4*norme2Y*nr)) / (2 * norme2Y)
            rho[0,r] = 1
            
            for j in range (p):     
                norme2X = np.sum(np.dot(np.transpose(Xtilde[:,j]), Xtilde[:,j]))                
                if (norme2X != 0):
                    Sj = rho[0,r] * np.dot(np.transpose(Xtilde[:,j]), Ytilde)
                    buff = np.multiply(np.dot (np.transpose(Xtilde[:,j]),Xtilde), np.transpose(phi[r,:]))
                    Sj = np.sum(buff) - buff[0,j] - Sj
        
                    threshold = n * myLambda*(pi[0,r]**lautreGamma)
                    
                    if (Sj > threshold):
                        phi[r,j] = (threshold - Sj) / norme2X
                    elif (Sj < -1 * threshold):
                        phi[r,j] = -1*(threshold + Sj) / norme2X
                    else:
                        phi[r,j] = 0
                 
    return phi, rho, pi