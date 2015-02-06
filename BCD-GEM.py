# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 11:04:03 2015

@author: user
"""

import numpy as np

# X : chaque ligne correspond à une réalisation
# Y : chaque ligne correspond à la variable observée
# Les lignes de X correspondent à celles de Y

   
# Parameters
delta = 0.1
myLambda = 0.01
lautreGamma = 0
petitSeuil = 0
seuilConvergence = 0.00001


def BCD_GEM (X, Y, k):
    X = np.matrix(X)
    Y = np.transpose(np.matrix(Y))
    n = X.shape[0]
    p = X.shape[1]
    
    # Initialisation (cf. article)
    # Picks the gaussian according to which which each data point is drawn
    indexes = np.random.randint(k, size = n)
    gamma = np.matrix(np.ones((n,k)))
    for i in range (n):
        gamma[i,indexes[i]] = 9
    gamma[:,:] = gamma / (k + 8)
    
    pi = np.matrix(np.ones(k) / k)
    phi = np.matrix(np.zeros((k,p)))
    rho = np.matrix(np.ones(k) * 2)
        
    while (True):
        # M Step
        new_phi, new_rho, new_pi = M_Step (phi, rho, pi, X, Y, gamma, n, p, k)
        
        # If convergence then break
        difference = np.linalg.norm(new_phi - phi) / (k*p) + np.linalg.norm(new_rho - rho) / k + np.linalg.norm(new_pi - pi) / k
        print difference
        if (difference < seuilConvergence):
            break
        
        # E Step 
        phi = new_phi
        rho = new_rho
        pi = new_pi
        gamma = E_Step(phi, rho, pi, X, Y, n, p, k)
        
    return phi, rho, pi
    

        
# For M-Step, adjustment of pi
def log_likelihood_pi (gamma, pi, phi, myLambda, lautreGamma):
    resultat = np.sum(np.dot(gamma, np.transpose(np.matrix(np.log(pi)))))
    resultat = -1 * resultat / gamma.shape[0]
    resultat = resultat + myLambda * np.dot(np.power(pi,lautreGamma), np.sum(np.abs(phi), axis = 1))
    return resultat
    
# Performs E-Step, reurns gamma
def E_Step (phi, rho, pi, X, Y, n, p, k):
    k = rho.shape[0]
    n = X.shape[0]
    A = np.dot(Y, rho)
    B = np.dot(X, np.transpose(phi))
    C = -1/2 * np.square(A-B)
    C = np.exp(C)
    product = np.multiply(pi, rho)
    product = np.dot(np.ones((n, 1)), product)
    C = np.multiply(product, C)
    gamma = np.divide(C, np.dot(np.sum(C, axis = 1), np.ones((1,k))))
    return gamma

# Generalized M Step, returns phi, rho, pi
def M_Step (phi, rho, pi, X, Y, gamma, n, p, k):
    # Adjust pi
    pi_barre = np.sum(gamma, axis = 0) / n
    valeur_initiale = log_likelihood_pi (gamma, pi, phi, myLambda, lautreGamma)
    t=1
    valeur = log_likelihood_pi(gamma, pi + t*(pi_barre - pi), phi, myLambda, lautreGamma)
    while valeur > valeur_initiale + petitSeuil:
        t = t * delta
        valeur = log_likelihood_pi(gamma, pi + t*(pi_barre - pi), phi, myLambda, lautreGamma)
    if (valeur < valeur_initiale):
        pi = pi + t*(pi_barre - pi)
        
    # Compute rho and phi
    for r in range (k):
        buff = np.dot(np.sqrt(gamma[:,r]), np.matrix(np.ones(p)) )
        Xtilde = np.multiply(buff, X)
        Ytilde = np.multiply(np.sqrt(gamma[:,r]), Y)
        prod = np.dot(np.transpose(Ytilde), np.dot(Xtilde, np.transpose(phi[r,:])))
        norme2 = np.dot(np.transpose(Ytilde),Ytilde)
        nr = np.sum(gamma[:,r])
        if (norme2 == 0):
            print "WAAAAAAAAAAAAH SAUVE QUI PEUT !"
        rho[0,r] = (prod + np.sqrt(prod**2 + 4*norme2*nr)) / (2 * norme2)
        
        for j in range (p):
            Sj = -1 * rho[0,r] * np.dot(np.transpose(Xtilde[:,j]), Ytilde)
            Sj = Sj + np.sum(np.multiply(np.dot (np.transpose(Xtilde[:,j]),Xtilde), np.transpose(phi[r,:])))
            Sj = Sj[0,0]
            threshold = n * myLambda*(pi[0,r])**lautreGamma
            print str(Sj) + " - " + str(threshold)
    
            if (Sj > threshold):
                print 'pof'
                phi[r,j] = (threshold - Sj) / np.dot(np.transpose(Xtilde[:,j]), Xtilde[:,j])
            elif (Sj < -1 * threshold):
                phi[r,j] = -1 * (threshold + Sj) /  np.dot(np.transpose(Xtilde[:,j]), Xtilde[:,j])
                print 'pof'
            else:
                phi[r,j] = 0
                
    return phi, rho, pi