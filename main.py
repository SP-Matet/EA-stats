# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 19:06:27 2015

@author: user
"""

from BCD_GEM import *
from compare import *
from generate_gaussians import *

N = 10000
P = 1
K = 2

X, Y, phi1, rho1, pi1 = generate_gaussian_mix (N, P, K)
phi2, rho2, pi2 = BCD_GEM (X,Y, K)
compare (phi1, phi2, pi1, pi2, rho1, rho2)