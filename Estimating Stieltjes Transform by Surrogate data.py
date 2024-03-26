# -*- coding: utf-8 -*-
"""Value of Stieltjes Transform.ipynb


import os
import sys
import time
import math
import numpy
import matplotlib.pyplot as plt
import pymanopt
import sklearn
import sklearn.datasets
from pymanopt.manifolds import PSDFixedRank, SymmetricPositiveDefinite
import autograd.numpy as np
from pymanopt import Problem
from pymanopt import optimizers
from pymanopt.optimizers import SteepestDescent
from numpy.linalg.linalg import matmul
import matplotlib.pyplot as plt
from numpy.core.numeric import identity
import math
from sklearn.datasets import make_spd_matrix
from scipy.stats import ortho_group
from numpy.random import RandomState

import random

"""n=400, p=2048; This surrogate case corresponds to n=25 and p=128"""

from numpy.core.numeric import identity
L = 1000
p = 4096
n_l = 50
sigma2 = 1
n_new = 800
lamb = p*sigma2/n_new

Omega = np.identity(p)*16
for i in range(p-1):
  Omega[i,i+1] = 5
  Omega[i+1,i] = 5
Sigmasqrt = np.identity(p)
## Generate the data
Ztr = np.matrix(np.random.normal(0,1,size=(n_new,p)))      ### generate Z data matrix
Xtr = np.matmul(Ztr,Sigmasqrt)                 ### generate X data matrix
Sigmahat = (np.array(Xtr)@np.array(Omega)@np.array(Xtr).T)/n_new
Stj = np.matrix.trace(np.linalg.inv(Sigmahat+lamb*np.identity(n_new)))/n_new
print(Stj)
Stjprime = np.matrix.trace(np.linalg.inv(Sigmahat+lamb*np.identity(n_new))@np.linalg.inv(Sigmahat+lamb*np.identity(n_new)))/n_new
print(Stjprime)
r_limit = (1/(lamb*Stj))*(sigma2+(lamb*n_new/p-sigma2)*(1-lamb*Stjprime/Stj))
print(r_limit)