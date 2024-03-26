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

## Esitmate using surrogate data

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


## Fixed point algorithm
import scipy.integrate as integrate
import scipy.special as special
from scipy.integrate import quad

def lsdfun(x,vt):
  return x/((1+vt*x)*np.pi*np.sqrt(100-(x-16)**2))
def lsdfun2(x,vt):
  return x**2/(((1+vt*x)**2)*np.pi*np.sqrt(100-(x-16)**2))
def risk_fpa(z,gamma1,sgm):
  v_previous = 0
  v_new = 0
  for i in range(20000):
    v_previous = v_new
    v_new = 1/(gamma1*quad(lsdfun,6,26,args=(v_new))[0]-z)
  v_prime = 1/(1/(v_new)**2-gamma1*quad(lsdfun2,6,26,args=(v_new))[0])
  risk_val = (1/(-z*v_new))*(sgm+(-z/gamma1-sgm)*(1+z*v_prime/v_new))
  print(v_new)
  print(v_prime)
  print(risk_val)
  return(risk_val)
