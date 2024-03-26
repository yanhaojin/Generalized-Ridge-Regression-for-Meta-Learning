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

from numpy.core.numeric import identity
L = 5000
p = 128
n_l = 100
sigma2 = 1
n_new = 100
lamb = p*sigma2/n_new

Omega = np.identity(p)*16
for i in range(p-1):
  Omega[i,i+1] = 5
  Omega[i+1,i] = 5
Sigmasqrt = np.identity(p)
manifold = SymmetricPositiveDefinite(p)

@pymanopt.function.autograd(manifold)
def CostOmega1(Omegatilde):
  Risk_Val = 0
  for k in range(L):
    Risk_Val=Risk_Val+(np.linalg.norm(np.array(y_list[k])@np.array(y_list[k]).T-(np.array(X_list[k])@Omegatilde@np.array(X_list[k].T))/(p),ord="fro"))
  return Risk_Val/L

number_runs = 25
EstRisk1 = []
EstRisk2 = []
EstRisk3 = []
MatErr = []
for i in range(number_runs):
  X_list = []
  y_list = []
  for l in range(L):
    Zl = np.matrix(np.random.normal(0,1,size=(n_l,p)))     ### generate Z data matrix
    Xl = np.matmul(Zl,Sigmasqrt)                ### generate X data matrix
    betal = np.matrix(np.random.multivariate_normal(mean=np.zeros(p),cov=Omega/p)).T
    Xbeta = np.matmul(Xl,betal)
    epsilonl = np.random.normal(0,sigma2,n_l)
    yl = Xbeta+np.matrix(epsilonl).T
    X_list.append(Xl)
    y_list.append(yl)                      ### append data matrix into list
  problem1 = Problem(manifold=manifold,cost=CostOmega1)
  solver1 = pymanopt.optimizers.SteepestDescent(max_time=2000,max_iterations=8000)
  Omega1 = solver1.run(problem1,initial_point=Omega).point
  ## generate the training sample
  Ztr = np.matrix(np.random.normal(0,1,size=(n_new,p)))     ### generate Z data matrix
  Xtr = np.matmul(Ztr,Sigmasqrt)                ### generate X data matrix
  betatr = np.matrix(np.random.multivariate_normal(mean=np.zeros(p),cov=Omega/p)).T
  Xbetatr = np.matmul(Xtr,betatr)
  epsilonltr = np.random.normal(0,sigma2,n_new)
  yltr = Xbetatr+np.matrix(epsilonltr).T
  ## generate the test sample
  Zt = np.matrix(np.random.normal(0,1,size=(200,p)))     ### generate Z data matrix
  Xt = np.matmul(Zt,Sigmasqrt)                ### generate X data matrix
  Xbetat = np.matmul(Xt,betatr)
  epsilont = np.random.normal(0,sigma2,200)
  yt = Xbetat+np.matrix(epsilont).T
  ## Calculate the estimated coefficients
  betahatL = np.linalg.inv(np.array(Xtr).T@np.array(Xtr)+n_new*lamb*np.identity(p))@np.array(Xtr).T@np.array(yltr)
  betahatL2 = np.linalg.inv(np.array(Xtr).T@np.array(Xtr)+n_new*lamb*np.linalg.inv(Omega1))@np.array(Xtr).T@np.array(yltr)
  betahatL3 = np.linalg.inv(np.array(Xtr).T@np.array(Xtr)+n_new*lamb*np.linalg.inv(Omega))@np.array(Xtr).T@np.array(yltr)
  ## Calculate the risk on unseen samples
  yhatL = np.array(np.array(Xt)@betahatL)
  RiskL = (np.linalg.norm(yhatL-yt)**2)/200
  yhatL2 = np.array(np.array(Xt)@betahatL2)
  RiskL2 = (np.linalg.norm(yhatL2-yt)**2)/200
  yhatL3 = np.array(np.array(Xt)@betahatL3)
  RiskL3 = (np.linalg.norm(yhatL3-yt)**2)/200
  EstRisk1.append(RiskL)
  EstRisk2.append(RiskL2)
  EstRisk3.append(RiskL3)
  MatErr.append(np.linalg.norm(Omega1-Omega,ord="fro"))
print(np.mean(MatErr))
print(np.mean(EstRisk1))
print(np.mean(EstRisk2))
print(np.mean(EstRisk3))
