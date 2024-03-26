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
 
L = 1000
p = 128
n_l = 50
sigma2 = 1
n_new = 25
clist = [0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2]


Omega = np.identity(p)*16
for i in range(p-1):
  Omega[i,i+1] = 5
  Omega[i+1,i] = 5
Sigmasqrt = np.identity(p)
manifold = SymmetricPositiveDefinite(p)
IniMat = np.identity(p)*2
for i in range(p-1):
  IniMat[i,i+1] = 1
  IniMat[i+1,i] = 1

@pymanopt.function.autograd(manifold)
def CostOmega1(Omegatilde):
  Risk_Val = 0 
  for k in range(L):
    Risk_Val=Risk_Val+(np.linalg.norm(np.array(y_list[k])@np.array(y_list[k]).T-(np.array(X_list[k])@Omegatilde@np.array(X_list[k].T))/(p),ord="fro"))
  Risk_Val = Risk_Val/L
  ## Adding penalty
  for k in range(p):
    for j in range(p):
      if (k!=j): 
        Risk_Val = Risk_Val+0.00035*abs(Omegatilde[k,j])
  return Risk_Val


number_runs = 50
MeanRisk1 = []
MeanRisk2 = []
MeanRisk3 = []
MeanErrornorm = []
EstRisk1 = []
EstRisk2 = []
EstRisk3 = []
Errornorm = []
for c in clist:
  print(c)
  EstRisk1 = []
  EstRisk2 = []
  EstRisk3 = []
  Errornorm = []
  lamb = c*p*sigma2/n_new
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
    solver1 = pymanopt.optimizers.SteepestDescent(max_time=21600,max_iterations=80000)  
    Omega1 = solver1.run(problem1,initial_point=np.identity(p)).point
    #Omega2 = solver1.run(problem1).point
    ## generate the training sample
    Ztr = np.matrix(np.random.normal(0,1,size=(n_new,p)))     ### generate Z data matrix
    Xtr = np.matmul(Ztr,Sigmasqrt)                 ### generate X data matrix 
    betatr = np.matrix(np.random.multivariate_normal(mean=np.zeros(p),cov=Omega/p)).T 
    Xbetatr = np.matmul(Xtr,betatr)
    epsilonltr = np.random.normal(0,sigma2,n_new) 
    yltr = Xbetatr+np.matrix(epsilonltr).T
    ## generate the test sample
    Zt = np.matrix(np.random.normal(0,1,size=(100,p)))      ### generate Z data matrix
    Xt = np.matmul(Zt,Sigmasqrt)                  ### generate X data matrix  
    Xbetat = np.matmul(Xt,betatr) 
    epsilont = np.random.normal(0,sigma2,100) 
    yt = Xbetat+np.matrix(epsilont).T
    ## Calculate the estimated coefficients
    betahatL = np.linalg.inv(np.array(Xtr).T@np.array(Xtr)+n_new*lamb*np.identity(p))@np.array(Xtr).T@np.array(yltr)
    betahatL2 = np.linalg.inv(np.array(Xtr).T@np.array(Xtr)+n_new*lamb*np.linalg.inv(Omega1))@np.array(Xtr).T@np.array(yltr)
    betahatL3 = np.linalg.inv(np.array(Xtr).T@np.array(Xtr)+n_new*lamb*np.linalg.inv(Omega))@np.array(Xtr).T@np.array(yltr)
    ## Calculate the risk on unseen samples
    yhatL = np.array(np.array(Xt)@betahatL)
    RiskL = (np.linalg.norm(yhatL-yt)**2)/100
    yhatL2 = np.array(np.array(Xt)@betahatL2)
    RiskL2 = (np.linalg.norm(yhatL2-yt)**2)/100
    yhatL3 = np.array(np.array(Xt)@betahatL3)
    RiskL3 = (np.linalg.norm(yhatL3-yt)**2)/100
    EstRisk1.append(RiskL)
    EstRisk2.append(RiskL2)
    EstRisk3.append(RiskL3) 
    Errornorm.append(np.linalg.norm(Omega1-Omega,ord="fro"))
  MeanErrornorm.append(np.mean(Errornorm))
  MeanRisk1.append(np.mean(EstRisk1))
  MeanRisk2.append(np.mean(EstRisk2))
  MeanRisk3.append(np.mean(EstRisk3)) 
print(MeanRisk1)
print(MeanRisk2)
print(MeanRisk3)
