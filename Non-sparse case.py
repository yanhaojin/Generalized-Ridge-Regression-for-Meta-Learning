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
from numpy.core.numeric import identity

### Settings in the experiment for Sigma being identity matrix and Omega being Toplitz
number_runs = 50
L = 10000
p = 128
n_l = 50
sigma2 = 1
n_new = 25
lamb = p*sigma2/n_new
Omega = np.identity(p)*16
for i in range(p-1):
  Omega[i,i+1] = 5
  Omega[i+1,i] = 5
Sigmasqrt = np.identity(p)

### spdm1 to spdm4 is four different initialization generated randomly by make_spd_matrix
manifold = SymmetricPositiveDefinite(p)
spdm1 = make_spd_matrix(n_dim=p, random_state=1)
spdm2 = make_spd_matrix(n_dim=p, random_state=2)
spdm3 = make_spd_matrix(n_dim=p, random_state=3)
spdm4 = make_spd_matrix(n_dim=p, random_state=4)

### Initialize the risk and error
MoM_1_EstRisk1 = []
MoM_1_EstRisk2 = [] 
MoM_2_EstRisk1 = []
MoM_2_EstRisk2 = [] 
MoM_3_EstRisk1 = []
MoM_3_EstRisk2 = [] 
MoM_4_EstRisk1 = []
MoM_4_EstRisk2 = [] 
MoM_5_EstRisk1 = []
MoM_5_EstRisk2 = [] 

MLE_1_EstRisk1 = []
MLE_1_EstRisk2 = [] 
MLE_2_EstRisk1 = []
MLE_2_EstRisk2 = [] 
MLE_3_EstRisk1 = []
MLE_3_EstRisk2 = [] 
MLE_4_EstRisk1 = []
MLE_4_EstRisk2 = [] 
MLE_5_EstRisk1 = []
MLE_5_EstRisk2 = [] 

MoM_1_Errornorm = []
MoM_2_Errornorm = []
MoM_3_Errornorm = []
MoM_4_Errornorm = []
MoM_5_Errornorm = [] 

MLE_1_Errornorm = []
MLE_2_Errornorm = []
MLE_3_Errornorm = []
MLE_4_Errornorm = []
MLE_5_Errornorm = [] 

MLE_6_EstRisk1 = []
MLE_6_EstRisk2 = [] 
MLE_7_EstRisk1 = []
MLE_7_EstRisk2 = [] 
MLE_8_EstRisk1 = []
MLE_8_EstRisk2 = [] 
MLE_9_EstRisk1 = []
MLE_9_EstRisk2 = [] 
MLE_10_EstRisk1 = []
MLE_10_EstRisk2 = [] 

MLE_6_Errornorm = []
MLE_7_Errornorm = []
MLE_8_Errornorm = []
MLE_9_Errornorm = []
MLE_10_Errornorm = []

### CostOmega1 and MLEfun are two functions returning the value of loss function for MLE and proposed estimator

@pymanopt.function.autograd(manifold)
def CostOmega1(Omegatilde):
  Risk_Val = 0 
  for k in range(L):
    Risk_Val=Risk_Val+(np.linalg.norm(np.array(y_list[k])@np.array(y_list[k]).T-(np.array(X_list[k])@Omegatilde@np.array(X_list[k].T))/(p)-sigma2*np.identity(n_l),ord="fro"))
  return Risk_Val/L

@pymanopt.function.autograd(manifold)
def MLEfun(Omegatilde):
  NegLL = 0 
  for k in range(L):
    NegLL=NegLL+np.linalg.norm(np.log(np.linalg.det(sigma2*np.identity(n_l)+(np.array(X_list[k])@Omegatilde@np.array(X_list[k].T))/(p)))+np.array(y_list[k]).T@np.linalg.inv(sigma2*np.identity(n_l)+(np.array(X_list[k])@Omegatilde@np.array(X_list[k].T))/(p))@np.array(y_list[k]))
  return NegLL

### For the MLE and proposed estimators, we use initialization by identity and four different random generated matrix above
    
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
    y_list.append(yl) 
  ### solving optimization problem for each cases 
  ### MoM and MLE with initialization by identity
  problem1 = Problem(manifold=manifold,cost=CostOmega1)
  solver1 = pymanopt.optimizers.SteepestDescent(max_time=21600,max_iterations=8000)  
  Omega1 = solver1.run(problem1,initial_point=np.identity(p)).point

  problem2 = Problem(manifold=manifold,cost=MLEfun)
  solver2 = pymanopt.optimizers.SteepestDescent(max_time=21600,max_iterations=8000)  
  Omega2 = solver2.run(problem2,initial_point=np.identity(p)).point
  ### MLE with other different initializations
  problem3 = Problem(manifold=manifold,cost=MLEfun)
  solver3 = pymanopt.optimizers.SteepestDescent(max_time=21600,max_iterations=8000)  
  Omega3 = solver3.run(problem3,initial_point=spdm1).point

  problem4 = Problem(manifold=manifold,cost=MLEfun)
  solver4 = pymanopt.optimizers.SteepestDescent(max_time=21600,max_iterations=8000)  
  Omega4 = solver4.run(problem4,initial_point=spdm2).point

  problem5 = Problem(manifold=manifold,cost=MLEfun)
  solver5 = pymanopt.optimizers.SteepestDescent(max_time=21600,max_iterations=8000)  
  Omega5 = solver5.run(problem5,initial_point=spdm3).point

  problem6 = Problem(manifold=manifold,cost=MLEfun)
  solver6 = pymanopt.optimizers.SteepestDescent(max_time=21600,max_iterations=8000)  
  Omega6 = solver6.run(problem6,initial_point=spdm4).point

  ## optimize MoM with different initializations
  problem7 = Problem(manifold=manifold,cost=CostOmega1)
  solver7 = pymanopt.optimizers.SteepestDescent(max_time=21600,max_iterations=8000)  
  Omega7 = solver7.run(problem7,initial_point=spdm1).point

  problem8 = Problem(manifold=manifold,cost=CostOmega1)
  solver8 = pymanopt.optimizers.SteepestDescent(max_time=21600,max_iterations=8000)  
  Omega8 = solver8.run(problem8,initial_point=spdm2).point

  problem9 = Problem(manifold=manifold,cost=CostOmega1)
  solver9 = pymanopt.optimizers.SteepestDescent(max_time=21600,max_iterations=8000)  
  Omega9 = solver9.run(problem9,initial_point=spdm3).point

  problem10 = Problem(manifold=manifold,cost=CostOmega1)
  solver10 = pymanopt.optimizers.SteepestDescent(max_time=21600,max_iterations=8000)  
  Omega10 = solver10.run(problem10,initial_point=spdm4).point 

  ## solving optimization for MLE with initialization given by MoM
  problem11 = Problem(manifold=manifold,cost=MLEfun)
  solver11 = pymanopt.optimizers.SteepestDescent(max_time=21600,max_iterations=8000)  
  Omega11 = solver11.run(problem11,initial_point=Omega1).point 

  ## generate the training sample in the new task
  Ztr = np.matrix(np.random.normal(0,1,size=(n_new,p)))     ### generate Z data matrix
  Xtr = np.matmul(Ztr,Sigmasqrt)                ### generate X data matrix 
  betatr = np.matrix(np.random.multivariate_normal(mean=np.zeros(p),cov=Omega/p)).T 
  Xbetatr = np.matmul(Xtr,betatr)
  epsilonltr = np.random.normal(0,sigma2,n_new) 
  yltr = Xbetatr+np.matrix(epsilonltr).T
  ## generate the test sample in the new task
  Zt = np.matrix(np.random.normal(0,1,size=(100,p)))     ### generate Z data matrix
  Xt = np.matmul(Zt,Sigmasqrt)                ### generate X data matrix  
  Xbetat = np.matmul(Xt,betatr) 
  epsilont = np.random.normal(0,sigma2,100) 
  yt = Xbetat+np.matrix(epsilont).T
  ## Calculate the estimated coefficients for MoM (use Omega1, Omega7,8,9,10)
  betahatMoM1_1 = np.linalg.inv(np.array(Xtr).T@np.array(Xtr)+n_new*lamb*np.identity(p))@np.array(Xtr).T@np.array(yltr)
  betahatMoM1_2 = np.linalg.inv(np.array(Xtr).T@np.array(Xtr)+n_new*lamb*np.linalg.inv(Omega1))@np.array(Xtr).T@np.array(yltr)

  betahatMoM2_1 = np.linalg.inv(np.array(Xtr).T@np.array(Xtr)+n_new*lamb*np.identity(p))@np.array(Xtr).T@np.array(yltr)
  betahatMoM2_2 = np.linalg.inv(np.array(Xtr).T@np.array(Xtr)+n_new*lamb*np.linalg.inv(Omega7))@np.array(Xtr).T@np.array(yltr) 

  betahatMoM3_1 = np.linalg.inv(np.array(Xtr).T@np.array(Xtr)+n_new*lamb*np.identity(p))@np.array(Xtr).T@np.array(yltr)
  betahatMoM3_2 = np.linalg.inv(np.array(Xtr).T@np.array(Xtr)+n_new*lamb*np.linalg.inv(Omega8))@np.array(Xtr).T@np.array(yltr) 

  betahatMoM4_1 = np.linalg.inv(np.array(Xtr).T@np.array(Xtr)+n_new*lamb*np.identity(p))@np.array(Xtr).T@np.array(yltr)
  betahatMoM4_2 = np.linalg.inv(np.array(Xtr).T@np.array(Xtr)+n_new*lamb*np.linalg.inv(Omega9))@np.array(Xtr).T@np.array(yltr) 

  betahatMoM5_1 = np.linalg.inv(np.array(Xtr).T@np.array(Xtr)+n_new*lamb*np.identity(p))@np.array(Xtr).T@np.array(yltr)
  betahatMoM5_2 = np.linalg.inv(np.array(Xtr).T@np.array(Xtr)+n_new*lamb*np.linalg.inv(Omega10))@np.array(Xtr).T@np.array(yltr) 
 
  ## Calculate the estimated coefficients for MLE
  betahatMLE1_1 = np.linalg.inv(np.array(Xtr).T@np.array(Xtr)+n_new*lamb*np.identity(p))@np.array(Xtr).T@np.array(yltr)
  betahatMLE1_2 = np.linalg.inv(np.array(Xtr).T@np.array(Xtr)+n_new*lamb*np.linalg.inv(Omega2))@np.array(Xtr).T@np.array(yltr) 

  betahatMLE2_1 = np.linalg.inv(np.array(Xtr).T@np.array(Xtr)+n_new*lamb*np.identity(p))@np.array(Xtr).T@np.array(yltr)
  betahatMLE2_2 = np.linalg.inv(np.array(Xtr).T@np.array(Xtr)+n_new*lamb*np.linalg.inv(Omega3))@np.array(Xtr).T@np.array(yltr)

  betahatMLE3_1 = np.linalg.inv(np.array(Xtr).T@np.array(Xtr)+n_new*lamb*np.identity(p))@np.array(Xtr).T@np.array(yltr)
  betahatMLE3_2 = np.linalg.inv(np.array(Xtr).T@np.array(Xtr)+n_new*lamb*np.linalg.inv(Omega4))@np.array(Xtr).T@np.array(yltr)

  betahatMLE4_1 = np.linalg.inv(np.array(Xtr).T@np.array(Xtr)+n_new*lamb*np.identity(p))@np.array(Xtr).T@np.array(yltr)
  betahatMLE4_2 = np.linalg.inv(np.array(Xtr).T@np.array(Xtr)+n_new*lamb*np.linalg.inv(Omega5))@np.array(Xtr).T@np.array(yltr)

  betahatMLE5_1 = np.linalg.inv(np.array(Xtr).T@np.array(Xtr)+n_new*lamb*np.identity(p))@np.array(Xtr).T@np.array(yltr)
  betahatMLE5_2 = np.linalg.inv(np.array(Xtr).T@np.array(Xtr)+n_new*lamb*np.linalg.inv(Omega6))@np.array(Xtr).T@np.array(yltr)

  betahatMLE6_1 = np.linalg.inv(np.array(Xtr).T@np.array(Xtr)+n_new*lamb*np.identity(p))@np.array(Xtr).T@np.array(yltr)
  betahatMLE6_2 = np.linalg.inv(np.array(Xtr).T@np.array(Xtr)+n_new*lamb*np.linalg.inv(Omega11))@np.array(Xtr).T@np.array(yltr)
  
  ## Calculate the risk on unseen samples for MoM
  yhatL = np.array(np.array(Xt)@betahatMoM1_1)
  RiskL = (np.linalg.norm(yhatL-yt)**2)/100
  yhatL2 = np.array(np.array(Xt)@betahatMoM1_2)
  RiskL2 = (np.linalg.norm(yhatL2-yt)**2)/100 
  MoM_1_EstRisk1.append(RiskL)
  MoM_1_EstRisk2.append(RiskL2) 
  MoM_1_Errornorm.append(np.linalg.norm(Omega1-Omega,ord="fro"))
  print("print MoM")
  print(Omega1)  
  print("Frobenius norm of Omegahat1-Omega")
  print(np.linalg.norm(Omega1-Omega,ord="fro")) 
  print(RiskL)
  print(RiskL2) 
  print(i)
  
  yhatL = np.array(np.array(Xt)@betahatMoM2_1)
  RiskL = (np.linalg.norm(yhatL-yt)**2)/100
  yhatL2 = np.array(np.array(Xt)@betahatMoM2_2)
  RiskL2 = (np.linalg.norm(yhatL2-yt)**2)/100 
  MoM_2_EstRisk1.append(RiskL)
  MoM_2_EstRisk2.append(RiskL2) 
  MoM_2_Errornorm.append(np.linalg.norm(Omega7-Omega,ord="fro"))
  print("print randomly generated SPD1")
  print(Omega7)  
  print("Frobenius norm of Omegahat7-Omega")
  print(np.linalg.norm(Omega7-Omega,ord="fro")) 
  print(RiskL)
  print(RiskL2) 
  print(i)

  yhatL = np.array(np.array(Xt)@betahatMoM3_1)
  RiskL = (np.linalg.norm(yhatL-yt)**2)/100
  yhatL2 = np.array(np.array(Xt)@betahatMoM3_2)
  RiskL2 = (np.linalg.norm(yhatL2-yt)**2)/100 
  MoM_3_EstRisk1.append(RiskL)
  MoM_3_EstRisk2.append(RiskL2) 
  MoM_3_Errornorm.append(np.linalg.norm(Omega8-Omega,ord="fro"))
  print("print randomly generated SPD2")
  print(Omega8)  
  print("Frobenius norm of Omegahat8-Omega")
  print(np.linalg.norm(Omega8-Omega,ord="fro")) 
  print(RiskL)
  print(RiskL2) 
  print(i)

  yhatL = np.array(np.array(Xt)@betahatMoM4_1)
  RiskL = (np.linalg.norm(yhatL-yt)**2)/100
  yhatL2 = np.array(np.array(Xt)@betahatMoM4_2)
  RiskL2 = (np.linalg.norm(yhatL2-yt)**2)/100 
  MoM_4_EstRisk1.append(RiskL)
  MoM_4_EstRisk2.append(RiskL2) 
  MoM_4_Errornorm.append(np.linalg.norm(Omega9-Omega,ord="fro"))
  print("print randomly generated SPD3")
  print(Omega9)  
  print("Frobenius norm of Omegahat9-Omega")
  print(np.linalg.norm(Omega9-Omega,ord="fro")) 
  print(RiskL)
  print(RiskL2) 
  print(i)

  yhatL = np.array(np.array(Xt)@betahatMoM5_1)
  RiskL = (np.linalg.norm(yhatL-yt)**2)/100
  yhatL2 = np.array(np.array(Xt)@betahatMoM5_2)
  RiskL2 = (np.linalg.norm(yhatL2-yt)**2)/100 
  MoM_5_EstRisk1.append(RiskL)
  MoM_5_EstRisk2.append(RiskL2) 
  MoM_5_Errornorm.append(np.linalg.norm(Omega10-Omega,ord="fro"))
  print("print randomly generated SPD4")
  print(Omega10)  
  print("Frobenius norm of Omegahat10-Omega")
  print(np.linalg.norm(Omega10-Omega,ord="fro")) 
  print(RiskL)
  print(RiskL2) 
  print(i) 


  ## Calculate the risk on unseen samples for MLE
  yhatL = np.array(np.array(Xt)@betahatMLE1_1)
  RiskL = (np.linalg.norm(yhatL-yt)**2)/100
  yhatL2 = np.array(np.array(Xt)@betahatMLE1_2)
  RiskL2 = (np.linalg.norm(yhatL2-yt)**2)/100 
  MLE_1_EstRisk1.append(RiskL)
  MLE_1_EstRisk2.append(RiskL2) 
  MLE_1_Errornorm.append(np.linalg.norm(Omega2-Omega,ord="fro"))
  print("print MLE (identity)")
  print(Omega2)  
  print("Frobenius norm of Omegahat2-Omega")
  print(np.linalg.norm(Omega2-Omega,ord="fro")) 
  print(RiskL)
  print(RiskL2) 
  print(i)


  yhatL = np.array(np.array(Xt)@betahatMLE2_1)
  RiskL = (np.linalg.norm(yhatL-yt)**2)/100
  yhatL2 = np.array(np.array(Xt)@betahatMLE2_2)
  RiskL2 = (np.linalg.norm(yhatL2-yt)**2)/100 
  MLE_2_EstRisk1.append(RiskL)
  MLE_2_EstRisk2.append(RiskL2) 
  MLE_2_Errornorm.append(np.linalg.norm(Omega3-Omega,ord="fro"))
  print("print randomly generated SPD1")
  print(Omega3)  
  print("Frobenius norm of Omegahat3-Omega")
  print(np.linalg.norm(Omega3-Omega,ord="fro")) 
  print(RiskL)
  print(RiskL2) 
  print(i)

  yhatL = np.array(np.array(Xt)@betahatMLE3_1)
  RiskL = (np.linalg.norm(yhatL-yt)**2)/100
  yhatL2 = np.array(np.array(Xt)@betahatMLE3_2)
  RiskL2 = (np.linalg.norm(yhatL2-yt)**2)/100 
  MLE_3_EstRisk1.append(RiskL)
  MLE_3_EstRisk2.append(RiskL2) 
  MLE_3_Errornorm.append(np.linalg.norm(Omega4-Omega,ord="fro"))
  print("print randomly generated SPD2")
  print(Omega4)  
  print("Frobenius norm of Omegahat4-Omega")
  print(np.linalg.norm(Omega4-Omega,ord="fro")) 
  print(RiskL)
  print(RiskL2) 
  print(i)

  yhatL = np.array(np.array(Xt)@betahatMLE4_1)
  RiskL = (np.linalg.norm(yhatL-yt)**2)/100
  yhatL2 = np.array(np.array(Xt)@betahatMLE4_2)
  RiskL2 = (np.linalg.norm(yhatL2-yt)**2)/100 
  MLE_4_EstRisk1.append(RiskL)
  MLE_4_EstRisk2.append(RiskL2) 
  MLE_4_Errornorm.append(np.linalg.norm(Omega5-Omega,ord="fro"))
  print("print randomly generated SPD3")
  print(Omega5)  
  print("Frobenius norm of Omegahat5-Omega")
  print(np.linalg.norm(Omega5-Omega,ord="fro")) 
  print(RiskL)
  print(RiskL2) 
  print(i)

  yhatL = np.array(np.array(Xt)@betahatMLE5_1)
  RiskL = (np.linalg.norm(yhatL-yt)**2)/100
  yhatL2 = np.array(np.array(Xt)@betahatMLE5_2)
  RiskL2 = (np.linalg.norm(yhatL2-yt)**2)/100 
  MLE_5_EstRisk1.append(RiskL)
  MLE_5_EstRisk2.append(RiskL2) 
  MLE_5_Errornorm.append(np.linalg.norm(Omega6-Omega,ord="fro"))
  print("print Omegahat4")
  print(Omega6)  
  print("Frobenius norm of Omegahat6-Omega")
  print(np.linalg.norm(Omega6-Omega,ord="fro")) 
  print(RiskL)
  print(RiskL2) 
  print(i)

  yhatL = np.array(np.array(Xt)@betahatMLE6_1)
  RiskL = (np.linalg.norm(yhatL-yt)**2)/100
  yhatL2 = np.array(np.array(Xt)@betahatMLE6_2)
  RiskL2 = (np.linalg.norm(yhatL2-yt)**2)/100 
  MLE_6_EstRisk1.append(RiskL)
  MLE_6_EstRisk2.append(RiskL2) 
  MLE_6_Errornorm.append(np.linalg.norm(Omega11-Omega,ord="fro"))
  print("print MLE by MoM initialization")
  print(Omega11)  
  print("Frobenius norm of Omegahat11-Omega")
  print(np.linalg.norm(Omega11-Omega,ord="fro")) 
  print(RiskL)
  print(RiskL2) 
  print(i)

  #### Print current average risk of each case
  print("MoM1 summary")
  print(np.mean(MoM_1_Errornorm))
  print(np.mean(MoM_1_EstRisk1))
  print(np.mean(MoM_1_EstRisk2)) 
  print(MoM_1_EstRisk1)
  print(MoM_1_EstRisk2) 
  print("MoM2 summary")
  print(np.mean(MoM_2_Errornorm))
  print(np.mean(MoM_2_EstRisk1))
  print(np.mean(MoM_2_EstRisk2)) 
  print(MoM_2_EstRisk1)
  print(MoM_2_EstRisk2) 
  print("MoM3 summary")
  print(np.mean(MoM_3_Errornorm))
  print(np.mean(MoM_3_EstRisk1))
  print(np.mean(MoM_3_EstRisk2)) 
  print(MoM_3_EstRisk1)
  print(MoM_3_EstRisk2) 
  print("MoM4 summary")
  print(np.mean(MoM_4_Errornorm))
  print(np.mean(MoM_4_EstRisk1))
  print(np.mean(MoM_4_EstRisk2)) 
  print(MoM_4_EstRisk1)
  print(MoM_4_EstRisk2) 
  print("MoM5 summary")
  print(np.mean(MoM_5_Errornorm))
  print(np.mean(MoM_5_EstRisk1))
  print(np.mean(MoM_5_EstRisk2)) 
  print(MoM_5_EstRisk1)
  print(MoM_5_EstRisk2)  
  
  print("MLE1 summary")
  print(np.mean(MLE_1_Errornorm))
  print(np.mean(MLE_1_EstRisk1))
  print(np.mean(MLE_1_EstRisk2)) 
  print(MLE_1_EstRisk1)
  print(MLE_1_EstRisk2) 
  print("MLE2 summary")
  print(np.mean(MLE_2_Errornorm))
  print(np.mean(MLE_2_EstRisk1))
  print(np.mean(MLE_2_EstRisk2)) 
  print(MLE_2_EstRisk1)
  print(MLE_2_EstRisk2) 
  print("MLE3 summary")
  print(np.mean(MLE_3_Errornorm))
  print(np.mean(MLE_3_EstRisk1))
  print(np.mean(MLE_3_EstRisk2)) 
  print(MLE_3_EstRisk1)
  print(MLE_3_EstRisk2) 
  print("MLE4 summary")
  print(np.mean(MLE_4_Errornorm))
  print(np.mean(MLE_4_EstRisk1))
  print(np.mean(MLE_4_EstRisk2)) 
  print(MLE_4_EstRisk1)
  print(MLE_4_EstRisk2) 
  print("MLE5 summary")
  print(np.mean(MLE_5_Errornorm))
  print(np.mean(MLE_5_EstRisk1))
  print(np.mean(MLE_5_EstRisk2)) 
  print(MLE_5_EstRisk1)
  print(MLE_5_EstRisk2)
  print("MLE with MoM init summary")
  print(np.mean(MLE_6_Errornorm))
  print(np.mean(MLE_6_EstRisk1))
  print(np.mean(MLE_6_EstRisk2)) 
  print(MLE_6_EstRisk1)
  print(MLE_6_EstRisk2)
