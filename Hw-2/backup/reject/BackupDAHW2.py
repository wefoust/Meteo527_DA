#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module creates a forecast
"""
import numpy as np
import matplotlib.pyplot as plt
from DAComputations import IntergratEuler,rmse,objinterp,getB


#%% Creating Initial Conditions
n = 40
F = 8
dt = .005
steps = 1000
data = np.random.normal(0,1,n)
#data = np.random.uniform(-1,1,n)
#data = np.random.rand(n)
initconditions = IntergratEuler(data,steps,dt,F)
initconditions = initconditions[-1,:]

#%% Creating Truth State
"""Calculates the truth state by iterating intergrating
   fowrad in time via the Eurler Method over 40 iterations
   (considered 1 cycle). The process is completed 100 times
   and the final value is considered Xtruth"""
cycles = 100
Xtruth = np.zeros((cycles,n)) #create 100x40 matrix
Xtruth[0,:]=initconditions    # first row =init conditions
for i in np.arange(1,cycles,1):    
    timesteps = IntergratEuler(Xtruth[i-1,:],n,dt,F)
    Xtruth[i,:] = timesteps[-1,:]
    
#%% Make Hovmöller Diagram
fig, ax = plt.subplots()
ax.contourf(np.arange(0,n,1),np.arange(0,cycles,1),Xtruth,cmap='bwr')
ax.set_aspect(.5)
ax.set_xticks([0,10,20,30])
ax.set_yticks([0,20,40,60,80,100])
ax.set_xlabel('X')
ax.set_ylabel('Cycle')
ax.set_title('Xtruth')

#%% Perturbing Init State and creating NoDa case
"""This section creates the NoDA case where it 
    perturbes the initial conditions"""
muXtrue = np.mean(Xtruth)
muXtrue_t0 = np.mean(Xtruth[0,:])
#perturbedIC = np.random.normal(muXtrue_t0,1,n)
perturbedIC = Xtruth[0,:]+np.random.normal(0,1,n)

NoDA = np.zeros((cycles,n))
NoDA[0,:] = perturbedIC
for i in np.arange(1,cycles,1):    
    timesteps = IntergratEuler(NoDA[(i-1),:],n,dt,F)
    NoDA[i,:] = timesteps[-1,:]
 
#%% Control Case
sigmao = 0.3
sigmab = 1
l = 1
xb = perturbedIC+0
H = np.identity(n) # n grid cells 
R = np.identity(n)*np.square(sigmao)
yo = np.random.normal(muXtrue,np.square(sigmao),size=(cycles,n))
B = getB(n,sigmab,l)

Control = np.zeros((cycles,n))
x_aa = np.zeros((cycles,n))
x_bb = np.zeros((cycles,n))



for i in np.arange(0,cycles-1):
    xa = objinterp(B,H,R,yo[i,:],xb)
    temp = IntergratEuler(xa,n,dt,F)
    xb = temp[-1,:]
    x_aa[i,:] = xa
    x_bb[i+1,:] = xb
x_bb[0,:] = perturbedIC
"""
for i in np.arange(0,cycles,1):
    Control[i,:] = objinterp(B,H,R,yo[i,:],xb[None,i])
    timesteps = IntergratEuler(Control[i,:],n,dt,F)
    xb = timesteps[-1,:]
    if i < cycles-1:    
        xbb.append(timesteps[-1,:])
"""
    
#%% RMSE Calculations
RMSE_b = np.zeros(cycles)
RMSE_a = np.zeros(cycles)
rmseConcat = []
for i in np.arange(0,cycles,1):
    RMSE_b[i] = rmse(x_bb[i,:],Xtruth[i,:])
    rmseConcat.append(RMSE_b[i])
    RMSE_a[i] = rmse(x_aa[i,:],Xtruth[i,:])
    rmseConcat.append(RMSE_a[i])
test = [RMSE_b,RMSE_a]
#%%Plotting the RMSE sawtooth graph
staggeredIndex = np.arange(0,cycles,.5)
for i in np.arange(1,2*cycles,2):
    staggeredIndex[i] = staggeredIndex[i]-.5+.0001

fig,ax = plt.subplots()
ax.plot(staggeredIndex[2:],rmseConcat[2:])
ax.set_xticks(np.arange(0,101,10).tolist())
ax.set_ylabel('RMSE')
ax.set_xlabel('t')
ax.set_title('RMSE of Xb and Xa')


fig,ax = plt.subplots()
ax.plot(staggeredIndex[2:42],rmseConcat[2:42])
ax.scatter(np.arange(1,21),RMSE_b[1:21],color='r')
ax.set_xticks(np.arange(0,21,5).tolist())
#ax.set_yticks([3,4,5,6,7,8])
ax.set_ylabel('RMSE')
ax.set_xlabel('t')
ax.set_title('RMSE of Xb and Xa')


#%%

B1 = getB(n,.1,l)
B2 = getB(n,.001,l)
Ex1 = np.zeros((cycles,n))
Ex2 = np.zeros((cycles,n))
RMSE_Ex1 = np.zeros(cycles)
RMSE_Ex2 = np.zeros(cycles)

for i in np.arange(0,cycles,1):
    Ex1[i,:] = objinterp(B1,H,R,yo[i,:],xb[i,:])
    Ex2[i,:] = objinterp(B2,H,R,yo[i,:],xb[i,:])
for i in np.arange(0,cycles,1):    
    RMSE_Ex1[i] = rmse(Ex1[i,:],Xtruth[i,:])
    rmseConcat.append(RMSE_Ex1[i])
    RMSE_Ex2[i] = rmse(Ex2[i,:],Xtruth[i,:])
    rmseConcat.append(RMSE_Ex2[i])
    
fig, ax =plt.subplots()
ax.plot(RMSE_Ex1)
ax.plot(RMSE_Ex2,color='r')    
ax.plot(RMSE_a,color='k')
ax.set_ylabel('RMSE')
ax.set_xlabel('t')
ax.set_title('RMSE \n $\sigma$ = 1., 0.1, and .001')



#%% 3D Var final
VarIncraments = 50
r = np.zeros((VarIncraments,n))
p = np.zeros((VarIncraments,n))
X = np.zeros((VarIncraments,n))

H = H
R_inv = np.linalg.inv(R)
yo = yo


b = H.T @ R_inv @ (yo[0,:]-NoDA[0,:]).T
A = np.linalg.inv(B) + (H.T @ R_inv @ H)
r[0,:] = (b - (A @ NoDA[0,:].T).T).T
p[0,:]=r[None,0,:]
X[0,:] = NoDA[0,:]

print(np.linalg.norm(r))
for i in np.arange(0,VarIncraments-1,1):
    alpha = float(r[None,i,:]@r[None,i,:].T)/float(p[None,i,:] @ A @ p[None,i,:].T)
    X[i+1,:] = X[i,:]+(alpha*p[i,:])
    r[i+1] = r[i,:] - alpha*(A @ p[i,:])
    Beta = float(r[None,i+1,:] @ r[None,i+1,:].T)\
          /float(r[None,i,:] @ r[None,i,:].T)
    p[i+1,:] = r[i+1,:] + Beta*p[i,:]
    #print(alpha)
    #print(np.matmul(r[i+1,:],r[i,:]))
    print(np.linalg.norm(r))
"""
        numerator = float(np.matmul(r[i,:],r[i,:].T))
        denomenator = np.matmul(np.matmul(p[i,:],A),p[i,:])
        alpha =numerator/denomenator
        r[i+1,:] = r[i,:] - alpha*np.matmul(A,p[i,:])
        betanum = float(np.matmul(r[i+1,:],r[i+1,:]))
        betadenom = float(np.matmul(r[i,:],r[i,:])) 
        Beta = betanum/betadenom
        p[i+1,:] = r[i+1,:] + Beta*p[i,:]
        print(np.matmul(r[i+1,:],r[i,:]))
        print(np.linalg.norm(r))
"""
#%% 3D Var
"""
H = predefined
R = predefined 
obs = yo
background = NoDA
b = H.T*R^-1*(obs-background)
A = B^-1 + H.T*R^-1*H #Hessian Matrix
r = b - A*X
alpha = (r.T*r)/(r.T*A*r)
"""


#GOOOD
#minsteps = 40
#r = np.zeros((minsteps,n))
#p = np.zeros((minsteps,n))
#H = np.asmatrix(H)
#R_inv = np.asmatrix(np.linalg.inv(R))
#yo = np.asmatrix(yo)
#background = np.asmatrix(NoDA)

#b = H.T*R*(yo[0,:]-background[0,:]).T
#A = np.linalg.inv(B) + (H.T*R_inv*H)
    #r = np.asmatrix(b - A*background[0,:].T)


#r[0,:] = np.asmatrix(b - A*background[0,:].T).T
#r[0,:] = b.T
#p[0,:]=r[0,:]
    #P = r
    #alpha = float(np.squeeze((r.T*r)/(P.T*A*P)))
    #minsteps = 20
#xMinimize=np.zeros((minsteps,n))
#xMinimize[0,:] = background[i,:]




"""
#test 2 
print(np.linalg.norm(r))
for i in np.arange(0,minsteps-1,1):
        numerator = float(np.matmul(r[i,:],r[i,:].T))
        denomenator = np.matmul(np.matmul(p[i,:],A),p[i,:])
        alpha =numerator/denomenator
        r[i+1,:] = r[i,:] - alpha*np.matmul(A,p[i,:])
        betanum = float(np.matmul(r[i+1,:],r[i+1,:]))
        betadenom = float(np.matmul(r[i,:],r[i,:])) 
        Beta = betanum/betadenom
        p[i+1,:] = r[i+1,:] + Beta*p[i,:]
        print(np.matmul(r[i+1,:],r[i,:]))
        print(np.linalg.norm(r))
        
"""
"""
#Test loop
print(np.linalg.norm(r))
for i in np.arange(0,10,1):
    alpha = float((r.T*r)/(r.T*A*r))
    rnew = r - (alpha*A*P)
    Beta = float((rnew.T*rnew)/(r.T*r))
    Pnew = rnew + Beta*P
    p = Pnew
    r = rnew
    print(np.linalg.norm(r))
"""


"""
for i in np.arange(0,minsteps-1,1):
    xMinimize[i+1,:] = xMinimize[i,:]+(alpha*P).T
    alpha = float((r.T*r)/(r.T*A*r))
    rnew = r - alpha*A*P
    Beta = float((rnew.T*rnew)/(r.T*r))
    Pnew = rnew +Beta*P
    
    p = Pnew
    r = rnew
    print(np.linalg.norm(r))
 """   
    
    