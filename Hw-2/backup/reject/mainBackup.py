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
Xtruth = np.zeros((cycles+1,n)) #create 100x40 matrix
Xtruth[0,:]=initconditions      # first row =init conditions
for i in np.arange(1,cycles+1,1):    
    timesteps = IntergratEuler(Xtruth[i-1,:],n,dt,F)
    Xtruth[i,:] = timesteps[-1,:]
    
#%% Make Hovmöller Diagram
fig, ax = plt.subplots()
ax.contourf(np.arange(0,40,1),np.arange(0,101,1),Xtruth,cmap='bwr')
ax.set_aspect(.5)
ax.set_xticks([0,10,20,30])
ax.set_yticks([20,40,60,80,100])
ax.set_xlabel('X')
ax.set_ylabel('Cycle')
ax.set_title('Xtruth')

#%% Perturbing Init State and creating NoDa case
"""This section creates the NoDA case where it 
    perturbes the initial conditions"""
muXtrue = np.mean(Xtruth)
muXtrue_t1 = np.mean(Xtruth[1,:])
perturbedIC = np.random.normal(muXtrue_t1,1,n)
#test=np.random.rand(n)
#perturbedIC = np.multiply(Xtruth[0,:],test)
NoDA = np.zeros((cycles+1,n))
NoDA[0,:] = perturbedIC
for i in np.arange(1,cycles+1,1):    
    timesteps = IntergratEuler(NoDA[i-1,:],n,dt,F)
    NoDA[i,:] = timesteps[-1,:]
 
#%% Control Case
sigmao = 0.3
sigmab = 1
l = 1
xb = NoDA
H = np.identity(n) # n grid cells 
R = np.identity(n)*np.square(sigmao)
yo = np.random.normal(muXtrue,np.square(sigmao),size=(cycles+1,n))
B = getB(n,sigmab,l)

Control = np.zeros((cycles+1,n))
for i in np.arange(0,cycles+1,1):
    Control[i,:] = objinterp(B,H,R,yo[i,:],xb[i,:])
   
RMSE_b = np.zeros(cycles+1)
RMSE_a = np.zeros(cycles+1)
rmseConcat = []
for i in np.arange(0,cycles+1,1):
    RMSE_b[i] = rmse(xb[i,:],Xtruth[i,:])
    rmseConcat.append(RMSE_b[i])
    RMSE_a[i] = rmse(Control[i,:],Xtruth[i,:])
    rmseConcat.append(RMSE_a[i])

#%%Plotting the RMSE sawtooth graph
staggeredIndex = np.arange(0,cycles+1,.5)
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
ax.set_xticks(np.arange(0,21,5).tolist())
ax.set_ylabel('RMSE')
ax.set_xlabel('t')
ax.set_title('RMSE of Xb and Xa')


#%%
B1 = getB(n,.1,l)
B2 = getB(n,.001,l)
Ex1 = np.zeros((cycles+1,n))
Ex2 = np.zeros((cycles+1,n))
RMSE_Ex1 = np.zeros(cycles+1)
RMSE_Ex2 = np.zeros(cycles+1)

for i in np.arange(0,cycles+1,1):
    Ex1[i,:] = objinterp(B1,H,R,yo[i,:],xb[i,:])
    Ex2[i,:] = objinterp(B2,H,R,yo[i,:],xb[i,:])
for i in np.arange(0,cycles+1,1):    
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


#%% 3D Var
"""
B = B
xb= N
yo=yo
distance = yo-xb
alpha = (r.T*r)/(P.T*A*P)
V= Aprime.T*bPrime
Xa = xb+BV
"""