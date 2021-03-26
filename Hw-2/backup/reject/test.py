#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 13:00:38 2021

@author: wef5056
"""
import numpy as np
import matplotlib.pyplot as plt
#from DAComputations import IntergratEuler,rmse,objinterp,getB

#%%
def IntergratEuler(data,steps,dt,F):
    """ This function integrates a 1-D model
    forward in time with the Euler Scheme"""
    xCells = len(data)
    update = np.zeros(xCells)
    for j in np.arange(1,steps,1):
        for i in np.arange(0,xCells,1):
            update[i] = ((data[(i+1)%xCells]-data[(i-2)%xCells])\
                                *data[(i-1)%xCells])-data[i]+F
        data = data + (dt*update)
    return data

def getB(n,sigma_b,l):
    B = np.zeros((n,n))
    for i in np.arange(0,n,1):
        for j in np.arange(0,n,1):
            distance = min((i-j)%n,(j-i)%n)
            B[i,j] = np.square(sigma_b)*np.exp(-(np.square(distance))/(2*l*l))
    return B

def objinterp(B,H,R,yo,xb):
    BHt = B@H.T
    HBHt = H@B@H.T
    inverse = np.linalg.inv(HBHt+R)
    W =  BHt@inverse
    delta = yo-H@xb
    xa = xb + W@delta
    return xa

def rmse(x,y):
    diff = x-y
    rmse = np.zeros(len(x))
    for i in np.arange(0,len(x)):
        rmse[i] = np.sqrt(np.sum(np.square(diff[i,:]))/len(x))
    #(np.sum(diff**2)/len(x))**.5)
    #diff = np.subtract(x,y)
    #rmse = np.sqrt(np.sum(np.square(diff))/len(x))
    return rmse 
#%% Creating Initial Conditions
n = 40
F = 8
dt = .005
steps = 1000
data = np.random.normal(0,1,n)
initconditions = IntergratEuler(data,steps,dt,F)

#%% Creating Truth State
"""Calculates the truth state by iterating intergrating
   fowrad in time via the Eurler Method over 40 iterations
   (considered 1 cycle). The process is completed 100 times
   and the final value is considered Xtruth"""
cycles = 100
Xtruth = np.zeros((cycles,n)) #create 100x40 matrix
Xtruth[0,:]=initconditions    # first row =init conditions
for i in np.arange(1,cycles,1):    
    Xtruth[i,:] = IntergratEuler(Xtruth[i-1,:],n,dt,F)
    
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

NoDA_xa = np.zeros((cycles,n))
NoDA_xa[0,:] = perturbedIC
for i in np.arange(1,cycles,1):    
    NoDA_xa[i,:] = IntergratEuler(NoDA_xa[(i-1),:],n,dt,F)
    
#%% Control Case
sigmao = 0.3
sigmab = 1
l = 1
xb = NoDA[0,:]+0
H = np.identity(n) # n grid cells 
R = np.identity(n)*np.square(sigmao)
yo = np.random.normal(muXtrue,np.square(sigmao),size=(cycles,n))
B = getB(n,sigmab,l)
Control = np.zeros((cycles,n))
Holder = np.zeros((cycles,n))
xb_c = perturbedIC
for i in np.arange(0,cycles,1):
    Control[i,:] = objinterp(B,H,R,yo[i,:],xb_c)
    xb_c = IntergratEuler(Control[i,:],n,dt,F)
    Holder[i,:] = xb_c
    
"""
for i in np.arange(0,cycles,1):
    Control[i,:] = objinterp(B,H,R,yo[i,:],xb)
    xb = IntergratEuler(Control[i,:],n,dt,F)
"""
#%%    
#RMSE_Control = np.zeros(cycles)
#RMSE_NoDA = np.zeros(cycles)
RMSE_Control = rmse(Control,Xtruth)
RMSE_NoDA = rmse(Holder,Xtruth)
RMSE = [RMSE_NoDA,RMSE_Control]
RMSE_plot = np.zeros(cycles*2)
for i in np.arange(0,len(RMSE_plot)):
    RMSE_plot[i] = RMSE[i%2][int(np.floor(i/2))]

#%%Plotting the RMSE sawtooth graph

staggeredIndex = np.arange(1,cycles+1,.5)
for i in np.arange(1,2*cycles,2):
    staggeredIndex[i] = staggeredIndex[i]-.5+.0001

fig,ax = plt.subplots()
ax.plot(staggeredIndex, RMSE_plot)
ax.set_xticks(np.arange(0,101,10).tolist())
ax.set_ylabel('RMSE')
ax.set_xlabel('t')
ax.set_title('RMSE of Xb and Xa')

fig,ax = plt.subplots()
ax.plot(staggeredIndex, RMSE_plot)
ax.set_xlim(0,21)
ax.set_xticks(np.arange(0,21,5).tolist())
ax.set_yticks([0,.5,1,1.5,2,2.5])
ax.set_ylabel('RMSE')
ax.set_xlabel('t')
ax.set_title('RMSE of Xb and Xa')
