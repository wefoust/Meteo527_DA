#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script completes homework 2
for METEO 527
"""
import numpy as np
import matplotlib.pyplot as plt
from DAComputations import IntergrateEuler,_OIDA, NoDA_RMSE,_3DVar

#%% Creating Initial Conditions
n = 40
F = 8
dt = .005
steps = 1000
data = np.round(np.random.normal(0,1,n),5)
initconditions = IntergrateEuler(data,steps,dt,F)

#%% Creating Truth State
"""Calculates the truth state by iterating intergrating
   fowrad in time via the Eurler Method over 40 iterations
   (considered 1 cycle). The process is completed 100 times
   and the final value is considered Xtruth"""
cycles = 100
Xtruth = np.zeros((cycles,n)) #create 100x40 matrix
Xtruth[0,:]=initconditions    # first row =init conditions
for i in np.arange(1,cycles,1):    
    Xtruth[i,:] = IntergrateEuler(Xtruth[i-1,:],n,dt,F)
    
#%% Make Hovmöller Diagram
fig, ax = plt.subplots()
ax.contourf(np.arange(0,n,1),np.arange(0,cycles,1),Xtruth,cmap='bwr')
ax.set_aspect(.5)
ax.set_xticks([0,10,20,30])
ax.set_yticks([0,20,40,60,80,100])
ax.set_xlabel('X')
ax.set_ylabel('Cycle')
ax.set_title('Xtruth')
fig.savefig('Hovmoller.png',
            bbox_inches='tight',
            dpi=250)
#%% Perturbing Init State and creating NoDa case
"""This section creates the NoDA case where it 
    perturbes the initial conditions"""
muXtrue = np.mean(Xtruth)
muXtrue_t0 = np.mean(Xtruth[0,:])
perturbedIC = np.round(Xtruth[0,:]+np.random.normal(0,1,n),5)
NoDA_xb = np.zeros((cycles,n))
NoDA_xb[0,:] = perturbedIC
for i in np.arange(1,cycles,1):    
    NoDA_xb[i,:] = IntergrateEuler(NoDA_xb[(i-1),:],n,dt,F)
    
#%% Control Case
""" This section runs the controll case"""

#Setting inputs
sigmao = 0.3
sigmab = 1
l = 1
H = np.identity(n) # n grid cells 
R = np.identity(n)*np.square(sigmao)
IC = perturbedIC+0
yo = np.round(Xtruth + np.random.normal(0,np.square(sigmao),
                               size=(cycles,n)),5)
# Performing DA with OI
RMSE_plot = _OIDA(cycles,sigmao,sigmab,l,H,yo,IC,dt,F,Xtruth)[1]

# Plotting
staggeredIndex = np.arange(1,cycles+1,.5)
for i in np.arange(1,2*cycles,2):
    staggeredIndex[i] = staggeredIndex[i]-.5+.0001
fig,(ax1,ax2) = plt.subplots(2)
ax1.plot(staggeredIndex, RMSE_plot)
ax1.set_xticks(np.arange(0,101,10).tolist())
ax1.set_yticks([0.5,1,1.5])
ax1.set_title('OI RMSE of Xb and Xa')
ax1.grid()
ax2.plot(staggeredIndex, RMSE_plot)
ax2.set_xlim(0,21)
ax2.set_xticks(np.arange(0,21,5).tolist())
ax2.set_yticks([0.5,1,1.5])
ax2.set_xlabel('Cycle')
fig.text(.05,.5,'RMSE',ha='center',va='center',rotation=90)
ax2.grid()
fig.savefig('OI_RMSE.png',
            bbox_inches='tight',
            dpi=250)

#%% Creating Plot of variable Sigmab
RMSE_plot1 = NoDA_RMSE(Xtruth,1,cycles,dt,F)
RMSE_plot2 = NoDA_RMSE(Xtruth,.01,cycles,dt,F)
RMSE_plot3 = NoDA_RMSE(Xtruth,.0001,cycles,dt,F)

fig,ax =plt.subplots()
line1 = ax.plot(RMSE_plot1,color='k',label = '$\sigma_b$= 1')
line2 = ax.plot(RMSE_plot2,color='r',label = '$\sigma_b$= .01')
line3 = ax.plot(RMSE_plot3,color='b',label = '$\sigma_b$= .0001')
ax.set_ylabel('RMSE')
ax.set_xlabel('Cycles')
ax.set_title('NoDA Error Growth')
ax.legend()
ax.grid(True)
ax.set_xlim(0,100)
ax.set_ylim(0)
fig.savefig('NoDA_RMSE.png',
            bbox_inches='tight',
            dpi=250)

#%% Creating 3D Var Plots
# Performing 3DVar
R = np.identity(n)*np.square(sigmao)
RMSE_3DVar_plot = _3DVar(sigmab,R,l,H,IC,yo,cycles,dt,F,Xtruth)[1]

fig,(ax1,ax2) = plt.subplots(2)
ax1.plot(staggeredIndex, RMSE_3DVar_plot)
ax1.set_xticks(np.arange(0,101,10).tolist())
ax1.set_yticks([0.5,1,1.5])
ax1.set_title('3D Var RMSE of Xb and Xa')
ax1.grid()
ax2.plot(staggeredIndex, RMSE_3DVar_plot)
ax2.set_xlim(0,21)
ax2.set_xticks(np.arange(0,21,5).tolist())
ax2.set_yticks([0.5,1,1.5])
ax2.set_xlabel('Cycle')
fig.text(.05,.5,'RMSE',ha='center',va='center',rotation=90)
ax2.grid()
fig.savefig('3DVar_RMSE.png',
            bbox_inches='tight',
            dpi=250)
#%% Perturbing Sigma Values Plot
sigmavalues = np.arange(.1,10,.1)
stepIndex =np.arange(0,len(sigmavalues),1)
timeMeanSigma = np.zeros(len(sigmavalues))
for i in stepIndex:
    timeMeanSigma[i] = np.mean(_3DVar(sigmavalues[i],sigmao,l,H,IC,yo,cycles,dt,F,Xtruth)[0][1])
    #timeMeanSigma[i] = np.mean(_OIDA(cycles,sigmao,sigmavalues[i],l,H,yo,IC,dt,F,Xtruth)[0][1])

fig,ax = plt.subplots()
ax.plot(sigmavalues,timeMeanSigma)
ax.set_xlabel('$\sigma_b$')
ax.set_ylabel('RMSE')
ax.set_title('Time Averaged RMSE of Background Error Variances')
ax.grid(True)
fig.savefig('BG_Error_Variances.png',
            bbox_inches='tight',
            dpi=250)
#%% Changing Obervation quantities

kVals = [1,2,4,8]
kPerterbedRMSE =[]
for i in np.arange(0,len(kVals)):
    H = np.identity(n)
    H_alt = H[0::kVals[i],:]
    R_alt = np.identity(int(n/kVals[i]))*np.square(sigmao)
    yo_alt = yo[:,0::kVals[i]]
    RMSE_plot = np.mean(_3DVar(sigmab,R_alt,l,H_alt,IC,yo_alt,cycles,dt,F,Xtruth)[0][1])
    #RMSE_plot = np.mean(_OIDA(cycles,sigmao,sigmab,l,H,yo,IC,dt,F,Xtruth)[0][1])
    kPerterbedRMSE.append(RMSE_plot) 

fig,ax1 = plt.subplots()
ax1.plot(kVals,kPerterbedRMSE)
ax1.set_xticks(np.arange(1,10,1))
ax1.set_xlim(0)
ax1.set_ylim(0)
ax1.set_xlabel('K')
ax1.set_ylabel('RMSE')
ax1.set_title('Values of K and Time Averaged RMSE')
ax1.grid()
fig.savefig('Time_Avg_K_MSE.png',
            bbox_inches='tight',
            dpi=250)
#%% 
R = np.identity(n)*np.square(sigmao)
lVals = np.arange(1,6).tolist()
lPerterbedRMSE =[]
for i in np.arange(0,len(lVals)):
    RMSE_l_plot = np.mean(_3DVar(sigmab,R,lVals[i],H,IC,yo,cycles,dt,F,Xtruth)[0][1])
    lPerterbedRMSE.append(RMSE_l_plot)
   
fig,ax1 = plt.subplots()
ax1.plot(lVals,lPerterbedRMSE)
ax1.set_xticks(np.arange(1,7,1))
ax1.set_xlim(0)
ax1.set_ylim(0)
ax1.set_xlabel('l')
ax1.set_ylabel('RMSE')
ax1.set_title('Values of l and Time Averaged RMSE')
ax1.grid()
fig.savefig('Time_Avg_l_RMSE.png',
            bbox_inches='tight',
            dpi=250)