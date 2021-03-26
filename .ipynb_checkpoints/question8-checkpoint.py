#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Question 8
"""
import matplotlib.pyplot as plt
import numpy as np

location =  np.array([0,140,240])


xb = np.array([18,15,20]).T
oldxb =xb
error = np.array([2,2,2]).T
yo = np.array([16,17]).T
H = np.array([[1,0,0],[0,1,0]])
B = np.array([[4,3.2,.8],[3.2,4,2],[.8,2,4]])
R = np.array([[1,0],[0,1]])


#These two constants are calculated based on the question
# In a real world case you will likely have to iterate over


#Creating 3x3 identity matrix
I = np.identity(3)


for i in range(len(yo)):
    sigma2yb = np.matmul(H[i],error)*np.matmul(H[i],error)   
    sigma2yo = 1
    W = np.matmul(B,H[i].T)*(1/(sigma2yb+sigma2yo))
    #W = 1/W
    print(W)
    xa = xb + W*(yo[i]-np.matmul(H[i],xb))
    intermediate = I-np.matmul(W,H[i])
    A = np.matmul(intermediate,B)
    xb = xa
    B = A

fig, ax = plt.subplots()
forecast = ax.plot(location,oldxb,color = 'k',linestyle='--', label='forecast')
obs = ax.scatter([location[0],location[1]],yo,color='r',label='obs')
analysis =ax.plot(location,xa,label='analysis')
#ax.set_ylim(10,25)
ax.set_ylim
ax.legend(loc="upper left")