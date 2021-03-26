#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Answer question 7
"""
import matplotlib.pyplot as plt
import numpy as np

location =  np.array([0,100,140,240])
xb = np.array([18,15,20]).T
yo = np.array([16,18,17]).T
error = np.array([2,2,2]).T
H = np.array([[1,0,0],[40/140,100/140,0],[0,1,0]])
B = np.array([[4,3.2,.8],[3.2,4,2],[.8,2,4]])
R = np.array([[1,0,0],[0,1,0],[0,0,1]])


#test = np.matmul(H,xb)

BHt = np.matmul(B,H.T)
HBHt = np.matmul(H,(np.matmul(B,H.T)))
inverse = np.linalg.inv(HBHt+R)
W =  np.matmul(BHt,inverse)
incrament = yo-np.matmul(H,xb)
xa = xb + np.matmul(W,incrament)

fig, ax = plt.subplots()
forecast = ax.plot([location[0],location[2],location[3]],xb,color = 'k',linestyle='--', label='forecast')
obs = ax.scatter(location[0:3],yo,color='r',label='obs')
analysis =ax.plot([location[0],location[2],location[3]],xa,label='analysis')
ax.set_ylim(10,25)
ax.legend(loc="upper left")