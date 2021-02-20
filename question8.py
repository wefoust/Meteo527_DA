#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Question 8 of the homework
"""
import matplotlib.pyplot as plt
import numpy as np

        
location =  np.matrix([0,140,140+240])
xb = np.matrix([18,15,20]).T
oldxb = np.matrix([18,15,20]).T
yo = np.matrix([16,17]).T
eb = np.matrix([2,2,2]).T
B = np.matrix([[4,3.2,.8],[3.2,4,2],[.8,2,4]])
R = np.matrix([[1,0],[0,1]])
H =np.matrix([[1,0,0],[0,1,0]])
I = np.identity(3)   
   
for i in list(range(2)):
    sigmab =np.square(np.matmul(H[i],eb))
    sigmao =R[i,i]
    W = np.matmul(B,H[i].T)*(1/(sigmab+sigmao))
    xa = xb + np.matmul(W,(yo[i]- np.matmul(H[i],xb)))
    #print(np.matmul(W,H[i].T))
    #step2 = (I-np.matmul(W,H[i])) 
    A = np.matmul(((I-np.matmul(W,H[i])) ),B)
    xb=xa
    B=A
    
    eb = np.sqrt(np.diagonal(B))
    print(eb)
  
fig, ax = plt.subplots()
forecast = ax.plot(np.squeeze(np.asarray(location)),np.squeeze(np.asarray(oldxb)),
                   color = 'k',
                   linestyle='--',
                   label='forecast')

obs = ax.scatter([np.squeeze(np.asarray(location))[0],np.squeeze(np.asarray(location))[1]],
                 np.squeeze(np.asarray(yo)),
                 color='r',
                 label='obs')
analysis =ax.plot(np.squeeze(np.asarray(location)),
                  np.squeeze(np.asarray(xa)),
                  label='analysis')

ax.set_ylim(10,25)
ax.set_ylim
ax.legend(loc="upper left")
ax.set_xticks(np.squeeze(np.asarray(location)))
ax.set_xticklabels(['Pitt','SC','NYC'])
ax.set_title('Temperature Analysis With 2 Observations (Sequential Algorithm)')
ax.set_ylabel('Temperature ($^\circ$F)')
ax.set_xlabel('Location')

fig.savefig('OI_2obs_Sequential.png',
                bbox_inches='tight',
                dpi=200)