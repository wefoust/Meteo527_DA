#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This modules holds compute functions for data assimilation
"""
import numpy as np

def IntergratEuler(data,steps,dt,F):
    """ This function integrates a 1-D model
    forward in time with the Euler Scheme"""
    xCells = len(data)
    update = np.zeros(xCells)
    modelOut = np.zeros((steps,xCells))
    modelOut[0,:] = data
    for j in np.arange(1,steps,1):
        for i in np.arange(0,xCells,1):
            update[i%xCells] = ((data[(i+1)%xCells]-data[(i-2)%xCells])\
                                *data[(i-1)%xCells])-data[i]+F
        data = data + (dt*update)
        modelOut[j,:]=data
    return modelOut

def rmse(x,y):
    diff = x-y
    rmse = (np.sum(diff**2)/len(x))**.5
    #diff = np.subtract(x,y)
    #rmse = np.sqrt(np.sum(np.square(diff))/len(x))
    return rmse 
    
def objinterp(B,H,R,yo,xb):
    BHt = np.matmul(B,H.T)
    HBHt = np.matmul(H,(np.matmul(B,H.T)))
    inverse = np.linalg.inv(HBHt+R)
    W =  np.matmul(BHt,inverse)
    delta = yo-np.matmul(H,xb)
    xa = xb + np.matmul(W,delta)
    return xa

def getB(n,sigma_b,l):
    B = np.zeros((n,n))
    for i in np.arange(0,n,1):
        for j in np.arange(0,n,1):
            distance = min((i-j)%n,(j-i)%n)
            B[i,j] = np.square(sigma_b)*np.exp(-(np.square(distance))/(2*l*l))
    return B

def sequentialAlg(xb,yo,steps,eb,B,R,H):
    I = np.identity(len(xb))
    for i in np.arange(0,steps,1):
        sigmab =np.square(np.matmul(H[i],eb))
        sigmao =R[i,i]
        W = np.matmul(B,H[i].T)*(1/(sigmab+sigmao))
        xa = xb + np.matmul(W,(yo[i]- np.matmul(H[i],xb))) 
        A = np.matmul(((I-np.matmul(W,H[i])) ),B)
        xb=xa
        B=A
        eb = np.sqrt(np.diagonal(B)) 
