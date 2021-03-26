"""
This module contains compute functions for HW2
"""
import numpy as np

#%%
def IntergrateEuler(data,steps,dt,F):
    """ This function integrates a 1-D Lorenz96 model 
    forward in time with the Euler Differentiation Scheme"""
    xCells = len(data)
    update = np.zeros(xCells)
    for j in np.arange(1,steps,1):
        for i in np.arange(0,xCells,1):
            update[i] = ((data[(i+1)%xCells]-data[(i-2)%xCells])\
                                *data[(i-1)%xCells])-data[i]+F
        data = data + (dt*update)
    return data

def getB(n,sigma_b,l):
    """ Gets B Matrix """
    B = np.zeros((n,n))
    for i in np.arange(0,n,1):
        for j in np.arange(0,n,1):
            distance = min((i-j)%n,(j-i)%n)
            B[i,j] = np.square(sigma_b)*np.exp(-(np.square(distance))/(2*l*l))
    return B

def objinterp(B,H,R,yo,xb):
    """ Calulates the analysis incrament """
    W = (B @ H.T) @ (np.linalg.inv(H @ B @ H.T + R))
    deltax = yo - (H @ xb)
    xa = xb + (W @ deltax)
    return xa

def rmse(x,y):
    """ Computes the RMSE between two arrays"""
    squarediff = (x-y)**2
    rmse = np.zeros(len(x))
    for i in np.arange(0,len(x)):
        rmse[i] = (np.sum(squarediff[i,:])/len(x))**.5
    return rmse

def _OIDA(cycles,sigmao,sigmab,l,H,yo,IC,dt,F,Xtruth):
    """ Integrates a model forward in time with OI
        data assimilation """ 
    n  = yo.shape[1]
    xa = np.zeros((cycles,n))
    xb = np.zeros((cycles,n))
    R = np.identity(n)*np.square(sigmao)
    B = getB(n,sigmab,l)
    xb[0,:] = IC
    for i in np.arange(0,cycles-1,1):
        xa[i,:] = objinterp(B,H,R,yo[i,:],xb[i,:])
        xb[i+1,:] = IntergrateEuler(xa[i,:],n,dt,F)
    xa[i+1,:] = objinterp(B,H,R,yo[i+1,:],xb[i+1,:])        
    
    RMSE_xa = rmse(xa,Xtruth)
    RMSE_xb = rmse(xb,Xtruth)
    RMSE = [RMSE_xb,RMSE_xa]
    RMSE_plot = np.zeros(cycles*2)
    for i in np.arange(0,len(RMSE_plot)):
        RMSE_plot[i] = RMSE[i%2][int(np.floor(i/2))]
    return RMSE,RMSE_plot

def NoDA_RMSE(Xtruth,sigmab,cycles,dt,F):
    """ Integrates a model forward in time with 
        no data assimilation """
    n = Xtruth.shape[1]
    perturbedIC = Xtruth[0,:]+np.random.normal(0,sigmab,n)
    NoDA_xa = np.zeros((cycles,n))
    NoDA_xa[0,:] = perturbedIC
    for i in np.arange(1,cycles,1):    
        NoDA_xa[i,:] = IntergrateEuler(NoDA_xa[(i-1),:],n,dt,F)
    RMSE = rmse(NoDA_xa,Xtruth)
    return RMSE

def _3DVar(sigmab,R,l,H,IC,yo,cycles,dt,F,Xtruth):
    """ Integrates model forward in time with 3DVar
        data assimilation"""
    n = H.shape[1]                        # Dynamically setting n = 40
    R_inv = np.linalg.inv(R)
    B = getB(n,sigmab,l)                  # function to get B
    U, D, VT = np.linalg.svd(B)
    L = U @ (np.sqrt(np.diag(D))) @ U.T
    I = np.identity(n)
    aPrime = I + (L.T @ H.T @ R_inv @ H @ L)  # @ symbol denotes matrix-matrix multiplication
    xb = IC
    V = np.zeros(n)                       # Preallocating V vector
    output_xa = np.zeros((cycles,n))
    output_xb = np.zeros((cycles,n))
    
    for i in np.arange(0,cycles):         # Looping Cycles
        d_ob = yo[i,:].T - (H @ xb)
        bPrime = L.T @ H.T @ R_inv @ d_ob   
        r = bPrime - (aPrime @ V)
        p = r
        while np.linalg.norm(r) >.00001: 
            alpha = float(r @ r)/float(p @ aPrime @ p)
            V2 = V + (alpha*p)
            r2 = r - alpha*(aPrime @ p)
            beta = float(r2 @ r2)/float(r @ r)
            p2 = r2 + (beta*p)
            p = p2
            r = r2
            V  = V2
        xa = xb + (L @ V)
        xb = IntergrateEuler(xa,n,dt,F)    # Function to Integrate model forward 
        output_xa[i,:] = xa                # Writing out xa
        if i < cycles-1:
            output_xb[i+1,:]=xb            # Writing out xb
    output_xb[0,:] = IC 
    
    RMSE_xa = rmse(output_xa,Xtruth)
    RMSE_xb = rmse(output_xb,Xtruth)
    RMSE = [RMSE_xb,RMSE_xa]
    RMSE_plot = np.zeros(cycles*2)
    for i in np.arange(0,len(RMSE_plot)):
        RMSE_plot[i] = RMSE[i%2][int(np.floor(i/2))]
    return RMSE,RMSE_plot