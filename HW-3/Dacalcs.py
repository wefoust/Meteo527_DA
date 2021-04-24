"""
This module contains compute functions for HW3
"""
import numpy as np

def getEnsemble(ensMembers,Xtruth):
    """ Creates a randomly generated ensemble based on Xtruth"""
    n = Xtruth.shape[1]
    ensemble = np.zeros((ensMembers,n))
    bgInitConditions = np.round(Xtruth[0,:]+np.random.normal(0,1,n),3)
    for i in np.arange(0,ensMembers):
        ensemble[i,:]  = np.round(bgInitConditions + np.random.normal(0,1,n),3)
    return bgInitConditions,ensemble
    
def getObs(obsDensity,sigmao,steps,Xtruth):
    """ Creates randomly generated observations, a linear observation operator
        and R matrix"""
        
    n = Xtruth.shape[1]
    yo = np.zeros((Xtruth.shape[0],Xtruth.shape[1]))
    for i  in np.arange(0,Xtruth.shape[0]):
        yo[i,:] = np.round(Xtruth[i,:]+np.random.normal(0,1,n),3)
    yo = yo[:,0::obsDensity]  # 10000x20
    yo= yo[0::steps,:]        # 1000x20

    H = np.identity(n)
    H = H[0::obsDensity,:]                             # 20x40
    R = np.identity(H.shape[0])*np.square(sigmao)      # 20x20
    return yo,H,R

def ESRF(ensembleBG,H,R,yo,cycles,Xtruth,steps,loc=False,cutoff=None,obsDensity=2,getPb=None):
    """ This code completes an Ensemble Square Root Filter for the Lorenz 96
        Model through time. Input parameters allow for localization or 
        error covariance matrix (Pb) calculations to be included"""
        
    if loc == True and cutoff == None:
        print('Must add cuttoff')
        return None
    
    output_xa = np.zeros((cycles,H.shape[1]))           # 1000x40
    output_xb = np.zeros((cycles,H.shape[1]))
    ensemble = ensembleBG+0                             # declare ensemble var
    n = ensemble.shape[1]                               # scalar = 40 
    I = np.identity(H.shape[1])                         # 40x40
    numObs = H.shape[0]                                 # scalar = 20
    numEnsMembers = ensemble.shape[0]                   # scalar = 80
    priorObs = np.zeros((numEnsMembers,1))                # 80x1
    priorObsPerturbation = np.zeros((numEnsMembers,1))  # 80x1
    priorPurturbation = np.zeros((numEnsMembers,n))     # 80x40
    updatedPerturbation = np.zeros((numEnsMembers,n))   # 80x40
    Pb = np.zeros((n,n,cycles))
    Pb_corr = np.zeros((n,n,cycles))
    spread = np.zeros(cycles)
    
    if loc == True:
        prerhoMatrix = np.zeros((n,n))
        for i in np.arange(0,n):
            for j in np.arange(0,n):
                distance = min((i-j)%n,(j-i)%n)
                r = distance/cutoff
                if r < 1:
                    rho = (-r**5/4)+(r**4/2)+(r**3*5/8)-(r**2*5/3)+1
                elif r <2 and r >=1:
                    rho = (r**5/12)-(r**4/2)+(r**3*5/8)+(r**2*5/3)-(5*r)+4-(2/3/r)
                elif r >= 2:
                    rho = 0    
                prerhoMatrix[i,j] = rho
        rhoMatrix = prerhoMatrix[::2,:]

    for i in np.arange(0,cycles):
        output_xb[i,:] = ensemble.mean(axis=0)
        for j in np.arange(0,numObs):
            priorMean = np.expand_dims(np.mean(ensemble,0),0)               
            priorPurturbation = ensemble - priorMean

            for k in np.arange(0,numEnsMembers):         
                priorObs[k] = (H[j:j+1,:] @ ensemble[k:k+1,:].T)           
    
            priorObsMean = np.mean(priorObs)                                    
            priorObsPerturbation = priorObs - priorObsMean
            obsErrorVariance = R[j][j]                                        
            obsPriorVariance = np.sum(priorObsPerturbation**2)/(numEnsMembers-1)           
            cov_x_y = np.sum((priorObsPerturbation * priorPurturbation),axis=0)\
                              / (numEnsMembers-1)

            K = cov_x_y/(obsPriorVariance+obsErrorVariance)                     
            K = np.expand_dims(K,axis=0)        
            if loc == True:
                K = rhoMatrix[j,:] * K

            cov_x_y = np.expand_dims(cov_x_y,axis=0)    
            updatedMean = priorMean + (K * (yo[i,j] - priorObsMean))         
            phi = 1 / (1 + np.sqrt(obsErrorVariance/(obsPriorVariance + obsErrorVariance)))   

            for k in np.arange(0,numEnsMembers):
                updatedPerturbation[k,:] = ((I - phi * (K.T @ H[j:j+1,:])) \
                                            @ priorPurturbation[k:k+1,:].T).T
               
            ensemble = updatedMean + updatedPerturbation
        
        output_xa[i,:] = np.mean(ensemble,0)                # updated ensemble mean
        for k in np.arange(0,numEnsMembers):
            ensemble[k,:] = IntergrateEuler(ensemble[k,:],steps,.005,8)
        
        spread[i] = np.sqrt((np.sum(updatedPerturbation**2))/(numEnsMembers-1))

        if getPb == True:        
            Pb[:,:,i] = updatedPerturbation.T @ updatedPerturbation
            for x in np.arange(0,n):
                for y in np.arange(0,n): 
                    Pb[x,y,i] = Pb[x,y,i]/(np.std(ensemble[:,x])*np.std(ensemble[:,y]))/(numEnsMembers-1)
    
    if getPb == True:
        Pb_corr = np.mean(Pb,axis = 2)

    RMSE_xa = np.zeros(cycles)
    RMSE_xb = np.zeros(cycles)
    
    for i in np.arange(0,cycles):
        RMSE_xa[i] = rmse(output_xa[i,:],Xtruth[i*steps,:])
        RMSE_xb[i] = rmse(output_xb[i,:],Xtruth[i*steps,:])

    RMSE = [RMSE_xb,RMSE_xa]
    RMSE_plot = np.zeros(cycles*2)
    for i in np.arange(0,len(RMSE_plot)):
        RMSE_plot[i] = RMSE[i%2][int(np.floor(i/2))]
    
    if getPb ==True:
        return output_xa, output_xb,Pb_corr,RMSE_plot,spread
    else:
        return output_xa, output_xb,RMSE_plot,spread

#%%
def IntergrateEuler(data,steps,dt,F):
    """ This function integrates a 1-D Lorenz96 model 
    forward in time with the Euler Differentiation Scheme"""
    xCells = len(data)
    update = np.zeros(xCells)
    for j in np.arange(0,steps):
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
    if squarediff.ndim == 2:
        for i in np.arange(0,len(x)):
            rmse[i] = (np.sum(squarediff[i,:])/len(x))**.5
    else:
        rmse = (np.sum(squarediff)/len(x))**.5
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

def _3DVar(sigmab,R,l,H,IC,yo,cycles,dt,F,Xtruth,steps):
    """ Integrates model forward in time with 3DVar
        data assimilation 
        sigmab = error background
        R = R matrix
        l = length scale
        H = Observation Operator
        IC = Initial Conditions
        yo = time x observations matrix
        cycles = cycles to run
        steps = steps to integrate model between cycles"""
    n = H.shape[1]                             # Dynamically setting n = 40
    R_inv = np.linalg.inv(R)
    B = getB(n,sigmab,l)                       # function to get B
    U, D, VT = np.linalg.svd(B)
    L = U @ (np.sqrt(np.diag(D))) @ U.T
    I = np.identity(n)
    aPrime = I + (L.T @ H.T @ R_inv @ H @ L)  # @ symbol denotes matrix-matrix multiplication
    xb = IC + 0
    V = np.zeros(n)                           # Preallocating V vector
    output_xa = np.zeros((cycles,n))
    output_xb = np.zeros((cycles,n))
    
    for i in np.arange(0,cycles):             # Looping Cycles
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
        xb = IntergrateEuler(xa,steps,dt,F)   # Function to Integrate model forward 
        output_xa[i,:] = xa                   # Writing out xa
        if i < cycles-1:
            output_xb[i+1,:]=xb               # Writing out xb
    output_xb[0,:] = IC
    
    RMSE_xa = np.zeros(cycles)
    RMSE_xb = np.zeros(cycles)
    for i in np.arange(0,cycles):
        RMSE_xa[i] = rmse(output_xa[i,:],Xtruth[i*steps,:])
        RMSE_xb[i] = rmse(output_xb[i,:],Xtruth[i*steps,:])

    RMSE = [RMSE_xb,RMSE_xa]
    RMSE_plot = np.zeros(cycles*2)
    for i in np.arange(0,len(RMSE_plot)):
        RMSE_plot[i] = RMSE[i%2][int(np.floor(i/2))]  
    return RMSE_plot
