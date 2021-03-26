"""
This code completes the data assimilation homework
"""
import matplotlib.pyplot as plt
import numpy as np

location =  np.array([0,140,240])


xb = np.array([18,15,20]).T
error = np.array([2,2,2]).T
yo = np.array([16,17]).T
H = np.array([[1,0,0],[0,1,0]])
B = np.array([[4,3.2,.8],[3.2,4,2],[.8,2,4]])
R = np.array([[1,0],[0,1]])

BHt = np.matmul(B,H.T)
HBHt = np.matmul(H,(np.matmul(B,H.T)))
inverse = np.linalg.inv(HBHt+R)
W =  np.matmul(BHt,inverse)
incrament = yo-np.matmul(H,xb)
xa = xb + np.matmul(W,incrament)



fig, ax = plt.subplots()
forecast = ax.plot(location,xb,color = 'k',linestyle='--', label='forecast')
analysis = ax.plot(location,xa,label='analysis')
obs = ax.scatter(location[0:1+1],yo,color='r',label='obs')
ax.set_ylim(10,25)
ax.legend(loc="upper left")


