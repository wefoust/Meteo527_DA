
"""
This code completes question 7 of the 1st data assimilation assignment
"""

#Importing libraries
import matplotlib.pyplot as plt
import numpy as np

#Setting Variables
location =  np.matrix([0,100,140,140+240])
xb = np.matrix([18,15,20]).T
yo = np.matrix([16,18,17]).T
error = np.matrix([2,2,2]).T
H = np.matrix([[1,0,0],[0,1,0],[40/140,100/140,0]])
B = np.matrix([[4,3.2,.8],[3.2,4,2],[.8,2,4]])
R = np.matrix([[1,0,0],[0,1,0],[0,0,1]])

#Matrix Calculations
BHt = np.matmul(B,H.T)
HBHt = np.matmul(H,(np.matmul(B,H.T)))
inverse = np.linalg.inv(HBHt+R)
W =  np.matmul(BHt,inverse)
delta = yo-np.matmul(H,xb)
xa = xb + np.matmul(W,delta)

#Figures
fig, ax = plt.subplots()
forecast = ax.plot([np.squeeze(np.asarray(location))[0],
                   np.squeeze(np.asarray(location))[2],
                   np.squeeze(np.asarray(location))[3]],
                   np.squeeze(np.asarray(xb)),
                   color = 'k',
                   linestyle='--',
                   label='forecast')

obs = ax.scatter([np.squeeze(np.asarray(location))[0],
                  np.squeeze(np.asarray(location))[1],
                  np.squeeze(np.asarray(location))[2]],
                  np.squeeze(np.asarray(yo)),
                  color='r',
                  label='obs')

analysis =ax.plot([np.squeeze(np.asarray(location))[0],
                   np.squeeze(np.asarray(location))[2],
                   np.squeeze(np.asarray(location))[3]],
                   np.squeeze(np.asarray(xa)),
                   label='analysis')

ax.set_ylim(10,25)
ax.legend(loc="upper left")
ax.set_xticks([np.squeeze(np.asarray(location))[0],
               np.squeeze(np.asarray(location))[1],
               np.squeeze(np.asarray(location))[2],
               np.squeeze(np.asarray(location))[3]])
ax.set_xticklabels(['Pitt','Altoona','SC','NYC'])
ax.set_title('Temperature Analysis With 3 Observations')
ax.set_ylabel('Temperature ($^\circ$F)')
ax.set_xlabel('Location')

fig.savefig('OI_3obs.png',
                bbox_inches='tight',
                dpi=200)