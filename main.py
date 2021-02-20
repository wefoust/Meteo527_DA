"""
This code completes the data assimilation homework
"""

#Importing Libraries
import matplotlib.pyplot as plt
import numpy as np

location =  np.array([0,140,140+240])
xb = np.matrix([18,15,20]).T
error = np.matrix([2,2,2]).T
yo = np.matrix([16,17]).T
H = np.matrix([[1,0,0],[0,1,0]])
#B = np.array([[4,3.2,.8],[3.2,4,2],[.8,2,4]])
#cor = [.8,.2,.5]
cor = [.8,.2,.1] #drop correlation of between SC&NYC, increase corr with Pitt&SC
B = np.array([[error[0]*error[0],error[0]*error[1]*cor[0],error[0]*error[2]*cor[1]],
              [error[1]*error[0]*cor[0],error[1]*error[1],error[1]*error[2]*cor[2]],
              [error[2]*error[0]*cor[1],error[2]*error[1]*cor[2],error[2]*error[2]]])
B = np.asmatrix(B)
R = np.matrix([[1,.6],[.6,1]])

BHt = np.matmul(B,H.T)
HBHt = np.matmul(H,(np.matmul(B,H.T)))
inverse = np.linalg.inv(HBHt+R)
W =  np.matmul(BHt,inverse)
delta = yo-np.matmul(H,xb)
xa = xb + np.matmul(W,delta)
incrament = xa-xb

fig, ax = plt.subplots()
forecast = ax.plot(np.squeeze(np.asarray(location)),np.squeeze(np.asarray(xb)),
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
ax.legend(loc="upper left")
ax.set_xticks(location)
ax.set_xticklabels(['Pitt','SC','NYC'])
ax.set_title('Temperature Analysis With 2 Observations')
ax.set_ylabel('Temperature ($^\circ$F)')
ax.set_xlabel('Location')

fig.savefig('OI_2obs.png',
                bbox_inches='tight',
                dpi=200)


