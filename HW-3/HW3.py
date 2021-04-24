"""
This is unworking Code for HW3. You must have the file DAComputations.py 
within the same directory to run this code. DaComputations.py contains
all necessary functions used in this code. 
"""
import numpy as np
import matplotlib.pyplot as plt
from DAcalcs import IntergrateEuler,_3DVar,getB,ESRF,getObs,getEnsemble
#%% Creating Truth Run
n = 40
F = 8
dt = .005
initsteps = 1000
data = np.round(np.random.normal(0,1,n),5)
initconditions = IntergrateEuler(data,initsteps,dt,F)

truthSteps = 10000
cycles = int(truthSteps/10)                              # Number of cycles
stepsXcycle = 10                                         # number of steps in cycle
Xtruth = np.zeros((truthSteps,n))                        # Preallocating Xtruth array
Xtruth[0,:]=initconditions                               # first row = init conditions
for i in np.arange(1,truthSteps,1):    
    Xtruth[i,:] = IntergrateEuler(Xtruth[i-1,:],1,dt,F)  # Outputs each cycle and steps between cycle =n=40


#%% Parameters for CNTRL Run
# "Namelist" variables 
steps = 10         # Steps till cycle
obsDensity = 2     # Set obs at every X grid cell
ensMembers = 80    # Ensemble Members
l = .5
sigmao = 1
sigmab = 1
cycles = int(truthSteps/steps) 

#Getting Obs
yo,H,R = getObs(obsDensity,sigmao,steps,Xtruth)

# Generating background state & ensemble
bgInitConditions, ensemble = getEnsemble(ensMembers,Xtruth)

#%% Performing 3DVar, ENSRF, and Getting B Matrix
RMSE_3DVar_plot = _3DVar(sigmab,R,l,H,bgInitConditions,yo,cycles,dt,F,Xtruth,steps)
xa,xb,Pb,RMSE_ENSRF_Plot,spread = ESRF(ensemble,H,R,yo,cycles,Xtruth,steps,getPb=True)
B = getB(n,sigmab,l) 

#%% Question 1A) Plot
staggeredIndex = np.zeros((2,int(cycles)))
staggeredIndex[0,:] = np.arange(1,cycles+1)
staggeredIndex[1,:] = np.arange(1.001,cycles+1)
staggeredIndex = np.squeeze(staggeredIndex.reshape((1,int(cycles*2)),order='F'))

fig,(ax1,ax2) = plt.subplots(2)
ax1.plot(staggeredIndex, RMSE_3DVar_plot, label='3DVar')
ax1.plot(staggeredIndex, RMSE_ENSRF_Plot, color = 'r',label='ENSRF')
ax1.set_xticks(np.arange(0,cycles+1,int(cycles/10)).tolist())
ax1.set_title('3DVar and ENSRF RMSE')
ax1.grid()
ax1.legend(loc='upper right')
ax1.set_ylabel('RMSE')
ax2.plot(staggeredIndex, RMSE_3DVar_plot, label='3DVar')
ax2.plot(staggeredIndex, RMSE_ENSRF_Plot, color ='r',label='ENSRF')
ax2.set_xlim(0,50)
ax2.set_xticks(np.arange(0,51,5))
ax2.grid()
ax2.legend(loc='upper right')
ax2.set_xlabel('Cycle')
ax2.set_ylabel('RMSE')

fig.savefig('./figures/HW3_3DVarandENSRF.png',
        bbox_inches='tight',
        dpi=250) 

#%% Question 1B) Plot 
fig,(ax1,ax2) = plt.subplots(1,2,sharey=True)
fig.suptitle('        Time Averaged Correlation Matrices')
Bmatrix = ax1.pcolor(B,edgecolor='k',vmin=-0,vmax=1,cmap='coolwarm')
ax1.xaxis.tick_top()
ax1.set_xlabel('B')
ax1.set_ylabel('Position')

Pbmatrix = ax2.pcolor(Pb,edgecolor='k',vmin=-0,vmax=1,cmap='coolwarm')
ax2.invert_yaxis()
ax2.xaxis.tick_top()
ax2.set_xlabel('Pb')
cbar_ax = fig.add_axes([0.99, 0.09, 0.05, 0.75])
fig.colorbar(Pbmatrix, cax=cbar_ax)
plt.tight_layout()
fig.savefig('./figures/HW3_B_Pb.png',
        bbox_inches='tight',
        dpi=250) 

#%% Seriel ENKF w/ Localization
cutoffs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
numEnsMembers = [10, 40]
RMSE_localization=[]

steps = 10          # Steps till cycle
obsDensity = 2      # Obs at every X grid cell
l = .5
sigmao = 1
sigmab = 1
cycles = int(truthSteps/steps) 
yo,H,R = getObs(obsDensity,sigmao,steps,Xtruth)

for j in numEnsMembers:
    bgInitConditions, ensemble = getEnsemble(j,Xtruth)        
    for i in cutoffs:
        xa,xb,RMSE_cutoff_Plot,spread = ESRF(ensemble,H,R,yo,cycles,Xtruth,steps,loc=True,cutoff=i)
        RMSE_localization.append(np.mean(RMSE_cutoff_Plot[1::2]))

#%% Question 2 Plot
fig,ax = plt.subplots()
ax.plot(cutoffs,RMSE_localization[0:len(cutoffs)],label = '10 Members')
ax.plot(cutoffs,RMSE_localization[len(cutoffs):],label = '40 Members')
ax.set_xlabel('Radius of Influence')
ax.set_ylabel('RMSE')
ax.grid()
ax.legend()
ax.set_title('RMSE by Ensemble Members and ROI')
fig.savefig('./figures/HW3_ROI_Ensemble.png',
        bbox_inches='tight',
        dpi=250) 

#%%% Question 3
steps = 1          # Steps till cycle
obsDensity = 2     # Obs at every X grid cell
ensMembers = 80    # Ensemble Members
l = .5
sigmao = 1
sigmab = 1
cycles = int(truthSteps/steps) 
yo,H,R = getObs(obsDensity,sigmao,steps,Xtruth)
bgInitConditions, ensemble = getEnsemble(ensMembers,Xtruth)
xa,xb,RMSE_ENSRF_Plot_1,spread_1 = ESRF(ensemble,H,R,yo,cycles,Xtruth,steps)

steps = 10         # Steps till cycle
cycles = int(truthSteps/steps) 
xa,xb,RMSE_ENSRF_Plot_10,spread_10 = ESRF(ensemble,H,R,yo,cycles,Xtruth,steps)

steps = 100         # Steps till cycle
cycles = int(truthSteps/steps) 
xa,xb,RMSE_ENSRF_Plot_100,spread_100 = ESRF(ensemble,H,R,yo,cycles,Xtruth,steps)




#%% Question 3 Plot
fig,ax = plt.subplots()
ax.plot(RMSE_ENSRF_Plot_1[1::2],color ='C0', label = 'm=1 RMSE')
ax.plot(spread_1,color='C0',linestyle='--', label = 'm=1 Spread')
ax.plot(np.arange(0,truthSteps,10),RMSE_ENSRF_Plot_10[1::2], color = 'r', label = 'm=10 RMSE' )
ax.plot(np.arange(0,truthSteps,10),spread_10, color = 'r', linestyle ='--',label = 'm=10 Spread')
ax.plot(np.arange(0,truthSteps,100),RMSE_ENSRF_Plot_100[1::2],color ='k', label = 'm=100 RMSE')
ax.plot(np.arange(0,truthSteps,100),spread_100,color ='k', linestyle = '--', label ='m=100 Spread')
ax.set_ylabel('RMSE')
ax.set_ylim(0,18)
ax.set_xlabel('Timestep')
ax.grid()
ax2=ax.twinx()
ax2.set_ylim(0,18)
ax2.set_ylabel('Ensemble Spread')
ax2.set_title('RMSE and Ensemble Spread')
ax2.grid()
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -.15), shadow=True, ncol=3)

fig.savefig('./figures/HW3_M_Spread.png',
        bbox_inches='tight',
        dpi=250) 