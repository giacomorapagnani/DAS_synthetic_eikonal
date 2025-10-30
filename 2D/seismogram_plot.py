import numpy as np
import matplotlib.pyplot as plt

s_salv=np.load('wellA_dt2ms.npy')                            # INSERT seismogram 

dz=5.
zax=np.arange(s_salv.shape[0]) * dz

dt=0.002
tax=np.arange(s_salv.shape[1]) * dt
'''
plt.figure('seismogram_salvus')
plt.title('Seismogram Salvus')
plt.imshow(s_salv, aspect='auto', cmap='seismic',
            extent=[tax[0], tax[-1], zax[-1] ,zax[0]  ])
plt.xlabel('Time [s]')
plt.ylabel('Depth [m]')
plt.xlim([0,2])
plt.ylim([1200,0])
plt.colorbar()
'''
s_eik=np.load('location/seismogram_A_rad.npy')                            # INSERT seismogram 
s_eik=s_eik/np.max(np.abs(s_eik))               #normalization


dz=5.
zax=np.arange(s_eik.shape[0]) * dz

dt=0.002
tax=np.arange(s_eik.shape[1]) * dt

'''
plt.figure('seismogram_eikonal')
plt.title('Seismogram Eikonal')
plt.imshow(s_eik, aspect='auto', cmap='seismic',
            extent=[tax[0], tax[-1], zax[-1] ,zax[0]  ])
plt.xlabel('Time [s]')
plt.ylabel('Depth [m]')
plt.colorbar()
'''

tmax=2.
zmax=1200.
itmax=int(round(tmax//dt))
izmax=int(round(zmax//dt))

s_salv=s_salv[0:izmax,0:itmax]
tax=tax[0:itmax]
zax=zax[0:izmax]

fig = plt.figure('subplot')

plt.subplot(2, 1, 1)
plt.imshow(s_salv, aspect='auto', cmap='seismic',
            extent=[tax[0], tax[-1], zax[-1] ,zax[0]  ],vmin=-1,vmax=1)
plt.ylabel('Depth [m]')
plt.colorbar()

plt.subplot(2, 1, 2)
plt.imshow(s_eik, aspect='auto', cmap='seismic',
            extent=[tax[0], tax[-1], zax[-1] ,zax[0]  ],vmin=-1,vmax=1)
plt.xlabel('Time [s]')
plt.ylabel('Depth [m]')
plt.colorbar()


plt.show()