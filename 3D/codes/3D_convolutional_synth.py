import numpy as np
import os
# %% FUNCTIONS

def load_focals(filename):
    return np.load(filename, allow_pickle=False)['focals']

def radiation_patterns(phif,delta,l,phir,ih,phase):
    """
    Calculate radiation pattern at fiber's channel given a FM.

    Args:
        phif  (float) : strike
        delta (float) : dip
        l     (float) : rake
        phir  (float) : receiver-azimuth
        ih    (float) : take-off angle
        phase         : P or S phase

    Returns:
        float: radiation amplitude for P or S phase.
    """

    Fp= ( np.cos(l) * np.sin(delta) * np.sin( 2 * ( phir - phif ) )
    - np.sin(l) * np.sin(2*delta) * (np.sin(phir - phif))**2 ) * ( np.sin(ih) )**2 
    + ( np.sin(l) * np.cos(2*delta) * np.sin(phir - phif) ) * np.sin(2*ih) 
    + np.sin(l) * np.sin(2*delta) * (np.cos(ih))**2
    
    Fsv= ( np.sin(l) * np.cos(2*delta) * np.sin(phir - phif) 
    - np.cos(l) * np.cos(delta) * np.cos(phir - phif) ) * np.cos(2* ih)
    + 0.5 * np.cos(l) *  np.sin(delta) * np.sin( 2 * ( phir - phif ) ) * np.sin(2*ih)
    - 0.5 * np.sin(l) *  np.sin(2*delta) * np.sin(2*ih) * ( 1 + (np.sin(phir-phif))**2 )
    
    Fsh=( np.cos(l) * np.cos(delta) * np.sin(phir - phif) 
    + np.sin(l) * np.cos(2*delta) * np.cos(phir - phif) ) * np.cos(ih)
    +  ( np.cos(l) *  np.sin(delta) * np.cos( 2 * ( phir - phif ) )
    - 0.5 * np.sin(l) *  np.sin(2*delta) * np.sin(2 * (phir-phif) ) ) * np.sin(ih)
	
    Fs = Fsv + Fsh

    if phase=='P':
        return Fp
    elif phase=='S':
        return Fs
    else:
        print('Error: in directvity_fiber Phase not correctly specified.\n choose P or S')
        exit
    return 

def ricker(f0,dt,nsamples,derivative=False):
    # if 'derivative=True' compute derivative of ricket wavelet
    if (nsamples%2):
        t=((nsamples-1)*dt)/2.
    else:
        nsamples +=1
        t=((nsamples-1)*dt)/2.
    x=f0*np.linspace(-t,t,nsamples)
    ric=( ( 1. - ( 2. * np.pi * (x**2.) ) ) * np.exp( -np.pi * (x**2.) ) )
    tax=x/f0
    if derivative:
        # ric'
        ric[1:]= ( ric[1:]-ric[0:-1] ) / dt
        ric[0]=0
        # normalization
        w_max = np.max(np.abs(ric))
        ric=ric/w_max
    return ric, tax

def directivity_fiber(ih,phase):
    """
    Calculate directivity at a given fiber's channel.

    Args:
        ih    (float) : take-off angle
        phase         : P or S phase

    Returns:
        float: directivity amplitude for P or S phase.
    """
    if phase=='P':
        dir_p = (np.cos(ih))**2
    elif phase=='S':
        dir_s = np.sin(2*ih)
    else:
        print('Error: in directvity_fiber Phase not correctly specified.\n choose P or S')
        exit
    return dir_p, dir_s

#%% LOADING DATA
workdir='../'
data_dir=os.path.join(workdir,'DATA')
travel_time_dir=os.path.join(workdir,'TRAVEL_TIMES')

# LOAD FOCAL MECHANISMS
switch_focal_mechanism_load=False                                           # SWITCH
if switch_focal_mechanism_load:  
    focal_mechanism_file=os.path.join(data_dir, 'focals_3d_20_deg.npz')     # CHANGE
    # list of dictionaries
    # keys : strike, dip, rake
    focal_mechanisms=load_focals(focal_mechanism_file)
else:
    fm = dict()
    fm['strike'] = 0.0
    fm['dip'] = 90.0
    fm['rake'] = 0.0

# LOAD FIBER GEOMETRY
fiber_geometry_dir=os.path.join(data_dir, 'fiber_geometry.npy')
# list of dictionaries
# keys : x, y, z
fiber_geometry=np.load(fiber_geometry_dir)

# LOAD TRAVEL TIME 
# ARRAY 3D : NxNxN
tt_file_p=os.path.join(travel_time_dir, 'tt_p.npy')
tt_file_s=os.path.join(travel_time_dir, 'tt_s.npy')
tt_p=np.load(tt_file_p)
tt_s=np.load(tt_file_s)

#%% SYNTHETIC GENERATION
#----------------------------------------------------------------------
# create list of dictionaries for every point in the fiber
# keys: x , y , z , tt_p , amp_p, tt_s , amp_s
n=len(fiber_geometry)
trace = np.zeros( n, dtype=[ ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
        ('tt_p', 'f4'), ('amp_p', 'f4'), ('tt_s', 'f4'), ('amp_s', 'f4') ] )
trace['x'] = fiber_geometry['x'].astype('f4')
trace['y'] = fiber_geometry['y'].astype('f4')
trace['z'] = fiber_geometry['z'].astype('f4')

tt_fiber_p=tt_p[fiber_geometry['x'],fiber_geometry['y'],fiber_geometry['z']]
trace['tt_p'] = tt_fiber_p.astype('f4')

tt_fiber_s=tt_s[fiber_geometry['x'],fiber_geometry['y'],fiber_geometry['z']]
trace['tt_s'] = tt_fiber_s.astype('f4')

# ih: incidence angle MISSING

# phir: reciver-azimuth MISSING

# calculate recorded amplitude
amp_p = 1 * directivity_fiber(ih,'P') * radiation_patterns(fm['stirke'],fm['dip'],fm['rake'],phir,ih,'P')
amp_s = 1 * directivity_fiber(ih,'S') * radiation_patterns(fm['stirke'],fm['dip'],fm['rake'],phir,ih,'S')

trace['amp_p'] = amp_p.astype('f4')
trace['amp_s'] = amp_s.astype('f4')

#----------------------------------------------------------------------
# Time axis
dt= 0.02                                        # ARBITRARY (?) 
tmax = np.max( np.abs( trace['tt'] ) )
tmax += tmax*0.20 # add 20% to max travel time
ns=int(round(tmax/dt))
tax=np.arange(ns)*dt

# Ricker wavelet
nsamples_w=200                                  # CHANGE
frequency_w=75                                  # CHANGE
wavelet,taxw=ricker(frequency_w,dt,nsamples_w,der=True)


# Time vs Channels matrix
dataP=np.zeros((np.size(trace),np.size(tax)))
dataS=np.zeros((np.size(trace),np.size(tax)))
trace_index=list(range(len(trace)))

time_index_p= int( round( trace['tt_p']/dt ) )
dataP[trace_index,time_index_p]=trace['amp_p']

time_index_s= int( round( trace['tt_s']/dt ) )
dataS[trace_index,time_index_s]=trace['amp_s']

# Convolution
for i in range(np.size(trace_index)):
    seisP=np.convolve(wavelet,dataP[i,:])
    seisS=np.convolve(wavelet,dataS[i,:])

data = dataP + dataS

noise = np.random.normal(0,1/30,(data.shape))
data = data + noise
#----------------------------------------------------------------------