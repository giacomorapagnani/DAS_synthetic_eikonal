import numpy as np
import os
# %% FUNCTIONS

def load_focals(filename):
    return np.load(filename, allow_pickle=False)['focals']

def radiation_patterns(phif,delta,l,phir,ih):
	# phif  : strike
	# delta : dip
	# l     : rake 
	# phir  : receiver-azimuth
	# ih    : take-off angle

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

    return Fp,Fs

def ricker(f0,dt,nsamples):
	if (nsamples%2):
		t=((nsamples-1)*dt)/2.
	else:
		nsamples +=1
		t=((nsamples-1)*dt)/2.
	x=f0*np.linspace(-t,t,nsamples)
	ric=( ( 1. - ( 2. * np.pi * (x**2.) ) ) * np.exp( -np.pi * (x**2.) ) )
	tax=x/f0
	return ric, tax

def directivity_fiber(ih):
    dir_p = (np.cos(ih))**2
    dir_s = np.sin(2*ih)
    return dir_p, dir_s

#%% LOADING DATA
workdir='../'
data_dir=os.path.join(workdir,'DATA')
travel_time_dir=os.path.join(workdir,'TRAVEL_TIMES')

focal_mechanism_file=os.path.join(data_dir, 'focals_3d_90_deg.npz')     # CHANGE
# list of dictionaries
# keys : strike, dip, rake
fm=load_focals(focal_mechanism_file)

fiber_geometry_dir=os.path.join(data_dir, 'fiber_geometry.npy')
# list of dictionaries
# keys : x, y, z
fiber_geometry=np.load(fiber_geometry_dir)

tt_file=os.path.join(travel_time_dir, 'tt.npy')
tt=np.load(tt_file)

#%% SYNTHETIC GENRATION