import numpy as np
import os
# %% CLASS

class ConvolutionalSynth:
    """
    Class to generate synthetic DAS data using convolution and NLL travel times.

    Attributes:
        event (list): list containing event coordinates and source parameters:
            EventName OriginTime Latitude(deg) Longitude(deg) Depth(km) Magnitude Strike Dip Rake 
        fiber_geometry (list): List fiber geometry :
            Station_name    Latitude    Longitude   Elevation_km
        tt_p (np.ndarray): 1D array of P-wave travel times.
        tt_s (np.ndarray): 1D array of S-wave travel times.
    """

    def __init__(self, events_path, fiber_geometry_path, tt_p, tt_s):
        self.event = self._load_events(events_path)
        self.fiber_geometry = self._load_fiber_geometry(fiber_geometry_path)
        self.tt_p = tt_p
        self.tt_s = tt_s

    def radiation_patterns(self,phase):
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
        phif = self.strike
        delta = self.dip
        l = self.rake

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

    def directivity_fiber(self,phase):
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

    def incidence_angle(self):
        # !!!MISSING!!!
        return  
    def receiver_azimuth(self):
        # !!!MISSING!!!
        return  
    
    def _load_fiber_geometry(self, filepath):
        # FIBER GEOMETRY -> list: channel_name, lat, lon, elev
        fiber_geometry = []
        with open(filepath, "r") as f:
            next(f)  # skip header
            for line in f:
                st_name, lat, lon, elev = line.split()
                fiber_geometry.append([st_name, float(lat), float(lon), float(elev)])
        return fiber_geometry
    
    def _load_events(self, filepath):
        # EVENTS -> list: event_name, tor, lat, lon, depth, mag, strike, dip, rake
        events = []
        with open(filepath, "r") as f:
            next(f)  # skip header
            for line in f:
                event_name, tor, lat, lon, depth, mag, strike, dip, rake = line.split()
                events.append([event_name, str(tor), float(lat), float(lon), float(depth), 
                               float(mag), float(strike), float(dip), float(rake)])
        return events

if __name__ == "__main__":

    workdir='../'
    # FIBER GEOMETRY
    fiber_geometry_dir=os.path.join(workdir,'FIBER_GEOMETRY')
    fiber_geometry_file=os.path.join(fiber_geometry_dir, 'flegrei_stations_geometry.txt')      ### CHANGE ###
    
    # EVENTS
    SWITCH_generate_new_catalogue=False
    if SWITCH_generate_new_catalogue:
        # generate new synthetic catalogue
        from synthetic_catalogue_class import Synthetic_catalogue
        ### Catalogue Parameters 
        nsources=2
        lat_min=40.775
        lat_max=40.855       
        lon_min=14.07           
        lon_max=14.175          
        dep_min=1000          
        dep_max=5000           
        t_min="2022-01-01"    
        t_max="2025-12-01"    
        mag_min=2.0           
        mag_max=4.5           
        inputs={'n_sources':nsources,'latmin':lat_min, 'latmax':lat_max, 'lonmin':lon_min, 'lonmax':lon_max, 'depmin':dep_min, 'depmax':dep_max, 
                    'tormin':t_min, 'tormax':t_max, 'magmin':mag_min, 'magmax':mag_max, 'focal_mechanism':"dc_random_uniform"}
        
        ### Catlogue directory
        cat_dir=os.path.join(workdir,'CAT')
        cat_name='flegrei_synth_'                                               ### NEW CATALOGUE NAME (CHANGE) ###
        cat_file=f'catalogue_{cat_name}{str(nsources)}_ev.txt'
    
        ### Generate Catalogue
        print(f'GENERATING NEW CATALOGUE: {cat_file}')
        dataset=Synthetic_catalogue(cat_dir, inputs, input_type='dict')
        dataset.gen_catalogue(cat_file, cat_name, seed=11)
        events_file=os.path.join(cat_dir,cat_file)
    else:
        # Read already existing catalogue
        event_dir=os.path.join(workdir,'CAT')
        filename_events='catalogue_flegrei_synth_1_ev.txt'                                ### CHANGE ###
        print(f'READING EVENTS FROM FILE: {filename_events}')
        events_file=os.path.join(event_dir,filename_events)
    
    #####################################################
    # SYNTHETIC GENERATION
    synth=ConvolutionalSynth(events_path=events_file, 
                             fiber_geometry_path=fiber_geometry_file, 
                             tt_p=, 
                             tt_s=)
    
    # TRAVEL TIME
    # ARRAY 1D : N
    tt_file_p=os.path.join(travel_time_dir, 'tt_p.npy')
    tt_file_s=os.path.join(travel_time_dir, 'tt_s.npy')
    tt_p=np.load(tt_file_p)
    tt_s=np.load(tt_file_s)
    
    # %%
    
    #%% SYNTHETIC GENERATION
    #----------------------------------------------------------------------
    # create list of dictionaries for every point in the fiber
    # keys: channel_name, x , y , z , tt_p , amp_p, tt_s , amp_s
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
    
    # ih: incidence angle !!!MISSING!!!
    
    # phir: reciver-azimuth !!!MISSING!!!
    
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