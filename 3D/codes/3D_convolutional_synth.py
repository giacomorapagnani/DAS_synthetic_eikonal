import numpy as np
import os
from traveltimes_NLL_class import Traveltimes
from traveltimes_event_class import TravelTimeEvent
# %% CLASS

class ConvolutionalSynth:
    """
    Class to generate synthetic DAS data using convolution and NLL travel times.
    Attributes:
        event (list): list containing event coordinates and source parameters:
            EventName OriginTime Latitude(deg) Longitude(deg) Depth(km) Magnitude Strike Dip Rake 
        fiber_geometry (list): List fiber geometry :
            Station_name    Latitude    Longitude   Elevation_km
    """

    def __init__(self, events_path, fiber_geometry_path):
        self.event = self._load_events(events_path)
        self.fiber_geometry = self._load_fiber_geometry(fiber_geometry_path)

    def _radiation_patterns(self,phase):
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
    
    def _directivity_fiber(self,phase):
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

    def trace_amplitude(self,phase):
        # Calculate trace amplitude at a given fiber's channel.
        dir_fib = self._directivity_fiber(phase)
        rad_pat = self._radiation_patterns(phase)
        amp = 1 * dir_fib * rad_pat
        return  amp

    def gen_data_matrix(self, dt, time_window):
        # Generates empty time vs channels data matrices for P and S waves
        self.dt = dt
        self.tax = self._gen_time_axis(time_window)
        self.data_p=np.zeros((np.size(self.fiber_geometry),np.size(tax)))
        self.data_s=np.zeros((np.size(self.fiber_geometry),np.size(tax)))
        return
    
    ######################################
    ######################################
    ######################################
    trace_index=list(range(len(trace)))
    time_index_p= int( round( trace['tt_p']/dt ) )
    dataP[trace_index,time_index_p]=trace['amp_p']
    ######################################
    ######################################
    ######################################

    def _gen_time_axis(self,tmax):
        ns=int(round(tmax/self.dt))
        tax=np.arange(ns)*dt
        return tax

    def ricker(self,f0, nsamples, dt=None, derivative=False):
        # if 'derivative=True' compute derivative of ricket wavelet
        if dt is None:
            dt=self.dt  
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
        self.ricker=ric
        self.ricker_tax=tax
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
    #%% INPUTS:
    workdir='../'
    #### 1 - FIBER GEOMETRY
    fiber_geometry_dir=os.path.join(workdir,'FIBER_GEOMETRY')
    fiber_geometry_file=os.path.join(fiber_geometry_dir, 'flegrei_stations_geometry.txt')      ### CHANGE ###

    #### 2 - EVENTS
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

    #### 3 - NLL grid parameters 
    # (CHANGE: should match those used to generate the traveltimes)
    nx=151 # number of grid points in x direction
    ny=151 # number of grid points in y direction
    nz=61  # number of grid points in z direction
    dx=0.1 # km
    dy=0.1 # km
    dz=0.1 # km
    ox=0.0      # origin x coordinate (km)
    oy=0.0      # origin y coordinate (km)
    oz=-1.0     # origin z coordinate (km), positive DOWN
    coord_origin_lat=40.777242 # latitude of the grid origin 
    coord_origin_lon=14.025848 # longitude of the grid origin
    coord_origin_ele=0.0       # elevation of the grid origin (km)
    NLL_grid_parameters = {
        'nx': nx,'ny': ny, 'nz': nz, 'dx': dx, 'dy': dy, 'dz': dz,
        'co_x': ox, 'co_y': oy, 'co_z': oz, 
        'co_lat': coord_origin_lat, 'co_lon': coord_origin_lon, 'co_ele': coord_origin_ele}

    #### 4 - NLL traveltime matrices
    db_path = '../NLL/FLEGREI/nll_grid'                                                    ### CHANGE ###
    hdr_filename = 'header.hdr'
    precision='single'
    model = 'time'
    tt_nll_obj = Traveltimes(db_path, hdr_filename)
    tt_nll_p = tt_nll_obj.load_traveltimes('P', model, precision)
    tt_nll_s = tt_nll_obj.load_traveltimes('S', model, precision)

    #### 5 - Time axis
    dt= 0.02                                        # ARBITRARY (?) 
    time_window= 120 #s                             # ARBITRARY (?)
    tax=time_win

    #### 6 - Ricker wavelet
    nsamples_w=200                                  # CHANGE
    frequency_w=75                                  # CHANGE
    wavelet,taxw=ricker()

    #%% SYNTHETIC GENERATION:
    # TravelTimeEvent class
    tt_class=TravelTimeEvent(NLL_grid_parameters= NLL_grid_parameters,
                             fiber_geometry_path=fiber_geometry_file)

    # ConvolutionalSynth class
    synth=ConvolutionalSynth(events_path=events_file, 
                             fiber_geometry_path=fiber_geometry_file)
    tt_p=tt_class.get_travel_time(event, tt_nll_p)
    tt_s=tt_class.get_travel_time(event, tt_nll_s)

    #----------------------------------------------------------------------

    # ih: incidence angle !!!MISSING!!!

    # phir: reciver-azimuth !!!MISSING!!!


    #----------------------------------------------------------------------



    # Time vs Channels matrix

    # Convolution
    for i in range(np.size(trace_index)):
        seisP=np.convolve(wavelet,dataP[i,:])
        seisS=np.convolve(wavelet,dataS[i,:])

    data = dataP + dataS

    noise = np.random.normal(0,1/30,(data.shape))
    data = data + noise
    #----------------------------------------------------------------------