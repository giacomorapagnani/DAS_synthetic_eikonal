import os
import numpy as np
import matplotlib.pyplot as plt
from pyrocko import trace, util, io
from traveltimes_NLL_class import Traveltimes
from traveltimes_event_class import TravelTimeEvent
# %% CLASS

class ConvolutionalSynth:
    """
    Class to generate synthetic DAS data using convolution and NLL travel times.
    """

    def __init__(self, events_path, fiber_geometry_path, 
                 NLL_grid_parameters, NLL_matrices_parameters, 
                 time_parameters):
        self.event = self._load_events(events_path)
        self.fiber_geometry = self._load_fiber_geometry(fiber_geometry_path)
        self.tt_class = TravelTimeEvent(NLL_grid_parameters, fiber_geometry_path)
        self._load_matrices_parameters(NLL_matrices_parameters)
        self._load_time_parameters(time_parameters)

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
        phir = self.receiver_azimuth()
        ih = self.take_off_angle()

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
        ih = self.take_off_angle()
        if phase=='P':
            dir_p = (np.cos(ih))**2
        elif phase=='S':
            dir_s = np.sin(2*ih)
        else:
            print('Error: in directvity_fiber Phase not correctly specified.\n choose P or S')
            exit
        return dir_p, dir_s

    def receiver_azimuth(self):
        # !!!MISSING!!!
        return  
    def take_off_angle(self):
        # !!!MISSING!!!
        return

    def trace_amplitude(self,phase, exclude_directivity=False, exclude_radiation_pattern=False):
        # Calculate trace amplitude at a given fiber's channel.
        if exclude_directivity:
            dir_fib = 1
        else:
            dir_fib = self._directivity_fiber(phase)
        if exclude_radiation_pattern:
            rad_pat = 1
        else:
            rad_pat = self._radiation_patterns(phase)
        amp = 1 * dir_fib * rad_pat
        return  amp

    def _gen_data_matrix(self):
        data_p=np.zeros( ( self.ns_ch, self.ns_tax ) )
        data_s=np.zeros( ( self.ns_ch, self.ns_tax ) )
        return data_p, data_s
    
    def __gen_time_axis(self):
        ns=int(round(self.time_window/self.dt))
        tax=np.arange(ns)*self.dt
        return tax ,ns

    def convolution(self,event):
        # generates P,S arrivals convolved with Ricker wavelet, scaled with amplitude
        tt_p=self.tt_class.get_travel_time(event, self.tt_nll_p)
        tt_s=self.tt_class.get_travel_time(event, self.tt_nll_s)
        tt_p_indices= np.round( tt_p / self.dt ).astype(int)
        tt_s_indices= np.round( tt_s / self.dt ).astype(int)
        
        dataP,dataS=self._gen_data_matrix()
        # trace_amplitude just gives one numebr = 1
        # MODIFY!!!
        dataP[self.ch_indices,tt_p_indices]= self.trace_amplitude('P',
                    exclude_directivity=True, exclude_radiation_pattern=True )
        dataS[self.ch_indices,tt_s_indices]= self.trace_amplitude('S',
                    exclude_directivity=True, exclude_radiation_pattern=True )
    
        for i in range(self.ns_ch):
            seisP=np.convolve(self.ricker_w,dataP[i,:],mode='same')
            seisS=np.convolve(self.ricker_w,dataS[i,:],mode='same')
            # cut head and tails of the convolved seismogram
            dataP[i,:]=seisP
            dataS[i,:]=seisS
        data = dataP + dataS
        data = self._add_noise(data, noise_type='gaussian')
        return data
    
    def _add_noise(self, data, noise_type='gaussian'):
        if noise_type=='gaussian':
            noise = np.random.normal(0,0.03,(data.shape))
        elif noise_type=='realistic':
            # !!!MISSING!!!
            noise = 0
        data += noise
        return data

    def __ricker(self):
        # select dt or dt_w (if provided)
        if self.dt_w is None:
            dt=self.dt  
        else:
            dt=self.dt_w
        # if nsamples_w even -> add 1 to make it odd
        ns_w=int(round(self.time_window_w/dt))
        if (ns_w%2)==0:
            ns_w +=1

        tax = np.arange(ns_w)*dt - self.time_window_w/2
        ric = (1 - 2 * (np.pi**2) * (self.frequency_w**2) * (tax**2)) \
                * np.exp(-(np.pi**2) * (self.frequency_w**2) * (tax**2))
        # if 'self.derivative_w=True' compute derivative of ricket wavelet
        if self.derivative_w:
            # ric' (same number of elements as ric)
            ric[1:]= ( ric[1:]-ric[0:-1] ) / dt
            ric[0]=0
            # normalization
            w_max = np.max(np.abs(ric))
            ric=ric/w_max
        
        #self.ricker_w_tax=tax
        check_wavelet=False
        if check_wavelet:
            plt.figure()
            plt.plot(tax,ric,'k.-')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.title('Ricker wavelet')
        return ric
    
    #---------------------------------------------
    ################### LOAD #####################
    #---------------------------------------------
    def _load_fiber_geometry(self, filepath):
        # FIBER GEOMETRY -> list: network_name channel_name, lat, lon, elev
        fiber_geometry = []
        with open(filepath, "r") as f:
            next(f)  # skip header
            for line in f:
                ntw_name, st_name, lat, lon, elev = line.split()
                fiber_geometry.append([ntw_name, st_name, float(lat), float(lon), float(elev)])
        # total number of channels
        self.ns_ch=len(fiber_geometry)
        # channel indices list
        self.ch_indices=list( range( self.ns_ch ) )
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
    
    def _load_matrices_parameters(self,NLL_matrices_parameters):
        db_path = NLL_matrices_parameters['db_path']
        hdr_filename = NLL_matrices_parameters['hdr_filename']
        precision = NLL_matrices_parameters['precision']
        model = NLL_matrices_parameters['model']
        # Load NLL traveltime matrices
        tt_nll_class = Traveltimes(db_path, hdr_filename)
        self.tt_nll_p = tt_nll_class.load_traveltimes('P', model, precision)
        self.tt_nll_s = tt_nll_class.load_traveltimes('S', model, precision)
        return
    
    def _load_time_parameters(self,time_parameters):
        self.dt = time_parameters['dt']
        self.time_window = time_parameters['time_window']
        # generate TIME AXIS (+ tax numer of samples)
        self.tax, self.ns_tax = self.__gen_time_axis()
        self.frequency_w = time_parameters['frequency_w']
        self.time_window_w = time_parameters['time_window_w']
        self.dt_w = time_parameters['dt_w']
        self.derivative_w = time_parameters['derivative_w']
        # generate RICKER WAVELET
        self.ricker_w = self.__ricker()
        return
    
    #---------------------------------------------
    ################## PLOT/SAVE #################
    #---------------------------------------------
    def plot_seismogram(self,seismogram,event,plot_fig=True,save_fig=False):
        plt.figure(f'{event[0]}', figsize=(13,7))
        plt.title(f'{event[0]}')
        plt.imshow(seismogram, aspect='auto', cmap='seismic', extent=[0, self.time_window, self.ns_ch, 0])
        plt.colorbar(label='Amplitude')
        plt.xlabel('Time (s)')
        plt.ylabel('Channel Number')
        if save_fig:
            plt.savefig(f'../PLOTS/{event[0]}.pdf')
            print(f'-\nSAVING FIGURE: {event[0]}.pdf')
        if plot_fig:
            plt.show()  
        return
    
    def save_seismogram_mseed(self,seismogram,event):
        # FIBER GEOMETRY -> list: channel_name, lat, lon, elev
        # EVENTS -> list: event_name, tor, lat, lon, depth, mag, strike, dip, rake
        #tmin = util.str_to_time(event[1])-5
        tmin = util.str_to_time(event[1].replace('T', ' ').replace('Z', ''))
        traces=[]
        for i,channel in enumerate(self.fiber_geometry): 
            data = seismogram[i,:]
            tr = trace.Trace(
                    network=channel[0],station=channel[1], channel='DAS',
                    deltat=self.dt, tmin=tmin, ydata=data)
            traces.append(tr)
        io.save(traces, f'../DATA/{event[0]}.mseed')
        print(f'-\nSAVING TRACE: {event[0]}.mseed')
        return

if __name__ == "__main__":
    #%% INPUTS:
    workdir='../'

    #### 1 - EVENTS (load or generate)
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
        print(f'---\nGENERATING NEW CATALOGUE: {cat_file}')
        dataset=Synthetic_catalogue(cat_dir, inputs, input_type='dict')
        dataset.gen_catalogue(cat_file, cat_name, seed=11)
        events_file=os.path.join(cat_dir,cat_file)
    else:
        # Read already existing catalogue
        event_dir=os.path.join(workdir,'CAT')
        filename_events='catalogue_flegrei_MT_final.txt'                                ### CHANGE ###
        print(f'---\nREADING EVENTS FROM FILE: {filename_events}')
        events_file=os.path.join(event_dir,filename_events)

    #### 2 - FIBER GEOMETRY (load)
    fiber_geometry_dir=os.path.join(workdir,'FIBER_GEOMETRY')
    fiber_geometry_file=os.path.join(fiber_geometry_dir, 'flegrei_stations_geometry.txt')      ### CHANGE ###

    #### 3 - NLL grid parameters (insert)
    # (CHANGE: must match those used to generate the NLL traveltimes)
    nx=151      # number of grid points in x direction
    ny=151      # number of grid points in y direction
    nz=61       # number of grid points in z direction
    dx=0.1      # km
    dy=0.1      # km
    dz=0.1      # km
    ox=0.0      # origin x coordinate (km)
    oy=0.0      # origin y coordinate (km)
    oz=-1.0     # origin z coordinate (km), positive DOWN
    coord_origin_lat=40.777242 # latitude of the grid origin
    coord_origin_lon=14.025848 # longitude of the grid origin
    coord_origin_ele=0.0       # elevation of the grid origin (km)
    NLL_grid_inputs = {
        'nx': nx,'ny': ny, 'nz': nz, 'dx': dx, 'dy': dy, 'dz': dz,
        'co_x': ox, 'co_y': oy, 'co_z': oz, 
        'co_lat': coord_origin_lat, 'co_lon': coord_origin_lon, 'co_ele': coord_origin_ele}

    #### 4 - NLL traveltime matrices (load)
    db_path = workdir + 'NLL/FLEGREI/nll_grid'                                                    ### CHANGE ###
    hdr_filename = 'header.hdr'
    precision='single'
    model = 'time'
    NLL_matrices_inputs = {
        'db_path': db_path, 'hdr_filename': hdr_filename,
        'precision': precision, 'model': model}

    #### 5 - TIME AXIS (generate)
    dt= 0.01  # == 100 Hz                           # ARBITRARY (?) 
    time_window= 10 #s after origin time            # CHANGE

    #### 6 - RICKER WAVELET (generate)
    frequency_w=3                                  # CHANGE
    time_window_w=1. # s                           # CHANGE
    dt_w=None                                      # if None, use dt
    derivative_w=False                             # if True, use derivative of Ricker

    time_inputs={
        'dt':dt, 'time_window':time_window,
        'frequency_w':frequency_w, 'time_window_w':time_window_w ,
        'dt_w':dt_w, 'derivative_w':derivative_w}

    #%% SYNTHETIC GENERATION:

    # ConvolutionalSynth class
    synth_class=ConvolutionalSynth(events_path = events_file, # 1 - EVENTS
                             fiber_geometry_path = fiber_geometry_file, # 2 - FIBER GEOMETRY
                             NLL_grid_parameters = NLL_grid_inputs, # 3 - NLL GRID PARAMETERS
                             NLL_matrices_parameters = NLL_matrices_inputs, # 4 - NLL MATRICES
                             time_parameters = time_inputs) # 5,6 - TIME AXIS + RICKER WAVELET

    #----------------------------------------------------------------------

    # ih: incidence angle !!!MISSING!!!

    # phir: reciver-azimuth !!!MISSING!!!

    #----------------------------------------------------------------------

    # synthetic seismogram of first event
    seis = synth_class.convolution(synth_class.event[0])

    #synth_class.plot_seismogram(seis,synth_class.event[0], plot_fig=True, save_fig=False)
    
    synth_class.save_seismogram_mseed(seis,synth_class.event[0])
    
    #save seism matrix 2d .npz
    # matrix, xax, yax