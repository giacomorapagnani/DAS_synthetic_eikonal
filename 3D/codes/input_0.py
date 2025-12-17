import os
from convolutional_synth_3d import ConvolutionalSynth
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

# synthetic seismogram of one event
ev_number=87            # CHANGE
seis = synth_class.convolution(synth_class.event[ev_number])

#synth_class.plot_seismogram(seis,synth_class.event[ev_number], plot_fig=True, save_fig=True)
    
synth_class.save_seismogram(seismogram = seis,
                            event = synth_class.event[ev_number],
                            file_prefix='synth_',
                            save_mseed=True,save_npy=True)
