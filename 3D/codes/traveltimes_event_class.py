import os
import numpy as np
from traveltimes_NLL_class import Traveltimes
import matplotlib.pyplot as plt
from latlon2cart_class import Coordinates

class TravelTimeEvent:
    def __init__(self, NLL_grid_parameters, fiber_geometry_path):
        self.fiber_geometry = self._load_fiber_geometry(fiber_geometry_path)
        self.nll_par = NLL_grid_parameters
        self.coord=Coordinates(self.nll_par['co_lat'], self.nll_par['co_lon'],  self.nll_par['co_ele'])

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
    
    def get_travel_time(self, event, tt_nll):
        # travel time for single event
        # event (list): event_name, tor, lat, lon, depth, mag, strike, dip, rake
        self.ev_lat = event[2]  #event latitude
        self.ev_lon = event[3]  #event longitude
        self.ev_depth = event[4]  #event depth in km (positive DOWN) 
        travel_times=[]
        for i in range (len(self.fiber_geometry)):
            ch_name=self.fiber_geometry[i][0]
            travel_times.append( self._compute_travel_time(ch_name, tt_nll) )
        travel_times=np.array(travel_times, dtype=np.float32)
        return travel_times

    def _compute_travel_time(self, ch_name, tt_nll):
        tt_cube= np.reshape(tt_nll[ch_name],(self.nll_par['nx'], self.nll_par['ny'], self.nll_par['nz']))
        ev_index=self.__compute_event_coord_index()
        tt=tt_cube[ev_index[0], ev_index[1], ev_index[2]]
        return tt
    
    def __compute_event_coord_index(self):
        source_x,source_y,source_z = self.coord.geo2cart(self.ev_lat, self.ev_lon, self.ev_depth) 
        source_x,source_y,source_z = source_x * 1e-3, source_y * 1e-3, source_z * 1e-3  # convert from m to km
        # Compute source indices in the grid
        source_x_index = int( round( (source_x - self.nll_par['co_x']) / self.nll_par['dx'] ) ) 
        source_y_index = int( round( (source_y - self.nll_par['co_y']) / self.nll_par['dy'] ) )
        source_z_index = int( round( (source_z - self.nll_par['co_z']) / self.nll_par['dz'] ) )
        return source_x_index, source_y_index, source_z_index
    
if __name__ == "__main__":

    ######### NLL grid parameters 
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
    #########

    ######### FIBER GEOMETRY
    fiber_geometry_dir='../FIBER_GEOMETRY'
    fiber_geometry_file=os.path.join(fiber_geometry_dir, 'flegrei_stations_geometry.txt')      ### CHANGE ###
    #########
    
    ######### NLL traveltime matrices
    db_path = '../NLL/FLEGREI/nll_grid'                                                    ### CHANGE ###
    hdr_filename = 'header.hdr'
    precision='single'
    model = 'time'
    tt_nll_obj = Traveltimes(db_path, hdr_filename)
    tt_nll_p = tt_nll_obj.load_traveltimes('P', model, precision)
    tt_nll_s = tt_nll_obj.load_traveltimes('S', model, precision)
    #########

    ######### EVENTS
    event_dir='../CAT'
    filename_events='catalogue_flegrei_synth_1_ev.txt'                                ### CHANGE ###
    events_file=os.path.join(event_dir,filename_events)
    #########

    ########################################################################
    ###################### TRAVEL TIME EVENT ###############################
    ########################################################################
    tt_class=TravelTimeEvent(NLL_grid_parameters= NLL_grid_parameters
                            ,fiber_geometry_path=fiber_geometry_file)
    
    events=tt_class._load_events(events_file)
    event=events[0]  # first event

    tt_p=tt_class.get_travel_time(event, tt_nll_p)
    tt_s=tt_class.get_travel_time(event, tt_nll_s)
    ########################################################################
    ########################################################################

    ######### PLOT TRAVEL TIMES
    plt.figure(figsize=(10,5))
    ch_names=[item[0] for item in tt_class.fiber_geometry]
    plt.plot(ch_names, tt_p, label='P-wave Travel Time', marker='o')
    plt.plot(ch_names, tt_s, label='S-wave Travel Time', marker='o')
    plt.xlabel('Channel Name')
    plt.ylabel('Travel Time (s)')
    plt.title('Travel Times for Event: {}'.format(event[0]))
    plt.xticks(rotation=90)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    #########
