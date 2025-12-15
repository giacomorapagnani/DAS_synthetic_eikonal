import os
import sys
from obspy.core import UTCDateTime
import numpy as num
from pyrocko import model
import pyrocko.moment_tensor as pmt

from fibonacci_fm_sampler_class import FocalMechanismSampler

class Synthetic_catalogue:

    def __init__(self, data_dir, inputs, input_type='file'):
        ''' inputs can be a python dictionay, an ascii or a pkl file '''
        self.data_dir=data_dir
        if input_type=='file':
            self._read_inputfile(data_dir, inputs)
        elif input_type=='dict':
            self.inputs=inputs
        else:
            print('inputs not passed!!!')
            self.inputs=None

    def _read_inputfile(self, input_file):
        ''' inputs is a file text with the following keys n_sources,latmin, latmax
        lonmin, lonmax, depmin, depmax, tormin, tormax, magmin, magmax'''
        if input_file[-3:]=='pkl':
            import pickle
            with open(self.data_dir+'/'+input_file,'rb') as f:
                inputs = pickle.load(f)
        else:
            with open(self.data_dir+'/'+input_file,'r') as f:
                f.readline() #skip the first line
                inputs={}
                for line in f:
                    toks=line.split()
                    inputs[toks[0]]=eval(toks[1])
        self.inputs=inputs

    def _write_catalogue(self, catfile, catname ):
        with open(self.data_dir+'/'+catfile,'w') as f:
            f.write('EventName OriginTime Latitude(deg) Longitude(deg) Depth(km) Magnitude Strike Dip Rake \n')
            for event_id in sorted(self.events.keys()):
                tor, lat, lon, dep, mag, strike, dip, rake=self.events[event_id]
                time_s=UTCDateTime(tor)
                evline=' %s %7.4f %7.4f %4.2f %3.2f %5.2f %5.2f %5.2f\n' %(str(time_s),lat,lon,dep/1000,mag,strike,dip,rake)
                f.write(catname+event_id+evline)
        events=[]
        for event_id in sorted(self.events.keys()):
            tor, lat, lon, dep, mag, strike, dip, rake=self.events[event_id]
            time_s=UTCDateTime(tor)
            ev=model.Event(name=catname+event_id, time=time_s,lat=lat, lon=lon,depth=dep, magnitude=mag)
            mt = pmt.MomentTensor(strike=strike, dip=dip, rake=rake, magnitude=mag)
            ev.moment_tensor = mt
            events.append(ev)
        model.dump_events(events, self.data_dir+'/'+catfile.replace('.txt','.pf'))

    def __gen_events(self):
        id, tor = self.__gen_evtime()
        lat, lon, dep =self.__gen_evcoords()
        mag=self.__gen_evmag()
        strike, dip, rake= self.fm_sampler.next()
        return id, tor, lat, lon, dep, mag, strike, dip, rake

    def __gen_evtime(self):
        tormin=UTCDateTime(self.inputs['tormin'])
        tormax=UTCDateTime(self.inputs['tormax'])
        id = (tormin + (tormax-tormin)*num.random.rand())
        tor=id.timestamp
        return id, tor

    def __gen_evcoords(self):
        lat = self.inputs['latmin'] + (self.inputs['latmax']-self.inputs['latmin'])*num.random.rand()
        lon = self.inputs['lonmin'] + (self.inputs['lonmax']-self.inputs['lonmin'])*num.random.rand()
        dep = self.inputs['depmin'] + (self.inputs['depmax']-self.inputs['depmin'])*num.random.rand()
        return lat, lon, dep
    
    def __gen_evmag(self):
        mag = self.inputs['magmin'] + (self.inputs['magmax']-self.inputs['magmin'])*num.random.rand()
        return mag
    
    def gen_catalogue(self, catfile, catname, return_object='False', seed=None):
        n_sources=self.inputs['n_sources']
        events={}
        self.fm_sampler=FocalMechanismSampler(n_sources)
        for i in range(n_sources):
            id, tor, lat, lon, dep, mag, strike, dip, rake=self.__gen_events()
            event_id = ((str(id).split(".")[0]).replace(':','')).replace('-','')
            events[event_id]=[tor, lat, lon, dep, mag, strike, dip, rake]
        self.events=events
        self._write_catalogue(catfile,catname)
        if seed is not None:
            num.random.seed(seed)
        if return_object:
            return events

if __name__ == "__main__":
    ### Catalogue Parameters (CHANGE) ###
    nsources=1
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
    cat_dir='../CAT'
    cat_name='flegrei_synth_'   ### CHANGE CATALOGUE NAME ###
    cat_file=f'catalogue_{cat_name}{str(nsources)}_ev.txt'

    ### Generate Catalogue
    dataset=Synthetic_catalogue(cat_dir, inputs, input_type='dict')
    dataset.gen_catalogue(cat_file, cat_name, seed=11)

