import numpy as num
import matplotlib.pyplot as plt

#%% FUNCTIONS
def radiation_pattern(phif,delta,lamb,ih,phase):
    phif= num.deg2rad(phif)
    delta= num.deg2rad(delta)
    lamb= num.deg2rad(lamb)

    if phase=='P':
        rp= (num.cos(lamb)*num.sin(delta)*num.sin(-2*phif)-num.sin(lamb)*num.sin(2*delta)*(num.sin(-phif))**2) *(num.sin(ih))**2 + \
        ( num.sin(lamb)*num.cos(2*delta)*num.sin(-phif) - num.cos(lamb)* num.cos(delta) * num.cos(-phif) )* num.sin(2*ih) + \
        num.sin(lamb)*num.sin(2*delta)* (num.cos(ih))**2

        return rp
    elif phase=='S':
        rsv= ( num.sin(lamb)*num.cos(2*delta)*num.sin(-phif) - num.cos(lamb)*num.cos(delta)*num.cos(-phif) )*num.cos(2*ih) + \
        0.5 * num.cos(lamb)*num.sin(delta)*num.sin(-2*phif)*num.sin(2*ih) - \
        0.5 * num.sin(lamb)*num.sin(2* delta)*num.sin(2*ih)*( 1 + (num.sin(-phif))**2 )

        rsh= ( num.cos(lamb)*num.cos(delta)*num.sin(-phif) + num.sin(lamb)*num.cos(2*delta)*num.cos(-phif) )*num.cos(ih) + \
        ( num.cos(lamb)*num.sin(delta)*num.cos(-2*phif) - 0.5*num.sin(lamb)*num.sin(2*delta)*num.sin(-2*phif) ) *num.sin(ih)
        rs=rsv+rsh
        return rs
    else:
        print('phase must be P os S')
        return None

def ricker(f0,dt,nsamples):
	if (nsamples%2):
		t=((nsamples-1)*dt)/2.
	else:
		nsamples +=1
		t=((nsamples-1)*dt)/2.
	x=f0*num.linspace(-t,t,nsamples)
	ric=num.array(x.size)
	ric=((1.-(2.*num.pi*(x**2.)))*num.exp(-num.pi*(x**2.)))
	tax=x/f0
	return ric, tax

#%% CODE
total_shots=5                                                               #CHANGE
shot_number=1                                                               #CHANGE
count=1

for source_z in range(1,total_shots+1):                                     
    for source_x in range(1,total_shots+1):

        #source position
        sz= source_z * 500.0 #borders != xoff                                   
        sx= source_x * 500.0 #borders != zoff                                   
         
        #import travel time matrix and velocity model
        file_tp='./shots_tt/5m_spacing/Vp/5_tt_mat_forge_' + str(int(sz)) + '_' + str(int(sx)) + '.npy'             
        file_ts='./shots_tt/5m_spacing/Vs/VS_source_tt_mat_forge_'+ str(int(sz)) + '_' + str(int(sx)) + '.npy'           
        file_vp='./forge_vel_mod/forge_vp_50.npy'
        file_vs='./forge_vel_mod/forge_vs_50.npy'
        file_dz='./forge_vel_mod/forge_depth_50.npy'

        spacing=5           #for Forge dataset
        das_length=1200     #for Forge dataset
        das_xposition=0                              #CHANGE

        ttp=num.load(file_tp)
        tts=num.load(file_ts)
        vp=num.load(file_vp)
        vs=num.load(file_vs)
        dz=num.load(file_dz)

        n,m=num.shape(ttp)
        xx=num.arange(n)*spacing
        yy=num.arange(m)*spacing

        izf=das_length//spacing         #depth of fiber
        ixf=das_xposition//spacing      #position of fiber

        # P,S travel times along fiber
        tp=ttp[0:izf,ixf]  # fiber geometry
        ts=tts[0:izf,ixf] # fiber geometry
        r=xx[0:izf]

        #time axis creation
        dt=0.001 #(arbitrary)
        tmax= num.max(ts)
        ns=int(round(tmax/dt + tmax/dt/5))
        tax=num.arange(ns)*dt
        
        dataP=num.zeros((num.size(r),num.size(tax)))
        dataS=num.zeros((num.size(r),num.size(tax)))

        #################### der ricker wavelet
        nsamples=200
        wavelet,taxw=ricker(75,dt,nsamples)
        wavelet[1:]= (wavelet[1:]-wavelet[0:-1])/dt
        wavelet[0]=0
        w_max=num.max(num.abs(wavelet))
        wavelet=wavelet/w_max
        #####################
        
        ################### exp wavelet
        #taxw=num.arange(200)*dt
        #wavelet=num.sin(taxw*num.pi*2*40)*num.exp(-taxw*30)
        #nsamples=num.size(taxw)-1
        #w_max=num.max(num.abs(wavelet))
        #wavelet=wavelet/w_max
        #####################
        
        nsamp=nsamples//2

        #Synthetic seismogram (P and S)
        phi_f_strike=12         # 0 Davide      [12] ->12
        delta_dip=40            #90             [40] -> 45
        lambda_rake=20          #0              [20] -> 0

        for i in range(num.size(r)):
            itp=int(tp[i]//dt)
            its=int(ts[i]//dt)
            incidance_h=num.arctan2(sx-das_xposition,sz-r[i])
            dataP[i,itp]= 1 * (num.cos(incidance_h))**2 * radiation_pattern(phi_f_strike,delta_dip,lambda_rake,incidance_h,'P') 
            dataS[i,its]= 1 * num.sin(2*incidance_h)    * radiation_pattern(phi_f_strike,delta_dip,lambda_rake,incidance_h,'S')
            seisP=num.convolve(wavelet,dataP[i,:])
            seisS=num.convolve(wavelet,dataS[i,:])
            nseis=num.size(seisP)
            dataP[i,:]=seisP[nsamp:-nsamp]
            dataS[i,:]=seisS[nsamp:-nsamp]

        data=dataP+dataS

        noise = num.random.normal(0,1/30,(data.shape))
        data= data+ noise

        #%% PLOT AND SAVE

        #num.save('synthetics/dir_and_rad/seismogram_' + str(int(sz)) + '_' + str(int(sx)) + '_rp',data)

        print('number of synthetics:',count)
        '''
        filename='synthetic_seismogram_'+ str(int(sz)) + '_' + str(int(sx)) + '.svg'
        plt.figure('synthetic_seismogram_'+ str(int(sz)) + '_' + str(int(sx)))
        plt.title('Synthetic Seismogram, Source at ' + str(int(sz)) + 'm Depth and ' + str(int(sx)) + 'm Distance')
        plt.imshow(data, aspect='auto', extent=[tax[0],tax[-1],r[-1],r[0]], cmap='seismic', vmin=-1, vmax=1)
        plt.xlabel('Time [s]')
        plt.ylabel('Depth [m]')
        plt.colorbar()
        #plt.savefig('synthetics/dir_and_rad/'+ filename, format="svg")
        #plt.show()
        plt.close('all')
        '''
        '''
        fig = plt.figure('subplot')
        plt.subplot(5, 5, count)
        plt.imshow(data, aspect='auto', extent=[tax[0],tax[-1],r[-1],r[0]], cmap='seismic', vmin=-1, vmax=1)
        '''
        
        if count==25:
             break
        if count<=6:
            fig = plt.figure('subplot1')
            plt.subplot(3, 2, count)
            plt.title('Z = ' + str(int(sz))+ 'm, X = ' + str(int(sx)) + 'm')
            plt.imshow(data, aspect='auto', extent=[tax[0],tax[-1],r[-1],r[0]], cmap='seismic', vmin=-1, vmax=1)
        if count>=7 and count<=12:
            fig = plt.figure('subplot2')
            plt.subplot(3, 2, count-6)
            plt.title('Z = ' + str(int(sz))+ 'm, X = ' + str(int(sx)) + 'm')
            plt.imshow(data, aspect='auto', extent=[tax[0],tax[-1],r[-1],r[0]], cmap='seismic', vmin=-1, vmax=1)
        if count>=13 and count<=18:
            fig = plt.figure('subplot3')
            plt.subplot(3, 2, count-12)
            plt.title('Z = ' + str(int(sz))+ 'm, X = ' + str(int(sx)) + 'm')
            plt.imshow(data, aspect='auto', extent=[tax[0],tax[-1],r[-1],r[0]], cmap='seismic', vmin=-1, vmax=1)
        if count>=19 and count<=24:
            fig = plt.figure('subplot4')
            plt.subplot(3, 2, count-18)
            plt.title('Z = ' + str(int(sz))+ 'm, X = ' + str(int(sx)) + 'm')
            plt.imshow(data, aspect='auto', extent=[tax[0],tax[-1],r[-1],r[0]], cmap='seismic', vmin=-1, vmax=1)    
              
        count+=1
plt.show()
           

