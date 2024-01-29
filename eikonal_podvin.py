#layer vel travel time. Borders stop included (method 2 working).
#debugging in progress
# %% functions
import numpy as np
import matplotlib.pyplot as plt
import time

start = time.time()


def tt_grid(xmin,xmax,zmin,zmax,spacing):
    '''
    output: inf values in tt_grid. dim(tt_grid) = n x m'''
    x_sample= index_pos((xmax-xmin),spacing) +1 # +1 to include zero
    xax= xmin + np.arange(x_sample)*spacing
    z_sample= index_pos((zmax-zmin),spacing) +1 #+1 to include zero
    zax= zmin + np.arange(z_sample)*spacing
    tt_grid=np.ones((z_sample,x_sample))*np.inf
    return tt_grid, xax, zax

def vel_grid_1D(x_sample,z_sample,vel_mod_1D,layer_depth,spacing):
    '''
    output: Layered vel_grid. dim(vel_grid)= n-1 x m-1 '''
    x=x_sample-1 ; z=z_sample-1
    vel_grid=np.zeros((z,x))
    layer_depth_ind=index_pos(layer_depth,spacing)
    for i in range(layer_depth.size-2):
        vel_grid[layer_depth_ind[i]:layer_depth_ind[i+1],:]=vel_mod_1D[i]
    vel_grid[layer_depth_ind[-2]:,:]=vel_mod_1D[-1] #last layer
    return vel_grid

def vel_grid_homo(x_sample,z_sample,velocity):
    '''
    output: homo vel_grid. dim(vel_grid)= n-1 x m-1 '''
    x=x_sample-1 ; z=z_sample-1
    vel_grid=np.ones((z,x))*velocity
    return vel_grid

def grid_border(grid):
    '''
    output: Adds inf values at borders. dim(grid_border) = ?+2 x ?+2'''
    grid_border=np.ones((grid.shape[0]+2,grid.shape[1]+2))*np.inf #inf at borders
    grid_border[1:-1,1:-1]=grid
    return grid_border

def head_waves(tt_grid,sl_grid,h):
    t=np.ones(4)*np.inf
    t[0]=tt_grid[0,1] + h* np.min((sl_grid[0,0],sl_grid[0,1]))
    t[1]=tt_grid[1,2] + h* np.min((sl_grid[0,1],sl_grid[1,1]))
    t[2]=tt_grid[2,1] + h* np.min((sl_grid[1,1],sl_grid[1,0]))
    t[3]=tt_grid[1,0] + h* np.min((sl_grid[1,0],sl_grid[0,0]))
    return t

def diffracted_waves(tt_grid,sl_grid,h):
    t=np.ones(4)*np.inf
    t[0]=tt_grid[0,0] + np.sqrt(2)*h* sl_grid[0,0]
    t[1]=tt_grid[0,2] + np.sqrt(2)*h* sl_grid[0,1]
    t[2]=tt_grid[2,2] + np.sqrt(2)*h* sl_grid[1,1]
    t[3]=tt_grid[2,0] + np.sqrt(2)*h* sl_grid[1,0]
    return t

def TTP(tn,tm,s,h):
    if (tn or tm or s)==np.inf:
        tp=np.inf
        return tp
    dtnm=tn-tm
    hsc=(h*s)/np.sqrt(2)
    if ((dtnm>=0) and (dtnm<=hsc)):
        tp= tn + np.sqrt( ( h*s )**2 - ( tn-tm )**2 )
        return tp
    tp=np.inf
    return tp

def transmitted_waves(tt_grid,sl_grid,h):    
    t=np.ones(8)*np.inf
    t[0]=TTP(tt_grid[0,1],tt_grid[0,0],sl_grid[0,0],h)
    t[1]=TTP(tt_grid[0,1],tt_grid[0,2],sl_grid[0,1],h)

    t[2]=TTP(tt_grid[1,2],tt_grid[0,2],sl_grid[0,1],h)
    t[3]=TTP(tt_grid[1,2],tt_grid[2,2],sl_grid[1,1],h)

    t[4]=TTP(tt_grid[2,1],tt_grid[2,2],sl_grid[1,1],h)
    t[5]=TTP(tt_grid[2,1],tt_grid[2,0],sl_grid[1,0],h)

    t[6]=TTP(tt_grid[1,0],tt_grid[2,0],sl_grid[1,0],h)
    t[7]=TTP(tt_grid[1,0],tt_grid[0,0],sl_grid[0,0],h)
    return t

def all_waves(tt_grid,sl_grid,h):
    '''
    input:
    tt_grid: 3x3 matrix
    sl_grid: 2x2 matrix
    h: spacing

    output:
    t: all travel times for central point'''
    t=np.ones(16)*np.inf
    t[0:4]=head_waves(tt_grid,sl_grid,h)
    t[4:8]=diffracted_waves(tt_grid,sl_grid,h)
    t[8:] =transmitted_waves(tt_grid,sl_grid,h)
    return t

def index_pos(x,dx): #!only positive values!
    ind= ( np.round( x/dx ) ).astype(int)
    return ind

def index_mini_grid(tt_grid,sl_grid,z,x):
    '''
    input: index of point to compute tt

    output:
    tt_grid: 3x3 matrix
    sl_grid: 2x2 matrix '''

    return tt_grid[z-1:z+2,x-1:x+2],sl_grid[z-1:z+1,x-1:x+1]

def index_ring_square(r,szi,sxi):
    '''
    input: source index and radious
    
    output: indices of the square of radious r '''
    U_indz=(np.ones(2*r+1)*(szi -r)).astype(int)
    R_indz=np.arange(szi -r , szi +r+1).astype(int)
    D_indz=(np.ones(2*r+1)*(szi + r)).astype(int)
    L_indz=np.arange(szi+ r , szi -r-1 ,-1).astype(int)

    U_indx=np.arange(sxi -r , sxi +r+1)
    R_indx=(np.ones(2*r+1)*(sxi + r)).astype(int)
    D_indx=np.arange(sxi+ r , sxi -r-1,-1)
    L_indx=(np.ones(2*r+1)*(sxi - r)).astype(int)

    return U_indz,R_indz,D_indz,L_indz,U_indx,R_indx,D_indx,L_indx

def index_sort(Z,X,grid,sort):
    if sort=='Z':
        dim_grid=grid.shape[0]
        ind_min=np.min(Z)
        nzeros= len(np.argwhere(Z==0))
        nmax=   len(np.argwhere(Z==dim_grid-2))
        Z=np.delete(Z, np.argwhere(Z==0))
        Z=np.delete(Z, np.argwhere(Z==dim_grid-2))
        if ind_min==0:
            Z = 1 + np.argsort(grid[Z,X[nzeros+nmax:]])
        else:
            Z = ind_min + np.argsort(grid[Z,X[nzeros+nmax:]])
        Z = np.append(Z, np.zeros(nzeros) )
        Z = np.append(Z, np.ones(nmax) * (dim_grid-2) )
    elif sort=='X':
        dim_grid=grid.shape[1]
        ind_min=np.min(X)
        nzeros= len(np.argwhere(X==0))
        nmax=   len(np.argwhere(X==dim_grid-2))
        X=np.delete(X, np.argwhere(X==0))
        X=np.delete(X, np.argwhere(X==dim_grid-2))
        if ind_min==0:
            X= 1 + np.argsort(grid[Z[nzeros+nmax:],X])
        else:
            X = ind_min + np.argsort(grid[Z[nzeros+nmax:],X])
        X = np.append(X, np.zeros(nzeros) )
        X = np.append(X, np.ones(nmax) * (dim_grid-2) )
    Z=Z.astype(int)
    X=X.astype(int)
    return Z,X

def index_sort_update(tt_grid,Z,X,dir,rad,limit):
    '''
    output: sort arrays from min to max, add 2 values each time'''
    if (dir=='D' or dir=='R'):
        limit=limit-1 # -1 because the 'D' or 'R' border go out of bounds 1 element before the edge
    if rad>=(limit): 
        Z=np.zeros( ( 2*( rad+1 ) +1 ),int )
        X=np.zeros( ( 2*( rad+1 ) +1 ),int )
        return Z,X

    if dir=='U':
        # sort X
        _,X=index_sort(Z,X,tt_grid,'X')

        ind_min=np.min(X)
        if ind_min!=0:
            X= np.insert(X,np.argwhere(X==ind_min)[0]+1,ind_min-1)
        else:
            X= np.insert(X,np.argwhere(X==0)[0]+1,0)
        
        ind_max=np.max(X)
        dim_grid=tt_grid.shape[1]
        if ind_max!=(dim_grid-2):
            X= np.insert(X,np.argwhere(X==ind_max)[0]+1,ind_max+1)
        else:
            X= np.insert(X,np.argwhere(X==dim_grid-2)[0]+1,dim_grid-2 )

        Z=np.append(Z,[Z[0],Z[0]])-1 #Z-1


    elif dir=='R':
        #sort Z
        Z,_=index_sort(Z,X,tt_grid,'Z')

        ind_min=np.min(Z)
        if ind_min!=0:
            Z= np.insert(Z,np.argwhere(Z==ind_min)[0]+1,ind_min-1)
        else:
            Z= np.insert(Z,np.argwhere(Z==0)[0]+1,0)
        
        ind_max=np.max(Z)
        dim_grid=tt_grid.shape[0]
        if ind_max!=(dim_grid-2):
            Z= np.insert(Z,np.argwhere(Z==ind_max)[0]+1,ind_max+1)
        else:
            Z= np.insert(Z,np.argwhere(Z==dim_grid-2)[0]+1,dim_grid-2 )

        X=np.append(X,[X[0],X[0]])+1 #X+1

        #print('R_Z:',Z)
        #print('R_X:',X)
        #print('num or zeros:',len(np.argwhere(Z==0)))
        #print('...')

    elif dir=='D':
        # sort X
        _,X=index_sort(Z,X,tt_grid,'X')

        ind_min=np.min(X)
        if ind_min!=0:
            X= np.insert(X,np.argwhere(X==ind_min)[0]+1,ind_min-1)
        else:
            X= np.insert(X,np.argwhere(X==0)[0]+1,0)
        
        ind_max=np.max(X)
        dim_grid=tt_grid.shape[1]
        if ind_max!=(dim_grid-2):
            X= np.insert(X,np.argwhere(X==ind_max)[0]+1,ind_max+1)
        else:
            X= np.insert(X,np.argwhere(X==dim_grid-2)[0]+1,dim_grid-2 )

        Z=np.append(Z,[Z[0],Z[0]])+1 #Z+1

        #print('D_Z:',Z)
        #print('D_X:',X)
        #print('num or zeros:',len(np.argwhere(X==0)))
        #print('...')

    elif dir=='L':
        #sort Z
        Z,_=index_sort(Z,X,tt_grid,'Z')

        ind_min=np.min(Z)
        if ind_min!=0:
            Z= np.insert(Z,np.argwhere(Z==ind_min)[0]+1,ind_min-1)
        else:
            Z= np.insert(Z,np.argwhere(Z==0)[0]+1,0)
        
        ind_max=np.max(Z)
        dim_grid=tt_grid.shape[0]
        if ind_max!=(dim_grid-2):
            Z= np.insert(Z,np.argwhere(Z==ind_max)[0]+1,ind_max+1)
        else:
            Z= np.insert(Z,np.argwhere(Z==dim_grid-2)[0]+1,dim_grid-2 )

        X=np.append(X,[X[0],X[0]])-1 #X-1

    X=X.astype(int)
    Z=Z.astype(int)  
    return Z,X

def travel_time(tt_grid,sl_grid,z,x,h):
    if ( z==0 or x==0 or z==(tt_grid.shape[0]-2) or x==(tt_grid.shape[1]-2) ):
        return
    mini_tt,mini_sl= index_mini_grid(tt_grid,sl_grid,z,x)
    tt=all_waves(mini_tt,mini_sl,h)
    tt_grid[z,x]=np.min(tt)
    return

###########################################################################
# %% parameters

########################################################################## FORGE VEL MODEL
#layer=np.load('forge_vel_mod/forge_depth_50.npy')
#vp=np.load('forge_vel_mod/forge_vp_50.npy')[:-1]
#vs=np.load('forge_vel_mod/forge_vs_50.npy')[:-1]

#spacing=10 #grid spacing 10m  

#x_min=0.
#x_max=3000.
#
#tt_mat,xoff,zoff= tt_grid(x_min,x_max,layer[0],layer[-1],spacing)
#
#tt_mat_bord=grid_border(tt_mat)
#
#vel_mat= vel_grid_1D(xoff.size,zoff.size,vp,layer,spacing)

#########################################################################

x_min=0.
x_max=3000.
z_min=0.
z_max=3000.
spacing=10.
tt_mat,xoff,zoff= tt_grid(x_min,x_max,z_min,z_max,spacing)
tt_mat_bord=grid_border(tt_mat)
############1D VEL
vel=np.array    ((1500.,3000.,10000.)) #dim(vel) != dim(depth)-1
depth=np.array  ((0.,300.,700.,3000.)) #max depth != zmax
vel_mat= vel_grid_1D(xoff.size,zoff.size,vel,depth,spacing)
############

############constant VEL
#vel=np.array    ((2500.))
#vel_mat=vel_grid_homo(xoff.size,zoff.size,vel)
############

############constant VEL + cube
#vel=np.array    ((2000.))
#vel_mat=vel_grid_homo(xoff.size,zoff.size,vel)
#vel_mat[220:240,50:70]=5000.
############

sl_mat=1/vel_mat
sl_mat_bord=grid_border(sl_mat)

#SOURCE POSITION
sx=0. #borders != xoff
sz=0. #borders != zoff

plt.figure('velocity model and source position')
plt.imshow(vel_mat,
           extent=[xoff[0], xoff[-1], zoff[-1] ,zoff[0]  ])
plt.plot(sx,sz,'*r')
plt.colorbar()

###########################################################################
# %% TRAVEL TIME

left_limit=sxi=index_pos(sx,spacing)    #left limit
rigth_limit=xoff.size-sxi               #rigth limit
up_limit=szi=index_pos(sz,spacing)      #up limit
down_limit=zoff.size-szi                #down limit
iterations=np.max((left_limit,rigth_limit,up_limit,down_limit))

tt_mat_bord[szi+1,sxi+1]=0 # +1 due to border

#init condition
U_z,R_z,D_z,L_z,U_x,R_x,D_x,L_x=index_ring_square(1,szi+1,sxi+1) # +1 due to border
for i in range( (2*1) + 1 ):
    travel_time(tt_mat_bord,sl_mat_bord,U_z[i],U_x[i],spacing)
    travel_time(tt_mat_bord,sl_mat_bord,R_z[i],R_x[i],spacing)
    travel_time(tt_mat_bord,sl_mat_bord,D_z[i],D_x[i],spacing)
    travel_time(tt_mat_bord,sl_mat_bord,L_z[i],L_x[i],spacing)
U_z,U_x= index_sort_update(tt_mat_bord,U_z,U_x,'U',1,up_limit)
R_z,R_x= index_sort_update(tt_mat_bord,R_z,R_x,'R',1,rigth_limit)
D_z,D_x= index_sort_update(tt_mat_bord,D_z,D_x,'D',1,down_limit)
L_z,L_x= index_sort_update(tt_mat_bord,L_z,L_x,'L',1,left_limit)

#propagation
for r in range(2,iterations+1): # +1 to make the last iteration on the farthest border
    for i in range( (2*r) + 1):
        travel_time(tt_mat_bord,sl_mat_bord,U_z[i],U_x[i],spacing)
        travel_time(tt_mat_bord,sl_mat_bord,R_z[i],R_x[i],spacing)
        travel_time(tt_mat_bord,sl_mat_bord,D_z[i],D_x[i],spacing)
        travel_time(tt_mat_bord,sl_mat_bord,L_z[i],L_x[i],spacing)
    U_z,U_x= index_sort_update(tt_mat_bord,U_z,U_x,'U',r,up_limit)
    R_z,R_x= index_sort_update(tt_mat_bord,R_z,R_x,'R',r,rigth_limit)
    D_z,D_x= index_sort_update(tt_mat_bord,D_z,D_x,'D',r,down_limit)
    L_z,L_x= index_sort_update(tt_mat_bord,L_z,L_x,'L',r,left_limit)

tt_mat=tt_mat_bord[1:-1,1:-1]

#wavefront visualization (window=0.2s)
wave_num= np.max(tt_mat[:-1,:-1])/20
wave_interval= np.max(tt_mat[:-1,:-1])/100
wave_front=  (tt_mat>=0) * (tt_mat<=wave_interval)
for i in range(1,20):
    wave_front+= (tt_mat>=wave_num*i) * (tt_mat<=wave_num*i + wave_interval)

# %% PLOTS

end = time.time()
print('execution time:',end - start,'s')

#np.save('tt_mat',tt_mat)

plt.figure('travel time grid')
plt.imshow(tt_mat,
            extent=[xoff[0], xoff[-1], zoff[-1] ,zoff[0]  ])
plt.plot(sx,sz,'*r')
plt.colorbar()
'''
plt.figure('travel time grid border')
plt.imshow(tt_mat_bord)
plt.colorbar()

plt.figure('grid color')
plt.imshow(grid_color,
            extent=[xoff[0], xoff[-1], zoff[-1] ,zoff[0]  ])
plt.plot(sx,sz,'*r')
plt.colorbar()

plt.figure('travel time grid ind')
plt.imshow(tt_mat)
plt.plot(sxi,szi,'*r')
plt.colorbar()

plt.figure('vertical travel time arrival')
plt.plot(tt_mat[:,0],zoff,'.-')
plt.gca().invert_yaxis()

plt.figure('surface travel time arrival')
plt.plot(xoff,tt_mat[0,:],'.-')
plt.gca().invert_yaxis()
'''
plt.figure('wave front')
plt.imshow(wave_front,
            extent=[xoff[0], xoff[-1], zoff[-1] ,zoff[0]  ])
plt.colorbar()


plt.show()
