import numpy as np
import matplotlib.pyplot as plt

class eikonal_travel_times:
    def __init__(self):
        
        print('Welcome to the Eikonal Travel Time Class!')
        print('Create or Load the Travel-Time-Matrix and the Velocity-Matrix: \n')
    ''' # load non implemented
    def load_tt_matrix(self,filename,zmin,xmin,spacing):
        self.tt_mat=np.load(filename)
        self.h= spacing
        self.z_sample= self.tt_mat.shape[0]
        self.x_sample= self.tt_mat.shape[1]
        self.zmin= zmin
        self.zmax= self.zmin + (self.z_sample -1) * self.h
        self.xmin= xmin
        self.xmax= self.xmin + (self.x_sample -1) * self.h
        print('Travel Time Matrix Loaded.')

    def load_vel_matrix(self,filename):
        self.vel_mat=np.load(filename)
        print('Velocity Matrix Loaded.')
    '''
    def index_pos(self,x,dx):
        ind= ( np.round( x/dx ) ).astype(int)
        return ind
    
    def create_tt_matrix(self,xmin,xmax,zmin,zmax,spacing):
        self.xmin=xmin
        self.xmax=xmax
        self.zmin=zmin
        self.zmax=zmax
        self.h=spacing
        self.x_sample= self.index_pos((self.xmax-self.xmin),self.h) +1 # +1 to include zero
        self.xoff= self.xmin + np.arange(self.x_sample)*self.h
        self.z_sample= self.index_pos((self.zmax-self.zmin),self.h) +1 #+1 to include zero
        self.zoff= self.zmin + np.arange(self.z_sample)*self.h
        self.tt_mat=np.ones((self.z_sample,self.x_sample))*np.inf

        self.create_grid_border('tt')

        print('Travel Time Matrix Created.')

    def create_homogeneous_vel_matrix(self,velocity):
        x=self.x_sample-1 ; z=self.z_sample-1
        self.vel_mat=np.ones((z,x))*velocity
        print('Velocity Matrix Created.')
        self.create_slowness_matrix()
        self.create_grid_border('sl')

    def create_1D_vel_matrix(self,vel_mod_1D,layer_depth):
        x=self.x_sample-1 ; z=self.z_sample-1
        vel_grid=np.zeros((z,x))
        layer_depth_ind=self.index_pos(layer_depth,self.h)
        for i in range(layer_depth.size-2):
            vel_grid[layer_depth_ind[i]:layer_depth_ind[i+1],:]=vel_mod_1D[i]
        vel_grid[layer_depth_ind[-2]:,:]=vel_mod_1D[-1] #last layer
        self.vel_mat=vel_grid
        print('Velocity Matrix Created.')
        self.create_slowness_matrix()
        self.create_grid_border('sl')

    def create_slowness_matrix(self):
        self.sl_mat= 1 / self.vel_mat
        print('Slowness Matrix Created.')

    def create_grid_border(self,tt_or_sl):
        if tt_or_sl=='tt':
            grid_border=np.ones((self.tt_mat.shape[0]+2,self.tt_mat.shape[1]+2))*np.inf #inf at borders
            grid_border[1:-1,1:-1]=self.tt_mat
            self.tt_mat_bord=grid_border.copy()
        elif tt_or_sl=='sl':
            grid_border=np.ones((self.sl_mat.shape[0]+2,self.sl_mat.shape[1]+2))*np.inf #inf at borders
            grid_border[1:-1,1:-1]=self.sl_mat
            self.sl_mat_bord=grid_border.copy()

    def insert_source(self,sz,sx):
        self.up_limit     = self.index_pos(sz,self.h)
        self.szi = self.index_pos(sz,self.h)
        self.down_limit  = self.z_sample - self.szi
        self.left_limit  = self.index_pos(sx,self.h)
        self.sxi = self.index_pos(sx,self.h)
        self.rigth_limit = self.x_sample - self.sxi
        self.iterations=np.max(( self.up_limit,self.down_limit,self.left_limit,self.rigth_limit ))

        self.tt_mat_bord[self.szi+1,self.sxi+1]=0 # +1 due to border
        print('Source Positioned.')
    
    def index_mini_grid(self,z,x):
        mini_z=self.tt_mat_bord[z-1:z+2,x-1:x+2].copy()
        mini_x=self.tt_mat_bord[z-1:z+1,x-1:x+1].copy()
        return mini_z,mini_x
    
    def index_ring_square(self,r):
        sxi=self.sxi + 1 # +1 due to border
        szi=self.szi + 1 # +1 due to border
        self.U_z=(np.ones(2*r+1)*(szi -r)).astype(int)
        self.R_z=np.arange(szi -r , szi +r+1).astype(int)
        self.D_z=(np.ones(2*r+1)*(szi + r)).astype(int)
        self.L_z=np.arange(szi+ r , szi -r-1 ,-1).astype(int)

        self.U_x=np.arange(sxi -r , sxi +r+1)
        self.R_x=(np.ones(2*r+1)*(sxi + r)).astype(int)
        self.D_x=np.arange(sxi+ r , sxi -r-1,-1)
        self.L_x=(np.ones(2*r+1)*(sxi - r)).astype(int)
    
    def index_sort(self,Z,X,sort):
        if sort=='Z':
            dim_grid=self.tt_mat_bord.shape[0]
            ind_min=np.min(Z)
            nzeros= len(np.argwhere(Z==0))
            nmax=   len(np.argwhere(Z==dim_grid-2))
            Z=np.delete(Z, np.argwhere(Z==0))
            Z=np.delete(Z, np.argwhere(Z==dim_grid-2))
            if ind_min==0:
                Z = 1 + np.argsort(self.tt_mat_bord[Z,X[nzeros+nmax:]])
            else:
                Z = ind_min + np.argsort(self.tt_mat_bord[Z,X[nzeros+nmax:]])
            Z = np.append(Z, np.zeros(nzeros) )
            Z = np.append(Z, np.ones(nmax) * (dim_grid-2) )
        elif sort=='X':
            dim_grid=self.tt_mat_bord.shape[1]
            ind_min=np.min(X)
            nzeros= len(np.argwhere(X==0))
            nmax=   len(np.argwhere(X==dim_grid-2))
            X=np.delete(X, np.argwhere(X==0))
            X=np.delete(X, np.argwhere(X==dim_grid-2))
            if ind_min==0:
                X= 1 + np.argsort(self.tt_mat_bord[Z[nzeros+nmax:],X])
            else:
                X = ind_min + np.argsort(self.tt_mat_bord[Z[nzeros+nmax:],X])
            X = np.append(X, np.zeros(nzeros) )
            X = np.append(X, np.ones(nmax) * (dim_grid-2) )
        Z=Z.astype(int)
        X=X.astype(int)
        return Z,X

    def index_update(self,A,update):
        if update=='Z':
            dim_grid=self.tt_mat_bord.shape[0]
        elif update=='X':
            dim_grid=self.tt_mat_bord.shape[1]

        ind_min=np.min(A)
        if ind_min!=0:
            A= np.insert(A,np.argwhere(A==ind_min)[0]+1,ind_min-1)
        else:
            A= np.insert(A,np.argwhere(A==0)[0]+1,0)

        ind_max=np.max(A)
        if ind_max!=(dim_grid-2):
            A= np.insert(A,np.argwhere(A==ind_max)[0]+1,ind_max+1)
        else:
            A= np.insert(A,np.argwhere(A==dim_grid-2)[0]+1,dim_grid-2 )

        return A

    def INDEX_SORT_UPDATE(self,dir,rad):
            # -1 because the 'D' or 'R' border go out of bounds 1 element before the edge
            if dir=='U':
                limit=self.up_limit
                if rad>=(limit):
                    self.U_z= np.zeros( ( 2*( rad+1 ) +1 ),int )
                    self.U_x= np.zeros( ( 2*( rad+1 ) +1 ),int )
                else:
                    _,self.U_x=self.index_sort(self.U_z,self.U_x,'X')
                    self.U_x= self.index_update(self.U_x,'X')
                    
                    self.U_z=np.append( self.U_z, [self.U_z[0],self.U_z[0]] ) - 1 #Z-1

            elif dir=='R':
                limit=self.rigth_limit -1
                if rad>=(limit):
                    self.R_z= np.zeros( ( 2*( rad+1 ) +1 ),int )
                    self.R_x= np.zeros( ( 2*( rad+1 ) +1 ),int )
                else:
                    self.R_z,_=self.index_sort(self.R_z,self.R_x,'Z')
                    self.R_z= self.index_update(self.R_z,'Z')

                    self.R_x=np.append( self.R_x , [self.R_x[0],self.R_x[0]] ) + 1 #X+1  

            elif dir =='D':
                limit=self.down_limit - 1
                if rad>=(limit):
                    self.D_z= np.zeros( ( 2*( rad+1 ) +1 ),int )
                    self.D_x= np.zeros( ( 2*( rad+1 ) +1 ),int )
                else:
                    _,self.D_x=self.index_sort(self.D_z,self.D_x,'X')
                    self.D_x= self.index_update(self.D_x,'X')

                    self.D_z=np.append( self.D_z , [self.D_z[0],self.D_z[0]] ) + 1 #Z+1
           
            elif dir=='L':
                limit=self.left_limit
                if rad>=(limit):
                    self.L_z= np.zeros( ( 2*( rad+1 ) +1 ),int )
                    self.L_x= np.zeros( ( 2*( rad+1 ) +1 ),int )            
                else:
                    self.L_z,_=self.index_sort(self.L_z,self.L_x,'Z')
                    self.L_z= self.index_update(self.L_z,'Z')

                    self.L_x=np.append( self.L_x , [self.L_x[0],self.L_x[0]] ) - 1 #X-1
    
    def travel_time(self,z,x): # z,x: current element position
        if ( z==0 or x==0 or z==(self.tt_mat_bord.shape[0]-2) or x==(self.tt_mat_bord.shape[1]-2) ):
            return  
        else: 
            self.mini_tt,self.mini_sl= self.index_mini_grid(z,x)
            tt=self.all_waves()
            self.tt_mat_bord[z,x]=np.min(tt)
    
    def all_waves(self):
        t=np.ones(16)*np.inf
        t[0:4]=self.head_waves()
        t[4:8]=self.diffracted_waves()
        t[8:] =self.transmitted_waves()
        print(t)
        print('...')
        return t

    def head_waves(self): #input are mini_grid
        th=np.ones(4)*np.inf
        th[0]=self.mini_tt[0,1] + self.h* np.min((self.mini_sl[0,0],self.mini_sl[0,1]))
        th[1]=self.mini_tt[1,2] + self.h* np.min((self.mini_sl[0,1],self.mini_sl[1,1]))
        th[2]=self.mini_tt[2,1] + self.h* np.min((self.mini_sl[1,1],self.mini_sl[1,0]))
        th[3]=self.mini_tt[1,0] + self.h* np.min((self.mini_sl[1,0],self.mini_sl[0,0]))
        return th
    
    def diffracted_waves(self): #input are mini_grid
        td=np.ones(4)*np.inf
        td[0]=self.mini_tt[0,0] + np.sqrt(2) *self.h* self.mini_sl[0,0]
        td[1]=self.mini_tt[0,2] + np.sqrt(2) *self.h* self.mini_sl[0,1]
        td[2]=self.mini_tt[2,2] + np.sqrt(2) *self.h* self.mini_sl[1,1]
        td[3]=self.mini_tt[2,0] + np.sqrt(2) *self.h* self.mini_sl[1,0]
        return td
    
    def transmitted_waves(self): #input are mini_grid  
        tp=np.ones(8)*np.inf
        tp[0]=self.TTP(self.mini_tt[0,1],self.mini_tt[0,0],self.mini_sl[0,0])
        tp[1]=self.TTP(self.mini_tt[0,1],self.mini_tt[0,2],self.mini_sl[0,1])

        tp[2]=self.TTP(self.mini_tt[1,2],self.mini_tt[0,2],self.mini_sl[0,1])
        tp[3]=self.TTP(self.mini_tt[1,2],self.mini_tt[2,2],self.mini_sl[1,1])
        
        tp[4]=self.TTP(self.mini_tt[2,1],self.mini_tt[2,2],self.mini_sl[1,1])
        tp[5]=self.TTP(self.mini_tt[2,1],self.mini_tt[2,0],self.mini_sl[1,0])
        
        tp[6]=self.TTP(self.mini_tt[1,0],self.mini_tt[2,0],self.mini_sl[1,0])
        tp[7]=self.TTP(self.mini_tt[1,0],self.mini_tt[0,0],self.mini_sl[0,0])
        return tp
    
    def TTP(self,tn,tm,s):
        if (tn or tm or s)==np.inf:
            tp=np.inf
            return tp
        dtnm=tn-tm
        hsc=(self.h*s)/np.sqrt(2)
        if ((dtnm>=0) and (dtnm<=hsc)):
            tp= tn + np.sqrt( ( self.h*s )**2 - ( tn-tm )**2 )
            return tp
        else:
            tp=np.inf ##ELSE???
            return tp
    
    def plot(self,matrix,title,xoff=None,zoff=None,sx=None,sz=None):
        if ( (sx==None or sz==None) and (xoff.any()==None or zoff.any()==None) ):
            plt.figure(title)
            plt.imshow(matrix)
            plt.colorbar()
        elif ( (sx==None or sz==None) and (xoff.all()!=None and zoff.all()!=None) ):
            plt.figure(title)
            plt.imshow(matrix,
                extent=[xoff[0], xoff[-1], zoff[-1] ,zoff[0]  ])
            plt.colorbar()
        elif ( (sx!=None and sz!=None) and (xoff.all()!=None and zoff.all()!=None) ):
            plt.figure(title)
            plt.imshow(matrix,
                extent=[xoff[0], xoff[-1], zoff[-1] ,zoff[0]  ])
            plt.plot(sx,sz,'*r')
            plt.colorbar()
        else:
            print('WARNING: Print Condition not Matched.')

    def first_propagation(self):
        r=1
        self.index_ring_square(r)
        for i in range( (2*r) +1 ): #number of elements per side of ring
            self.travel_time(self.U_z[i],self.U_x[i])
            self.travel_time(self.R_z[i],self.R_x[i])
            self.travel_time(self.D_z[i],self.D_x[i])
            self.travel_time(self.L_z[i],self.L_x[i])
        self.INDEX_SORT_UPDATE('U',r)
        self.INDEX_SORT_UPDATE('R',r)
        self.INDEX_SORT_UPDATE('D',r)
        self.INDEX_SORT_UPDATE('L',r)

    def PROPAGATION(self):
        self.first_propagation()
        for r in range(2,self.iterations+1): # +1 to make the last iteration on the farthest border
            for i in range( (2*r) + 1):
                self.travel_time(self.U_z[i],self.U_x[i])
                self.travel_time(self.R_z[i],self.R_x[i])
                self.travel_time(self.D_z[i],self.D_x[i])
                self.travel_time(self.L_z[i],self.L_x[i])
            self.INDEX_SORT_UPDATE('U',r)
            self.INDEX_SORT_UPDATE('R',r)
            self.INDEX_SORT_UPDATE('D',r)
            self.INDEX_SORT_UPDATE('L',r)

        self.tt_mat=self.tt_mat_bord[1:-1,1:-1]
        print('Propagation Completed!')

#########################################################################

x_min=0.
x_max=3000.
z_min=0.
z_max=3000.
spacing=20.

e = eikonal_travel_times()

e.create_tt_matrix(x_min,x_max,z_min,z_max,spacing)

############constant VEL
vel=np.array    ((2500.))
e.create_homogeneous_vel_matrix(vel)
############

############1D VEL
#vel=np.array    ((1500.,3000.,6000.)) #dim(vel) != dim(depth)-1
#depth=np.array  ((0.,300.,700.,3000.)) #max depth != zmax
#e.create_1D_vel_matrix(vel,depth)
############

############constant VEL + cube
#vel=np.array    ((2000.))
#e.create_homogeneous_vel_matrix(vel)
#e.vel_mat[220:240,50:70]=5000.
############

#SOURCE POSITION
sz=600.
sx=600.
e.insert_source(sz,sx)

e.plot(e.vel_mat,'velocity matrix and source position',
       e.xoff,e.zoff,sx,sz)
#plt.show()

e.PROPAGATION()

######################################
# PLOTS
wave_num= np.max(e.tt_mat[:-1,:-1])/20
wave_interval= np.max(e.tt_mat[:-1,:-1])/100
wave_front=  (e.tt_mat>=0) * (e.tt_mat<=wave_interval)
for i in range(1,20):
    wave_front+= (e.tt_mat>=wave_num*i) * (e.tt_mat<=wave_num*i + wave_interval)

e.plot(e.tt_mat,'Travel Time Matrix',
       e.xoff,e.zoff,sx,sz)

e.plot(wave_front,'Wave Front',
       e.xoff,e.zoff)

plt.show()