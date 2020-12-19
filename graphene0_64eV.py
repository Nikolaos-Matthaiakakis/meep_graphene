##Scattering from graphene plasmonic nanoresonator (by Nikolaos Matthaiakakis)

#This is a coordinate transform example for obtaining scattering spectra from a graphene plasmonic resonator with Ef=0.64 and μ=1000cm^2/V.s.

#The resonator is square shaped with a size of 75nm x 65nm and z=1/(resolution*stretch), giving a single pixel thickness after the coordinate transform. Detailed convergence testing hasn't been completed but the results for a thickness of 5nm (resolution=10, stretching=20, alpha=1um) are already very good.

#In order to obtain correct results, careful testing of the distance of the structure from the PML is required. The air block surrounding the graphene structure must also be carefully defined when changing the stretch factor.

#Normalising to get a proper value of the scattering cross section might need some tuning, but the shape and frequency of the plasmon peak is pretty much correct.

#Please verify that your results are correct when using this code. As the author I claim no responsibility for any errors. 

import meep as mp
import math
import numpy as np
import matplotlib.pyplot as plt


### User input-------------1.0


## Define simulation runtime
field_min=5e-5 # Stop simulation after reaching this field value

## Define wavelength range 
alpha=1.0 # Simulation unit for dimension, 1 corresponds to 1um 
wvl_min=6 # um
wvl_max=10 # um
nfrq = 100

## Mesh/timestep parameters (at least 8 pixels per smallest wavelength, i.e. np.floor(8/wvl_min))
resolution = 10 # number of pixels for alpha

## Define structure properties (for block, go to code to change shape)
tr_x=20 # Zoom in resolution xtimes
tr_y=20 # Zoom in resolution xtimes
tr_z=20 # Zoom in resolution xtimes
lx=0.07# um
ly=0.065# um
lz=1/resolution/tr_z # um

## Cell parameters
dpml = 0.5*wvl_max # um
dair_x = 0.26*wvl_max # um
dair_y = 0.26*wvl_max # um
dair_z = 0.35*wvl_max # um 
sx = 2*(dpml+dair_x+lx) # um, cell size
sy = 2*(dpml+dair_y+lx) # um, cell size
sz = 2*(dpml+dair_z+lx) # um, cell size

# Transformation settings and scaling parameters
gref_x=3.5# um Air block X dimension
gref_y=3.5# um Air block Y dimension 
gref_d=3# um Air block thickness

## Material 
# Drude Parameters
Omega_p= 3.225094 # eV, value for 0.34nm thickness (https://iopscience.iop.org/article/10.1088/1742-6596/129/1/012004/meta),  Graphene Fermi level=0.64eV 
Gama= 0.0103 # eV, 1000cm^2/V.s
epsilon_inf= 5.5 # epsilon infinity
Sigma= None
Sigma_diag = mp.Vector3(1, 1, 0)

## Source settings
amp1=1 # Source 1 amplitude
src_co=3 # source cutoff
src_cmpt=mp.Ex # Source 1 field direction

## Scattering monitor dimensions
lxm=2.9 # um
lym=2.9 # um
lzm=2.9 # um
    
## Movie/Field monitor settings
mon_cmpt=mp.Ex # Field to monitor   
vid_x=sx-2*dpml# Video monitor X
vid_y=sy-2*dpml# Video monitor Y
vid_z=sz-2*dpml# Video monitor Z
dx=alpha/resolution
vid_t=0.1*dx*40# Video time step

## Plot settings
range_fl=2 # 1 plots wavelenght, 2 plots energy, 3 plots frequency


### Reference simulation-------------1.1


## Calculate simulation values
c=3*10**8 #m/s speed of light
Omega_p=Omega_p*np.sqrt(0.34*10**-9)/np.sqrt(alpha/resolution*10**-6)/np.sqrt(alpha/tr_z) # eV (normalize to meep thickness 1/(resolution*stretch) taking into account resolution and stretch
courant=0.5/tr_x  # Courant factor

## Convert to frequency (meep units)
frq_min = 1/wvl_max
frq_max = 1/wvl_min
frq_cen = 0.5*(frq_min+frq_max)
dfrq = frq_max-frq_min
um_scale = alpha# default unit length is 1 um
eV_um_scale = um_scale/1.23984193# conversion factor for eV to 1/um [=1/hc]
h_plankeV=4.135667662*10**-15 # eVs

## Define material susceptibility 
# Drude
drude_range = mp.FreqRange(min=um_scale/(wvl_max*2), max=um_scale/(wvl_min/2)) # Valid frequency range um_scale/wavelength
drude_frq = Omega_p*eV_um_scale # converts energy to meep frequency
drude_gam = Gama*eV_um_scale # converts energy to meep frequency
drude_sig = Sigma
drude_sig_diag=Sigma_diag
epsilon = epsilon_inf # epsilon infinity
drude_susc = [mp.DrudeSusceptibility(frequency=drude_frq, gamma=drude_gam, sigma=drude_sig, sigma_diag=drude_sig_diag)]
drude = mp.Medium(epsilon_diag=mp.Vector3(epsilon, epsilon, epsilon), E_susceptibilities=drude_susc, valid_freq_range=drude_range)

c1=mp.Vector3(tr_x, 0.0, 0.0)
c2=mp.Vector3(0.0, tr_y, 0.0)
c3=mp.Vector3(0.0, 0.0, tr_z)
m_tr=mp.Matrix(c1,c2,c3)
drude.transform(m_tr)
material_set=drude
    
## Create geometry
lx=lx*tr_x
ly=ly*tr_y
lz=lz*tr_z
material_ref=mp.Medium(epsilon_diag=mp.Vector3(1, 1, 1),mu_diag=mp.Vector3(1,1,1))
material_ref.transform(m_tr)
geometry_ref = [mp.Block(size=mp.Vector3(gref_x,gref_y,gref_d), material=material_ref)] 
geometry = [mp.Block(size=mp.Vector3(gref_x,gref_y,gref_d), material=material_ref),
            mp.Block(size=mp.Vector3(lx,ly,lz), material=material_set)] 

## Create cell
pml_layers = [mp.PML(thickness=dpml)]
cell_size = mp.Vector3(sx,sy,sz)

## Define source symmetries
if src_cmpt == mp.Ex: # Source 1 
    vid_dim=mp.Vector3(vid_x,0,vid_z)
    vid_dim2=mp.Vector3(lxm,0,lxm)
    pt = mp.Vector3(0,lx/2+dx+0.005,0)
    symmetries = [mp.Mirror(mp.X,phase=-1),
                  mp.Mirror(mp.Y,phase=+1)]
elif src_cmpt == mp.Ey: # Source 2 
    vid_dim=mp.Vector3(0,vid_y,vid_z)
    vid_dim2=mp.Vector3(0,lxm,lxm)
    pt = mp.Vector3(ly/2+dx+0.005,0,0)
    symmetries = [mp.Mirror(mp.X,phase=+1),
                  mp.Mirror(mp.Y,phase=-1)]
elif src_cmpt == mp.Ez:
    symmetries = [mp.Mirror(mp.X,phase=+1),
                  mp.Mirror(mp.Y,phase=+1)]


## Create source
# is_integrated=True necessary for any planewave source extending into PML
sources = [mp.Source(mp.GaussianSource(frq_cen,fwidth=dfrq,is_integrated=True,start_time=0,
                     cutoff=src_co),
                     center=mp.Vector3(0,0,-0.5*sz+dpml),
                     size=mp.Vector3(sx,sy,0),
                     component=src_cmpt)]
    
## Simulation object
sim = mp.Simulation(resolution=resolution,
                     geometry=geometry_ref, 
                     cell_size=cell_size,
                     boundary_layers=pml_layers,
                     sources=sources,
                     k_point=mp.Vector3(),
                     Courant=courant,
                     symmetries=symmetries,
                     eps_averaging=False,
                    )
   
## Scattering monitor ref
box_x1 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(x=-lxm/2),size=mp.Vector3(0,2*lym/2,2*lzm/2)))
box_x2 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(x=+lxm/2),size=mp.Vector3(0,2*lym/2,2*lzm/2)))
box_y1 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(y=-lym/2),size=mp.Vector3(2*lxm/2,0,2*lzm/2)))
box_y2 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(y=+lym/2),size=mp.Vector3(2*lxm/2,0,2*lzm/2)))
box_z1 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(z=-lzm/2),size=mp.Vector3(2*lxm/2,2*lym/2,0)))
box_z2 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(z=+lzm/2),size=mp.Vector3(2*lxm/2,2*lym/2,0)))
                                
### Run simulation-------------1.2

## Field movie settings
volume=mp.Block(size=vid_dim, center=mp.Vector3())
f = plt.figure(dpi=100)
Animate_ref = mp.Animate2D(sim, fields=mon_cmpt, f=f,output_plane=volume, realtime=False, normalize=True)
plt.close()

# Run 
sim.run(mp.at_every(vid_t,Animate_ref),until_after_sources=mp.stop_when_fields_decayed(2,src_cmpt,pt,field_min))    
plt.close()

## Scattering monitor Fourier-transformed fields in ram
freqs = mp.get_flux_freqs(box_x1)
box_x1_data = sim.get_flux_data(box_x1)
box_x2_data = sim.get_flux_data(box_x2)
box_y1_data = sim.get_flux_data(box_y1)
box_y2_data = sim.get_flux_data(box_y2)
box_z1_data = sim.get_flux_data(box_z1)
box_z2_data = sim.get_flux_data(box_z2)
box_z1_flux0 = mp.get_fluxes(box_z1)


### Main simulation-------------2.1


## Create simulation object
sim.reset_meep()
sim = mp.Simulation(resolution=resolution,
                    cell_size=cell_size,
                    boundary_layers=pml_layers,
                    sources=sources,
                    k_point=mp.Vector3(),
                    symmetries=symmetries,
                    geometry=geometry,
                    Courant=courant,
                    eps_averaging=False
                    )
    
# Scattering monitor
box_x1 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(x=-lxm/2),size=mp.Vector3(0,2*lym/2,2*lzm/2)))
box_x2 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(x=+lxm/2),size=mp.Vector3(0,2*lym/2,2*lzm/2)))
box_y1 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(y=-lym/2),size=mp.Vector3(2*lxm/2,0,2*lzm/2)))
box_y2 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(y=+lym/2),size=mp.Vector3(2*lxm/2,0,2*lzm/2)))
box_z1 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(z=-lzm/2),size=mp.Vector3(2*lxm/2,2*lym/2,0)))
box_z2 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(z=+lzm/2),size=mp.Vector3(2*lxm/2,2*lym/2,0)))

# Remove Fourier-transformed fields from first simulation
sim.load_minus_flux_data(box_x1, box_x1_data)
sim.load_minus_flux_data(box_x2, box_x2_data)
sim.load_minus_flux_data(box_y1, box_y1_data)
sim.load_minus_flux_data(box_y2, box_y2_data)
sim.load_minus_flux_data(box_z1, box_z1_data)
sim.load_minus_flux_data(box_z2, box_z2_data)

## Simulation region preview
figure1=plt.figure(dpi=100)
plt.subplot(2, 2, 1)
volume=mp.Block(size=mp.Vector3(sx,sy,0), center=mp.Vector3())
sim.plot2D(output_plane=volume)
## 
plt.subplot(2, 2, 2)
volume=mp.Block(size=mp.Vector3(sx,0,sz), center=mp.Vector3())
sim.plot2D(output_plane=volume)
## 
plt.subplot(2, 2, 3)
volume=mp.Block(size=mp.Vector3(lxm+dx,lxm+dx,0), center=mp.Vector3())
sim.plot2D(output_plane=volume)
## 
plt.subplot(2, 2, 4)
volume=mp.Block(size=vid_dim2, center=mp.Vector3())
sim.plot2D(output_plane=volume)
if mp.am_master():
    figure1.tight_layout(pad=3.0)
    plt.savefig("simulation.png")
        
### Run simulation-------------2.2

## Field movie (simulation)
#volume=mp.Block(size=mp.Vector3(sx,sy,0), center=mp.Vector3())
volume=mp.Block(size=vid_dim, center=mp.Vector3())
f = plt.figure(dpi=100)
Animate_XY = mp.Animate2D(sim, fields=mon_cmpt, f=f,output_plane=volume, realtime=False, normalize=True)
plt.close()

# Run 
sim.run(mp.at_every(vid_t,Animate_XY),until_after_sources=mp.stop_when_fields_decayed(2,src_cmpt,pt,field_min))    
plt.close()

## Scattering monitor fluxes
freqs = mp.get_flux_freqs(box_x1)
box_x1_flux = mp.get_fluxes(box_x1)
box_x2_flux = mp.get_fluxes(box_x2)
box_y1_flux = mp.get_fluxes(box_y1)
box_y2_flux = mp.get_fluxes(box_y2)
box_z1_flux = mp.get_fluxes(box_z1)
box_z2_flux = mp.get_fluxes(box_z2)

## Scattering calculation
scatt_flux = np.asarray(box_x1_flux)-np.asarray(box_x2_flux)+np.asarray(box_y1_flux)-np.asarray(box_y2_flux)+np.asarray(box_z1_flux)-np.asarray(box_z2_flux)
intensity = np.asarray(box_z1_flux0)/(lxm*lym)
scatt_cross_section = np.divide(scatt_flux,intensity)


### Plot results-------------3


## Unit conversions
if range_fl==1: # Wavelength um
    plt_range=1/np.asarray(freqs)
    plt_label='Wavelength, um'
elif range_fl==2: # Energy
    plt_range=h_plankeV*c/(1/np.asarray(freqs)*10**-6)
    plt_label='Energy, eV'
else: # Frequency
    plt_range=c/(1/np.asarray(freqs)*10**-6)/10**12
    plt_label='Frequency, THz'

## Permittivity data
eps_dat=[]
for n in np.arange(frq_min,frq_max,0.001):
    temp_n=n    
    eps_dat=np.append(eps_dat,sim.get_epsilon_point(pt=mp.Vector3(0,0,0), frequency=temp_n))
n_mt=np.sqrt(eps_dat)
n_m=n_mt.real
k_m=n_mt.imag

## Plot permittivity
if mp.am_master():
    plt.figure(dpi=150,figsize=(6,7))
    plt.subplot(2, 1, 1)
    plt.plot(1/np.arange(frq_min,frq_max,0.001),n_m,'b-',label='n')
    plt.plot(1/np.arange(frq_min,frq_max,0.001),k_m,'r-',label='k')
    plt.grid(True,which="both",ls="-")
    plt.xlabel('wavelength, um')
    plt.ylabel('n,k')
    plt.legend(loc='lower left')
    plt.title('Dispersion')
    
    plt.subplot(2, 1, 2)
    plt.plot(h_plankeV*c/(1/np.arange(frq_min,frq_max,0.001)*10**-6),n_m,'b-',label='n')
    plt.plot(h_plankeV*c/(1/np.arange(frq_min,frq_max,0.001)*10**-6),k_m,'r-',label='k')
    plt.grid(True,which="both",ls="-")
    plt.xlabel('Energy, eV')
    plt.ylabel('n,k')
    plt.legend(loc='lower left')
    plt.title('Dispersion')
    
    plt.tight_layout()
    plt.savefig("Dispersion.png")    

## Plot scattering 
if mp.am_master():
    plt.figure(dpi=150)
    plt.plot(plt_range,abs(scatt_cross_section),'bo-',label='Meep')
    plt.grid(True,which="both",ls="-")
    plt.xlabel(plt_label)
    plt.ylabel('scatt_cross_section, au')
    plt.legend(loc='upper right')
    plt.title('scattering')
    plt.tight_layout()
    plt.savefig("scattering.png")
    
## Simulation region E-field
figure2=plt.figure(dpi=100)
plt.subplot(2, 2, 1)
volume=mp.Block(size=mp.Vector3(sx,sy,0), center=mp.Vector3())
sim.plot2D(fields=mon_cmpt,output_plane=volume)
## 
ex_data = sim.get_array(center=mp.Vector3(), size=mp.Vector3(sx,0,sz), component=mp.Ex)
plt.subplot(2, 2, 2)
volume=mp.Block(size=vid_dim, center=mp.Vector3())
sim.plot2D(fields=mon_cmpt,output_plane=volume)
## 
ex_data = sim.get_array(center=mp.Vector3(), size=mp.Vector3(lxm,lxm,0), component=mp.Ex)
plt.subplot(2, 2, 3)
volume=mp.Block(size=mp.Vector3(lxm,lxm,0), center=mp.Vector3())
sim.plot2D(fields=mon_cmpt,output_plane=volume)
## 
plt.subplot(2, 2, 4)
volume=mp.Block(size=vid_dim2, center=mp.Vector3())
sim.plot2D(fields=mon_cmpt,output_plane=volume)
##    
if mp.am_master():
    figure2.tight_layout(pad=3.0)
    plt.savefig("E_field.png")

## Generate movies
if mp.am_master():
    filename = "movie_ref_E.mp4"
    Animate_ref.to_mp4(10,filename)
    
    filename = "movie_E_field.mp4"
    Animate_XY.to_mp4(10,filename)