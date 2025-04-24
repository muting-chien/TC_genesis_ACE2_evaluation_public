##########################
# This is the extracted code from Tropical_wave.ipynb
# Main reason is to avoid kernel crashed in Tropical_wave.ipynb
# Generate WK-diagram for 11 members of precip anomaly
# 2024.12.3
# Mu-Ting Chien
########################################
# Import pacakges
import numpy as np
from netCDF4 import Dataset
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.util as cartopy_util
import cartopy.crs as ccrs
import os
import sys
sys.path.append('/barnes-engr-scratch1/c832572266/Function/')
from scipy import signal
#import KW_diagnostics as KW
import mjo_mean_state_diagnostics_uw as MJO
import create_my_colormap as mycolor
RWB = mycolor.red_white_blue()
sys.path.append('/home/c832572266/code/function/')
import KW_diagnostics_new as KW

# Set constants
d2s = 86400

# Set path for figure
iexp = 0 # 0 to 2 'm2K' '2K','CTL'
iexp_spectrum = 1
expname_list = list(['ace2'])
expname = expname_list[iexp]
dt  = 4 # hot many data per day

DIR = '/barnes-engr-scratch1/c832572266/'
fig_dir = DIR + 'figure/ace2_fig/' # Reomte direcotry for figures 
fig_dir_sub = fig_dir + 'power_spectrum/'
os.makedirs(fig_dir, exist_ok=True) 
os.makedirs(fig_dir_sub, exist_ok=True)
file_dir_multi_yr = DIR + 'data_output/ace2/ace2_output/10yr/'
file_name = 'autoregressive_predictions.nc'

saved_precip_ano = 1 # 0 or 1
calc_rsym = 0 # 0 or 1 is spectrum calculated or not
plot_rsym = 1 # 1 or 0: plot spectrum

if calc_rsym == 1:
    # Load precip saved from Save_precip.py (xarray)
    if saved_precip_ano == 0:
        ds = xr.open_dataset(file_dir_multi_yr + "PRECIP_"+expname+"_10yr.nc")
        lat = ds['lat']
        lat_15SN = ds['lat_15SN'].sel(lat=slice(-15,15))
        lon = ds['lon']
        time = ds['time']
        mem = ds['sample'][:]#[0:2]
        # Load not-meridionally-averaged precip (only extract 15S-15N), note that in this data, only 10S-10N data is saved, so data in other latitudes are np.nan automatically!
        PRECIP = ds['PRECIP'][:,:,:,:].sel(lat=slice(-15,15))*d2s # (sample, time, lat, lon) #[0:2]

        nt               = np.size(time)
        nlat_15SN        = np.size(lat_15SN)
        nlon             = np.size(lon)
        nmem             = np.size(mem)

        # Remove annual cycle and diurnal cycle of precip (not meridionally averaged)

        # Remove diurnal cycle
        nday             = int(nt/dt)
        V                = PRECIP.transpose("time","lat","lon","sample").values # New dim: (time, lon, mem) # change this if different variables
        V_reshape        = np.reshape(V, (nday, dt, nlat_15SN, nlon, nmem))
        print(np.shape(V_reshape))
        diurnal_cyc      = np.tile( np.nanmean( V_reshape,0).squeeze(), (nday,1,1,1,1))
        print(np.shape(diurnal_cyc))
        #print('nday:', nday, ', nt:', nt, ', nlon:', nlon, ', nmem:',nmem)
        diurnal_cyc_flat = np.reshape(diurnal_cyc, (nday*dt, nlat_15SN, nlon, nmem))
        V_ano            = V - diurnal_cyc_flat # (nday*dt, nlat_15SN, nlon, nmem)
        print(np.shape(V_ano))

        plot_test_fig = 1
        if plot_test_fig == 1: # plot removing diurnal cycle
            t = np.arange(0, dt*10) # 10 days
            plt.plot(t, V[t,0,0,0], 'k-o')
            plt.plot(t, diurnal_cyc_flat[t,0,0,0], 'b-o')
            plt.legend(['raw','diurnal cycle'])
            plt.xlabel('hours')
            plt.show()

        # Remove annual cycle
        V_ano_final, cyc_final = MJO.remove_anncycle_4d( signal.detrend(V_ano, 0), time, lat_15SN, lon, mem, 1/dt) 
        # Note that 1/dt is not included in the current function, but it is included in the function on olympus (UW)

        if plot_test_fig == 1:
            ts = np.arange(0, 365*4*2)
            plt.subplot(2,1,1)
            plt.plot(ts, V[ts, 1, 1, 0], 'k')
            plt.plot(ts, cyc_final[ts,1, 1, 0] + diurnal_cyc_flat[ts, 1, 1, 0], 'g')
            plt.legend(['raw','diurnal + seasonal cyc'])
            plt.subplot(2,1,2)
            plt.plot(ts, V_ano[ts, 1, 1, 0], 'r')
            plt.legend(['ano'])
            plt.show()

        np.savez(file_dir_multi_yr+'pr_ano_15SN.npz', pr_ano_15SN=V_ano_final, time=time, lon=lon, \
                 mem=mem, lat_15SN=lat_15SN)
        print('Finish saving precip anomaly')
    else:
        data = np.load(file_dir_multi_yr+'pr_ano_15SN.npz')
        V_ano_final = data['pr_ano_15SN']
        lat_15SN    = data['lat_15SN']
        time        = data['time']
        lon         = data['lon']
        nt          = np.size(time)
        nlon        = np.size(lon)
        nlat_15SN   = np.size(lat_15SN)
        print('Finish loading precip anomaly')

    # Generate Wheeler-Kiladis spectrum using 11 members
    '''
    V_ano_11mem = V_ano_final[:,:,:,0:11]
    V_ano_11mem = np.transpose(V_ano_11mem, (3,0,1,2) )
    if saved_precip_ano == 0:
        V_ano_11mem = np.reshape(V_ano_11mem, (nday*dt*11, nlat_15SN, nlon))
    else:
        V_ano_11mem = np.reshape(V_ano_11mem, (nt*11, nlat_15SN, nlon))
    print(np.shape(V_ano_11mem))
    '''

    # (1) Calculate precip specturm 
    power_pr_sym, power_pr_asy, power_background, r_sym, r_asy, x, y, freq, zonalwnum, dof \
        = KW.calculate_power_spectrum(V_ano_final[:,:,:,0], \
            kw_meridional_proj=0, Fs_t=dt, Fs_lon=1, output_sym_only=0)
    
    # save data
    np.savez(file_dir_multi_yr+'pr_wavenum_freq_10yr_1mem.npz', power_pr_sym=power_pr_sym, \
             power_pr_asy=power_pr_asy, \
             r_sym=r_sym, r_asy=r_asy, power_background=power_background,\
             x=x, y=y, freq=freq, zonalwnum=zonalwnum)
    print('Finish saving spectrum')

else: # load saved spectrum
    data = np.load(file_dir_multi_yr+'pr_wavenum_freq_10yr_1mem.npz')
    power_pr_sym = data['power_pr_sym']
    r_sym        = data['r_sym']
    x            = data['x']
    y            = data['y']
    power_pr_asy = data['power_pr_asy']
    r_asy        = data['r_asy']
    power_background = data['power_background']


# Plot spectrum
if plot_rsym == 1:

    # PLotting, CHANGE ICASE & CASE_SHORT:
    clev = np.arange(-1.5,0.1,0.1)
    cticks = clev
    vname = 'precip'

    # clev for signal strength
    clev_r = np.arange(1.1,2.05,0.05) #6,4.6,0.2)
    cticks_r = np.arange(1.1,2.1,0.1)
    
    # Plot symmetric spectrum
    KW.plot_raw_spectrum(power_pr_sym, x, y, clev, cticks, fig_dir_sub, vname, iexp_spectrum, \
                         expname+'-Mem01', sym_asy_background=0)
    KW.plot_signal_strength(r_sym, x, y, clev_r, cticks_r, fig_dir_sub, vname, iexp_spectrum, \
                            expname+'-Mem01', sym_asy=0)

    # Plot anti-symmetric spectrum
    KW.plot_raw_spectrum(power_pr_asy, x, y, clev, cticks, fig_dir_sub, vname, iexp_spectrum, \
                         expname+'-Mem01', sym_asy_background=1)
    KW.plot_signal_strength(r_asy, x, y, clev_r, cticks_r, fig_dir_sub, vname, iexp_spectrum, \
                            expname+'-Mem01', sym_asy=1)

    # Plot background spectrum
    KW.plot_raw_spectrum(power_background, x, y, clev, cticks, fig_dir_sub, vname, iexp_spectrum, \
                         expname+'-Mem01', sym_asy_background=2)