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
from datetime import datetime

# Set constants
d2h = 24

# Set path for figure
iexp = 0 
iexp_spectrum = 1
expname_list = list(['imerg'])
expname = expname_list[iexp]
dt  = 4 # hot many data per day

DIR = '/barnes-engr-scratch1/c832572266/'
fig_dir = DIR + 'figure/ace2_fig/' # Reomte direcotry for figures 
fig_dir_sub = fig_dir + 'power_spectrum/imerg/'
os.makedirs(fig_dir, exist_ok=True) 
os.makedirs(fig_dir_sub, exist_ok=True)
file_dir_multi_yr = DIR + 'data_output/ace2/obs_compare_with_ace2/'
file_name = 'imerg_preicp_1deg_6_hourly_2001_2010.nc'

saved_precip_ano = 0 # 0 or 1
calc_rsym = 1 # 0 or 1 is spectrum calculated or not
plot_rsym = 1 # 1 or 0: plot spectrum

if calc_rsym == 1:
    # Load precip saved from Save_precip.py (xarray)
    if saved_precip_ano == 0:
        ds = xr.open_dataset(file_dir_multi_yr + file_name)
        lat = ds['lat']
        lat_15SN = ds['lat'].sel(lat=slice(-15,15))
        lon = ds['lon']
        time = ds['time']
        #mem = ds['sample'][:]#[0:2]
        # Load not-meridionally-averaged precip (only extract 15S-15N), note that in this data, only 10S-10N data is saved, so data in other latitudes are np.nan automatically!
        PRECIP = ds['precipitationCal'][:,:,:].sel(lat=slice(-15,15))*d2h # (time, lat, lon) original unit:mm/hr

        nt               = np.size(time)
        nlat_15SN        = np.size(lat_15SN)
        nlon             = np.size(lon)
        #nmem             = np.size(mem)

        # Remove annual cycle and diurnal cycle of precip (not meridionally averaged)

        # Remove diurnal cycle
        nday             = int(nt/dt)
        V                = PRECIP.transpose("time","lat","lon").values # New dim: (time, lon, mem) # change this if different variables
        #V                = PRECIP.values
        V_reshape        = np.reshape(V, (nday, dt, nlat_15SN, nlon))
        print(np.shape(V_reshape))
        diurnal_cyc      = np.tile( np.nanmean( V_reshape,0).squeeze(), (nday,1,1,1))
        print(np.shape(diurnal_cyc))
        #print('nday:', nday, ', nt:', nt, ', nlon:', nlon)
        diurnal_cyc_flat = np.reshape(diurnal_cyc, (nday*dt, nlat_15SN, nlon))
        V_ano            = V - diurnal_cyc_flat # (nday*dt, nlat_15SN, nlon)
        print(np.shape(V_ano))

        plot_test_fig = 1
        if plot_test_fig == 1: # plot removing diurnal cycle
            t = np.arange(0, dt*10) # 10 days
            plt.plot(t, V[t,0,0], 'k-o')
            plt.plot(t, diurnal_cyc_flat[t,0,0], 'b-o')
            plt.legend(['raw','diurnal cycle'])
            plt.xlabel('hours')
            plt.show()

        # Remove annual cycle
        V_ano_final, cyc_final = MJO.remove_anncycle_3d( signal.detrend(V_ano, 0), time, lat_15SN, lon, 1/dt) 
        # Note that 1/dt is not included in the current function, but it is included in the function on olympus (UW)

        if plot_test_fig == 1:
            ts = np.arange(0, 365*4*2)
            plt.subplot(2,1,1)
            plt.plot(ts, V[ts, 1, 1], 'k')
            plt.plot(ts, cyc_final[ts,1, 1] + diurnal_cyc_flat[ts, 1, 1], 'g')
            plt.legend(['raw','diurnal + seasonal cyc'])
            plt.subplot(2,1,2)
            plt.plot(ts, V_ano[ts, 1, 1], 'r')
            plt.legend(['ano'])
            plt.show()

        # Change time data into formatted string: YYYY-MM-DDTHH:MM:SS (original format is cftime.DatetimeJulian(2001,1,1,6,0,0 has_year_zero=False))
        # Reason for changing the time data is that the current format cannot be saved as npz file
        # Convert to datetime and format
        def convert_to_datetime(time_obj):
            # Convert to Python datetime object
            py_datetime = datetime(time_obj.year, time_obj.month, time_obj.day, 
                                    time_obj.hour, time_obj.minute, time_obj.second)
            # Format it to 'yyyy-mm-ddThh:mm:ss'
            return py_datetime.strftime('%Y-%m-%dT%H:%M:%S')

        # If time is an array, apply the conversion to each element
        time_strings = [convert_to_datetime(t) for t in time.values]

        # Output the formatted string
        print(time_strings)

        np.savez(file_dir_multi_yr+'pr_ano_15SN_imerg_2001-2010.npz', pr_ano_15SN=V_ano_final, time=time_strings, lon=lon, \
                 lat_15SN=lat_15SN)
        print('Finish saving precip anomaly')
    else:
        data = np.load(file_dir_multi_yr+'pr_ano_15SN_imerg_2001-2010.npz')
        V_ano_final = data['pr_ano_15SN']
        lat_15SN    = data['lat_15SN']
        time        = data['time']
        lon         = data['lon']
        nt          = np.size(time)
        nlon        = np.size(lon)
        nlat_15SN   = np.size(lat_15SN)
        print('Finish loading precip anomaly')

    # Generate Wheeler-Kiladis spectrum 

    # (1) Calculate precip specturm 
    power_pr_sym, power_pr_asy, power_background, r_sym, r_asy, x, y, freq, zonalwnum, dof \
        = KW.calculate_power_spectrum(V_ano_final[:,:,:], \
            kw_meridional_proj=0, Fs_t=dt, Fs_lon=1, output_sym_only=0)
    
    # save data
    np.savez(file_dir_multi_yr+'pr_wavenum_freq_10yr_imerg_2001-2010.npz', power_pr_sym=power_pr_sym, \
             power_pr_asy=power_pr_asy, \
             r_sym=r_sym, r_asy=r_asy, power_background=power_background,\
             x=x, y=y, freq=freq, zonalwnum=zonalwnum)
    print('Finish saving spectrum')

else: # load saved spectrum
    data = np.load(file_dir_multi_yr+'pr_wavenum_freq_10yr_imerg_2001-2010.npz')
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
                         expname, sym_asy_background=0)
    KW.plot_signal_strength(r_sym, x, y, clev_r, cticks_r, fig_dir_sub, vname, iexp_spectrum, \
                            expname, sym_asy=0)

    # Plot anti-symmetric spectrum
    KW.plot_raw_spectrum(power_pr_asy, x, y, clev, cticks, fig_dir_sub, vname, iexp_spectrum, \
                         expname, sym_asy_background=1)
    KW.plot_signal_strength(r_asy, x, y, clev_r, cticks_r, fig_dir_sub, vname, iexp_spectrum, \
                            expname, sym_asy=1)

    # Plot background spectrum
    KW.plot_raw_spectrum(power_background, x, y, clev, cticks, fig_dir_sub, vname, iexp_spectrum, \
                         expname, sym_asy_background=2)