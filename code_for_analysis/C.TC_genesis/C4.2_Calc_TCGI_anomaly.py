####################################
# Goal of this code is to calculate anomaly of TCGI (3-dimension)
# so that these fields can be composited by KW phase
# to generate KW composite TCGI
# 2025.2.12
# Mu-Ting Chien
##########################

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('/barnes-engr-scratch1/c832572266/Function/')
from scipy import signal
#import KW_diagnostics as KW
import mjo_mean_state_diagnostics_uw as MJO
#import create_my_colormap as mycolor
#RWB = mycolor.red_white_blue()
#sys.path.append('/home/C832572266/code/function/')
#import KW_diagnostics_new as KW


DIR = '/barnes-engr-scratch1/c832572266/'

ace2_era5 = 1 #0:ace2 or 1:era5 or 2:ace2 (100yr)
tcgi_raw_fix1var = 1 # 0: original TCGI, 1: TCGI with 1 variables fixed

if ace2_era5 == 0:
    raw_fix_list = list(['','_change-1-variable_new_time'])
elif ace2_era5 >=1:
    raw_fix_list = list(['','_change-1-variable'])


if ace2_era5 == 0:
    fig_dir = DIR + 'figure/ace2_fig/' # Reomte direcotry for figures 
    fig_dir_sub = fig_dir + 'test_anomaly/'
    file_dir_multi_yr = DIR + 'data_output/ace2/ace2_output/10yr/' # input and output directory
    file_name = 'TCGI_ACE2_2001_2010_2.5deg_6h' # change this if other files
    nsub = 1
elif ace2_era5 == 1:
    fig_dir = DIR + 'figure/ace2_fig/' # Reomte direcotry for figures 
    fig_dir_sub = fig_dir + 'test_anomaly_era5/'
    file_dir_multi_yr = DIR + 'data_output/ERA5_TCGI/' # input and output directory
    file_name = 'TCGI_ERA5_2001_2010_2.5deg_6h'
    nsub = 1
elif ace2_era5 == 2:
    fig_dir = DIR + 'figure/ace2_fig/' # Reomte direcotry for figures 
    fig_dir_sub = fig_dir + 'test_anomaly/'
    sub_dir = list(['2001','2002','2003','2004','2005','2006','2007','2008','2009','2010'])
    nsub = 10

os.makedirs(fig_dir, exist_ok=True) 
os.makedirs(fig_dir_sub, exist_ok=True)

dimsize = 3 # or 4 (dimension of the variables), change this!
calc_lat_avg = 0 # or 1 (whether calc meridional average in the end or not), change this!
dt  = 4 # hot many data per day (6-hourly)

if tcgi_raw_fix1var == 0:
    vname_list = list(['TCGI','absvor_clipped','shear','col_rh','relsst','sst','absvor']) # change this if other files
else:
    vname_list = list(['TCGI_absvor_clipped','TCGI_shear','TCGI_col_rh','TCGI_relsst'])

vname_out_list = list
nv = np.size(vname_list)

for isub in range(0, nsub):

    if ace2_era5 == 2:
        file_dir_multi_yr = DIR + 'data_output/ace2/ace2_output/repeat_2001-2010/'+sub_dir[isub]+'/' # input and output directory
        file_name = 'TCGI_ACE2_2001_2010_2.5deg_6h'+raw_fix_list[tcgi_raw_fix1var]+'_'+sub_dir[isub] # change this if other files
        print(sub_dir[isub])

    if tcgi_raw_fix1var == 0:

        ds     = xr.open_dataset(file_dir_multi_yr + file_name +'.nc') #(time, ensemble_member, lat, lon)
    
    elif tcgi_raw_fix1var == 1:
        
        if ace2_era5 == 0 or ace2_era5==1:
            ds     = xr.open_dataset(file_dir_multi_yr + file_name +raw_fix_list[tcgi_raw_fix1var]+'.nc')
        elif ace2_era5 == 2:
            ds     = xr.open_dataset(file_dir_multi_yr + file_name +'.nc')
            

    #ds_time = xr.open_dataset(file_dir_multi_yr + file_name+'.nc')
    #time   = ds_time['time'][:] 
    
    if ace2_era5 == 2:
        ds = ds.isel(time=slice(0, -3))
    
    time   = ds['time'][:]
    lat    = ds['lat'][:]
    lon    = ds['lon'][:]
    if dimsize == 4:
        plev   = ds['plev'][:-1] # don't use 1000 hPa
        nlev    = np.size(plev)
    nt          = np.size(time)
    nlat        = np.size(lat)
    nlon        = np.size(lon)
    if isub == 0:
        print(nt)

    for iv in range(0, nv):#

        V = ds[vname_list[iv]][:]

        #####################
        # Calculate anomaly
        #######################
        # Remove annual cycle and diurnal cycle of precip (not meridionally averaged)
        # Remove diurnal cycle
        nday             = int(nt/dt)
        
        # get information of data array (will used to save nc in the end)
        coords           = {dim: V.coords[dim].values for dim in V.dims}
        long_name        = V.attrs.get("long_name", None) 
        name             = V.name
        dim              = V.dims
        units            = V.attrs.get("units", None) 

        V                = V.values 
        if dimsize == 4:
            V_reshape        = np.reshape(V, (nday, dt, nlat, nlon, nlev))
            diurnal_cyc      = np.tile( np.nanmean( V_reshape,0).squeeze(), (nday,1,1,1,1))
            diurnal_cyc_flat = np.reshape(diurnal_cyc, (nday*dt, nlat, nlon, nlev))
        elif dimsize == 3:
            V_reshape        = np.reshape(V, (nday, dt, nlat, nlon))
            diurnal_cyc      = np.tile( np.nanmean( V_reshape,0).squeeze(), (nday,1,1,1))
            diurnal_cyc_flat = np.reshape(diurnal_cyc, (nday*dt, nlat, nlon))
        
        #print(np.shape(V_reshape))
        #print(np.shape(diurnal_cyc))
        #print('nday:', nday, ', nt:', nt, ', nlon:', nlon, ', nmem:',nmem)

        V_ano            = V - diurnal_cyc_flat # (nday*dt, nlat, nlon, nlev) or (nday*dt, nlat, nlon)
        #print(np.shape(V_ano))

        plot_test_fig = 1
        if plot_test_fig == 1: # plot removing diurnal cycle
            t = np.arange(0, dt*10) # 10 days
            if dimsize == 4:
                plt.plot(t, V[t,0,0,0], 'k-o')
                plt.plot(t, diurnal_cyc_flat[t,0,0,0], 'b-o')
            elif dimsize == 3:
                plt.plot(t, V[t,0,0], 'k-o')
                plt.plot(t, diurnal_cyc_flat[t,0,0], 'b-o')            
            plt.legend(['raw','diurnal cycle'])
            plt.xlabel('hours')
            plt.savefig(fig_dir_sub+'Test_anomaly_'+vname_list[iv]+'.png')
            plt.show()

        # Detrend (only non nan can be detrended, otherwise there would be error)
        V_ano2 = np.where(np.isnan(V_ano)==1, -10**10, V_ano)
        V_ano2 = signal.detrend(V_ano2, 0)
        V_ano2 = np.where(np.isnan(V_ano)==1, np.nan, V_ano2)
        del V_ano

        # Remove annual cycle
        if dimsize == 4:
            V_ano_final, cyc_final = MJO.remove_anncycle_4d( V_ano2, time, lat, lon, plev, 1/dt) 
        elif dimsize == 3:
            V_ano_final, cyc_final = MJO.remove_anncycle_3d( V_ano2, time, lat, lon, 1/dt) 
        # Note that 1/dt is not included in the current function, but it is included in the function on olympus (UW)

        # Transform back to data array
        V_ano_final = xr.DataArray(V_ano_final, coords=coords, dims=dim, name=name, \
                        attrs={"units": units, "long_name":long_name+', remove time mean, seasonal cycle, and diurnal cycle'})

        if calc_lat_avg == 1:
            #######################################
            # Calculate meridional average of V_ano_final
            ########################################
            # Convert latitude to radians for cosine weighting
            lat_radians = np.deg2rad(lat)
            # Create weights based on cos(latitude)
            weights = np.cos(lat_radians)

            lat_0N = ds.lat.sel(lat=0.5, method="nearest")
            lat_0S = ds.lat.sel(lat=-0.5, method="nearest")
            lat_10N = ds.lat.sel(lat=9.5, method="nearest")
            lat_10S = ds.lat.sel(lat=-9.5, method="nearest")
            V_0_10N_lat_avg = V_ano_final.sel(lat=slice(lat_0N, lat_10N)).weighted(weights).mean(dim='lat') # 0.5~10.5
            V_0_10S_lat_avg = V_ano_final.sel(lat=slice(lat_10S, lat_0S)).weighted(weights).mean(dim='lat') # -10.5~-0.5
            
            # Save anomaly
            ds_out = xr.Dataset({
                vname_list[iv]+'_ano': V_ano_final,
                vname_list[iv]+'_ano_0_10N_latavg':V_0_10N_lat_avg,
                vname_list[iv]+'_ano_0_10S_latavg':V_0_10S_lat_avg,
                "lat":lat,
                "lon":lon,
                "time":time
            })
        else:
            # Save anomaly
            ds_out = xr.Dataset({
                vname_list[iv]+'_ano': V_ano_final,
                "lat":lat,
                "lon":lon,
                "time":time,
            })

            
        if ace2_era5 == 2 and tcgi_raw_fix1var == 1:
            ds_out.to_netcdf(file_dir_multi_yr + file_name + '_ano.nc',"a")
        else:
            ds_out.to_netcdf(file_dir_multi_yr + file_name+raw_fix_list[tcgi_raw_fix1var]+'_ano.nc',"a")
        print('Finish saving '+vname_list[iv]+' anomaly')

    