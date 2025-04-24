#############################################
# Goal of this code is to save precip data from ACE2
# 2024.12.3
# Mu-Ting Chien
#########################################
# Import pacakges
import numpy as np
import xarray as xr
import os

# Set constants
d2s = 86400

# Set path for figure
expname = 'ace2' 
imem = 0 # Just choose 1 ensemble member
mem_str = f"{imem+1:02d}" 

DIR = '/barnes-engr-scratch1/c832572266/'
fig_dir = DIR + 'figure/ace2_fig/' # Reomte direcotry for figures 
os.makedirs(fig_dir,exist_ok=True) 
file_name = 'autoregressive_predictions.nc'

# Load precip 
load_precip = 1 # 1 or 0
first_time_execution = 1
if expname == 'ace2':

    if first_time_execution == 1:
        # Load yr 1 data
        yr_str = '1'
        file_dir = DIR + 'data_output/ace2/ace2_output/yr'+yr_str+'/'
        ds     = xr.open_dataset(file_dir + file_name) #(time, ensemble_member, lat, lon)

    # Find the index of the nearest latitudes to -15 and 15
    lat_15S = ds.lat.sel(lat=-15, method="nearest")
    lat_15N = ds.lat.sel(lat=15, method="nearest")

    # Select data between these latitudes
    ds = ds.sel(lat=slice(lat_15S, lat_15N))
    time_1 = ds['time'][:] 
    lat_15SN = ds['lat'][:]
    
    nt_1 = np.size(time_1)

    # Load yr 2-5 data
    yr_str = '2-5'
    file_dir2 = DIR + 'data_output/ace2/ace2_output/yr'+yr_str+'/'
    ds2 = xr.open_dataset(file_dir2 + file_name).sel(lat=slice(lat_15S, lat_15N))
    time_2 = ds2['time'][:]
    nt_2   = np.size(time_2)

    # Load yr 6-10 data
    yr_str = '6-10'
    file_dir2 = DIR + 'data_output/ace2/ace2_output/yr'+yr_str+'/'
    ds3 = xr.open_dataset(file_dir2 + file_name).sel(lat=slice(lat_15S, lat_15N))
    time_3 = ds3['time'][:]
    nt_3   = np.size(time_3)

    nt = nt_1 + nt_2 + nt_3

    # Load variables: PRECIP 
    if load_precip == 1:
        #   Precipitaiton at surface (original unit: kg/m2/s), *d2s will change unit into mm/day
        PRECIP_1 = ds['PRATEsfc'][:,:,:,:] 
        PRECIP_2 = ds2['PRATEsfc'][:,:,:,:] 
        PRECIP_3 = ds3['PRATEsfc'][:,:,:,:]
        PRECIP = xr.concat([PRECIP_1, PRECIP_2, PRECIP_3], dim='time')
        del PRECIP_1, PRECIP_2, PRECIP_3


mem  = ds['sample']
lat  = ds['lat']
lon  = ds['lon']
nmem = np.size(mem)
nlat = np.size(lat)
nlon = np.size(lon)
print('Finish loading precip')

#######################################
# Calculate meridional average of precip
########################################
# Convert latitude to radians for cosine weighting
lat_radians = np.deg2rad(lat_15SN)
# Create weights based on cos(latitude)
weights = np.cos(lat_radians)

PRECIP_15SN_lat_avg     = PRECIP.weighted(weights).mean(dim='lat')

#############################
# Save precip output
############################
file_dir_multi_yr = DIR + 'data_output/ace2/ace2_output/10yr/'
os.makedirs(file_dir_multi_yr, exist_ok=True) 

ds = xr.Dataset({
    "PRECIP":PRECIP,
    "PRECIP_15SN_lat_avg":PRECIP_15SN_lat_avg,
    "lat_15SN":lat_15SN,
})
ds.to_netcdf(file_dir_multi_yr + "PRECIP_"+expname+"_10yr.nc")
print('Finish saving precip')