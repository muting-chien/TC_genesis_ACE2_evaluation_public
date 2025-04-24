###############################
# Goal of this code is to save the global mean timeseries of 
# each variable in autoregressive.nc to another nc file
# 2024.12.2
# Mu-Ting Chien
##########################

import xarray as xr
import numpy as np

# Open the original NetCDF file
file_dir = '/barnes-engr-scratch1/mchien/data_output/ace2/ace2_output/'
yr_list = list(['yr1','yr2-5','yr6-10'])
input_file = "autoregressive_predictions.nc"
output_file = "global_mean_timeseries.nc"

for iyr in range(1,3):
    print(yr_list[iyr])
    file_dir_sub = file_dir + yr_list[iyr] + '/'
    ds = xr.open_dataset(file_dir_sub + input_file)

    # Prepare to save the global mean timeseries
    mean_ds = xr.Dataset()

    # Loop through all variables
    for var_name, da in ds.data_vars.items():
        if {'time', 'sample','lat', 'lon'}.issubset(da.dims):  # Check if the variable has required dimensions
            # Weight by cosine of latitude
            weights = np.cos(np.deg2rad(ds['lat']))
            weights = weights / weights.mean()  # Normalize weights
            # Compute weighted global mean
            global_mean = (da * weights).mean(dim=['lat', 'lon'])
            mean_ds[var_name] = global_mean

    # Copy time coordinate to the new dataset
    mean_ds['time'] = ds['time']

    # Save the global mean time series to a new NetCDF file
    mean_ds.to_netcdf(file_dir_sub + output_file)
    print(f"Global mean timeseries saved to {output_file}")