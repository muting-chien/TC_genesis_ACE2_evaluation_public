######################
# Goal of this code is to save precip data from the below two sources to compare with ACE2
#  2. IMERG from UW olympus /home/disk/eos4/muting/data/Tb/imerg.1deg.8x.fill.0119.nc -->Trnaform from 3-hourly into 6-hourly
# 2024.12.4
# Mu-Ting Chien
########################

import numpy as np
from netCDF4 import Dataset
import xarray as xr
import os

# Set output directory
file_dir_out = "/barnes-engr-scratch1/mchien/data_output/ace2/obs_compare_with_ace2/"

# Load ERA5 precip from barnes-engr-scratch1
file_dir = "/barnes-engr-scratch1/DATA/Tb_precip_obs/"
filename = "imerg.1deg.8x.fill.0119.nc"

# Open files
ds = xr.open_dataset(file_dir+filename)
ds = ds.sel(time=slice("2001-01-01","2010-12-31"))

# Calculate 6-hourly mean
ds_6hourly = ds.resample(time="6H").mean()

# Save to a new NetCDF file
output_file = "imerg_preicp_1deg_6_hourly_2001_2010.nc"
ds_6hourly.to_netcdf(file_dir_out+output_file)

print(f"Processed data saved to {output_file}")