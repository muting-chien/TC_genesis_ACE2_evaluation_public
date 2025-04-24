###########################
# Reorganize variable before applying hybrid_to_pressure_coord_transformation_ace2
# resave the file (separate each ensemble member, put all levels together)
# 2024.12.11
# Mu-Ting Chien
#######################################
import numpy as np
import xarray as xr
from xgcm import Grid
import os
import numpy as np

# Set experiment name:
DIR           = '/barnes-engr-scratch1/c832572266/'
expname_list  = list(['ace2'])
icase         = 0 # (which exp, Change this!)
expname       = expname_list[icase]
dir_in_output = DIR + 'data_output/'+expname+'/ace2_output/' 
#
# Set variable name:
vname_list      = list(['u','v','T','q'])
vname_long_list = list(['eastward_wind','northward_wind','air_temperature','specific_total_water'])

#nt_small = 10 (for testing)

# Load original data
filename = 'autoregressive_predictions.nc'
# Name of the sub directory for the data named "filename"
if icase == 0:
    dir_sub_list = list(['yr1','yr2-5','yr6-10'])
nsub = np.size(dir_sub_list)

# Pile data from all levels into 1 variable
for isub in range(0, nsub):
    dir_in_sub = dir_in_output + dir_sub_list[isub]+'/'
    dir_out = dir_in_sub + '3D_field/'
    os.makedirs(dir_out, exist_ok=True)

    ds = xr.open_dataset(dir_in_sub + filename)
    time = ds['time']#[0:nt_small]
    mem  = ds['sample']
    lat  = ds['lat']
    lon  = ds['lon']
    nt   = np.size(time)
    nmem = np.size(mem)
    nlat = np.size(lat)
    nlon = np.size(lon)

    for iv in range(1,4):# which variable, change this!
        vname           = vname_list[iv]
        vname_long      = vname_long_list[iv]

        if vname !='q':
            nlev_org = 6 # for air_temperature and wind
            level_indx = np.array([1,2,3,5,6,7]) # for air temperautre and wind
        else:
            nlev_org = 5 # for specific humidity
            level_indx = np.array([1,2,4,5,7]) # for specific humidity

        for imem in range(0,nmem):#nmem-1): #(0,nmem)
            mem_str = f"{imem+1:02d}"
            print("mem:",mem_str)
            V_list = []
            for ilev in range(0, nlev_org):
                print(ilev)
                V_lev = ds[vname_long+'_'+str(level_indx[ilev])][imem,:,:,:] #[imem, 0:nt_small,:,:]
                V_list.append(V_lev)
            
            # Concatenate along a new 'level' dimension
            V = xr.concat(V_list, dim='level')

            # Optionally assign level coordinates based on level_indx
            V = V.assign_coords(level=level_indx)
            print(np.shape(V))

            file_out = vname+"_level_"+expname+"_mem"+mem_str+".nc"

            # Check if file exists and delete it if so
            if os.path.exists(dir_in_sub + file_out):
                os.remove(dir_in_sub + file_out)


            # Resave data

            ds_out = xr.Dataset({
                vname_long_list[iv]:(['level','time','lat','lon'],V.data)},
                coords={
                    "level":level_indx,
                    "time":time,
                    "lat":lat,
                    "lon":lon,
                }
            )
            ds_out.to_netcdf(dir_out + file_out)
