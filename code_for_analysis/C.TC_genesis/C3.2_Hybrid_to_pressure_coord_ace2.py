#########################
# Goal of this code is to transform the original data in hybrid coordinate (original output of ACE2) to pressure coordinate
# so that we can further analyze
# This code is modified from Hybrid_to_pressure_coord.py (for ACE1)
# 2024.12.11
# Mu-Ting Chien
##########################

# For this code
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from numpy import dtype
import os

# For function
from xgcm import Grid

# Load constant
p_target = np.arange(100,1100,100) # This is the target pressure level after conversion

# Set experiment name and directory
DIR = '/barnes-engr-scratch1/c832572266/'
fig_dir = DIR + 'figure/ace2_fig/hybrid-to-pressure-transform/'
os.makedirs(fig_dir,exist_ok=True) 
expname_list       = list(['ace2'])
vname_list         = list(['u','v','T','q'])
vname_long_list    = list(['eastward_wind','northward_wind','air_temperature','specific_total_water'])
vname_figname_list = list(['Zonal_wind','Meridional_wind','Temperature','Specific_total_water'])
unit_list          = list(['m/s','m/s','K','kg/kg'])

for icase in range(0,1): # ace2

    expname       = expname_list[icase]
    print(expname)

    #
    dir_in_output  = DIR + 'data_output/'+expname+'/ace2_output/'
    dir_in_forcing = DIR + 'data_output/'+expname+'/ace2_forcing/'

    # Name of the sub directory for the data named "filename"
    if icase == 0:
        dir_sub_list = list(['yr1','yr2-5','yr6-10'])
    nsub = np.size(dir_sub_list)

    for isub in range(0,nsub):
        print(dir_sub_list[isub])
        dir_in_sub = dir_in_output + dir_sub_list[isub]+'/'
        dir_out    = dir_in_sub+'pressure_coordinate/'
        os.makedirs(dir_out, exist_ok=True)

        # Load coefficient ak and bk in hybrid coordinate (copied from Table 5 in Watts_Mayer et al. 2024)
        ak = np.array([0, 5119.9, 13881.3, 19343.5, 20087.1,\
                       15596.7, 8880.45, 3057.27, 0])
        bk = np.array([0, 0, 0.00537781, 0.0597284, 0.203491,\
                       0.438391, 0.680643, 0.873929, 1])

        # load dimension info from original file 
        filename = 'autoregressive_predictions.nc'
        ds = xr.open_dataset(dir_in_sub + filename)
        time = ds['time']
        mem  = ds['sample']
        lat  = ds['lat']
        lon  = ds['lon']
        nt   = np.size(time)
        nmem = np.size(mem)
        nlat = np.size(lat)
        nlon = np.size(lon)

        nt_small = nt # or 10 (10 is for testing only)
        time = time[:nt_small]


        # Load variable saved from Reorgnaize_variable_before_coord-transform.py (this variable will be transformed into pressure coordinate)
        for iv in range(1, np.size(vname_list)): # (Change this! Which variable.)
            
            vname         = vname_list[iv]
            vname_long    = vname_long_list[iv]
            vname_figname = vname_figname_list[iv]
            unit          = unit_list[iv]
            print('vname:',vname)
            
            #########################
            # Only select ak and bk which contains the level that actual data exists
            ###########################
            # Note that this is because my ACE simulation has some missing data in some levels for different variables
            if vname != 'q': # 'T' or 'u' or 'v'
                level_indx_correct = np.array([1,2,3,5,6,7]) # This is the level with data, level 0,4 are missing
            else:
                level_indx_correct = np.array([1,2,4,5,7]) # level 0, 3, 6 are missing
            ############################################
            nlev_org = np.size(level_indx_correct)
            ak_extracted = ak[level_indx_correct]
            bk_extracted = bk[level_indx_correct]
            print('Finish preprocessing, start hybrid-to-pressure conversion:')

            # Start converting:
            for imem in range(0,nmem):

                mem_str = f"{imem+1:02d}"
                print('Mem:',mem_str)

                # Load surface pressure
                PS = ds['PRESsfc'][imem,0:nt_small,:,:]

                # Use surface pressure to transform the real pressure
                p = np.empty([nt_small, nlev_org, nlat, nlon])
                for ilev in range(0,nlev_org):
                    #print(ilev, level_indx_correct[ilev])
                    p[:,ilev,:,:] = ak_extracted[ilev]/100 + bk_extracted[ilev] * PS/100 # Unit is hPa
                
                ds2 = xr.open_dataset(dir_in_sub + '3D_field/'+vname+"_level_"+expname+"_mem"+mem_str+".nc")
                level_indx = ds2['level']

                # Convert p from numpy array into xarray data array
                time = ds2['time'][0:nt_small]
                lat = ds2['lat']
                lon = ds2['lon']
                level = ds2['level']
                coords = {
                    'time':time,
                    'level':level_indx,
                    'lat':lat,
                    'lon':lon,    
                }
                p_xr = xr.DataArray(p, coords=coords, dims=['time','level','lat','lon'])

                ####################################################
                # Transform variable from hybrid to pressure coordinate
                ds2 = ds2.assign({'p': np.log(p_xr)})
                grid = Grid(ds2, coords={'Z': {'center': 'level'}}, periodic=False)
                    
                dsout = xr.Dataset(coords={"plev": ("plev", p_target),
                                            "time": ("time", ds2.time.data),
                                            "lat": ("lat", ds2.lat.data),
                                            "lon": ("lon", ds2.lon.data)})
                ds2 = ds2.drop_vars(['time','lat','lon','level']) # To avoid these variables also go through the coord-transformation

                for var in ds2.data_vars:
                    # Removing time dimension by taking mean along 'time' for target_data
                    target_data_no_time = ds2.p.mean(dim='time')
                        
                    # Applying transformation with time-independent target_data
                    varout = grid.transform(
                        ds2[var],
                        'Z',
                        np.log(p_target),
                        target_data=target_data_no_time
                    )
                    varout = varout.rename({"p": "plev"})
                    varout = varout.assign_coords({'plev': p_target})

                    dsout = dsout.assign({var: varout})
                    #print('Varout.sizes:',varout.sizes)

                # Save data
                file_out = 'P-coord_'+expname+'_'+vname+'_mem'+mem_str+'.nc'
                dsout.to_netcdf(dir_out + file_out, mode='w')

                if imem == 0:
                    # Plot result of coordinate transform
                    figname = 'Hybrid_to_pressure_transform_'+vname+'_Mem01_'+expname+'.png'
                    Torg = ds2[vname_long][:,0,90,0] # Choose time = 0 , equator, lon = 0
                    Tnew = dsout[vname_long][0,90,0,:]
                    plev = dsout['plev']
                    plt.plot(Torg, p_xr[0,:,90,0],'k-o')
                    plt.plot(Tnew, plev, 'r-o')
                    plt.legend(['Original hybrid coordinate','Pressure coordinate'])
                    plt.xlabel(vname_figname+' ('+unit+')')
                    plt.ylabel('Pressure (hPa)')
                    plt.title(expname+', Mem'+mem_str+', @(0E, 0.49N), t=0')
                    plt.yticks(np.arange(100,1100,100))
                    plt.gca().invert_yaxis()
                    #plt.ylim([700,900])
                    plt.savefig(fig_dir + figname)
                    plt.show()
            print('Finish coordinate conversion!')


