##############################
# Goal of this code is to calculate and save TCGI from ACE2 (repeating 2001-2010)
# This code is modified from the companion code for ace2 (Calc_TCGI_ace2.py)
# 2025.2.10
# Mu-Ting Chien
############################

'''
# Calculate TCGI from Carmargo et al. 2014 (Highresmip)
mu = exp(b + b_absvor * absvor + b_colrh * colrh + b_relsst * relsst + b_shear * shear + log cos(lat) )
- mu: expected number of TCs per month in a 40-year period
- b: constanat
- absvor: clipped absolute vorticity at 850 hPa (1/s): min(actual abs vorticity, 3.7*10**(-5))
- colrh: column relative humidity (%)
- relsst: relative sst (C): difference between sst at each grid point and tropical mean SST
- shear: vertical wind shear between 850 and 200 hPa (m/s)

From Tippett et al. (2011), using reanalysis and the observed TCs data, the constants are obtained:
- b = -11.96
- b_absvor = 1.12
- b_colrh = 0.12
- b_relsst = 0.46
- b_shear = -0.13
'''

import numpy as np
import xarray as xr
import metpy as mp
import matplotlib.pyplot as plt
import sys
sys.path.append('/barnes-engr-scratch1/c832572266/Function/')
from scipy import signal
import mjo_mean_state_diagnostics_uw as MJO
import atmosphere_general as ATM
from netCDF4 import Dataset
import numpy as np
from numpy import dtype

# Load data (ACE2, 2001-2010)
DIR   = '/barnes-engr-scratch1/c832572266/data_output/ace2/ace2_output/repeat_2001-2010/'
dir_in_era5_TCGI = '/barnes-engr-scratch1/mchien/data_output/ERA5_TCGI/'
#dir_out = DIR + '10yr/'
vname = list(['u','v','q','T'])
nv    = np.size(vname)
nt = 365*10*4 -1

# Interpolate the data into 2.5 deg * 2.5 deg (match ERA5)
lat_2p5deg = np.arange(-30, 32.5, 2.5)
lon_2p5deg = np.arange(0, 360, 2.5)

sub_dir = list(['2001','2002','2003','2004','2005','2006','2007','2008','2009','2010'])
nsub    = np.size(sub_dir)

# Find indices to find the correct SST (only select the correct year)
nday_yr = np.array([365, 365, 365, 366, 365, 365, 365, 366, 365, 365])
for i in range(0, nsub):
    if i == 0:
        itmin_sst = np.zeros([10])
        itmax_sst = np.zeros([10])
    if i == 0:
        itmin_sst[i] = 0
    else:
        itmin_sst[i] = itmax_sst[i-1]
    itmax_sst[i] = itmin_sst[i]+nday_yr[i]*4
    #print(itmin_sst[i], itmax_sst[i])

# Which year of the ensemble runs to diagnose (each has 10 years)
for i in range(0, nsub):#0, nsub):
    print(sub_dir[i])

    itmin_sst_int = int(itmin_sst[i])
    itmax_sst_int = int(itmax_sst[i])

    dir_out = DIR + sub_dir[i] +'/'
    ############################
    # 1. Load data: u850, u200, v850, v200, q, T
    #############################
    tname = list(['yr1-5','yr6-10'])
    itmin = 0
    dir_in = DIR + sub_dir[i] + '/pressure_coordinate/'
    for it in range(0, 2): # first 5 years or later 5 years

        if i!=0:
            ds     = xr.open_dataset(dir_in + 'P-coord_ace2_u_'+sub_dir[i]+'_'+tname[it]+'.nc')
        else:
            ds     = xr.open_dataset(dir_in + 'P-coord_ace2_u_'+sub_dir[i]+'.nc')
        ds1    = ds.sel(plev=slice(800, 900), lat=slice(-31, 31))
        ds1    = ds1.interp(lat=lat_2p5deg, lon=lon_2p5deg, method='linear')
        lat    = ds1['lat']
        lon    = ds1['lon']
        nlat   = np.size(lat)
        nlon   = np.size(lon)
        if i!=0:
            time_small = ds['time']
            nt_small   = np.size(time_small)

        if it == 0:
            u850 = np.empty([nt, nlat, nlon])
            u200 = np.empty([nt, nlat, nlon])
            v850 = np.empty([nt, nlat, nlon])
            v200 = np.empty([nt, nlat, nlon])

        # Load u (select 850 and 200 hPa)
        if i!=0:
            u850[itmin : itmin+nt_small, :, :] = ds1['eastward_wind'].mean(dim="plev")
        else:
            u850                               = ds1['eastward_wind'].mean(dim="plev")
        ds3                                    = ds.sel(plev=200, lat=slice(-31, 31), drop=True)
        ds3                                    = ds3.interp(lat=lat_2p5deg, lon=lon_2p5deg, method='linear')
        if i!=0:
            u200[itmin : itmin+nt_small, :, :] = ds3['eastward_wind']
        else:
            u200                               = ds3['eastward_wind']

        # Load v (select 850 and 200 hPa)
        ds                                   = xr.open_dataset(dir_in + 'P-coord_ace2_v_'+sub_dir[i]+'_'+tname[it]+'.nc')
        ds1                                  = ds.sel(plev=slice(800, 900), lat=slice(-31, 31))
        ds1                                  = ds1.interp(lat=lat_2p5deg, lon=lon_2p5deg, method='linear')
        if i == 0:
            time_small = ds['time']
            nt_small   = np.size(time_small)        
        v850[itmin : itmin+nt_small, :, :]   = ds1['northward_wind'].mean(dim="plev")
        ds3                                  = ds.sel(plev=200, lat=slice(-31, 31), drop=True)
        ds3                                  = ds3.interp(lat=lat_2p5deg, lon=lon_2p5deg, method='linear')
        v200[itmin : itmin+nt_small, :, :]   = ds3['northward_wind']

        # Load q
        ds     = xr.open_dataset(dir_in + 'P-coord_ace2_q_'+sub_dir[i]+'_'+tname[it]+'.nc')
        ds     = ds.sel(lat=slice(-31, 31))
        ds     = ds.interp(lat=lat_2p5deg, lon=lon_2p5deg, method='linear')
        plev   = ds['plev'][:-1]
        nlev   = np.size(plev)

        if it == 0:
            q = np.empty([nt, nlev, nlat, nlon])
            T = np.empty([nt, nlev, nlat, nlon])
            time = np.empty([nt])
        tmp  = ds['specific_total_water'][:,:,:,:-1] # do not use 1000 hPa because there are nans
        q[itmin : itmin+nt_small, :,:,:] = tmp.transpose('time','plev','lat','lon')

        # Load T
        ds     = xr.open_dataset(dir_in + 'P-coord_ace2_T_'+sub_dir[i]+'_'+tname[it]+'.nc')
        ds     = ds.sel(lat=slice(-31, 31))
        ds     = ds.interp(lat=lat_2p5deg, lon=lon_2p5deg, method='linear')
        tmp    = ds['air_temperature'][:,:,:,:-1] # do not use 1000 hPa because there are nans
        T[itmin : itmin+nt_small, :,:,:]  = tmp.transpose('time','plev','lat','lon')

        # Load time
        ds     = Dataset(DIR + '/Time_ace2_2001-2010.nc', 'r')
        time[itmin : itmin+nt_small] = ds.variables['time'][itmin : itmin+nt_small]

        itmin = itmin + nt_small

    print('Finish loading input data')

    ########################################
    # 2. Calculate each variable: vorticity 
    #######################################
    lat_repmat = np.tile(lat, (nt, nlon, 1))
    lat_repmat = np.transpose(lat_repmat, (0, 2, 1))

    # a. Normalize by latitude
    v850_norm = v850/np.cos(lat_repmat*2*np.pi/360)

    # b. Assign horizontal resolution of the data (1 deg * 1 deg) 
    #     Note: This is different from ERA5 which is 2.5 deg * 2.5 deg
    dx_ace2 = 1
    dy_ace2 = 1

    # c. Normalize by latitude
    v850_norm = v850/np.cos(lat_repmat*2*np.pi/360)

    # d. Calculate relative vorticity
    dx = 111*1000*dx_ace2
    dy = 111*1000*dy_ace2
    vor = ATM.calculate_vorticity(u850, v850_norm, dx, dy)

    # e. Calculate absolute vorticity
    omega = 7.29 * 10**(-5) 
    absvor = vor + 2 * omega * np.sin(lat_repmat*2*np.pi/360)

    # f. Calculate the clipped vorticity
    # if the absolute vorticity < absvor_crit, use absolute vorticity
    # otherwise, use absvor_crit
    absvor_crit = 3.7*10**(-5)
    absvor_clipped = np.where(np.abs(absvor)<absvor_crit, np.abs(absvor), absvor_crit) 
    print('Finish calc abs vorticity')

    ####################################
    # 3. Calculate vertical wind shear
    ####################################
    shear = ( (u200-u850)**2 + (v200-v850)**2 )**0.5
    print('Finish calc shear')

    ########################################
    # 4. Calculate column relative humidity
    ########################################
    col_rh, col_qv, col_qvs = ATM.calc_saturation_fraction(T, q, plev)
    col_rh = col_rh*100 # Change unit of col_rh into %
    print('Finish calc column rh')

    #######################################
    # 5. Load relative SST calcualted from ERA5.py, start here!
    ######################################
    if i!=9: # for none 2010
        fname = list(['6h','monthly'])
        vname_in_file = list(['','_monthly'])
        for k in range(0, 1): # 6hourly or monthly
            filename = 'TCGI_ERA5_2001_2010_2.5deg_'+fname[k]+'.nc'
            ds = xr.open_dataset(dir_in_era5_TCGI + filename)
            relsst = ds['relsst'][itmin_sst_int:itmax_sst_int, :, :]
            sst    = ds['sst'][itmin_sst_int:itmax_sst_int, :, :]

    else: # for 2010 (need to recalculate relsst because the above data miss the last day (2010.12.31))
        dir_forcing = "/barnes-engr-scratch1/mchien/data_output/ace2/ace2_forcing/"
        ds          = xr.open_dataset(dir_forcing + 'forcing_2010.nc')
        ds          = ds.sel(latitude=slice(-31, 31))
        ds          = ds.interp(latitude=lat, longitude=lon, method='linear') # coarsen to 2.5 deg, change this if using other data!
        land_frac   = ds['land_fraction'][:,:] # I use this because this is only (lat, lon), no time dependence
        # Make sure no nan at lon=0
        land_frac[:,0] = (land_frac[:,1].values + land_frac[:,-1].values)/2
        #plt.imshow(land_frac[::-1, :])

        # Load SST
        TS = ds['surface_temperature'][:,:,:]
        nt_small = np.size(TS,0)

        # b. stack land_frac with time
        land_frac_time = np.tile(land_frac, (nt_small, 1, 1))
        #print(np.shape(land_frac_time))

        sst = np.empty([nt_small, nlat, nlon])
        sst[:] = np.nan

        # Make sure no nan at lon = 0 (get value at lon=0 from lon=2.5 and lon=357.5)
        TS[:,:,0] = (TS[:,:,1].values + TS[:,:,-1].values)/2

        # Mask out land for TS to get SST
        land_frac_time_tmp = np.tile(land_frac, (nt_small, 1, 1))
        sst[:, :, :] = np.where(land_frac_time_tmp>0.5, np.nan, TS) # at least land_frac>0.5, then we define as land, otherwise, it is ocean

        # Find 20S-20N averaged SST
        lat_tmp       = ds['latitude']
        lat_tmp       = lat_tmp.values
        dlat          = lat_tmp-(-20)
        ilatmin       = np.argwhere(np.abs(dlat)==np.min(np.abs(dlat))).squeeze()
        dlat          = lat_tmp-(20)
        ilatmax       = np.argwhere(np.abs(dlat)==np.min(np.abs(dlat))).squeeze()

        # Calculate relative SST (sst_local - sst_tropical_mean)
        sst_trop_mean  = np.nanmean( np.nanmean( sst[:,ilatmin:ilatmax+1,:], 2), 0) 
        sst_trop_mean  = MJO.mer_ave( sst_trop_mean, lat_tmp[ilatmin:ilatmax+1], 0)
        relsst         = sst - sst_trop_mean

    # Repeating 10 times (10 years)
    relsst = np.tile(relsst, (10, 1, 1))
    sst    = np.tile(sst, (10, 1, 1))

    # remove the last index because only 14599
    relsst = relsst[:nt, :, :]
    sst    = sst[:nt, :, :]

    if (np.shape(col_rh)!=np.shape(sst)) or (np.shape(col_rh)!=np.shape(relsst)):
        print('Shape mismatch in SST!')

    ##########################################################
    # 6. Calculate TCGI based on (1)6-hourly and (2)monthly data
    #########################################################
    # Use constants derived from Tippet et al. 2011
    b        = -11.96
    b_absvor = 1.12
    b_colrh  = 0.12
    b_relsst = 0.46
    b_shear  = -0.13

    # A. Calculate TCGI based on 6-hourly data
    # Unit of TCGI: number of TC per month in 40-year period
    TCGI = np.exp( b + b_absvor*absvor_clipped[:,:,:]*10**5 + b_colrh*col_rh[:,:,:] +\
                    b_relsst*relsst[:,:,:] + b_shear*shear[:,:,:] +np.log( np.cos(lat_repmat[:,:,:]/360*2*np.pi) ) )

    # B. Calculate TCGI based on monthly mean
    # Calculate monthly mean of each variable
    day_per_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    day_per_month_leap = 29
    nyr = 10

    ist = 0
    for iyr in range(0, nyr):
        for imon in range(0, 12):
            #print('Mon:', imon+1)
            if (iyr==3 and imon==1) or (iyr==7 and imon==1): # 2004 and 2008, Feb
                day_per_month_tmp = day_per_month_leap*4
            else:
                day_per_month_tmp = day_per_month[imon]*4

            if imon == 0 and iyr == 0:
                absvor_clipped_monthly = np.empty([12, nyr, nlat, nlon])
                absvor_monthly         = np.empty([12, nyr, nlat, nlon])
                col_rh_monthly         = np.empty([12, nyr, nlat, nlon])
                relsst_monthly         = np.empty([12, nyr, nlat, nlon])
                sst_monthly            = np.empty([12, nyr, nlat, nlon])
                shear_monthly          = np.empty([12, nyr, nlat, nlon])

            iend = ist + day_per_month_tmp
            #print(ist, iend)
            
            absvor_clipped_monthly[imon,iyr,:,:] = np.mean(absvor_clipped[ist: iend, :, :], 0)
            absvor_monthly[imon,iyr,:,:]      = np.mean(absvor[ist: iend, :, :],         0)
            col_rh_monthly[imon,iyr,:,:]      = np.mean(col_rh[ist: iend, :, :],         0)
            relsst_monthly[imon,iyr,:,:]      = np.mean(relsst[ist: iend, :, :],         0)
            sst_monthly[imon,iyr,:,:]         = np.mean(sst[ist: iend, :, :],            0)
            shear_monthly[imon,iyr,:,:]       = np.mean(shear[ist: iend, :, :],          0)
            ist = ist + day_per_month_tmp

    # Calculate monthly TCGI
    lat_repmat_monthly = np.tile(lat, (12, nyr, nlon, 1)).transpose((0,1,3,2))
    TCGI_monthly = np.exp( b + b_absvor*absvor_clipped_monthly*10**5 + b_colrh*col_rh_monthly + b_relsst*relsst_monthly + \
                        b_shear*shear_monthly +np.log( np.cos(lat_repmat_monthly/360*2*np.pi)) )

    year = np.arange(2001, 2011)
    month = np.arange(1, 13)
    print('Finish calc TCGI')

    ###################################
    # 7. Save each variable and TCGI: absvor, absvor_clipped, shear, relsst, col_rh, TCGI_6h (6-hourly)
    ###################################
    # Save 6-hourly or monthly
    fname = list(['6h','monthly'])
    vname_in_file = list(['','_monthly'])
    for k in range(0, 2): # 6hourly or monthly

        filename = 'TCGI_ACE2_'+sub_dir[i]+'_2.5deg_'+fname[k]+'.nc'
        data_out = Dataset(dir_out+filename, 'w', format='NETCDF4')

        # define axis size
        if k == 0:
            data_out.createDimension('time', nt)
        elif k == 1:
            data_out.createDimension('month', 12)
            data_out.createDimension('year', nyr)
        data_out.createDimension('lat', nlat)
        data_out.createDimension('lon', nlon)

        if k == 0:
            # define time
            time2 = data_out.createVariable('time' , dtype('int').char, ('time'))
            time2.units = 'hours since 1900-01-01 00:00:00'  # Replace with the appropriate units
            time2.long_name = 'start from 2001-01-01 00 h, 6-hourly'
            time2.calendar = 'gregorian'
            time2.axis = 't'
        else:
            # define month
            mon2 = data_out.createVariable('month' , dtype('int').char, ('month'))
            mon2.units = 'none'  # Replace with the appropriate units
            mon2.long_name = 'month'
            mon2.axis = 'mon'    
            # define year
            yr2 = data_out.createVariable('year' , dtype('int').char, ('year'))
            yr2.units = 'none'  # Replace with the appropriate units
            yr2.long_name = 'year'
            yr2.axis = 'yr'      

        # define lat
        lat2 = data_out.createVariable('lat' , dtype('float').char, ('lat'))
        lat2.units = 'deg'  # Replace with the appropriate units
        lat2.long_name = 'latitude'
        lat2.axis = 'y'

        # define lon
        lon2 = data_out.createVariable('lon' , dtype('float').char, ('lon'))
        lon2.units = 'deg'  # Replace with the appropriate units
        lon2.long_name = 'longitude'
        lon2.axis = 'x'

        vname_out = list(['absvor','absvor_clipped','shear','col_rh','relsst','sst','TCGI'])
        unit      = list(['1/s', '1/s', 'm/s', '%', 'K','K','#/month/40 years'])
        vname_long = list(['Absolute vorticity at 850 hPa = relative vorticity + planetary vorticity','Absolute vorticity (clipped) at 850 hPa: <=3.7*10**(-5)',\
                        'Vertical wind shear from 850 to 200 hPa','Column relative humidity','Relative sea surface temperature (anomaly compared to 20S-20N climatology)',\
                        'Sea surface temperature',\
                        'Number of TC genesis based on environmental conditions, coefficients obtained from Tippett et al. (2011), TCGI = exp(b + b_absvor * absvor + b_colrh * colrh + b_relsst * relsst + b_shear * shear + log cos(lat) ), \
                                b = -11.96, b_absvor = 1.12, b_colrh = 0.12, b_relsst = 0.46, b_shear = -0.13'])

        nv = np.size(vname_out)
        for iv in range(0, nv):
            if k == 0:
                V2 = data_out.createVariable(vname_out[iv] , dtype('float').char, ('time','lat','lon'))
            else:
                V2 = data_out.createVariable(vname_out[iv] , dtype('float').char, ('month','year','lat','lon'))
            V2.units = unit[iv]  # Replace with the appropriate units
            V2.long_name = vname_long[iv]
            tmp    = globals()[vname_out[iv]+vname_in_file[k]]
            V2[:]  = tmp[:]

        if k == 0:
            time2[:] = time[:]
        else:
            yr2[:] = year[:]
            mon2[:] = month[:]
        lon2[:]  = lon[:]
        lat2[:]  = lat[:]

    print('Finish saving output')