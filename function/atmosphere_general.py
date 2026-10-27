###################
# This file documents the commonly used function for atmospheric science
# for example, vertical integration, calculate column relative humidity
# 2025.1.28
# Mu-Ting Chien
###############################

import numpy as np

######################################
def calculate_vorticity(U, V, dx, dy):
    """
    Calculate the vertical vorticity using the U and V wind components.

    Parameters:
    U (numpy.ndarray): U wind component (time, lat, lon).
    V (numpy.ndarray): V wind component (time, lat, lon).
    dx (float): Grid spacing in the x-direction.
    dy (float): Grid spacing in the y-direction.

    Returns:
    numpy.ndarray: Vorticity field.

    ###############
    # CAUTION!!!!
    ###############
    # V_org has to normalize by latitude before input to this function. This is because dx in this function does not account for different length in different latitude
    
    # V = V_org/np.cos(lat*2*np.pi/360)
    """

    dvdx = np.gradient(V, dx, axis=2)  # Calculate dV/dx
    dudy = np.gradient(U, dy, axis=1)  # Calculate dU/dy

    vorticity = dvdx - dudy

    return vorticity


########################################
def z_integrate_4d(x,plev): #(t,lev,lat,lon)
    # Note that plev should be in Pa
    
    g = 9.8 #m/s^2
    nt   = np.size(x,0)
    nlev = np.size(x,1)
    nlat = np.size(x,2)
    nlon = np.size(x,3)
    Pmat = np.tile(plev,[nlat, nlon,1]) #equivalent to matlab: repmat
    Pmat = np.transpose(Pmat, (2, 0, 1)) # equivalent to matlab: Pmat = permute(Pmat,[2,3,1]);
    dp = np.zeros([nlev-1, nlat, nlon])

    for ii in range(0,nlev-1):
        dp[ii,:,:] = Pmat[ii+1,:,:]-Pmat[ii,:,:]

    x_mid = np.zeros([nt,nlev-1,nlat, nlon])
    for ii in range(0,nlev-1): #lev
        x_mid[:,ii,:,:] = (x[:,ii+1,:,:]+x[:,ii,:,:])/2 # Takes the midpoint of each pressure level

    # Vertical Integration
    xdP = np.zeros([nt,nlev-1,nlat,nlon])
    for tt in range(0,nt):
        temp1 = x_mid[tt,:,:,:]
        xdP[tt,:,:,:] = temp1*dp  # Pa
        del temp1
    Col_x = (1/g)*np.squeeze(np.sum(xdP,1))
    if dp[0,0,0]<0:
        Col_x = -Col_x
        
    return Col_x



########################################
def z_integrate_3d(x,plev): #(t,lev,lon)
    # Note that plev should be in Pa

    g = 9.8 #m/s^2
    nt   = np.size(x,0)
    nlev = np.size(x,1)
    nlon = np.size(x,2)
    Pmat = np.tile(plev,[nlon,1]) #equivalent to matlab: repmat
    Pmat = np.transpose(Pmat, (1, 0)) # equivalent to matlab: Pmat = permute(Pmat,[2,3,1]);
    dp = np.zeros([nlev-1, nlon])

    for ii in range(0,nlev-1):
        dp[ii,:] = Pmat[ii+1,:]-Pmat[ii,:]

    x_mid = np.zeros([nt,nlev-1,nlon])
    for ii in range(0,nlev-1): #lev
        x_mid[:,ii,:] = (x[:,ii+1,:]+x[:,ii,:])/2 # Takes the midpoint of each pressure level

    # Vertical Integration
    xdP = np.zeros([nt,nlev-1,nlon])
    for tt in range(0,nt):
        temp1 = x_mid[tt,:,:]
        xdP[tt,:,:] = temp1*dp  # Pa
        del temp1
    Col_x = (1/g)*np.squeeze(np.sum(xdP,1))
    if dp[0,0]<0:
        Col_x = -Col_x

    return Col_x

######################################
def cc_equation(T, plev): # T in K
    
    # Define constant
    epsilon = 0.622
    
    # Check dimension
    if np.size(np.shape(T))==4: # If 4D variable
        nt = np.size(T, 0)
        nlat = np.size(T, 2)
        nlon = np.size(T, 3)
        plev_large = np.tile(plev, (nt, nlat, nlon, 1))
        plev_large = np.transpose(plev_large, (0, 3, 1, 2))
    elif np.size(np.shape(T))==3: # If 3D variable
        nt = np.size(T,0)
        nlon = np.size(T,2)
        plev_large = np.tile(plev, (nt,nlon,1))
        plev_large = np.transpose(plev_large, (0,2,1))
    elif np.size(np.shape(T))==2: # If 2D variable
        nt = np.size(T,0)
        plev_large = np.tile(plev, (nt, 1))
    elif np.size(np.shape(T))==1: # If 1D variable
        plev_large = plev
    
    e = 6.1094*np.exp( 17.625*(T-273.15)/(T-273.15+243.04) ) # hPa
    qvs = epsilon*e/plev_large
    
    return e, qvs



#############################################
def calc_saturation_fraction(T, qv, plev):
    
    # Calculate saturated vapor pressure 
    e, qvs = cc_equation(T, plev)  #CC equation, hPa
    del e
    
    # Calculate saturation fraction
    if np.size(np.shape(T))==4:
        col_qv  = z_integrate_4d(qv, plev*100)
        col_qvs = z_integrate_4d(qvs, plev*100)
    elif np.size(np.shape(T))==3:
        col_qv  = z_integrate_3d(qv, plev*100)
        col_qvs = z_integrate_3d(qvs, plev*100)
    else:
        print('T is not 3D or 4D, fix the function to include 2D and 1D!')
    sf = col_qv/col_qvs # fraction
    
    return sf, col_qv, col_qvs



def genDispersionCurves(nWaveType=6, nPlanetaryWave=50, rlat=0, Ahe=[90, 25, 8]):
    """
    Function to derive the shallow water dispersion curves. Closely follows NCL version.

    input:
        nWaveType : integer, number of wave types to do
        nPlanetaryWave: integer
        rlat: latitude in radians (just one latitude, usually 0.0)
        Ahe: [50.,25.,12.] equivalent depths
              ==> defines parameter: nEquivDepth ; integer, number of equivalent depths to do == len(Ahe)

    returns: tuple of size 2
        Afreq: Frequency, shape is (nWaveType, nEquivDepth, nPlanetaryWave)
        Apzwn: Zonal savenumber, shape is (nWaveType, nEquivDepth, nPlanetaryWave)

    notes:
        The outputs contain both symmetric and antisymmetric waves. In the case of
        nWaveType == 6:
        0,1,2 are (ASYMMETRIC) "MRG", "IG", "EIG" (mixed rossby gravity, inertial gravity, equatorial inertial gravity)
        3,4,5 are (SYMMETRIC) "Kelvin", "ER", "IG" (Kelvin, equatorial rossby, inertial gravity)
    """
    nEquivDepth = len(Ahe) # this was an input originally, but I don't know why.
    pi    = np.pi
    radius = 6.37122e06    # [m]   average radius of earth
    g     = 9.80665        # [m/s] gravity at 45 deg lat used by the WMO
    omega = 7.292e-05      # [1/s] earth's angular vel
    # U     = 0.0   # NOT USED, so Commented
    # Un    = 0.0   # since Un = U*T/L  # NOT USED, so Commented
    ll    = 2.*pi*radius*np.cos(np.abs(rlat))
    Beta  = 2.*omega*np.cos(np.abs(rlat))/radius
    fillval = 1e20

    # NOTE: original code used a variable called del,
    #       I just replace that with `dell` because `del` is a python keyword.

    # Initialize the output arrays
    Afreq = np.empty((nWaveType, nEquivDepth, nPlanetaryWave))
    Apzwn = np.empty((nWaveType, nEquivDepth, nPlanetaryWave))

    for ww in range(1, nWaveType+1):
        for ed, he in enumerate(Ahe):
            # this loops through the specified equivalent depths
            # ed provides index to fill in output array, while
            # he is the current equivalent depth
            # T = 1./np.sqrt(Beta)*(g*he)**(0.25) This is close to pre-factor of the dispersion relation, but is not used.
            c = np.sqrt(g * he)  # phase speed
            L = np.sqrt(c/Beta)  # was: (g*he)**(0.25)/np.sqrt(Beta), this is Rossby radius of deformation

            for wn in range(1, nPlanetaryWave+1):
                s  = -20.*(wn-1)*2./(nPlanetaryWave-1) + 20.
                k  = 2.0 * pi * s / ll
                kn = k * L

                # Anti-symmetric curves
                if (ww == 1):       # MRG wave
                    if (k < 0):
                        dell  = np.sqrt(1.0 + (4.0 * Beta)/(k**2 * c))
                        deif = k * c * (0.5 - 0.5 * dell)

                    if (k == 0):
                        deif = np.sqrt(c * Beta)

                    if (k > 0):
                        deif = fillval


                if (ww == 2):       # n=0 IG wave
                    if (k < 0):
                        deif = fillval

                    if (k == 0):
                        deif = np.sqrt( c * Beta)

                    if (k > 0):
                        dell  = np.sqrt(1.+(4.0*Beta)/(k**2 * c))
                        deif = k * c * (0.5 + 0.5 * dell)


                if (ww == 3):       # n=2 IG wave
                    n=2.
                    dell  = (Beta*c)
                    deif = np.sqrt((2.*n+1.)*dell + (g*he) * k**2)
                    # do some corrections to the above calculated frequency.......
                    for i in range(1,5+1):
                        deif = np.sqrt((2.*n+1.)*dell + (g*he) * k**2 + g*he*Beta*k/deif)


                # symmetric curves
                if (ww == 4):       # n=1 ER wave
                    n=1.
                    if (k < 0.0):
                        dell  = (Beta/c)*(2.*n+1.)
                        deif = -Beta*k/(k**2 + dell)
                    else:
                        deif = fillval

                if (ww == 5):       # Kelvin wave
                    deif = k*c

                if (ww == 6):       # n=1 IG wave
                    n=1.
                    dell  = (Beta*c)
                    deif = np.sqrt((2. * n+1.) * dell + (g*he)*k**2)
                    # do some corrections to the above calculated frequency
                    for i in range(1,5+1):
                        deif = np.sqrt((2.*n+1.)*dell + (g*he)*k**2 + g*he*Beta*k/deif)

                eif  = deif  # + k*U since  U=0.0
                P    = 2.*pi/(eif*24.*60.*60.)  #  => PERIOD
                # dps  = deif/k  # Does not seem to be used.
                # R    = L #<-- this seemed unnecessary, I just changed R to L in Rdeg
                # Rdeg = (180.*L)/(pi*6.37e6) # And it doesn't get used.

                Apzwn[ww-1,ed-1,wn-1] = s
                if (deif != fillval):
                    # P = 2.*pi/(eif*24.*60.*60.) # not sure why we would re-calculate now
                    Afreq[ww-1,ed-1,wn-1] = 1./P
                else:
                    Afreq[ww-1,ed-1,wn-1] = fillval
    return  Afreq, Apzwn
