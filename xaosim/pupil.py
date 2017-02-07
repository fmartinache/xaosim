import numpy as np
import pdb
from scipy.ndimage import rotate

shift = np.fft.fftshift
fft   = np.fft.fft2
ifft  = np.fft.ifft2

dtor = np.pi / 180.0 # convert degrees to radians

# ==================================================================
def get_prad(fsz=512, ldim=8.0, wl=1.6e-6, pscale=10.0):
    ''' ----------------------------------------------------------
    returns the pixel size that, in a Fourier simulation,
    would result in the requested sampling requirements:

    - fsz    : size of the square Fourier array (default: 512x512)
    - ldim   : linear dimension in meters (default: 8.0)
    - wl     : wavelength in meters (default: 1.6e-6)
    - pscale : plate scale in mas / pixel (default: 10.0)
    ---------------------------------------------------------- '''
    ld = wl/ldim * 648e6 / np.pi / pscale # l/D in pixels
    return(fsz/ld/2)

# ==================================================================
def spectral_sampling(wl1, wl2, nl, wavenum=False):
    ''' ----------------------------------------------------------
    Returns an array of regularly sampled wavelength between 
    wl1 and wl2, optionally equally spaced in wave numbers.
    - wl1: shorter wavelength (in meters)
    - wl2: longer wavelength (in meters)
    - nl: number of spectral bins
    - wavenum: True if sampling equally spaced in wave numbers
    ---------------------------------------------------------- '''
    if (not wavenum):
        return (wl1 + (wl2-wl1) * np.arange(nl) / (nl-1))
    else:
        k1, k2 = 1./wl1, 1./wl2
        kk = k1 + (k2-k1) * np.arange(nl) / (nl-1)
        return (1./kk)

# ==================================================================
def hex_pup_coords(srad, nr):
    ''' ----------------------------------------------------------
    produces a list of x,y coordinates for a hex grid made of
    nr concentric rings, with a step srad
    ---------------------------------------------------------- '''
    rmax   = np.round(nr * srad)
    xs, ys = np.array(()), np.array(())

    for i in range(1-nr, nr, 1):
        for j in xrange(1-nr, nr, 1):
            x = srad * (i + 0.5 * j)
            y = j * np.sqrt(3)/2.*srad
            if (abs(i+j) < nr):
                xs = np.append(xs, x)
                ys = np.append(ys, y)

    xx, yy = xs.copy(), ys.copy()
    xs, ys = np.array(()), np.array(())

    for i in range(xx.size): 
        if (0.5*srad <= np.sqrt(xx[i]**2 + yy[i]**2) < rmax*0.97*np.sqrt(3)/2.):
            xs = np.append(xs, xx[i])
            ys = np.append(ys, yy[i])
    return(xs, ys)

# ==================================================================
def F_test_figure((ys, xs), ww):
    ''' ------------------------------------------------------------
    Returns an (ys x xs) size array with an uppercase F drawn inside
    The F letter overall size is 5x3 times the thickness of the line.

    To be used when lost in the woods, when it comes to orienting
    arrays and images, for instance after Fourier Transforms.

    The orientation convention: origin in the bottom left corner, to
    get the F to be upright.

    Parameters:
    ----------
    - (ys, xs): the dimensions of the array
    - ww      : the thickness of the line
    --------------------------------------------------------------------
    '''
    res = np.zeros((ys, xs))
    xx, yy = np.meshgrid(np.arange(xs)-xs/2+ww/2, np.arange(ys) - ys/2 + ww/2)
    res[(xx > -ww) * (xx <=    0) * (yy > -2*ww) * (yy <= 3*ww)] = 1.0
    res[(xx >=  0) * (xx <= 2*ww) * (yy >  2*ww) * (yy <= 3*ww)] = 1.0
    res[(xx >=  0) * (xx <=   ww) * (yy >     0) * (yy <=   ww)] = 1.0

    return(res)

# ==================================================================
def uniform_disk((ys, xs), radius):
    ''' ---------------------------------------------------------
    returns an (ys x xs) array with a uniform disk of radius "radius".
    ---------------------------------------------------------  '''
    xx,yy  = np.meshgrid(np.arange(xs)-xs/2, np.arange(ys)-ys/2)
    mydist = np.hypot(yy,xx)
    res = np.zeros_like(mydist)
    res[mydist <= radius] = 1.0
    return(res)

# ==================================================================
def four_spider_mask((ys, xs), pix_rad, pdiam, odiam=0.0, 
                     beta=45.0, thick=0.45, offset=0.0,
                     spiders=True, split=False):
    ''' ---------------------------------------------------------
    tool function called by other routines to generate specific
    pupil geometries. Although the result is scaled by pix_rad in 
    pixels, telescope specifics are provided in meters.

    Parameters:
    ----------
    - (ys, xs) : dimensions of the 2D array      (in pixels)
    - pix_rad  : radius of the circular aperture (in pixels)
    - pdiam    : diameter of the aperture        (in meters)
    - odiam    : diameter of the obstruction     (in meters)
    - beta     : angle of the spiders            (in degrees)
    - thick    : thickness of the spiders        (in meters)
    - offset   : spider intersect point distance (in meters)
    - spiders  : flag to true to include spiders (boolean)
    - split    : split the mask into four parts  (boolean)
    --------------------------------------------------------- '''

    beta    = beta * dtor # converted to radians
    ro      = odiam / pdiam
    xx,yy   = np.meshgrid(np.arange(xs)-xs/2, np.arange(ys)-ys/2)
    mydist  = np.hypot(yy,xx)

    thick  *= pix_rad / pdiam
    offset *= pix_rad / pdiam

    x0      = thick/(2 * np.sin(beta)) + offset 
    y0      = thick/(2 * np.cos(beta)) - offset * np.tan(beta)
    
    if spiders:
        # quadrants left - right
        a = ((xx >=  x0) * (np.abs(np.arctan(yy/(xx-x0+1e-8))) < beta))
        b = ((xx <= -x0) * (np.abs(np.arctan(yy/(xx+x0+1e-8))) < beta))
        # quadrants up - down
        c = ((yy >= 0.0) * (np.abs(np.arctan((yy-y0)/(xx+1e-8))) > beta))
        d = ((yy <  0.0) * (np.abs(np.arctan((yy+y0)/(xx+1e-8))) > beta))
        
    # pupil outer and inner edge
    e = (mydist < pix_rad)
    if odiam > 0.0:
        e *= (mydist > ro * pix_rad)

    if split:
        res = np.array([a*e, b*e, c*e, d*e])
        return(res)
    
    if spiders:
        return((a+b+c+d)*e)
    else:
        return(e)

# ======================================================================
def four_spider_grid_model(th1=0.0, th2=90.0, apdiam=8.0, odiam=2.3,
                           beta=51.75, thick=0.45, offset=0.0,
                           spiders=True, split=False):
    ''' -------------------------------------------------------------
    Tool that corresponds produces a discrete representation of the
    pupil produced by the four_spider_mask() routine.

    Returns an array (possibly split into quadrants) of coordinates
    of points contained within the aperture.
    ------------------------------------------------------------- '''
    return(0)
    
# ======================================================================
def HST((xs,ys), radius, spiders=True):
    ''' -------------------------------------------------------------
    Draws the Hubble Space Telescope pupil of given radius in a array

    Parameters:
    ----------

    - (xs, ys) : dimensions of the 2D array
    - radius   : outer radius of the aperture (in pixels)
    - spiders  : boolean (w/ or w/out spiders)
    -------------------------------------------------------------  '''
    # pupil description
    pdiam, odiam = 2.4, 0.792 # tel. and obst. diameters (meters)
    thick  = 0.20             # adopted spider thickness (meters)
    beta   = 45.0             # spider angle
    offset = 0.0
    return(four_spider_mask((ys, xs), radius, pdiam, odiam, 
                            beta=beta, thick=thick, offset=offset, 
                            spiders=spiders))

# ==================================================================
def subaru((n,m), radius, spiders=True):
    ''' ---------------------------------------------------------
    returns an array that draws the pupil of the Subaru Telescope
    at the center of an array of size (n,m) with radius "radius".

    Symbols and values used for the description come from the
    document sent to Axsys for the fabrication of the SCExAO
    PIAA lenses, circa 2009.
    --------------------------------------------------------- '''

    # Subaru pupil description
    # ------------------------
    pdiam, odiam = 7.92, 2.3  # tel. and obst. diameters (meters)
    thick  = 0.45             # adopted spider thickness (meters)
    offset = 1.278            # spider intersection offset (meters)
    beta   = 51.75            # spider angle beta

    return(four_spider_mask((m, n), radius, pdiam, odiam, 
                            beta, thick, offset, spiders))

# ==================================================================
def subaru_dbl_asym((xs, ys), radius, spiders=True, PA1=0.0, PA2=90.0,
                    thick1=0.15, thick2=0.15):
    ''' -------------------------------------------------------------
    Returns a pupil mask with *two* asymmetric arms for two distinct
    position angles.

    Parameters:
    ----------

    - (xs, ys) : dimensions of the 2D array
    - radius   : outer radius of the aperture
    - spiders  : boolean (w or without spiders)
    - PA1      : position angle of arm #1 (in degrees)
    - PA2      : position angle of arm #2 (in degrees)
    - thick1   : asymm. arm #1 thickness (% of aperture diameter)
    - thick2   : asymm. arm #2 thickness (% of aperture diameter)
    ------------------------------------------------------------- '''
    a = subaru_asym((xs, ys), radius, spiders=spiders, PA=PA1, thick=thick1)
    b = subaru_asym((xs, ys), radius, spiders=spiders, PA=PA2, thick=thick2)
    return(a*b)

# ==================================================================
def radial_arm((xs,ys), radius, PA=0.0, thick=0.15):
    ''' -------------------------------------------------------------
    Produces a pupil mask for an occulting radial arm for a given
    position angle and thickness.

    Parameters:
    - (xs, ys) : dimensions of the 2D array   (integer # of pixels)
    - radius   : outer radius of the aperture (integer # of pixels)
    - PA       : position angle of the arm    (degrees)
    - thick    : fraction of the ap. diameter (float)
    ------------------------------------------------------------- '''
    res = np.ones((ys, xs))
    ang = np.mod(PA, 360.0)
    xx,yy  = np.meshgrid(np.arange(xs)-xs/2, np.arange(ys)-ys/2)
    zone = (xx > 0) * (np.abs(yy) <= thick*radius)
    res[zone] = 0.0
    res = rotate(res, ang, order=0, reshape=False)
    return(res.astype('int'))

# ==================================================================
def subaru_asym((xs, ys), radius, spiders=True, PA=0.0, thick=0.15):
    ''' -------------------------------------------------------------
    Returns a pupil mask with an asymmetric arm that mostly follows
    the geometry of the original APF-WFS.

    Parameters:
    ----------

    - (xs, ys) : dimensions of the 2D array
    - radius   : outer radius of the aperture
    - spiders  : boolean (w or without spiders)
    - PA       : position angle of the arm (in degrees)
    - thick    : asymm. arm thickness (% of aperture diameter)
    ------------------------------------------------------------- '''

    #th = np.mod(PA, 360.0) * dtor # convert PA into radians
    #th0 = np.mod(th, np.pi)
    #xx,yy  = np.meshgrid(np.arange(xs)-xs/2, np.arange(ys)-ys/2)
    #h = thick * radius / np.abs(np.cos(th0))

    pup = subaru((xs,ys), radius, spiders)
    arm = radial_arm((xs,ys), radius, PA=PA, thick=thick)

    return(pup * arm)

# ======================================================================
def segmented(sz, prad, srad, gap=False):
    '''Returns a segmented pupil image, and a list of coordinates
    for each segment.
    '''
    nr   = 50 # number of rings within the pupil

    xs = np.array(())
    ys = np.array(())

    for i in range(1-nr, nr, 1):
        for j in xrange(1-nr, nr, 1):
            x = srad * (i + 0.5 * j)
            y = j * np.sqrt(3)/2.*srad
            if (abs(i+j) < nr):
                xs = np.append(xs, x)
                ys = np.append(ys, y)
    
    print ("%d" % (xs.size))
    xx, yy = xs.copy(), ys.copy()        # temporary copies
    xs, ys = np.array(()), np.array(())  # start from scratch again
    
    for i in xrange(xx.size):
        thisrad = np.sqrt(xx[i]**2 + yy[i]**2)
        #print(thisrad)
        if (thisrad < prad):
            xs = np.append(xs, xx[i]+sz/2)
            ys = np.append(ys, yy[i]+sz/2)

    pup = np.zeros((sz,sz))
    for i in xrange(xs.size):
        pup[xs[i], ys[i]] = 1.0

    print ("%d" % (xs.size))
    return(pup)

# ======================================================================
def golay9(sz, prad, hrad):
    ''' Returns a square "sz x sz" NR Golay 9 pupil model of radius 
    "prad" and hole diameter "hrad".
    '''
    xx,yy  = np.meshgrid(np.arange(sz)-sz/2, np.arange(sz)-sz/2)
    mydist = np.hypot(yy,xx)
    dstep  = prad / 3.5

    xs = np.array([])
    ys = np.array([])

    pup = np.zeros((sz,sz))
    for i in xrange(3):
        theta = 2.0 * i * np.pi / 3.0

        for k in xrange(1,3):
            xs = np.append(xs, (k+1) * dstep * np.cos(theta))
            ys = np.append(ys, (k+1) * dstep * np.sin(theta))

        xs = np.append(xs,  2 * dstep)
        ys = np.append(ys, -2 * dstep * np.sqrt(3) / 2)

        xs = np.append(xs,  -3 * dstep * np.sqrt(3) / 2)
        ys = np.append(ys, - dstep * np.sqrt(3) / 2)

        xs = np.append(xs, 0.5 * dstep * np.sqrt(3) / 2)
        ys = np.append(ys, 3 * dstep * np.sqrt(3) / 2)

    xs = np.cast['int'](np.round(xs))
    ys = np.cast['int'](np.round(ys))
        
    for i in xrange(xs.size):
        pup = np.roll(np.roll(pup, -xs[i], 0), -ys[i], 1)
        pup[mydist < hrad] = 1.0
        pup = np.roll(np.roll(pup,  xs[i], 0),  ys[i], 1)

    return(pup)


# ======================================================================
def hex_grid(sz, prad, srad, gap=False):
    '''Returns a segmented pupil image, and a list of coordinates
    for each segment.
    '''
    nr   = 50 # number of rings within the pupil

    xs = np.array(())
    ys = np.array(())

    for i in range(1-nr, nr, 1):
        for j in xrange(1-nr, nr, 1):
            x = srad * (i + 0.5 * j)
            y = j * np.sqrt(3)/2.*srad
            if (abs(i+j) < nr):
                xs = np.append(xs, x)
                ys = np.append(ys, y)
    
    print ("%d" % (xs.size))
    xx, yy = xs.copy(), ys.copy()        # temporary copies
    xs, ys = np.array(()), np.array(())  # start from scratch again
    
    for i in xrange(xx.size):
        thisrad = np.sqrt(xx[i]**2 + yy[i]**2)
        #print(thisrad)
        if (thisrad < prad):
            xs = np.append(xs, xx[i]+sz/2)
            ys = np.append(ys, yy[i]+sz/2)

    pup = np.zeros((sz,sz))
    for i in xrange(xs.size):
        pup[xs[i], ys[i]] = 1.0

    print ("%d" % (xs.size))
    return(pup)

# ======================================================================
def mklwe_bank(sz):
    quads = four_spider_mask((sz, sz), sz/2, 8.0, 0.0,
                             beta=51.75, thick=0.0, offset=1.28,
                             spiders=True, split=True)
    xx, yy = np.meshgrid(np.arange(sz)-sz/2, np.arange(sz)-sz/2)
    
    nm = 12
    bank = np.zeros((nm, sz, sz))
    for ii in xrange(nm):
        if ((ii % 3) == 0):
            bank[ii] = 1.0 * quads[ii / 3]
        elif ((ii % 3) == 1):
            temp = xx - xx[quads[ii / 3]].mean()
            bank[ii] = temp * quads[ii / 3]
        elif ((ii % 3) == 2):
            temp = yy - yy[quads[ii / 3]].mean()
            bank[ii] = temp * quads[ii / 3]
        bank[ii] /= bank[ii].std()
    return(bank)

# ======================================================================
def kolmo(rnd1, rnd2, fc, ld0, correc=1e0, rms=0.1):
    '''Does a Kolmogorov wavefront simulation with partial AO correction.
    
    Wavefront simulation of total size "size", following Kolmogorov statistics
    with a Fried parameter "r0", with partial AO correction up to a cutoff 
    frequency "fc". 

    Parameters:
    ----------

    - rnd1, rnd2 : arrays of normally distributed numbers
    - fc         : cutoff frequency (in lambda/D)
    - ld0        : lambda/D (in pixels)
    - correc     : correction of wavefront amplitude (factor 10, 100, ...)
    - std        : rms

    Note1: after applying the pupil mask, the Strehl is going to vary a bit
    Note2: one provides rnd1 and rn2 from outside so that the same experiment
           can be repeated with the same random numbers.
    '''

    ys,xs = rnd1.shape
    xx,yy  = np.meshgrid(np.arange(xs)-xs/2, np.arange(ys)-ys/2)
    myarr = shift(np.hypot(yy,xx))
    temp = np.zeros(rnd1.shape, dtype=complex)
    temp.real = rnd1
    temp.imag = rnd2

    in_fc = (myarr < (fc*ld0))
    out_fc = True - in_fc

    myarr[0,0] = 1.0 # trick to avoid div by 0!
    myarr = myarr**(-11./6.)
    myarr[in_fc] /= correc
    
    test = (ifft(myarr * temp)).real
    
    test -= np.mean(test)
    test *= rms/np.std(test)

    return test

