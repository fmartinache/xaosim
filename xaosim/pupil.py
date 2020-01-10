import numpy as np
import pdb
from scipy.ndimage import rotate

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
def ring_grid_coords(nel=6, rad=10.0, central=True, rot=0.0):
    ''' ----------------------------------------------------------
    returns a 2D array of "nel" real set of x,y coordinates for 
    points located on a circle of radius "rad". Optionally, the
    original central point can be discarded and the whole thing
    can be rotated by an angle "rot" in radians.

    Parameters:
    ----------
    - nel     : the number of points along the circle
    - rad     : the radius of the ring (float)
    - central : include or not the central point (boolean)
    - rot     : a rotation angle (in radians)
    ---------------------------------------------------------- '''
    th = 2*np.pi * np.arange(nel)/float(nel) + rot
    ring = np.array([rad * np.cos(th), rad * np.sin(th)])
    if central is False:
        return ring
    else:
        oring = np.zeros((2,nel+1))
        oring[:,1:] = ring
        return oring

# ==================================================================
def meta_ring_grid_coords(xy, nel=6, rad=10.0, central=True, rot=0.0):
    '''------------------------------------------------------------------- 
    returns a single 2D array of real x,y coordinates for a ring of nel 
    point surrounding each point (x,y) coordinate provided in the input 
    xy array.

    Parameters;
    ----------
    - xy      : a 2D array of (x,y) coordinates (float)
    - nel     : the number of points along the circle
    - rad     : the radius of the ring (float)
    - central : include or not the central point (boolean)
    - rot     : a rotation angle of circle coords (in radians)
    ------------------------------------------------------------------- '''

    xs = np.array(())
    ys = np.array(())

    npt = np.max(xy.shape) # number of points in the input grid
    xy0 =xy.copy() # local copy
    if xy0.shape[0] == npt:
        xy0 = xy0.T

    temp = ring_grid_coords(nel=nel, rad=rad, central=central, rot=rot)
    
    for k in range(npt):
        xs = np.append(xs, temp[0,:] + xy0[0,k])
        ys = np.append(ys, temp[1,:] + xy0[1,k])

    if xy.shape[0] == npt:
        return(np.array((xs, ys)).T)
    else:
        return(np.array((xs, ys)))

# ==================================================================
def hex_grid_coords(nr=1, radius=10, rot=0.0):
    ''' ----------------------------------------------------------
    returns a 2D array of real x,y coordinates for a regular 
    hexagonal grid that fits within a hexagon.

    Parameters:
    ----------
    - nr     : the number of "rings" (integer)
    - radius : the radius of a ring (float)
    - rot    : a rotation angle (in radians)
    ---------------------------------------------------------- '''
    xs = np.array(())
    ys = np.array(())

    RR = np.array([[np.cos(rot), -np.sin(rot)],
                   [np.sin(rot),  np.cos(rot)]])

    for i in range(1-nr, nr, 1):
        for j in range(1-nr, nr, 1):
            x = radius * (i + 0.5 * j)
            y = j * np.sqrt(3)/2 * radius
            if abs(i+j) < nr:
                xs = np.append(xs, x)
                ys = np.append(ys, y)
    return(RR.dot(np.array((xs, ys))))

# ==================================================================
def meta_hex_grid_coords(xy, nr=1, radius=10):
    '''------------------------------------------------------------------- 
    returns a single 2D array of real x,y coordinates for a regular 
    hexagonal grid of nr rings around each point (x,y) coordinate provided
    in the input xy array.

    Intended use: build a hex grid of hex grids, for instance to help with
    the discretization of a hexagonal segmented aperture.

    Parameters;
    ----------
    - xy    : a 2D array of (x,y) coordinates (float)
    - nr    : the number of hex "rings" to be created around each point
    - radius: the radius of a ring (float)
    ------------------------------------------------------------------- '''

    xs = np.array(())
    ys = np.array(())

    npt = np.max(xy.shape) # number of points in the input grid
    xy0 =xy.copy() # local copy
    if xy0.shape[0] == npt:
        xy0 = xy0.T

    temp = hex_grid_coords(nr, radius)
    for k in range(npt):
        xs = np.append(xs, temp[0,:] + xy0[0,k])
        ys = np.append(ys, temp[1,:] + xy0[1,k])

    if xy.shape[0] == npt:
        return(np.array((xs, ys)).T)
    else:
        return(np.array((xs, ys)))

# ==================================================================
def hex_mirror_model(nra, nrs, step, fill=False, rot=0.0):
    ''' -------------------------------------------------------------------
    - nra : the number of rings for the global aperture (rings)
    - nrs : the number of rings per segment
    - step: the minimum distance between two segments centers (float)
    - fill: adds points at gaps between segments and on the edges
    ------------------------------------------------------------------- '''

    RR = np.array([[np.cos(rot), -np.sin(rot)],
                   [np.sin(rot),  np.cos(rot)]])
    
    # the (coarse) *array* geometry
    # =============================
    seg1 = hex_grid_coords(nra, step)
    keep = (np.abs(seg1[0,:]) > 1e-3) + (np.abs(seg1[1,:]) > 1e-3)
    seg1 = seg1[:,keep]

    radius = step / (np.sqrt(3) * nrs)
    if fill is True:
        nrs += 1
    res = RR.dot(meta_hex_grid_coords(np.flipud(seg1), nrs, radius))
    return(res)

# ==================================================================
def F_test_figure(ys, xs, ww):
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
def uniform_rect(ys, xs, yw, xw, between_pix=True):
    ''' ---------------------------------------------------------
    returns an (ys x xs) array with a centered uniform rectangle
    of size (yw x xw)

    Parameters:
    ----------
    - ys, xs: the size of the array
    - yw, xw: the size of the rectangle
    - between_pix: flag
    ---------------------------------------------------------  '''
    if between_pix is False:
        xx,yy  = np.meshgrid(np.arange(xs)-xs/2, np.arange(ys)-ys/2)
    else:
        xx,yy  = np.meshgrid(np.arange(xs)-xs/2+0.5, np.arange(ys)-ys/2+0.5)
    res = np.zeros_like(xx)
    res[(np.abs(xx) <= xw/2)*(np.abs(yy) <= yw/2)] = 1.0

    #res[np.abs(xx) <= xw/2] = 1.0
    #res[np.abs(yy) <= yw/2] = 1.0
    return(res)

# ==================================================================
def uniform_disk(ys, xs, radius, between_pix=False):
    ''' ---------------------------------------------------------
    returns an (ys x xs) array with a uniform disk of radius "radius".
    ---------------------------------------------------------  '''
    if between_pix is False:
        xx,yy  = np.meshgrid(np.arange(xs)-xs/2, np.arange(ys)-ys/2)
    else:
        xx,yy  = np.meshgrid(np.arange(xs)-xs/2+0.5, np.arange(ys)-ys/2+0.5)
    mydist = np.hypot(yy,xx)
    res = np.zeros_like(mydist)
    res[mydist <= radius] = 1.0
    return(res)

# ==================================================================
def uniform_hex(ys, xs, hrad, between_pix=False):
    ''' ---------------------------------------------------------
    returns an (ys x xs) array with a uniform hexagon of radius "hrad"
    ---------------------------------------------------------  '''
    if between_pix is False:
        xx,yy  = np.meshgrid(np.arange(xs)-xs/2, np.arange(ys)-ys/2)
    else:
        xx,yy  = np.meshgrid(np.arange(xs)-xs/2+0.5, np.arange(ys)-ys/2+0.5)

    res = np.ones_like(xx)
    res[np.abs(yy) > hrad - xx / np.sqrt(3)]  = 0.0
    res[np.abs(yy) > hrad + xx / np.sqrt(3)]  = 0.0
    res[np.abs(xx) > 0.5 * np.sqrt(3) * hrad]  = 0.0
    return res

# ==================================================================
def four_spider_mask(ys, xs, pix_rad, pdiam, odiam=0.0, 
                     beta=45.0, thick=0.25, offset=0.0,
                     spiders=True, split=False, between_pix=True):
    ''' ---------------------------------------------------------
    tool function called by other routines to generate specific
    pupil geometries. Although the result is scaled by pix_rad in 
    pixels, telescope specifics are provided in meters.

    Parameters:
    ----------
    - ys, xs  : dimensions of the 2D array      (in pixels)
    - pix_rad : radius of the circular aperture (in pixels)
    - pdiam   : diameter of the aperture        (in meters)
    - odiam   : diameter of the obstruction     (in meters)
    - beta    : angle of the spiders            (in degrees)
    - thick   : thickness of the spiders        (in meters)
    - offset  : spider intersect point distance (in meters)
    - spiders : flag to true to include spiders (boolean)
    - split   : split the mask into four parts  (boolean)
    --------------------------------------------------------- '''

    beta    = beta * dtor # converted to radians
    ro      = odiam / pdiam
    xx,yy   = np.meshgrid(np.arange(xs)-xs/2, np.arange(ys)-ys/2)
    if between_pix is True:
        xx,yy   = np.meshgrid(np.arange(xs)-xs/2+0.5, np.arange(ys)-ys/2+0.5)
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
    e = (mydist <= np.round(pix_rad))
    if odiam > 1e-3: # threshold at 1 mm
        e *= (mydist > np.round(ro * pix_rad))
        
    if split:
        res = np.array([a*e, b*e, c*e, d*e])
        return(res)
    
    if spiders:
        return((a+b+c+d)*e)
    else:
        return(e)

# ======================================================================
def four_quadrant_model_split(mask_xy, beta=51.75, thick=0.45, offset=1.28):
    ''' -------------------------------------------------------------
    Tool that corresponds takes a set of coordinates (model mask) 
    such as the ones used by XARA, and given the spider geometry 
    parameters, splits it into quadrants.

    Parameters:
    ----------

    - mask_xy: a (Nx2) array of (x,y) coordinate points (in meters)
    - beta: spider angle (in degrees)
    - thick: the thickness of the spiders (in meters)
    - offset: spider intersect offset (in meters)

    Note:
    ----

    Default values are set for SCExAO.
    ------------------------------------------------------------- '''
    xx = mask_xy[:,0]
    yy = mask_xy[:,1]

    beta1 = beta * dtor

    x0  = thick / (4 * np.sin(beta1)) + 0.5 * offset
    y0  = thick / (4 * np.cos(beta1)) - 0.5 * offset * np.tan(beta1)

    quad1 = (xx >=   0) * (np.abs(np.arctan(yy/(xx-x0+1e-8))) < beta1)
    quad2 = (xx <= -x0) * (np.abs(np.arctan(yy/(xx+x0+1e-8))) < beta1)
    quad3 = (yy >= 0.0) * (np.abs(np.arctan((yy-y0)/(xx+1e-8))) > beta1)
    quad4 = (yy <= 0.0) * (np.abs(np.arctan((yy+y0)/(xx+1e-8))) > beta1)
    
    res = np.array([[xx[quad1], yy[quad1]],
                    [xx[quad2], yy[quad2]],
                    [xx[quad3], yy[quad3]],
                    [xx[quad4], yy[quad4]]])
    return(res)

# ======================================================================
def lwe_mode_vector(split_xyq, iQuad, iMode):
    ''' -------------------------------------------------------------
    Tool that buids a vector containing a LWE mode of index i0, for
    a given split pupil coordinates model, resulting from the
    four_quadrant_model_split() function.

    Parameters:
    ----------

    - split_xyq : array of xy - coordinates (in meters)
    - iQuad     : index of quadrant (0 - 4)
    - iMode     : index of mode (0: piston, 1: tip, 2: tilt)
    ------------------------------------------------------------- '''

    nq = split_xyq.shape[0]

    vector = np.array([])
    for iq in range(nq):
        if iq != iQuad:
            vector = np.append(vector, np.zeros(split_xyq[iq,0].size))
        else:
            if iMode == 0:
                vector = np.append(vector, np.ones(split_xyq[iq,0].size))
            elif iMode == 1:
                vector = np.append(vector, split_xyq[iq,0])
            else:
                vector = np.append(vector, split_xyq[iq,1])

    vector /= vector.std()
    return(vector)
        
# ======================================================================
def HST(xs,ys, radius, spiders=True, between_pix=True):
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
    return(four_spider_mask(ys, xs, radius, pdiam, odiam, 
                            beta=beta, thick=thick, offset=offset, 
                            spiders=spiders, between_pix=between_pix))

# ==================================================================
def JWST(sz, pscale=0.1, aperture="CLEARP"):
    ''' ---------------------------------------------------------
    Returns a square (sz x sz) array filled with a representation
    of the JWST aperture.

    Parameters:
    ----------
    - sz       : size of the array (integer)
    - pscale   : pupil pixel scale in meter / pixel (float)
    - aperture : identifier (string: default "CLEARP")

    Remarks:
    -------
    Using the longer wavelength filters ("F277W" and longer)
    requires the insertion of the pupil alignment reference (PAR)
    which alters the pupil shape from CLEAR to CLEARP.

    CLEARP is assumed by default. Anything else will use CLEAR.
    --------------------------------------------------------- '''
    
    xx, yy = np.meshgrid(np.arange(sz)-sz/2, np.arange(sz)-sz/2)
    sp_th = 0.1  # true thickness spider in meters
    trad  = 1.32 # true segment radius in meters
    orad  = 1.00 # true PAR obstruction radius in meters
    SP_th = 0.33 # true PAR spider thickness in meters
    
    seg_rad  = int(trad / pscale) # segment radius in pixels
    spi_hthk = sp_th / pscale / 2 # TEL spider half-thickness in pixels
    SPI_hthk = SP_th / pscale / 2 # PAR spider half-thickness in pixels
    PAR_rad  = orad / pscale      # PAR obstruction radius in pixels
    
    res = np.zeros((sz, sz))

    # segmented aperture first
    # ------------------------
    scoords = hex_grid_coords(nr=3, radius=seg_rad, rot=0)
    scoords = np.delete(scoords, scoords.shape[1]//2, axis=1) # no central segment
    scoords = scoords.astype('int')

    seg0 = uniform_hex(sz,sz,seg_rad/np.sqrt(3)-1)
    seg0 = seg0.T
    
    for ii in range(scoords.shape[1]):
        res += np.roll(np.roll(seg0, scoords[0,ii], axis=0),
                       scoords[1,ii], axis=1)

    # adding spider arm shadow
    # ------------------------
    sp1 = 1 - (np.abs(xx) <= spi_hthk) * (yy > 0)
    sp2 = rotate(sp1, 150.0, order=0, reshape=False)
    sp3 = np.fliplr(sp2)

    res *= sp1 * sp2 * sp3


    # CLEARP or CLEAR?
    # ----------------
    if aperture == "CLEARP":
        mask_obs = 1-uniform_disk(sz,sz, PAR_rad)

        sp1 = 1 - (np.abs(xx) <= SPI_hthk) * (yy > 0)
        sp2 = rotate(sp1, 120.0, order=0, reshape=False)
        sp3 = np.fliplr(sp2)

        mask_obs *= sp1 * sp2 * sp3
        res *= mask_obs
    return res

# ==================================================================
def JWST_NRM(sz,pscale=0.0064486953125):
    ''' ---------------------------------------------------------
    Returns a square (sz x sz) array filled with a representation
    of the JWST aperture with the non-redundant mask.

    Parameters:
    ----------
    - sz       : size of the array (integer)
    - pscale   : pupil pixel scale in meter / pixel (float)

    Credit:
    ------
    Written by J. Marquez (2020)
    --------------------------------------------------------- '''

    ###		Hole positions (meters)			###
    X_m = [ 1.99,     1.32,     0.0,     1.99,    1.32,   -1.32,   -2.64]
    Y_m = [-1.14583, -2.28895, -2.28895, 1.14583, 2.28895, 2.28895, 0.0]
    
    hole_radius = 0.4650001189 # radius hole (meter)
    hexa_f = np.zeros([sz,sz]) # Final NRM
    
    # First Hole
    #-----------
    hexa = uniform_hex(sz, sz, hole_radius/pscale, between_pix=False).T
    #hexa = rotate(hexa, 90.0, order=0, reshape=False)

    for i in range(len(X_m)):
        hexa_p = np.roll(np.roll(hexa,int(X_m[i]/pscale),0),int(Y_m[i]/pscale),1)
        hexa_f += hexa_p
    return(hexa_f)

# ==================================================================
def PHARO(PSZ, rad, mask="std", between_pix=True, ang=0):
    ''' ---------------------------------------------------------
    returns an array that draws the pupil of the PHARO camera
    of the Palomar Hale telescope, at the center of an array of
    size "PSZ" with radius "radius" (both in pixels).

    Parameters:
    ----------
    - PSZ:     size of the array (assumed to be square)
    - rad:     radius of the standard pupil (in pixels)
    - mask:    aperture mask used
      + "std": standard cross (default)
      + "med": medium cross
    - between_pix: flag
    - ang:     global rotation of the pupil (in degrees)

    Notes:
    -----
    The reference is the standard cross radius, which corresponds
    to an actual 4.978 m clear aperture on the telescope. The
    central obstruction diameter is then 1.841 m (standard cross).

    The med_cross corresponds to a 4.646 m clear aperture,
    and a 2.293 central obstruction.
    
    See Hayward et al (2000) in PASP for reference.
    --------------------------------------------------------- '''
    xx,yy  = np.meshgrid(np.arange(PSZ)-PSZ/2, np.arange(PSZ)-PSZ/2)

    if between_pix is True:
        xx += 0.5
        yy += 0.5

    mydist = np.hypot(yy,xx)

    res = np.zeros_like(mydist)

    if mask is "med":
        res[mydist <= 0.933 * rad]   = 1.0 # undersized aperture
        res[mydist <= 0.460 * rad]   = 0.0 # oversized obstruction
        res[np.abs(xx) < 0.05 * rad] = 0.0 # fat spiders
        res[np.abs(yy) < 0.05 * rad] = 0.0 # fat spiders

    else:
        res[mydist <= rad]             = 1.0 # std clear aperture
        res[mydist <= 0.370 * rad]     = 0.0 
        res[np.abs(xx) <= 0.015 * rad] = 0.0
        res[np.abs(yy) <= 0.015 * rad] = 0.0

    if ang is not 0:
        res = rotate(res, ang, order=0, reshape=False)
    return res

# ==================================================================
def VLT(n,m, radius, spiders=True, between_pix=True):
    ''' ---------------------------------------------------------
    returns an array that draws the pupil of the VLT
    at the center of an array of size (n,m) with radius "radius".

    Parameters describing the pupil were deduced from a pupil mask
    description of the APLC coronograph of SPHERE, by Guerri et al, 
    2011. 

    http://cdsads.u-strasbg.fr/abs/2011ExA....30...59G
    --------------------------------------------------------- '''

    # VLT pupil description
    # ---------------------
    pdiam, odiam = 8.00, 1.12  # tel. and obst. diameters (meters)
    thick  = 0.04              # adopted spider thickness (meters)
    offset = 1.11              # spider intersection offset (meters)
    beta   = 50.5              # spider angle beta

    return(four_spider_mask(m, n, radius, pdiam, odiam, 
                            beta, thick, offset, spiders=spiders,
                            between_pix=between_pix))

# ==================================================================
def keck(sz, spiders=True, between_pix=True):
    tmp = segmented_aperture(sz, 3, sz//7, rot=0.0)
    xx, yy = np.meshgrid(np.arange(sz)-sz/2, np.arange(sz)-sz/2)
    dist = np.hypot(yy, xx)
    obst = np.ones((sz,sz))

    #pdb.set_trace()
    obst[dist <= 0.121 * sz] = 0.0
    tmp *= obst

    thick = 0.00116 * sz
    for ii in range(6):
        ang = 30.0 + 60.0 * ii
        zone = (xx > 0) * (np.abs(yy) <= thick)
        res = np.ones((sz,sz))
        res[zone] = 0.0
        res = rotate(res, ang, order=0, reshape=False)
        tmp *= res
    return(tmp)

# ==================================================================
def subaru(n,m, radius, spiders=True, between_pix=True):
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
    thick  = 0.25             # adopted spider thickness (meters)
    offset = 1.278            # spider intersection offset (meters)
    beta   = 51.75            # spider angle beta

    return(four_spider_mask(m, n, radius, pdiam, odiam=odiam, 
                            beta=beta, thick=thick, offset=offset,
                            spiders=spiders,
                            between_pix=between_pix))

# ==================================================================
def subaru_dbl_asym(xs, ys, radius, spiders=True, PA1=0.0, PA2=90.0,
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
    a = subaru_asym(xs, ys, radius, spiders=spiders, PA=PA1, thick=thick1)
    b = subaru_asym(xs, ys, radius, spiders=spiders, PA=PA2, thick=thick2)
    return(a*b)

# ==================================================================
def radial_arm(xs,ys, radius, PA=0.0, thick=0.15):
    ''' -------------------------------------------------------------
    Produces a pupil mask for an occulting radial arm for a given
    position angle and thickness.

    Parameters:
    - xs, ys : dimensions of the 2D array   (integer # of pixels)
    - radius : outer radius of the aperture (integer # of pixels)
    - PA     : position angle of the arm    (degrees)
    - thick  : fraction of the ap. diameter (float)
    ------------------------------------------------------------- '''
    res = np.ones((ys, xs))
    ang = np.mod(PA, 360.0)
    xx,yy  = np.meshgrid(np.arange(xs)-xs/2, np.arange(ys)-ys/2)
    zone = (xx > 0) * (np.abs(yy) <= thick*radius)
    res[zone] = 0.0
    res = rotate(res, ang, order=0, reshape=False)
    return(res.astype('int'))

# ==================================================================
def subaru_asym(xs, ys, radius, spiders=True, PA=0.0, thick=0.15):
    ''' -------------------------------------------------------------
    Returns a pupil mask with an asymmetric arm that mostly follows
    the geometry of the original APF-WFS.

    Parameters:
    ----------

    - xs, ys  : dimensions of the 2D array
    - radius  : outer radius of the aperture
    - spiders : boolean (w or without spiders)
    - PA      : position angle of the arm (in degrees)
    - thick   : asymm. arm thickness (% of aperture diameter)
    ------------------------------------------------------------- '''

    pup = subaru(xs,ys, radius, spiders)
    arm = radial_arm(xs,ys, radius, PA=PA, thick=thick)

    return(pup * arm)

# ======================================================================
def segmented_aperture(sz, nr, srad, rot=0.0):
    ''' ----------------------------------------------------------------
    Returns the square sz x sz image of a segmented aperture.

    Parameters:
    ----------
    - sz   : the size of the array
    - nr   : the number of rings (annuli) making up the aperture
    - srad : the radius of a segment (in pixels)
    - rot  : a rotation angle (in radians)
    ---------------------------------------------------------------- '''
    xy = hex_mirror_model(nr+1, srad, srad, rot=rot).astype(np.int)+sz//2
    pup = np.zeros((sz,sz))
    for i in range(xy.shape[1]):
        try:
            pup[xy[0,i],xy[1,i]] = 1.0
        except:
            pass
    return(pup)

# ======================================================================
def golay9_coords(prad, rot=0.0):
    ''' -------------------------------------------------------------
    Returns the holes coordinates of a 9-hole Golay array

    Parameters:
    ----------
    - prad: the radius of the circular aperture to be masked
    - rot: mask azimuth (in radians)
    ------------------------------------------------------------- '''
    dstep = prad / 3.5 # why exactly 3.5 here?
    d1    = dstep*np.sqrt(7)
    th0   = np.arctan(np.sqrt(3)/2)

    xs = np.array([])
    ys = np.array([])
    
    for i in range(3):
        theta = 2.0 * i * np.pi / 3.0 + rot
        
        for k in range(2,4):
            xs = np.append(xs, k * dstep * np.cos(theta))
            ys = np.append(ys, k * dstep * np.sin(theta))

        xs = np.append(xs,  d1 * np.cos(theta - th0))
        ys = np.append(ys,  d1 * np.sin(theta - th0))

    return np.array([xs, ys]).T

# ======================================================================
def golay9(sz, prad, hrad, between_pix=True, rot=0.0):
    ''' -------------------------------------------------------------
    Returns a square "sz x sz" NR Golay 9 pupil model

    Parameters:
    ----------
    - sz:          size of the 2D array to produce (in pixels)
    - prad:        size of the pupil the mask fits into (in pixels)
    - hrad:        sub-aperture radius (in pixels)
    - between_pix: centers the array between 4 pixels (boolean)
    - rot:         mask azimuth (in radians)
    ------------------------------------------------------------- '''
    off = 0
    if between_pix is True:
        off = 0.5
    xx,yy  = np.meshgrid(np.arange(sz)-sz/2+off, np.arange(sz)-sz/2+off)
    mydist = np.hypot(yy,xx)
    pup    = np.zeros((sz,sz))

    coords = golay9_coords(prad, rot)
    xs = np.cast['int'](np.round(coords[:,0]))
    ys = np.cast['int'](np.round(coords[:,1]))
    
    for i in range(xs.size):
        pup = np.roll(np.roll(pup, -ys[i], 0), -xs[i], 1)
        pup[mydist < hrad] = 1.0
        pup = np.roll(np.roll(pup,  ys[i], 0),  xs[i], 1)

    return(pup)

# ======================================================================
def lwe_mode_bank_2D(sz, odiam=8.0, beta=51.75, offset=1.28):
    ''' -------------------------------------------------------------
    Builds a 3D array containing pupil images of the 12 raw LWE
    modes: piston, tip and tilt, for the four expected quadrants
    of a pupil like Subaru, VLT, ...

    Parameters:
    ----------

    - sz: size of the 2D pupil images to be produced
    - odiam: pupil outer diameter     (default 8.0,   in meters)
    - beta: spider angle              (default 51.75, in degrees)
    - offset: spider intersect offset (default: 1.28, in meters)
    ------------------------------------------------------------- '''
    quads = four_spider_mask(sz, sz, sz/2, odiam, 0.0,
                             beta=beta, thick=0.0, offset=offset,
                             spiders=True, split=True)
    xx, yy = np.meshgrid(np.arange(sz)-sz/2, np.arange(sz)-sz/2)
    
    nm = 12
    bank = np.zeros((nm, sz, sz))
    for ii in range(nm):
        if ((ii % 3) == 0):
            bank[ii] = 1.0 * quads[ii // 3]
        elif ((ii % 3) == 1):
            temp = xx - xx[quads[ii // 3]].mean()
            bank[ii] = temp * quads[ii // 3]
        elif ((ii % 3) == 2):
            temp = yy - yy[quads[ii // 3]].mean()
            bank[ii] = temp * quads[ii // 3]
        bank[ii] /= bank[ii].std()
    return(bank)
