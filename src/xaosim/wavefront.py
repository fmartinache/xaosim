import numpy as np
from .pupil import uniform_rect as urect
from .pupil import _xyic
from scipy.ndimage import rotate

shift = np.fft.fftshift
fft = np.fft.fft2
ifft = np.fft.ifft2


# ======================================================================
def sin_map(sz, ncx, ncy, phi=0.0):
    ''' -----------------------------------------------------------
    Produces the 2D map of a sinusoidal modulation with a given
    number of cycles across the aperture.

    Parameters:
    ----------
    - sz : the size of the map in pixels (sz x sz)
    - ncx: the horizontal number of cycles
    - ncy: the vertical number of cycles
    - phi: the phase of the modulation (in radians)

    Remarks:
    -------
    phi = 0       means a true sinusoidal modulation.
    phi = np.pi/2 means cosinusoidal
    ----------------------------------------------------------- '''
    ramp = np.arange(sz)/(sz)-0.5
    xx, yy = np.meshgrid(ramp, ramp)
    res = np.sin(2*np.pi*(xx*ncx + yy*ncy) + phi)
    return res.astype(np.float32)


# ======================================================================
def mksin_vector(xymask, ncx, ncy, phi=0.0):
    ''' -----------------------------------------------------------
    returns a 1D vector of size xymask.shape[0], containing the
    local values of a sinusoidal wavefront computed for the xymask
    set of coordinates.
    ----------------------------------------------------------- '''
    xmax = np.abs(xymask[:, 0]).max()
    ymax = np.abs(xymask[:, 1]).max()
    fcx = ncx / xmax
    fcy = ncy / ymax
    res = np.sin(np.pi*(fcx * xymask[:, 0] + fcy * xymask[:, 1]) + phi)
    return res


# ======================================================================
def poke_map(sz, dx, dy, ww=3):
    ''' -----------------------------------------------------------
    Produces a 2D poke map for the aperture at the position dx, dy

    Parameters:
    ----------
    - sz: the size of the map in pixels (sz x sz)
    - dx: the horizontal offset of the poke (in pixels)
    - dy: the vertical offset of the poke (in pixels)
    - ww: the width of the poke (in pixels)

    Remarks:
    -------
    To add a border width eliminates the risk for a poke
    applied on one edge of the aperture to also appear on the
    opposite edge
    ----------------------------------------------------------- '''
    bw = 2*ww
    poke = urect(sz+2*bw, sz+2*bw, ww, ww, between_pix=False)
    poke = np.roll(np.roll(poke, dy, axis=1), dx, axis=0)
    return poke[bw:bw+sz, bw:bw+sz]


# ======================================================================
def piston_map(sz, coords, hrad, between_pix=True, piston=None):
    ''' -----------------------------------------------------------
    Produces a 2D map with circular holes where a piston is applied

    Parameters:
    ----------
    - sz          : the size of the map in pixels (sz x sz)
    - coords      : the array of hole coordinates (in pixels)
    - hrad        : the hole radius (in pixels)
    - between_pix : to align the map between pixels (bool)
    - piston      : a 1D array (or list) of pistons

    Remarks:
    -------
    If different than None, "piston" should contain at least as
    many values as there are holes. If None, a random set of
    values is genreated.
    ----------------------------------------------------------- '''

    off = 0
    if between_pix is True:
        off = 0.5
    xx, yy = np.meshgrid(np.arange(sz)-sz/2+off, np.arange(sz)-sz/2+off)
    mydist = np.hypot(yy, xx)
    pmap = np.zeros((sz, sz))

    p0 = np.zeros(coords.shape[0])
    if piston is not None:
        p0 = piston
    else:
        p0 = np.random.randn(coords.shape[0])

    xs = coords[:, 0]
    ys = coords[:, 1]
    for i in range(xs.size):
        pmap = np.roll(np.roll(pmap, -ys[i], 0), -xs[i], 1)
        pmap[mydist < hrad] = p0[i]
        pmap = np.roll(np.roll(pmap,  ys[i], 0),  xs[i], 1)
    return pmap


# ======================================================================
def kolmo(rnd, fc, ld0, correc=1e0, rms=0.1):
    ''' -----------------------------------------------------------
    WARNING: "legacy" function kept here as a reference.

    Use atmo_screen() instead!

    This function was adapted to a FFT based diffraction simulation,
    which is no longer used by this package.

    Does a Kolmogorov wavefront simulation with partial AO correction.

    Wavefront simulation of total size "size", following Kolmogorov
    statistics with a Fried parameter "r0", with partial AO correction
    up to a cutoff frequency "fc".

    Parameters:
    ----------

    - rnd     : array of uniformly distributed numbers [0,1)
    - fc      : cutoff frequency (in lambda/D)
    - ld0     : lambda/D (in pixels)
    - correc  : correction of wavefront amplitude (factor 10, 100, ...)
    - rms     : rms over the entire computed wavefront

    Note1: after applying the pupil mask, the Strehl is going to vary a bit
    Note2: one provides rnd from outside the routine so that the same
    experiment can be repeated with the same random numbers.
    ----------------------------------------------------------- '''

    ys, xs = rnd.shape
    xx, yy = np.meshgrid(np.arange(xs) - xs/2, np.arange(ys) - ys/2)
    rr = shift(np.hypot(yy, xx))

    in_fc = (rr < (fc*ld0))

    rr[0, 0] = 1.0  # trick to avoid div by 0!
    modul = rr**(-11./6.)
    modul[in_fc] /= correc

    test = (ifft(modul * np.exp(1j * 2*np.pi * (rnd - 0.5)))).real

    test -= np.mean(test)
    test *= rms/np.std(test)

    return test


# ==================================================================
def atmo_screen(isz, ll, r0, L0, fc=25, correc=1.0, pdiam=None):
    ''' -----------------------------------------------------------
    The Kolmogorov - Von Karman phase screen generation algorithm.

    Adapted from the work of Carbillet & Riccardi (2010).
    http://cdsads.u-strasbg.fr/abs/2010ApOpt..49G..47C

    Kolmogorov screen can be altered by an attenuation of the power
    by a correction factor *correc* up to a cut-off frequency *fc*
    expressed in number of cycles across the phase screen

    Parameters:
    ----------

    - isz    : the size of the array to be computed (in pixels)
    - ll     :  the physical extent of the phase screen (in meters)
    - r0     : the Fried parameter, measured at a given wavelength (in meters)
    - L0     : the outer scale parameter (in meters)
    - fc     : DM cutoff frequency (in lambda/D)
    - correc : correction of wavefront amplitude (factor 10, 100, ...)
    - pdiam  : pupil diameter (in meters)

    Returns: two independent phase screens, available in the real and
    imaginary part of the returned array.

    Remarks:
    -------
    If pdiam is not specified, the code assumes that the diameter of
    the pupil is equal to the extent of the phase screen "ll".
    ----------------------------------------------------------- '''

    phs = 2*np.pi * (np.random.rand(isz, isz) - 0.5)

    xx, yy = np.meshgrid(np.arange(isz)-isz/2, np.arange(isz)-isz/2)
    rr = np.hypot(yy, xx)
    rr = shift(rr)
    rr[0, 0] = 1.0

    modul = (rr**2 + (ll/L0)**2)**(-11/12.)

    if pdiam is not None:
        in_fc = (rr < fc * ll / pdiam)
    else:
        in_fc = (rr < fc)

    modul[in_fc] /= correc

    screen = ifft(modul * np.exp(1j*phs)) * isz**2
    screen *= np.sqrt(2*0.0228)*(ll/r0)**(5/6.)

    screen -= screen.mean()
    return(screen)


# ==================================================================
def noll_variance(iz, D, r0):
    '''Computes the Noll residual variance (in rad**2) for a partly
    corrected wavefront.

    Adapted from the work of Marcel Carbillet for CAOS Library 5.4

    Itself based on (Noll R.J., JOSA, 66, 3 (1976))

    Parameters:
    ----------
    - iz : Zernike mode until which the wavefront is ideally corrected
    - D  : aperture diameter (in meters)
    - r0 : Fried parameter (in meters)
    -------------------------------------------------------------------
    '''
    noll_var = [1.0299, 0.582,  0.134, 0.111, 0.0880,
                0.0648, 0.0587, 0.0525, 0.0463, 0.0401,
                0.0377, 0.0352, 0.0328, 0.0304, 0.0279,
                0.0267, 0.0255, 0.0243, 0.0232, 0.022,
                0.0208]

    try:
        noll = noll_var[iz]
    except IndexError:
        noll = 0.2944*(iz+1.)**(-np.sqrt(3)/2.)

    return(noll*(D/r0)**(5./3))


# ==================================================================
def noll_rms(iz, D, r0, wl=None):
    '''Computes the RMS for a partly corrected wavefront.

    Adapted from the work of Marcel Carbillet for CAOS Library 5.4
    Itself based on (Noll R.J., JOSA, 66, 3 (1976))

    Uses the Noll residual variance "noll_variance()" function
    introduced above.

    Parameters:
    ----------
    - iz : Zernike mode until which the wavefront is ideally corrected
    - D  : aperture diameter (in meters)
    - r0 : Fried parameter (in meters)

    Optional parameter:
    ------------------
    - wl: the wavelength (unit TBD by user)

    If wl is provided, the result (otherwise in radians) is converted
    into OPD, expressed in matching units
    -------------------------------------------------------------------
    '''
    vari = noll_variance(iz, D, r0)
    if wl is None:
        res = np.sqrt(vari)
    else:
        res = np.sqrt(vari) * wl / (2*np.pi)
    return(res)


# =============================================================================
def sector_modes(sz, ns=4, ponly=False, rot=0.0):
    ''' ---------------------------------------------------------------
    Returns a cube of 2D sector wavefront modes

    Intended to simulate and/or fit LWE (low wind effect) like modes
    on a telescope. Going counterclockwise, from horizontal right

    Parameters:
    - sz    : size (integer) of the array returned (sz x sz)
    - ns    : number (integer) of sectors (ns=4 or 6)
    - ponly : piston only or piston + tip-tilt (default)
    - rot   : rotation angle (in degrees)
    --------------------------------------------------------------- '''
    nm = ns if ponly else ns * 3
    res = np.empty((nm, sz, sz))
    res[:] = np.nan
    yy, xx = _xyic(sz, sz)
    dist = np.hypot(yy, xx)

    th = np.linspace(0, 2*np.pi, ns, endpoint=False)

    sector = (xx >= 0) * (yy >= 0.0) * np.abs(np.arctan(yy/(xx+1e-8)) < th[1])
    sector = rotate(sector, -rot, order=0, reshape=False)
    sector[dist > sz // 2] = False
    if ponly:
        for ii in range(ns):
            res[ii] = rotate(sector, -360/ns*ii, order=0, reshape=False)
    else:
        for ii in range(ns):
            tmp = rotate(sector, -360/ns*ii, order=0, reshape=False)
            res[3*ii] = tmp
            res[3*ii+1] = xx * tmp
            res[3*ii+1, tmp] -= res[3*ii+1, tmp].mean()
            res[3*ii+2] = yy * tmp
            res[3*ii+2, tmp] -= res[3*ii+2, tmp].mean()
    for ii in range(nm):
        res[ii] *= 1.0/res[ii].std()
    return res
