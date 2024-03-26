import numpy as np
from scipy.special import factorial as fac
from scipy import *


def dist(n, m):
    ''' -----------------------------------------------------------------------
    Returns the distance (in pixels) relative to the center of the array

    Parameters:
    ----------
    - (n,m): size of the array in pixels

    Remarks:
    -------
    - for odd size, center of array = central pixel
    - for even size, center of array is in between pixels!

    - equivalent of the convenient IDL dist function!
    ----------------------------------------------------------------------- '''
    if (n % 2 == 0):
        xl = np.arange(n) - n//2 + 0.5
    else:
        xl = np.arange(n) - n//2

    if (m % 2 == 0):
        yl = np.arange(m) - m//2 + 0.5
    else:
        yl = np.arange(m) - m//2

    xx, yy = np.meshgrid(xl, yl)
    return(np.hypot(yy, xx))


def azim(n, m):
    ''' -----------------------------------------------------------------------
    Returns the azimuth in radians of points in an array of size (n, m) with
    respect to the center of the array.

    Parameters:
    ----------
    - (n,m): size of the array in pixels

    Remarks:
    -------
    - for odd size, center of array = central pixel
    - for even size, center of array is in between pixels!

    - equivalent of the convenient IDL dist function!
    ----------------------------------------------------------------------- '''
    if (n % 2 == 0):
        xl = np.arange(n) - n//2 + 0.5
    else:
        xl = np.arange(n) - n//2

    if (m % 2 == 0):
        yl = np.arange(m) - m//2 + 0.5
    else:
        yl = np.arange(m) - m//2

    xx, yy = np.meshgrid(xl, yl)
    return np.arctan2(xx, yy)


def zer_coeff(n, m):
    ''' -----------------------------------------------------------------------
    Returns the Zernike coefficients and exponents for a given mode (n,m)
    ----------------------------------------------------------------------- '''
    coeffs, pows = [], []

    for s in range((n-m)//2+1):
        coeffs.append((-1.0)**s *
                      fac(n-s) /
                      (fac(s) * fac((n+m)/2.0 - s) * fac((n-m)/2.0 - s)))
        pows.append(n-2.0*s)
    return coeffs, pows


def noll_2_zern(j):
    '''------------------------------------------
    Noll index converted to Zernike indices

    j: Noll index
    n: radial Zernike index
    m: azimuthal Zernike index
   ------------------------------------------ '''
    if (j == 0):
        raise ValueError("Noll indices start at 1")

    n = 0
    j1 = j-1
    while (j1 > n):
        n += 1
        j1 -= n

    m = (-1)**j * ((n % 2) + 2 * int((j1+((n+1) % 2)) / 2.0))
    return (n, m)


def mkzer(n, m, size, rad, limit=False):
    '''----------------------------------------------------
    returns a 2D array of size sz,
    containing the n,m Zernike polynomial
    within a disk of radius rad

    Parameters:
    ----------
    - n, m: the Zernike indices for the polynomial
    - size: the size of the 2D array
    - rad: the radius of the circular area covered
    - limit: boolean (if True, mode extends beyond "rad"
   ---------------------------------------------------- '''
    res = np.zeros((size, size))
    (coeffs, pows) = zer_coeff(n, np.abs(m))

    rho = dist(size, size)/float(rad)
    outp = np.where(rho > 1.0)
    inp = np.where(rho <= 1.0)

    for i in range(np.size(coeffs)):
        res += coeffs[i] * (rho)**pows[i]

    if (limit is not False):
        res[outp] = 0.0

    azi = azim(size, size)

    if m > 0:
        res *= np.cos(m * azi)
    if m < 0:
        res *= np.sin(-m * azi)

    # normalization
    rms0 = np.std(res[inp])
    res /= rms0
    return res


def zer_mode_bank_2D(sz, i0, i1, limit=True):
    ''' ------------------------------------------
    Returns a 3D array containing 2D (sz x sz)
    maps of Zernike modes for Noll index going
    from i0 to i1 included.

    Parameters:
    ----------
    - sz: the size of the contained 2D arrays
    - i0: the first Zernike index to be used
    - i1: the last Zernike index to be used
    - limit: boolean (if True, mode confined to "rad")
    ------------------------------------------ '''
    dZ = i1 - i0 + 1
    res = np.zeros((dZ, sz, sz))
    for i in range(i0, i1+1):
        res[i-i0] = mkzer1(i, sz, sz/2, limit=limit)
    return(res)


def zer_name_list(i0, i1):
    ''' ------------------------------------------
    Returns a list of names of zernike modes for
    the Noll index going from i0 to i1 included.

    Note that the first Noll index is 1.
    ------------------------------------------ '''
    if not i1 > i0 > 0:
        return []
    zer_names = ['piston', 'x-tilt', 'y-tilt', 'defocus',
                 'obliq. astig.', 'vertc. astig.',
                 'vertc. coma', 'horiz. coma',
                 'vertc. trefoil', 'obliq. trefoil',
                 'prim. spherical',
                 'vertc. sec. astig.', 'obliq. sec. astig.',
                 'vertc. quadrafoil', 'obliq. quadrafoil']

    zer_list = [f"zernike_{ii:02d}" for ii in range(20)]
    for ii, lbl in enumerate(zer_names):
        zer_list[ii] = lbl
    return zer_list[i0-1:i1]


def mkzer1(jj, sz, rad, limit=False):
    ''' ----------------------------------------------------
    returns a 2D array of size sz,
    containing the j^th Zernike polynomial
    within a disk of radius rad

    Parameters:
    ----------
    - jj: the Noll index for the polynomial
    - sz: the size of the 2D array
    - rad: the radius of the circular area covered
    - limit: boolean (if True, mode confined to "rad")
   ---------------------------------------------------- '''
    (n, m) = noll_2_zern(jj)
    if jj == 1:
        return np.ones((sz, sz))
    else:
        return mkzer(n, m, sz, rad, limit)  # .astype(np.float32)


def mkzer_vector(n, m, xymask):
    '''------------------------------------------
    returns a 1D vector of size xymask.shape(0),
    containing the n,m Zernike polynomial
   ------------------------------------------ '''
    (coeffs, pows) = zer_coeff(n, np.abs(m))
    res = np.zeros(xymask.shape[0])
    rho = np.sqrt(xymask[:, 0]**2+xymask[:, 1]**2)
    rho /= rho.max()

    for i in range(np.size(coeffs)):
        res += coeffs[i] * (rho)**pows[i]

    azi = np.arctan2(xymask[:, 0], xymask[:, 1])

    if m > 0:
        res *= np.cos(m * azi)
    if m < 0:
        res *= np.sin(-m * azi)

    # normalization
    rms0 = np.std(res)
    res /= rms0
    return res


def mkzer1_vector(j, xymask):
    '''------------------------------------------
    returns a 1D vector of size xymask.shape(0),
    containing the j^th Zernike polynomial
   ------------------------------------------ '''
    (n, m) = noll_2_zern(j)
    return(mkzer_vector(n, m, xymask))


def mk_pattern(n, m):
    '''------------------------------------------
    returns an ad-hoc pattern which I do not
    remember what it corresponds to. Probably
    should discard this function.
   ------------------------------------------ '''
    x, y = np.meshgrid(np.arange(n)-n/2, np.arange(m)-m/2)
    dd = np.roll(np.hypot(y, x), 8, axis=0)
    b = np.zeros_like(dd)
    b[dd < 12] = 1.0
    b[dd < 7] = 0.0
    b[x < 0] = 0.0
    b[10:45, 20:25] = 1.0
    b[10:15, 17:28] = 1.0
    return(b)
