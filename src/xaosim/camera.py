#!/usr/bin/env python3

'''
===============================================================================
                   This is the camera module of XAOSIM.

It defines the generic camera class and the classes that inherit from the
generic camera like the Shack Hartman camera.
===============================================================================

'''
import numpy as np
import threading
from .pupil import uniform_disk as ud, _xyic
from  .sft import sft
from .shmlib import shm
import time

# import fft module and create short hand calls
import numpy.fft as fftmod

fft = fftmod.fft2
ifft = fftmod.ifft2
shift = fftmod.fftshift

dtor = np.pi/180.0  # to convert degrees to radians
i2pi = 1j*2*np.pi   # complex phase factor

# =============================================================================
# =============================================================================


class Cam(object):
    ''' Generic monochoromatic camera class

    ===========================================================================
    The camera is connected to the other objects (DM- and atmosphere- induced
    wavefronts) via shared memory data structures, after instantiation of
    the camera, when it is time to take an image.

    Thoughts:
    --------

    I am also considering another class to xaosim to describe the astrophysical
    scene and the possibility to simulate rather simply images of complex
    objects.

    Using a generic convolutive approach would work all the time, but may be
    overkill for 90% of the use cases of this piece of software so I am not
    sure yet.
    ===========================================================================

    '''

    # =========================================================================
    def __init__(self, name="SCExAO_chuck", csz=200, ysz=256, xsz=320,
                 pupil=None,
                 pdiam=7.92, pscale=10.0, wl=1.6e-6,
                 shmf="scexao_ircam.im.shm", shdir="/dev/shm/"):
        ''' Default instantiation of a cam object:

        -------------------------------------------------------------------
        Parameters are:
        --------------
        - name    : a string describing the camera ("instrument + camera name")
        - csz     : array size for Fourier computations
        - (ys,xs) : the dimensions of the actually produced image
        - pupil   : a csz x csz array containing the pupil
        - pscale  : the plate scale of the image, in mas/pixel
        - wl      : the central wavelength of observation, in meters
        - shmf    : the name of the file used to point the shared memory
        - shdir   : the name of the shared memory directory
        ------------------------------------------------------------------- '''

        self.name = name
        self.csz = csz              # Fourier computation size
        self.ysz = ysz              # camera vertical dimension
        self.xsz = xsz              # camera horizontal dimension
        self.isz = max(ysz, xsz)    # max image dimension

        # possible crop values (to match true camera image sizes)
        self.x0 = (self.isz - self.xsz) // 2
        self.y0 = (self.isz - self.ysz) // 2
        self.x1 = self.x0 + self.xsz
        self.y1 = self.y0 + self.ysz

        if pupil is None:
            self.pupil = ud(csz, csz, csz//2, True)
        else:
            self.pupil = pupil

        self.pdiam = pdiam                # pupil diameter in meters
        self.pscale = pscale              # plate scale in mas/pixel
        self.wl = wl                      # wavelength in meters
        self.frm0 = np.zeros((ysz, xsz))  # initial camera frame
        self.shmf = shdir+shmf            # the shared memory "file"
        self.shdir = shdir                # the shared memory directory

        self.btwn_pixel = False           # fourier comp. centering option
        self.phot_noise = False           # photon noise flag
        self.signal = 1e6                 # default # of photons in frame
        self.keepgoing = False            # flag for the camera server
        self.dm_shmf = None               # associated DM shared memory file
        self.atmo_shmf = None             # idem for atmospheric phase screen
        self.corono = False               # if True: perfect coronagraph
        self.wft_unit = "micron"          # default unit choice for wavefronts
        self.unit_factor = 1e-6

        # allocate/connect shared memory data structure
        self.shm_cam = shm(self.shmf, data=self.frm0, verbose=False)

        self.tlog = TimeLogger(lsize=20)

        # final tune-up
        self.update_cam()


    # =========================================================================
    def __str__(self):
        msg = f"Camera size : {self.xsz} x {self.ysz} pixels\n"
        msg += f"Wavelengtn  : {self.wl * 1e6:.2f} microns\n"
        msg += f"Plate scale : {self.pscale:.2f} mas/pixel\n"
        return msg

    # =========================================================================
    def update_cam(self, wl=None, pscale=None,
                   between_pixel=None, wft_unit=None, pview=None):
        ''' -------------------------------------------------------------------
        Change the filter, the plate scale or the centering of the camera

        Parameters:
        - pscale        : the plate scale of the image, in mas/pixel
        - wl            : the central wavelength of observation, in meters
        - between_pixel : whether FT are centered between four pixels or not
        - wft_unit      : "micron" or "nanometer"
        - pview         : number of pixels for a pupil viewing camera
        ------------------------------------------------------------------- '''
        wasgoing = False

        if self.keepgoing:
            wasgoing = True
            self.stop()
            time.sleep(2*self.delay)  # just to make sure

        if wl is not None:
            self.wl = wl
            try:
                del self._A1
            except AttributeError:
                print("sft aux array to be refreshed")
                pass

        if pscale is not None:
            self.pscale = pscale
            try:
                del self._A1
            except AttributeError:
                print("SFT aux array to be refreshed")
                pass

        if between_pixel is not None:
            self.btwn_pixel = between_pixel
            try:
                del self._A1
            except AttributeError:
                print("SFT aux array to be refreshed")
                pass

        self.ld0 = self.wl/self.pdiam*3.6e6/dtor/self.pscale  # l/D (in pixels)
        self.nld0 = self.isz / self.ld0           # nb of l/D across the frame

        if wft_unit is not None:
            if wft_unit == "micron":
                self.unit_factor = 1e-6
                print("wavefront unit set to 'micron'")
            elif wft_unit == "nanometer":
                self.unit_factor = 1e-9
                print("wavefront unit set to 'nanometer'")
            else:
                print("wavefront unit: 'micron' or 'nanometer' only!")

        if pview is not None:
            self.pview = pview
            print(f"pupil viewing mode set to {pview} pixels")

        if wasgoing:
            self.start(
                delay=self.delay,
                dm_shmf=self.dm_shmf, atmo_shmf=self.atmo_shmf)

    # =========================================================================
    def update_signal(self, nph=1e6):
        ''' Update the strength of the signal

        Automatically sets the *phot_noise* flag to *True*
        *IF* the value provided is negative, it sets the *phot_noise* flag
        back to *False* and sets the signal back to 1e6 photons

        Parameters:
        ----------
        - nph: the total number of photons inside the frame
        ------------------------------------------------------------------- '''
        if (nph > 0):
            self.signal = nph
            self.phot_noise = True
        else:
            self.signal = 1e6
            self.phot_noise = False

    # =========================================================================
    def get_image(self, ):
        ''' Returns the image currently avail on shared memory '''
        return(self.shm_cam.get_data())

    # =========================================================================
    def sft(self, A2):
        ''' Class specific implementation of the explicit Fourier Transform

        -------------------------------------------------------------------
        The algorithm is identical to the function in the sft module,
        except that intermediate FT arrays are stored for faster
        computation.

        For a more generic implementation, refer to the sft module of this
        package.

        Assumes the original array is square.
        No need to "center" the data on the origin.
        -------------------------------------------------------------- '''
        try:
            _ = self._A1  # look for existence of auxilliary arrays
        except AttributeError:
            print("updating the Fourier auxilliary arrays")
            NA = self.csz
            NB = self.isz
            m = self.nld0
            self._coeff = m/(NA*NB)

            U = np.zeros((1, NB))
            X = np.zeros((1, NA))

            offset = 0
            if self.btwn_pixel is True:
                offset = 0.5
            X[0, :] = (1./NA)*(np.arange(NA)-NA/2.0+offset)
            U[0, :] = (m/NB)*(np.arange(NB)-NB/2.0+offset)

            sign = -1.0

            self._A1 = np.exp(sign*i2pi*np.dot(np.transpose(U), X))
            self._A3 = np.exp(sign*i2pi*np.dot(np.transpose(X), U))
            self._A1 *= self._coeff

        B = (self._A1.dot(A2)).dot(self._A3)
        return np.array(B)

    # =========================================================================
    def off_pointing(self, offx, offy):
        ''' Produces the phase map that will induce the specified off axis
        pointing in number of pixels along the x and y axes.

        Parameters:
        ----------
        - offx: horizontal pointing offset (in pixels)
        - offy: vertical pointing offset (in pixels)

        Returns:
        -------
        a 2D array that can directly be fed as the *opdmap* argument of
        make_image()
        '''
        yy, xx = _xyic(self.csz, self.csz, between_pix=self.btwn_pixel)
        offset = (offx*xx + offy*yy)*self.wl/(self.csz*self.ld0)
        return offset

    # =========================================================================
    def total_phase(self, opdmap=None, dmmap=None):
        ''' Combines the two opd maps and computes the resulting phase

        Parameters:
        ----------
        - opdmap  : (optional) an OPD (optical path displacement) map
        - dmmap   : (optional) a deformable mirror displacement map

        Comment: Attempts to make code reusable in specialized cameras
        '''
        phs = np.zeros((self.csz, self.csz), dtype=np.float64)  # phase map

        if dmmap is not None:  # a DM map was provided (x2 for reflection)
            phs = 4 * np.pi / self.wl * dmmap * self.unit_factor

        if opdmap is not None:  # an OPD map was provided
            phs += 2 * np.pi / self.wl * opdmap
        return phs

    # =========================================================================
    def make_image(self, opdmap=None, dmmap=None, nochange=False):
        ''' Produces an image, given a certain number of maps (opd & DM)
        and updates the shared memory data structure that the camera
        instance is linked to with that image

        If you need something that returns the image, you have to use the
        class member method get_image(), after having called this method.
        -------------------------------------------------------------------

        Parameters:
        ----------
        - opdmap  : (optional) an OPD (optical path displacement) map
        - dmmap   : (optional) a deformable mirror displacement map
        - nochange: (optional) a flag to skip the computation!
        ------------------------------------------------------------------- '''

        # nothing to do? skip the computation!
        if (nochange is True) and (self.phot_noise is False):
            return

        phs = self.total_phase(opdmap=opdmap, dmmap=dmmap)

        if self.corono:  # perfect coronagraph simulation !
            wf = 0+1j*phs
        else:
            wf = np.exp(1j*phs)

        wf *= np.sqrt(self.signal / self.pupil.sum())  # signal scaling
        wf *= self.pupil                               # apply the pupil mask
        self._phs = phs * self.pupil                   # store total phase
        self.fc_pa = self.sft(wf)                      # focal plane cplx ampl
        img = np.abs(self.fc_pa)**2                    # intensity
        frm = img[self.y0:self.y1, self.x0:self.x1]    # image crop

        if self.phot_noise:  # need to be recast to fit original format
            frm = np.random.poisson(lam=frm.astype(np.float64), size=None)

        # push the image to shared memory
        self.shm_cam.set_data(frm.astype(self.shm_cam.npdtype))

    # =========================================================================
    def start(self, delay=0.1, dm_shmf=None, atmo_shmf=None):
        ''' ----------------------------------------
        Starts an independent thread that looks for
        changes on the DM, atmo and qstatic and
        updates the image

        Parameters:
        ----------
        - delay      : time (in seconds) between exposures
        - dm_shmf    : shared mem file for DM
        - atmo_shmf  : shared mem file for atmosphere
        ---------------------------------------- '''
        self.delay = delay

        if not self.keepgoing:
            self.dm_shmf = dm_shmf
            self.atmo_shmf = atmo_shmf

            self.keepgoing = True
            t = threading.Thread(
                target=self.__loop__,
                args=(delay, self.dm_shmf, self.atmo_shmf))
            t.start()
            print("The *CAMERA* server was started")
        else:
            print("The *CAMERA* server is already running")

    # =========================================================================
    def stop(self,):
        ''' ----------------------------------------
        Simple high-level accessor to interrupt the
        thread of the camera server infinite loop
        ---------------------------------------- '''
        if self.keepgoing:
            print("The *CAMERA* server was stopped")
            self.keepgoing = False
        else:
            print("The *CAMERA* server was not running")

    # =========================================================================
    def __loop__(self, delay=0.1, dm_shm=None, atmo_shm=None):

        ''' ----------------------------------------
        Thread (infinite loop) that monitors changes
        to the DM, atmo, and qstatic data structures
        and updates the camera image.

        Parameters:
        ----------
        - delay     : time in seconds between exposures
        - dm_shm    : shared mem file for DM
        - atmo_shm  : shared mem file for atmosphere
        - qstat_shm : shared mem file for qstatic error
        --------------------------------------------

        Do not use directly: use self.start_server()
        and self.stop_server() instead.
        ---------------------------------------- '''
        dm_cntr = 0      # counter to keep track of updates
        atm_cntr = 0     # on the phase screens
        dm_map = None    # arrays that store current phase
        atm_map = None   # screens, if they exist
        nochange = True  # lazy flag!

        # 1. read the shared memory data structures if present
        # ----------------------------------------------------
        if dm_shm is not None:
            dm_map = shm(dm_shm)

        if atmo_shm is not None:
            atm_map = shm(atmo_shm)

        # 2. enter the loop
        # ----------------------------------------------------
        while self.keepgoing:
            nochange = True  # lazy flag up!

            if dm_map is not None:
                test = dm_map.get_counter()
                dmmap = dm_map.get_data()
                if test != dm_cntr:
                    dm_cntr = test
                    nochange = False
            else:
                dmmap = None

            if atm_map is not None:
                test = atm_map.get_counter()
                atmomap = atm_map.get_data()
                if test != atm_cntr:
                    atm_cntr = test
                    nochange = False
            else:
                atmomap = None

            self.make_image(opdmap=atmomap, dmmap=dmmap, nochange=nochange)
            self.tlog.log()
            time.sleep(self.delay)

    # =========================================================================
    def close(self,):
        ''' ----------------------------------------
        Closes the linked shared memory structure
        ---------------------------------------- '''
        self.shm_cam.close()

# =============================================================================
# =============================================================================


class CoroCam(Cam):
    ''' Coronagraphic camera class definition

    Unlike the ideal coronagraph option of the generic Cam class, this
    is an attempt to implement a complete coronagraphic simulation,
    featuring:
    - an apodizing aperture mask
    - a focal plane mask
    - a Lyot stop

    The default setting for now will be a classical Lyot coronagraph but
    the idea is to make it easy to simulate FQPM, phase masks, but I
    won't bother yet with PIAA-like concepts just yet.

    This class inherits from the more generic Cam class.
    '''

    # =========================================================================
    def __init__(self, name="SCExAO_coro", csz=400, ysz=256, xsz=320,
                 pupil=None, fpm=None, lstop=None, pview=None,
                 pdiam=7.92, pscale=10.0, wl=1.6e-6,
                 shmf="scexao_coro.im.shm", shdir="/dev/shm/"):
        ''' Default instantiation of a coronagraphic cam object:
        -------------------------------------------------------------------

        Parameters are:
        --------------
        - name    : a string describing the camera ("instrument + camera name")
        - csz     : array size for Fourier computations
        - (ys,xs) : the dimensions of the actually produced image
        - pupil   : (csz x csz) the pupil (possibly apodized)
        - fpm     : (csz x csz) the focal plane mask
        - lstop   : (csz x csz) the lyot stop
        - pview   : (psz) size of pupil viewing mode in pixels? (ex: ZELDA)

        - pscale  : the plate scale of the image, in mas/pixel
        - wl      : the central wavelength of observation, in meters
        - shmf    : the name of the file used to point the shared memory
        - shdir   : the name of the shared memory directory

        Note:
        ----
        For a pupil viewing camera (eg. ZELDA), one should adapt the plate scale
        in the intermediate focal plane along with the computation resolution
        (csz) to compute things with enough finesse... that can be better than
        Nyquist.
        ------------------------------------------------------------------- '''
        super(CoroCam, self).__init__(
            name=name, csz=csz, ysz=ysz, xsz=xsz, pupil=pupil,
            pdiam=pdiam, pscale=pscale, wl=wl, shmf=shmf, shdir=shdir)

        del self.corono  # no ideal coronagraph here!
        self.btwn_pixel = True
        self.pview = pview
        self.isz = self.csz  # full Fourier computation
        self.x0 = (self.isz - self.xsz) // 2
        self.y0 = (self.isz - self.ysz) // 2
        self.x1 = self.x0 + self.xsz
        self.y1 = self.y0 + self.ysz
        self.nld0 = self.isz / self.ld0           # nb of l/D across the frame

        # handling of the coronagraph parts
        if fpm is None:
            self.fpm = 1.0 - ud(csz, csz, 4*self.ld0, between_pix=True)
            print("Default focal plane mask: 4 l/D Lyot radius")
        else:
            self.fpm = fpm

        if lstop is None:
            self.lstop = ud(csz, csz, 0.9*csz//2, between_pix=True)
            print("Default lyot stop: undersized 10 %")
        else:
            self.lstop = lstop

    # =========================================================================
    def make_image(self, opdmap=None, dmmap=None, nochange=False):
        ''' Produces a CORONAGRAPHIC image, given a certain number of
        phase screens, and updates the shared memory data structure that
        the camera instance is linked to with that image

        If you need something that returns the image, you have to use the
        class member method get_image(), after having called this method.
        -------------------------------------------------------------------

        Parameters:
        ----------
        - opdmap  : (optional) an OPD (optical path displacement) map
        - dmmap   : (optional) a deformable mirror displacement map
        - nochange: (optional) a flag to skip the computation!

        Note:
        ----
        Implementing a camera pupil viewing mode for this coronagraphic 
        camera that is useful to simulate a ZELDA setup.
        ------------------------------------------------------------------- '''

        # nothing to do? skip the computation!
        if (nochange is True) and (self.phot_noise is False):
            return

        phs = self.total_phase(opdmap=opdmap, dmmap=dmmap)

        wf = np.exp(1j*phs)
        wf *= np.sqrt(self.signal / self.pupil.sum())  # signal scaling
        wf *= self.pupil                               # apply the pupil mask
        self._b4m = self.sft(wf)
        self._afm = self._b4m * self.fpm

        if self.pview is not None:  # pupil viewing camera!
            self._cca = sft(self._afm, self.pview, self.nld0, inv=True, btwn_pix=True)
            frm = np.abs(self._cca)**2  # intensity

        else:  # actual coronagraph!
            self._b4l = self.sft(self._afm)
            self._afl = self._b4l * self.lstop
            self._cca = self.sft(self._afl)
            img = np.abs(self._cca)**2  # intensity
            frm = img[self.y0:self.y1, self.x0:self.x1]  # image crop

        if self.phot_noise:  # need to be recast to fit original format
            frm = np.random.poisson(lam=frm.astype(np.float64), size=None)

        # push the image to shared memory
        self.shm_cam.set_data(frm.astype(self.shm_cam.npdtype))

# =============================================================================
# =============================================================================


class SHCam(Cam):
    ''' Shack-Hartman specialized form of camera

    ===========================================================================
    This class inherits from the more generic Cam class.
    ===========================================================================
    '''
    # ==================================================
    def __init__(self, name, csz=256, dsz=128, mls=10,
                 pupil=None, wl=0.8e-6,
                 shmf='SHcam.im.shm', shdir='/dev/shm/'):

        ''' Instantiation of a SH camera

        -------------------------------------------------------------------
        Parameters:
        ----------
        - name    : a string describing the instrument
        - sz      : an array size for Fourier computations
        - dsz     : size of the detector in pixels
        - mls     : # of lenses in micro-lens array (mls X mls)
        - pupil   : a csz x csz array containing the pupil
        - wl      : the central wavelength of the sensor, in meters
        - shmf    : the name of the file used to point the shared memory
        - shdir   : the shared memory directory
        ------------------------------------------------------------------- '''
        self.name = name
        self.sz = csz
        self.xs = dsz
        self.ys = dsz
        self.wl = wl
        self.mls = mls                 # u-lens array size (in lenses)

        if pupil is None:
            self.pupil = ud(csz, csz, csz//2, True)
        else:
            self.pupil = pupil

        self.shdir = shdir
        self.shmf = shdir+shmf            # the shared memory "file"
        self.frm0 = np.zeros((dsz, dsz))  # initial camera frame

        self.px0 = (self.sz-self.xs)/2  # pixel offset for image in array
        self.py0 = (self.sz-self.ys)/2  # pixel offset for image in array

        self.signal = 1e6        # default number of photons in frame
        self.phot_noise = False  # default state for photon noise
        self.keepgoing = False   # flag for the camera server

        self.dm_shmf = None      # associated shared mem file for DM
        self.atmo_shmf = None    # idem for atmospheric phase screen

        self.shm_cam = shm(self.shmf, data=self.frm0, verbose=False)

        self.cdiam = self.sz / np.float(self.mls)  # oversized u-lens size
        self.rcdiam = np.round(self.cdiam).astype(int)

        self.tlog = TimeLogger(lsize=20)

    # ==================================================
    def make_image(self, opdmap=None, dmmap=None, nochange=False):
        ''' Produce a SH image, given a certain number of phase screens
        -------------------------------------------------------------------
        Parameters:
        ----------
        - atmo    : (optional) atmospheric phase screen
        - qstatic : (optional) a quasi-static aberration
        - dmmap   : (optional) a deformable mirror displacement map

        -------
        ------------------------------------------------------------------- '''
        # nothing to do? skip the computation!
        if (nochange is True) and (self.phot_noise is False):
            return

        mu2phase = 4.0 * np.pi / self.wl / 1e6  # microns to phase (x2)

        mls = self.mls
        cdiam = int(self.cdiam)     # SH cell computation "diameter"
        idiam = int(self.xs / mls)  # SH cell image size "diameter"

        phs = np.zeros((self.sz, self.sz))  # full phase map
        frm = np.zeros((self.ys, self.xs))  # oversized array

        # -------------------------------------------------------------------
        if dmmap is not None:  # a DM map was provided
            phs = mu2phase * dmmap

        # -------------------------------------------------------------------
        if opdmap is not None:  # a phase screen was provided
            phs += opdmap

        # -------------------------------------------------------------------
        wf = np.exp(1j*phs)
        wf[self.pupil == 0.0] = 0+0j  # re-apply the pupil map

        xl0 = int(cdiam - idiam / 2)

        for ii in range(mls * mls):  # cycle ove rthe u-lenses
            wfs = np.zeros((2*cdiam, 2*cdiam), dtype=complex)
            li, lj = ii // mls, ii % mls  # i,j indices for the u-lens
            pi0 = int(np.round(li * self.xs / mls))
            pj0 = int(np.round(lj * self.xs / mls))  # image corner pixel

            ci0 = li * cdiam
            cj0 = lj * cdiam

            wfs[xl0:xl0+cdiam, xl0:xl0+cdiam] = wf[cj0:cj0+cdiam,
                                                   ci0:ci0+cdiam]
            # compute the image by the u-lens
            iml = shift(np.abs(fft(shift(wfs)))**2)
            frm[pj0:pj0+idiam, pi0:pi0+idiam] = iml[xl0:xl0+idiam,
                                                    xl0:xl0+idiam]

        # -------------------------------------------------------------------
        if frm.sum() > 0:
            frm *= self.signal / frm.sum()

        if self.phot_noise:  # poisson + recast
            tmp = np.random.poisson(lam=frm, size=None)
            frm = tmp.astype(self.shm_cam.npdtype)

        self.shm_cam.set_data(frm)  # push the image to shared memory


# =============================================================================
# =============================================================================


class TimeLogger(object):
    ''' Utility class to keep track of things like frame rates.
    ================================================================
    Append to a class instance that runs an independent threaded
    loop and call the log() function at every iteration of the loop.

    The get_rate() function returns the average time of the last
    *lsize* time differences, where *lsize* is specified at creation.
    ================================================================ '''

    def __init__(self, lsize=20):
        ''' TimeLogger Class constructor.

        Parameters:
        ----------
        - lsize (integer, default=20)
          Specify the size of the time log to maintain '''
        self.lsize = lsize
        self.times = []
        self.rate = 0

    def log(self,):
        self.times.append(time.time())
        if len(self.times) > self.lsize:
            self.times.pop(0)

    def get_rate(self,):
        '''Returns the rate at which the data was logged in Hz'''
        # compute time differences
        dtimes = list(map(lambda x, y: x-y, self.times[1:], self.times[:-1]))
        # return the inverse of the average logging rate in Hz
        self.rate = 1/np.mean(dtimes)
        print("Refresh rate = %.3f Hz" % (self.rate))
        return
