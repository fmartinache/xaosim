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
from . import pupil
from .shmlib import shm
import time

dtor  = np.pi/180.0 # to convert degrees to radians

try:
    import pyfftw
    shift = pyfftw.interfaces.numpy_fft.fftshift
    fft   = pyfftw.interfaces.numpy_fft.fft2
    ifft  = pyfftw.interfaces.numpy_fft.ifft2
    print("using pyfftw library!")
except:
    shift = np.fft.fftshift # short-hand for FFTs
    fft   = np.fft.fft2
    ifft  = np.fft.ifft2

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
                 shmf="scexao_ircam.im.shm", shdir="/dev/shm"):
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

        self.name    = name
        self.csz     = csz
        self.ysz     = ysz
        self.xsz     = xsz

        if pupil is None:
            self.pupil = pupil.uniform_disk(csz, csz, csz//2, True)
        else:
            self.pupil = pupil

        self.pdiam   = pdiam               # pupil diameter in meters
        self.pscale  = pscale              # plate scale in mas/pixel
        self.wl      = wl                  # wavelength in meters
        self.frm0    = np.zeros((ysz,xsz)) # initial camera frame
        self.shmf    = shdir+shmf          # the shared memory "file"
        self.shdir   = shdir               # the shared memory directory

        self.phot_noise = False            # default state for photon noise
        self.signal     = 1e6              # default number of photons in frame
        self.keepgoing  = False            # flag for the camera server

        self.dm_shmf    = None             # associated shared memory file for DM
        self.atmo_shmf  = None             # idem for atmospheric phase screen
        
        self.corono     = False            # if True: perfect coronagraph

        self.isz        = max(ysz, xsz)   # max image dimension (for computation)

        # possible crop values (to match true camera image sizes)
        self.x0 = (self.isz - self.xsz) // 2
        self.y0 = (self.isz - self.ysz) // 2
        self.x1 = self.x0 + self.xsz
        self.y1 = self.y0 + self.ysz

        # allocate/connect shared memory data structure
        self.shm_cam = shm(self.shmf, data = self.frm0, verbose=False)

        # final tune-up
        self.update_cam()
        
    # =========================================================================
    def update_cam(self, wl=None, pscale=None):
        ''' -------------------------------------------------------------------
        Change the filter or the plate scale of the camera

        Parameters:
        - pscale  : the plate scale of the image, in mas/pixel
        - wl      : the central wavelength of observation, in meters
        ------------------------------------------------------------------- '''
        wasgoing = False
        
        if self.keepgoing:
            wasgoing = True
            self.stop()
            time.sleep(2*self.delay) # just to make sure

        if wl is not None:
            self.wl = wl
            try:
                del self._A1
            except:
                print("sft aux array to be refreshed")
                pass
            
        if pscale is not None:
            self.pscale = pscale
            try:
                del self._A1
            except:
                print("sft aux array to be refreshed")
                pass

        self.ld0 = self.wl/self.pdiam*3.6e6/dtor/self.pscale # l/D (in pixels)
        self.nld0 = self.isz / self.ld0           # nb of l/D across the frame

        tmp = self.sft(np.zeros((self.csz, self.csz)))

        if wasgoing:
            self.start(delay=self.delay,
                       dm_shmf=self.dm_shmf, atmo_shmf=self.atmo_shmf)

    # =========================================================================
    def update_signal(self, nph=1e6):
        ''' Update the strength of the signal

        Automatically sets the *phot_noise* flag to *True*
        *IF* the value provided is negative, it sets the *phot_noise* flag
        back to *False*

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
            test = self._A1 # look for existence of auxilliary arrays
        except:
            print("updating the Fourier auxilliary arrays")
            NA    = self.csz
            NB    = self.isz
            m     = self.nld0
            self._coeff = m/(NA*NB)
    
            U = np.zeros((1,NB))
            X = np.zeros((1,NA))
    
            X[0,:] = (1./NA)*(np.arange(NA)-NA/2.)
            U[0,:] =  (m/NB)*(np.arange(NB)-NB/2.)
    
            sign = -1.0
        
            self._A1 = np.exp(sign * 2j*np.pi* np.dot(np.transpose(U),X))
            self._A3 = np.exp(sign * 2j*np.pi* np.dot(np.transpose(X),U))

        #B  = np.dot(np.dot(self._A1,A2),self._A3)
        B = self._A1.dot(A2).dot(self._A3)
        return self._coeff * np.array(B)

    # =========================================================================
    def make_image(self, phscreen=None, dmmap=None):
        ''' Produce an image, given a certain number of phase screens
        -------------------------------------------------------------------
        Parameters:
        ----------
        - atmo    : (optional) atmospheric phase screen
        - qstatic : (optional) a quasi-static aberration
        - dmmap   : (optional) a deformable mirror displacement map
        ------------------------------------------------------------------- '''

        # mu2phase: DM displacement in microns to radians (x2 reflection)
        # nm2phase: phase screen in nm to radians (no x2 factor)

        mu2phase = 4.0 * np.pi / self.wl / 1e6 # convert microns to phase
        nm2phase = 2.0 * np.pi / self.wl / 1e9 # convert microns to phase

        phs = np.zeros((self.csz, self.csz), dtype=np.float64)  # full phase map

        if dmmap is not None: # a DM map was provided
            phs = mu2phase * dmmap

        if phscreen is not None: # a phase screen was provided
            phs += phscreen

        if self.corono: # perfect coronagraph simulation ! 
            wf = 0+1j*phs
        else:
            wf = np.exp(1j*phs)

        wf *= np.sqrt(self.signal / self.pupil.sum()) # signal scaling
        wf *= self.pupil                              # apply the pupil mask
        self._phs = phs * self.pupil                  # store total phase
        self.fc_pa = self.sft(wf)                     # focal plane cplx ampl
        img = np.abs(self.fc_pa)**2                   # intensity
        frm = img[self.y0:self.y1, self.x0:self.x1]   # image crop

        if self.phot_noise: # need to be recast to fit original format
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
            t = threading.Thread(target=self.__loop__, 
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
        updt     = True
        dm_cntr  = 0 # counter to keep track of updates
        atm_cntr = 0 # on the phase screens
        qst_cntr = 0 #

        dm_map  = None # arrays that store current phase
        atm_map = None # screens, if they exist

        #self.dmtype = self.DM.dmtype
        
        # 1. read the shared memory data structures if present
        # ----------------------------------------------------
        if dm_shm is not None:
            try:
                dm_map = shm(dm_shm)
            except:
                print("SHM file %s is not valid?" % (dm_shm,))

        if atmo_shm is not None:
            try:
                atm_map = shm(atmo_shm)
            except:
                print("SHM file %s is not valid?" % (atm_shm,))
                

        # 2. enter the loop
        # ----------------------------------------------------
        while self.keepgoing:
            cmd_args = "" # commands to be sent to self.make_image()

            if dm_map is not None:
                test = dm_map.get_counter()
                if test != dm_cntr:
                    cmd_args += "dmmap = dm_map.get_data(),"
                    #cmd_args += "dmmap = self.DM.wft.get_data(),"

            if atm_map is not None:
                test = atm_map.get_counter()
                if test != atm_cntr:
                    myphscreen = atm_map.get_data()
                    cmd_args += "phscreen = myphscreen,"

            exec("self.make_image(%s)" % (cmd_args,))

            time.sleep(self.delay)

# =============================================================================
# =============================================================================

class SHCam(Cam):
    ''' Shack-Hartman specialized form of camera

    ===========================================================================
    This class inherits from the more generic Cam class.
    ===========================================================================
    '''
    # ==================================================
    def __init__(self, name, csz = 256, dsz = 128, mls = 10,
                 pupil=None, wl = 0.8e-6,
                 shmf = 'SHcam.im.shm', shdir='/dev/shm/'):

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
        self.name    = name
        self.sz      = csz
        self.xs      = dsz
        self.ys      = dsz
        self.wl      = wl
        self.mls     = mls                 # u-lens array size (in lenses)

        if pupil is None:
            self.pupil = pupil.uniform_disk(csz, csz, csz//2, True)
        else:
            self.pupil = pupil

        self.shmf    = shdir+shmf          # the shared memory "file"
        self.shdir   = shdir
        self.frm0    = np.zeros((dsz,dsz)) # initial camera frame

        self.px0     = (self.sz-self.xs)/2 # pixel offset for image within array
        self.py0     = (self.sz-self.ys)/2 # pixel offset for image within array

        self.phot_noise = False            # default state for photon noise
        self.signal     = 1e6              # default number of photons in frame
        self.keepgoing  = False            # flag for the camera server

        self.dm_shmf    = None             # associated shared mem file for DM
        self.atmo_shmf  = None             # idem for atmospheric phase screen
        

        self.shm_cam = shm(self.shmf, data = self.frm0, verbose=False)

        self.cdiam = self.sz / np.float(self.mls) # oversized u-lens (in pixels)
        self.rcdiam = np.round(self.cdiam).astype(int)

        
    # ==================================================
    def make_image(self, phscreen=None, dmmap=None):
        ''' Produce a SH image, given a certain number of phase screens
        -------------------------------------------------------------------
        Parameters:
        ----------
        - atmo    : (optional) atmospheric phase screen
        - qstatic : (optional) a quasi-static aberration
        - dmmap   : (optional) a deformable mirror displacement map

        Important:   ONLY THE DM WORKS AT THE MOMENT HERE!
        -------
        ------------------------------------------------------------------- '''

        mu2phase = 4.0 * np.pi / self.wl / 1e6 # microns to phase (x2)
        nm2phase = 2.0 * np.pi / self.wl / 1e9 # nanometers to phase

        mls = self.mls
        cdiam = int(self.cdiam) # SH cell computation "diameter"
        idiam = int(self.xs / mls) # SH cell image size "diameter"

        phs = np.zeros((self.sz, self.sz))      # full phase map
        frm = np.zeros((self.ys, self.xs)) # oversized array

        # -------------------------------------------------------------------
        if dmmap is not None: # a DM map was provided
            phs = mu2phase * dmmap

        # -------------------------------------------------------------------
        if phscreen is not None: # a phase screen was provided
            phs += phscreen #* nm2phase
            
        # -------------------------------------------------------------------        
        wf = np.exp(1j*phs)
        wf[self.pupil == False] = 0+0j # re-apply the pupil map

        xl0 = int(cdiam - idiam / 2)

        for ii in range(mls * mls): # cycle ove rthe u-lenses
            wfs = np.zeros((2*cdiam, 2*cdiam), dtype=complex)
            li, lj   = ii // mls, ii % mls # i,j indices for the u-lens
            pi0 = int(np.round(li * self.xs / mls))
            pj0 = int(np.round(lj * self.xs / mls)) # image corner pixel

            ci0 = li * cdiam
            cj0 = lj * cdiam

            wfs[xl0:xl0+cdiam, xl0:xl0+cdiam] = wf[cj0:cj0+cdiam,
                                                   ci0:ci0+cdiam]
            # compute the image by the u-lens
            iml = shift(np.abs(fft(shift(wfs)))**2)
            frm[pj0:pj0+idiam, pi0:pi0+idiam] = iml[xl0:xl0+idiam,
                                                    xl0:xl0+idiam]

        # -------------------------------------------------------------------
        # temp0 = Image.fromarray(frm)
        # temp1 = temp0.resize((self.ys, self.xs), resample=1)
        # frm = np.array(temp1).astype(self.shm_cam.npdtype)


        if frm.sum() > 0:
            frm *= self.signal / frm.sum()

        if self.phot_noise: # poisson + recast
            tmp = np.random.poisson(lam=frm, size=None)
            frm = tmp.astype(self.shm_cam.npdtype)

        self.shm_cam.set_data(frm) # push the image to shared memory

