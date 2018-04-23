import numpy as np
import threading
import time
import pupil
from shmlib import shm
from PIL import Image
import pdb

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

# ===========================================================
# ===========================================================
class instrument(object):
    # ==================================================
    def __init__(self, name="SCExAO"):
        ''' Default instantiation of an instrument object

        In the context of xaosim, an instrument is simply an assembly of 
        several of the following basic elements (see class definitions):
        - a deformable mirror
        - an atmospheric phase screen
        - a camera
        -------------------------------------------------------------------
        Usage:
        -----

        The default is to rely on a pre-defined template, like the one of
        SCExAO or CIAO, that is specified by the instrument name passed as 
        a parameter to the constructor.
        
        >> myinstrument = xaosim.instrument("SCExAO")

        If the name is not one of the possible templates, like for instance:

        >> myinstrument = xaosim.instrument("sauerkraut")

        the returned object is an empty shell that will have to be manually
        fed, using the class constructors below.
        ------------------------------------------------------------------- '''
        self.name = name
        if self.name == "SCExAO":
            print("Creating %s" % (self.name,))
            arr_size = 512
            dms = 50
            self.DM  = DM(self.name, dms, 8)
            self.cam = cam(self.name, arr_size, (320,256), 10.0, 1.6e-6)
            self.atmo = phscreen(self.name, arr_size, self.cam.ld0, dms, 1500.0)

        elif self.name == "CIAO":
            arr_size = 128
            dms      = 11
            self.DM  = DM(self.name, dms, 8)

            self.cam = SHcam(self.name, sz = 128, dsz = 128, mls = 10,
                             pscale = 36.56, wl = 0.8e-6, 
                             shmf = '/tmp/SHcam.im.shm')

            self.atmo = phscreen(self.name, arr_size, 10, dms, 500.0)
            
        elif self.name == "NIRC2":
            arr_size = 256
            dms = 50
            self.DM = DM(self.name, dms, 4)
            self.cam = cam(self.name, arr_size, (128,128), 10.0, 3.776e-6)
            self.atmo = phscreen(self.name, arr_size, self.cam.ld0, dms, 1500.0)
            
        else:
            print("""No template for '%s':
            check your spelling or... 
            specify characteristics by hand!""" % (self.name))
            self.DM  = None
            self.cam = None
            self.atmo = None

    # ==================================================
    def update_wavelength(self, wl=1.6e-6):
        ''' A function that changes the wavelength of operation

        This function is an instrument feature and not just a camera
        as it impacts the size of the computation of the atmospheric
        phase screen.

        Parameter:
        ---------
        - wl: the wavelength (in meters)
        ------------------------------------------------------------------- '''
        camwasgoing = False
        atmwasgoing = False

        if self.cam.keepgoing:
            camwasgoing = True
            self.cam.stop()
            
        if self.atmo.keepgoing:
            atmwasgoing = True
            self.atmo.stop()
            if (self.atmo.shm_phs.fd != 0):
                self.atmo.shm_phs.close()

        prev_atmo_shmf = self.cam.atmo_shmf
        prev_dm_shmf   = self.cam.dm_shmf
        
        self.cam = cam(self.name, self.cam.sz,
                       (self.cam.xs, self.cam.ys), self.cam.pscale, wl)
        self.atmo = phscreen(self.name, self.cam.sz, self.cam.ld0, self.atmo.rms)

        if camwasgoing:
            self.cam.start(dm_shmf=prev_dm_shmf, atmo_shmf=prev_atmo_shmf)

        if atmwasgoing:
            self.atmo.start()

    # ==================================================
    def snap(self):
        ''' image snap produced without starting servers.

        Mode useful for simulations not requiring the semi real-time
        features of xaosim.
        ------------------------------------------------------------------- '''

        cmd_args = ""
        
        if self.DM is not None:
            self.DM.update(verbose=True)
            cmd_args += "dmmap = self.DM.dmd,"

        if self.atmo is not None:
            cmd_args += 'phscreen = self.atmo.shm_phs.get_data()'

        exec "self.cam.make_image(%s)" % (cmd_args,)

        return(self.cam.get_image())
    
    # ==================================================
    def start(self, delay=0.1):
        ''' A function that starts all the components *servers*
        
        To each component is associated a server that periodically updates
        information on the global DM shape and the camera image, based on
        the status of the atmospheric phase screen and the different DM
        channels.
        -------------------------------------------------------------------
        Parameter:
        ---------
        - delay: (float) a time delay in seconds that sets a common cadence

        Usage:
        -----

        >> myinstrument.start()

        When doing things by hand, for an instrument that is not a preset,
        one needs to be careful when plugging the camera to the right shared
        memory data structures.

        Refer to the code below and the component class definitions to see
        how to proceed with your custom system.
        ------------------------------------------------------------------- '''
        if self.DM is not None:
            self.DM.start(delay)
            
        if self.atmo is not None:
            self.atmo.start(delay)
        
        if (self.name == "SCExAO"):
            self.cam.start(delay,
                           "/tmp/dmdisp.im.shm",
                           "/tmp/phscreen.im.shm")

        if  (self.name == "CIAO"):
            self.cam.start(delay, "/tmp/dmdisp.im.shm")

    # ==================================================
    def stop(self,):
        ''' A function that turns off all servers (and their threads)

        After this, the python session can safely be closed.
        -------------------------------------------------------------------
        Usage:
        -----
        
        >> myinstrument.stop()

        Simple no?
        ------------------------------------------------------------------- '''
        if self.atmo is not None:
            self.atmo.stop()
        if self.cam is not None:
            self.cam.stop()
        if self.DM is not None:
            self.DM.stop()

    # ==================================================
    def close(self,):
        ''' A function to call after the work with the severs is over
        -------------------------------------------------------------------
        To properly release all the file descriptors that point toward
        the shared memory data structures.
        ------------------------------------------------------------------- '''
        # --- just in case ---
        self.stop()
        
        # --- the camera itself ---
        if (self.cam.shm_cam.fd != 0):
            self.cam.shm_cam.close()

        # --- the atmospheric phase screen ---
        if (self.atmo.shm_phs.fd != 0):
            self.atmo.shm_phs.close()

        # --- the different DM channels ---
        for i in xrange(self.DM.nch):
            exec "test = self.DM.disp%d.fd" % (i,)
            if (test != 0):
                exec "self.DM.disp%d.close()" % (i,)

        # --- more DM simulation relevant files ---
        if (self.DM.disp.fd != 0):
            self.DM.disp.close()

        if (self.DM.volt.fd != 0):
            self.DM.volt.close()

        self.cam = None
        self.DM = None
        self.atmo = None
            
# ===========================================================
# ===========================================================
class phscreen(object):
    '''Atmospheric Kolmogorov-type phase screen.

    ====================================================================

    Class Attributes:
    ----------------
    - sz      : size (sz x sz) of the phase screen         (in pixels)
    - pdiam   : diameter of the aperture within this array (in pixels)
    - rndarr  : uniformly distributed random array           (sz x sz)
    - kolm    : the original phase screen                    (sz x sz)
    - kolm2   : the oversized phase screen    ((sz + pdiam) x (sz + pdiam))
    - qstatic : an optional quasi static aberration    (pdiam x pdiam)
    - rms     : total phase screen rms value           (in nanometers)
    - rms_i   : instant rms inside the pupil           (in nanometers)

    Comment:
    -------
    While the attributes are documented here for reference, the prefered
    way of interacting with them is via the functions defined within the
    class.
    ====================================================================

    '''
    # ==================================================
    def __init__(self, name, sz = 512, ld0 = 10, dms = 50, rms = 100.0,
                 shmf='/tmp/phscreen.im.shm'):

        ''' Kolmogorov type atmosphere + qstatic error

        -----------------------------------------------------
        Parameters:
        ----------
        - name : a string describing the instrument
        - sz   : the size of the Fourier array
        - ld0  : lambda/D for the camera (in pixels)
        - dms  : the size of the DM (to simulate AO correction)
        - rms  : the RMS wavefront error in nm
        - shmf : file name to point to shared memory
        -----------------------------------------------------
        '''
        self.shmf    = shmf
        self.sz      = sz
        self.rms     = np.float(rms)
        self.rms_i   = np.float(rms)
        self.rndarr  = np.random.rand(sz,sz)
        self.correc  = 1.0 # at first, no AO correction
        self.fc      = dms / 2.0
        self.ld0     = ld0
        self.kolm    = pupil.kolmo(self.rndarr, self.fc, self.ld0, 
                                   self.correc, self.rms)

        self.pdiam = np.round(sz / ld0).astype(int)
        self.qstatic = np.zeros((self.pdiam, self.pdiam))
        self.shm_phs = shm(shmf, data = self.qstatic, verbose=False)

        self.kolm2   = np.tile(self.kolm, (2,2))
        self.kolm2   = self.kolm2[:self.sz+self.pdiam,:self.sz+self.pdiam]

        self.keepgoing = False

        self.offx = 0 # x-offset on the "large" phase screen array
        self.offy = 0 # y-offset on the "large" phase screen array

        self.ttc     = False         # Tip-tilt correction flag
        
        # auxilliary array (for tip-tilt correction)
        self.xx, self.yy  = np.meshgrid(np.arange(self.pdiam)-self.pdiam/2,
                                        np.arange(self.pdiam)-self.pdiam/2)
        self.xxnorm2 = np.sum(self.xx**2)
        self.yynorm2 = np.sum(self.yy**2)
        
    # ==============================================================
    def start(self, delay=0.1):
        ''' ----------------------------------------
        High-level accessor to start the thread of 
        the phase screen server infinite loop
        ---------------------------------------- '''
        if not self.keepgoing:

            self.kolm2   = np.tile(self.kolm, (2,2))
            self.kolm2   = self.kolm2[:self.sz+self.pdiam,:self.sz+self.pdiam]

            self.keepgoing = True
            t = threading.Thread(target=self.__loop__, args=(delay,))
            t.start()
            print("The *ATMO* phase screen server was started")
        else:
            print("The *ATMO* phase screen server is already running")

    # ==============================================================
    def freeze(self):
        ''' ----------------------------------------
        High-level accessor to interrupt the thread 
        of the phase screen server infinite loop
        ---------------------------------------- '''
        if self.keepgoing:
            self.keepgoing = False
        else:
            print("The *ATMO* server was frozen")


    # ==============================================================
    def stop(self):
        ''' ----------------------------------------
        High-level accessor to interrupt the thread
        of the phase screen server infinite loop
        ---------------------------------------- '''
        if self.keepgoing:
            self.kolm2[:] = 0.0
            time.sleep(0.5)
            self.keepgoing = False
            print("The *ATMO* server was stopped")
        else:
            print("The *ATMO* server was not running")


    # ==============================================================
    def update_screen(self, correc=None, rms=None, fc=None):
        ''' ------------------------------------------------
        Generic update of the properties of the phase-screen
        
        ------------------------------------------------ '''
        if rms is not None:
            self.rms = rms
            
        if correc is not None:
            self.correc = correc

        if fc is not None:
            self.fc = fc
            
        self.kolm    = pupil.kolmo(self.rndarr, self.fc, self.ld0, 
                                   self.correc, self.rms)

        self.kolm2   = np.tile(self.kolm, (2,2))
        self.kolm2   = self.kolm2[:self.sz+self.pdiam,:self.sz+self.pdiam]

        if self.keepgoing is False:
            # case that must be adressed:
            # amplitude changed when atmo is frozen!
            subk = self.kolm2[self.offx:self.offx+self.pdiam,
                              self.offy:self.offy+self.pdiam].copy()
            
            if self.ttc is True:            
                ttx = np.sum(subk*self.xx) / self.xxnorm2
                tty = np.sum(subk*self.yy) / self.yynorm2
                subk -= ttx * self.xx + tty * self.yy

            self.rms_i = subk.std()
            self.shm_phs.set_data(subk)

    # ==============================================================
    def update_rms(self, rms):
        ''' ------------------------------------------
        Update the rms of the phase screen on the fly

        Parameter:
        ---------
        - rms: the rms of the phase screen (in nm)

        Special case of the update_screen() call, kept
        for legacy reasons. TB discarded in the near future.
        -----------------------------------------  '''
        self.update_screen(rms=rms)
                
    # ==============================================================
    def __loop__(self, delay=0.1):
        ''' ------------------------------------------
        Main loop: frozen screen slid over the aperture

        Options:
        ---------
        - delay: the time delay between refresh  (0.1 sec)
        -----------------------------------------  '''

        while self.keepgoing:
            self.offx += 2
            self.offy += 1
            self.offx = self.offx % self.sz
            self.offy = self.offy % self.sz

            subk = self.kolm2[self.offx:self.offx+self.pdiam,
                              self.offy:self.offy+self.pdiam].copy()

            if self.ttc is True:
                ttx = np.sum(subk*self.xx) / self.xxnorm2
                tty = np.sum(subk*self.yy) / self.yynorm2
                subk -= ttx * self.xx + tty * self.yy

            self.rms_i = subk.std()
            self.shm_phs.set_data(subk)
            time.sleep(delay)


# ===========================================================
# ===========================================================
class cam(object):
    ''' Generic monochoromatic camera class

    ====================================================================

    Class Attributes:
    ----------------

    Comment:
    -------
    While the attributes are documented here for reference, the prefered
    way of interacting with them is via the functions defined within the
    class.
    ====================================================================

    '''
    # ==================================================
    def __init__(self, name, sz = 512, 
                 (xs, ys) = (320, 256), pscale = 10.0, wl = 1.6e-6,
                 shmf = '/tmp/ircam.im.shm'):
        ''' Default instantiation of a cam object:

        -------------------------------------------------------------------
        Parameters are:
        --------------
        - name    : a string describing the instrument
        - sz      : an array size for Fourier computations (>= than xs,ys!)
        - (xs,ys) : the dimensions of the actually produced image
        - pscale  : the plate scale of the image, in mas/pixel
        - wl      : the central wavelength of observation, in meters
        - shmf    : the name of the file used to point the shared memory
        ------------------------------------------------------------------- '''

        self.name    = name
        self.sz      = sz
        self.xs      = xs
        self.ys      = ys
        
        if ((sz < xs) or (sz < ys)):
            print("Array size %d should be > image sizs (%d,%d)" % (sz, xs, ys))
            return(-1)

        self.pscale  = pscale              # plate scale in mas/pixel
        self.wl      = wl                  # wavelength in meters
        self.frm0    = np.zeros((ys, xs))  # initial camera frame
        self.shmf    = shmf                # the shared memory "file"

        self.px0     = (self.sz-self.xs)/2 # pixel offset for image within array
        self.py0     = (self.sz-self.ys)/2 # pixel offset for image within array

        self.phot_noise = False            # default state for photon noise
        self.signal     = 1e6              # default number of photons in frame
        self.keepgoing  = False            # flag for the camera server

        self.dm_shmf    = None             # associated shared memory file for DM
        self.atmo_shmf  = None             # idem for atmospheric phase screen

        self.corono     = False            # if True: perfect coronagraph
        self.self_update()

    # ==================================================
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

    # ==================================================
    def self_update(self):
        ''' Separate call after updating the wavelength, pscale or pupil.

        Self-update: no parameters are required!
        ------------------------------------------------------------------- '''
        wasgoing = False
        
        if self.keepgoing:
            wasgoing = True
            self.stop()

        self.shm_cam = shm(self.shmf, data = self.frm0, verbose=False)
        self.prebin = 1
        
        if  "SCExAO" in self.name:
            self.pdiam = 7.92          # Subaru Telescope diameter (in meters)
            self.prebin = 5
        elif "CIAO" in self.name:
            self.pdiam = 1.0           # C2PU Telescope diameter (in meters)
        elif "HST" in self.name:
            self.pdiam = 2.4           # Hubble Space Telescope
        elif "NIRC2" in self.name:
            self.pdiam = 10.2          # Keck II Telescope "diameter"
        else:
            self.pdiam = 8.0           # default size: 8-meter telescope

        self.ld0     = self.wl / self.pdiam
        self.ld0    *= 3.6e6 / dtor / self.pscale # lambda_0/D   (in pixels)
        self.prad0   = self.sz/self.ld0/2         # pupil radius (in pixels)

        self.pupil   = self.get_pupil(self.name, self.sz, self.prad0,
                                      rebin=self.prebin)

        if wasgoing:
            self.start()
        
    # ==================================================
    def get_pupil(self, name="", size=256, radius=50, rebin=1):
        ''' Choose the pupil function call according to name
        -------------------------------------------------------------------
        Parameters:
        ----------
        - name : a string describing the instrument
        - size : the square size of the array
        - radius: the radius of the pupil for this array
        ------------------------------------------------------------------- '''
        rsz = size * rebin
        rrad = radius * rebin
        if name == "SCExAO":
            res = pupil.subaru((rsz, rsz), rrad, spiders=True)

        elif name == "NICMOS":
            res = pupil.HST((rsz, rsz), rrad, spiders=True)
            
        elif name == "NIRC2":
            th0 = -20.5*np.pi/180.0 # pupil angle
            res = pupil.segmented_aperture(rsz, 3, int(rrad/3), rot=th0)
        else:
            print("Default: unobstructed circular aperture")
            res = pupil.uniform_disk((rsz, rsz), rrad)

        res = res.reshape(size, rebin, size, rebin).mean(3).mean(1)
        return(res)

    # ==================================================
    def get_image(self, ):
        '''Returns the image currently avail on shared memory
        -------------------------------------------------------
        '''
        return(self.shm_cam.get_data())

    # ==================================================
    def make_image(self, phscreen=None, dmmap=None):
        '''Produce an image, given a certain number of phase screens
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

        phs = np.zeros((self.sz, self.sz), dtype=np.float128)  # full phase map

        if dmmap is not None: # a DM map was provided
            dms = dmmap.shape[0]
            zoom = self.prad0 / (dms/2.0) # scaling factor for DM 2 WF array
            rwf = int(np.round(zoom*dms)) # resized wavefront
        
            x0 = (self.sz-rwf)/2
            x1 = x0 + rwf

            phs0 = Image.fromarray(mu2phase * dmmap)   # phase map
            phs1 = phs0.resize((rwf, rwf), resample=1) # resampled phase map
            phs[x0:x1,x0:x1] = phs1


        if phscreen is not None: # a phase screen was provided
            phs[x0:x1,x0:x1] += phscreen * nm2phase

        if self.corono:
            wf = 0+1j*phs
        else:
            wf = np.exp(1j*phs)

        wf *= np.sqrt(self.signal / self.pupil.sum())
        wf[self.pupil == False] = 0+0j # re-apply the pupil map

        self.fc_pa = fft(shift(wf)) / self.sz # focal plane complex amplitude

        img = shift(np.abs(self.fc_pa)**2)
               
        frm = img[self.py0:self.py0+self.ys, self.px0:self.px0+self.xs]

        if self.phot_noise: # need to be recast to fit original format
            frm = np.random.poisson(lam=frm.astype(np.float64), size=None)
            
        # push the image to shared memory
        self.shm_cam.set_data(frm.astype(self.shm_cam.npdtype))


    # ==================================================
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
        if not self.keepgoing:
            self.dm_shmf = dm_shmf
            self.atmo_shmf = atmo_shmf
            
            self.keepgoing = True
            t = threading.Thread(target=self.__loop__, 
                                 args=(delay,self.dm_shmf, self.atmo_shmf))
            t.start()
            print("The *CAMERA* server was started")
        else:
            print("The *CAMERA* server is already running")

    # ==================================================
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

    # ==================================================
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

            if atm_map is not None:
                test = atm_map.get_counter()
                if test != atm_cntr:
                    cmd_args += "phscreen = atm_map.get_data(),"

            exec "self.make_image(%s)" % (cmd_args,)

            time.sleep(delay)

# ===========================================================
# ===========================================================
class SHcam(cam):
    # ==================================================
    def __init__(self, name, sz = 256, dsz = 128, mls = 10,
                 pscale = 36.56, wl = 0.8e-6,
                 shmf = '/tmp/SHcam.im.shm'):

        ''' Instantiation of a SH camera

        -------------------------------------------------------------------
        Parameters:
        ----------
        - name    : a string describing the instrument
        - sz      : an array size for Fourier computations
        - dsz     : size of the detector in pixels
        - mls     : # of lenses in micro-lens array (mls X mls)
        - pscale  : the plate scale of the image, in mas/pixel
        - wl      : the central wavelength of observation, in meters
        - shmf    : the name of the file used to point the shared memory

        ------------------------------------------------------------------- '''
        self.name    = name
        self.sz      = sz
        self.xs      = dsz
        self.ys      = dsz
        self.wl      = wl
        self.pscale  = pscale
        self.mls     = mls                 # u-lens array size (in lenses)
        
        self.shmf    = shmf                # the shared memory "file"
        self.frm0    = np.zeros((dsz,dsz)) # initial camera frame

        self.px0     = (self.sz-self.xs)/2 # pixel offset for image within array
        self.py0     = (self.sz-self.ys)/2 # pixel offset for image within array

        self.phot_noise = False            # default state for photon noise
        self.signal     = 1e6              # default number of photons in frame
        self.keepgoing  = False            # flag for the camera server

        self.dm_shmf    = None             # associated shared mem file for DM
        self.atmo_shmf  = None             # idem for atmospheric phase screen
        
        self.pupil   = self.get_pupil(self.name, self.sz, self.sz/2)

        #super(SHcam, self).self_update()
        self.shm_cam = shm(self.shmf, data = self.frm0, verbose=False)

        self.cdiam = self.sz / np.float(self.mls) # oversized u-lens (in pixels)
        self.rcdiam = np.round(self.cdiam).astype(int)

        #np.round(2*self.prad0/self.mls) # u-lens size (in pixels)
        
    # ==================================================
    def make_image(self, phscreen=None, dmmap=None):
        ''' For test purposes only?

        Produce a SH image, given a certain number of phase screens
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

        mls = self.mls
        cdiam = self.cdiam
        rcdiam = self.rcdiam

        phs = np.zeros((self.sz, self.sz))      # full phase map
        frm = np.zeros((self.ys, self.xs)) # oversized array

        # -------------------------------------------------------------------
        if dmmap is not None: # a DM map was provided
            dms = dmmap.shape[0]
            #zoom = self.prad0 / (dms/2.0) # scaling factor for DM 2 WF array
            rwf = self.sz#int(np.round(zoom*dms)) # resized wavefront
        
            x0 = (self.sz-rwf)/2
            x1 = x0 + rwf

            xx,yy  = np.meshgrid(np.arange(rwf)-rwf/2, np.arange(rwf)-rwf/2)
            mydist = np.hypot(yy,xx)

            phs0 = Image.fromarray(mu2phase * dmmap)   # phase map
            phs1 = phs0.resize((rwf, rwf), resample=1) # resampled phase map
            phs[x0:x1,x0:x1] = phs1

        # -------------------------------------------------------------------
        if phscreen is not None: # a phase screen was provided
            phs[x0:x1,x0:x1] += phscreen * nm2phase

        # -------------------------------------------------------------------
        wf = np.exp(1j*phs)
        wf[self.pupil == False] = 0+0j # re-apply the pupil map

        xl0 = np.round(cdiam/2)

        for i in xrange(mls * mls): # cycle ove rthe u-lenses
            wfs = np.zeros((2*cdiam, 2*cdiam), dtype=complex)
            li, lj = i / mls, i % mls # i,j indices for the u-lens
            pi0, pj0 = np.round(li * cdiam), np.round(lj * cdiam)

            wfs[xl0:xl0+rcdiam, xl0:xl0+rcdiam] = wf[pi0:pi0+rcdiam, pj0:pj0+rcdiam]

            # compute the image by the u-lens
            iml = shift(np.abs(fft(shift(wfs)))**2)
            
            frm[pi0:pi0+rcdiam, pj0:pj0+rcdiam] = iml[xl0:xl0+rcdiam, xl0:xl0+rcdiam]

        # -------------------------------------------------------------------
        temp0 = Image.fromarray(frm)
        temp1 = temp0.resize((self.ys, self.xs), resample=1)
        frm = np.array(temp1).astype(self.shm_cam.npdtype)


        if frm.sum() > 0:
            frm *= self.signal / frm.sum()

        if self.phot_noise: # need to be recast to fit original format
            frm = np.random.poisson(lam=frm,
                                    size=None).astype(self.shm_cam.npdtype)

        self.shm_cam.set_data(frm) # push the image to shared memory


# ===========================================================
# ===========================================================

class DM(object):
    ''' -------------------------------------------------------------------
    Deformable mirror class

    The displacement map *self.dmd* is in microns
    ------------------------------------------------------------------- '''

    # ==================================================
    def __init__(self, instrument="SCExAO", dms=50, nch=8, 
                 shm_root="/tmp/dmdisp"):
        ''' -----------------------------------------
        Constructor for instance of deformable mirror
        Parameters:
        ----------
        - instrument : a string
        - dms: an integer (linear size of the DM)
        - nch: number of channels
        - shm_root: the root name for shared mem files
        ----------------------------------------- '''
        self.keepgoing = False
        self.dms = dms # deformable mirror size of (dms x dms) actuators
        self.nch = nch # numbers of channels to drive the DM
        self.dmd0 = np.zeros((dms, dms), dtype=np.float32)
        self.shm_cntr = np.zeros(nch) - 1
        self.disp = shm('%s.im.shm' % (shm_root,), 
                        data=self.dmd0, verbose=False)

        for i in xrange(nch):
            exec '''self.disp%d = shm(fname='%s%d.im.shm', 
            data=self.dmd0, verbose=False)''' % (i,shm_root,i)

        if "SCExAO" in instrument:
            self.volt = shm("/tmp/dmvolt.im.shm", 
                            data=self.dmd0, verbose=False)

    # ==================================================
    def get_counter_channel(self, chn):
        ''' ----------------------------------------
        Return the current channel counter value.
        Reads from the already-opened shared memory
        data structure.
        ---------------------------------------- '''
        if chn < self.nch:
            exec "cnt = self.disp%d.get_counter()" % (chn,)
        else:# chn == nch:
            cnt = self.disp.get_counter()
        return(cnt)

    # ==================================================
    def start(self,delay=0.1):
        ''' ----------------------------------------
        Starts an independent thread that looks for
        changes on all channels, and updates the 
        actual DM shape.
        ---------------------------------------- '''
        if not self.keepgoing:
            self.keepgoing = True
            t = threading.Thread(target=self.__loop__, args=(delay,))
            t.start()
            print("The *DM* server was started")
        else:
            print("The *DM* server is already running")

    # ==================================================
    def stop(self,):
        ''' ----------------------------------------
        Simple high-level accessor to interrupt the
        thread of the DM server infinite loop
        ---------------------------------------- '''
        if self.keepgoing:
            self.keepgoing = False
            print("The *DM* server was stopped")
        else:
            print("The *DM* server was not running")

    # ==================================================
    def update(self, verbose=False):
        ''' ----------------------------------------
        DM state update.

        Reads all existing channels and combines
        them to update the actual DM shape.
        ---------------------------------------- '''
        combi = np.zeros_like(self.disp0.get_data())
        for i in xrange(self.nch):
            exec "combi += self.disp%d.get_data()" % (i,)
        self.dmd = combi
        self.disp.set_data(combi)
        if verbose:
            print("DM shape updated!")
        
    # ==================================================
    def __loop__(self, delay=0.1):
        ''' ----------------------------------------
        Thread (infinite loop) that updates the DM
        shape until told to stop.

        Do not use directly: call start_server()
        and stop_server() instead.
        ---------------------------------------- '''
        updt = True
        while self.keepgoing:
            for i in xrange(self.nch):
                test = self.get_counter_channel(i)
                if test != self.shm_cntr[i]:
                    self.shm_cntr[i] = test
                    updt = True
            if updt:
                updt = False
                self.update()
            time.sleep(delay)

