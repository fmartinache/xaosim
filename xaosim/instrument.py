import numpy as np
import threading
import time
import pupil
from shmlib import shm
from PIL import Image
import pdb

dtor  = np.pi/180.0 # to convert degrees to radians

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
            self.DM  = DM(self.name, 50, 8)
            self.cam = cam(self.name, arr_size, (320,256), 10.0, 1.6e-6)
            self.atmo = phscreen(self.name, arr_size, self.cam.ld0, 500.0)

        elif self.name == "CIAO":
            arr_size = 512
            self.DM  = DM(self.name, 11, 8)
            self.cam = cam(self.name, arr_size, (320, 256), 60.0, 0.8e-6,
            shmf = '/tmp/ciao_cam.im.shm')
            self.atmo = phscreen(self.name, arr_size, self.cam.ld0, 500.0)
        else:
            print("""No template for '%s':
            check your spelling or... 
            specify characteristics by hand!""" % (self.name))
            self.DM  = None
            self.cam = None
            self.atmo = None

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
        if self.DM != None:
            self.DM.start(delay)
            
        if self.atmo != None:
            self.atmo.start(delay)
        
        if ((self.name == "SCExAO") or (self.name == "CIAO")):
            self.cam.start(delay,
                           "/tmp/dmdisp.im.shm",
                           "/tmp/phscreen.im.shm")


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
        if self.atmo != None:
            self.atmo.stop()
        if self.cam != None:
            self.cam.stop()
        if self.DM != None:
            self.DM.stop()


# ===========================================================
# ===========================================================
class phscreen(object):
    # ==================================================
    def __init__(self, name, sz = 512, ld0 = 10, rms = 100.0,
                 shmf='/tmp/phscreen.im.shm'):

        ''' Kolmogorov type atmosphere + qstatic error

        -----------------------------------------------------
        Parameters:
        ----------
        - name : a string describing the instrument
        - sz   : the size of the Fourier array
        - ld0  : lambda/D (in pixels)
        - rms  : the RMS wavefront error in nm
        - shmf : file name to point to shared memory
        -----------------------------------------------------
        '''
        self.shmf    = shmf
        self.sz      = sz
        self.r1      = np.random.randn(sz,sz)
        self.r2      = np.random.randn(sz,sz)
        self.kolm    = pupil.kolmo(self.r1, self.r2, 5.0, ld0, 
                                   1.0, rms)
        self.pdiam = np.round(sz / ld0).astype(int)
        self.qstatic = np.zeros((self.pdiam, self.pdiam))
        self.shm_phs = shm(shmf, data = self.qstatic, verbose=False)

        self.kolm2   = np.tile(self.kolm, (2,2))
        self.kolm2   = self.kolm2[:self.sz+self.pdiam,:self.sz+self.pdiam]

        self.keepgoing = False

        self.offx = 0 # x-offset on the "large" phase screen array
        self.offy = 0 # y-offset on the "large" phase screen array

    # ==============================================================
    def start(self, delay=0.1):
        if not self.keepgoing:

            self.kolm2   = np.tile(self.kolm, (2,2))
            self.kolm2   = self.kolm2[:self.sz+self.pdiam,:self.sz+self.pdiam]

            self.keepgoing = True
            t = threading.Thread(target=self.__loop__, args=(delay,))
            t.start()
            print("phase screen server started!")
        else:
            print("phase screen server already running!")

    # ==============================================================
    def freeze(self):
        ''' ----------------------------------------
        Simple high-level accessor to interrupt the
        thread of the phase screen server infinite loop
        ---------------------------------------- '''
        if self.keepgoing:
            self.keepgoing = False
        else:
            print("The *ATMO* server was frozen")


    # ==============================================================
    def stop(self):
        ''' ----------------------------------------
        Simple high-level accessor to interrupt the
        thread of the phase screen server infinite loop
        ---------------------------------------- '''
        if self.keepgoing:
            self.kolm2[:] = 0.0
            time.sleep(0.5)
            self.keepgoing = False
        else:
            print("The *ATMO* server was not running")


    # ==============================================================
    def __loop__(self, delay = 0.1):

        while self.keepgoing:
            self.offx += 2
            self.offy += 1
            self.offx = self.offx % self.sz
            self.offy = self.offy % self.sz

            subk = self.kolm2[self.offx:self.offx+self.pdiam,
                              self.offy:self.offy+self.pdiam]
            self.shm_phs.set_data(subk)
            time.sleep(delay)


# ===========================================================
# ===========================================================
class cam(object):
    # ==================================================
    def __init__(self, name, sz = 512, 
                 (xs, ys) = (320, 256), pscale = 10.0, wl = 1.6e-6,
                 shmf = '/tmp/ircam.im.shm'):
        ''' Default instantiation of a cam object:

        -------------------------------------------------------------------
        Parameters are:
        --------------
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
            print("Array size %d should be greater than image dimensions (%d,%d)" % (sz, xs, ys))
            return(-1)

        self.pscale  = pscale              # plate scale in mas/pixel
        self.wl      = wl                  # wavelength in meters
        self.frm0    = np.zeros((ys, xs))  # initial camera frame
        self.shmf    = shmf                # the shared memory "file"

        self.px0     = (self.sz-self.xs)/2 # pixel offset for image within array
        self.py0     = (self.sz-self.ys)/2 # pixel offset for image within array
        self.self_update()

        self.phot_noise = False            # default state for photon noise
        self.signal     = 1e6              # default number of photons in frame
        self.keepgoing  = False            # flag for the camera server

    # ==================================================
    def self_update(self):
        ''' Separate call after updating the wavelength, pscale or pupil.

        Self-update: no parameters are required!
        ------------------------------------------------------------------- '''
        self.shm_cam = shm(self.shmf, data = self.frm0, verbose=False)

        if  "SCExAO" in self.name:
            self.pdiam = 7.92          # Subaru Telescope diameter (in meters)
        elif "CIAO" in self.name:
            self.pdiam = 1.0           # C2PU Telescope diameter (in meters)
        elif "HST" in self.name:
            self.pdiam = 2.4           # Hubble Space Telescope
        else:
            self.pdiam = 8.0           # default size: 8-meter telescope

        self.ld0     = self.wl / self.pdiam
        self.ld0    *= 3.6e6 / dtor / self.pscale # lambda_0/D   (in pixels)
        self.prad0   = self.sz/self.ld0/2         # pupil radius (in pixels)

        self.pupil   = self.get_pupil(self.name, self.sz, self.prad0)

    # ==================================================
    def get_pupil(self, name="", size=256, radius=50):
        ''' Choose the pupil function call according to name
        -------------------------------------------------------------------
        Parameters:
        ----------
        - name : a string describing the instrument
        - size : the square size of the array
        - radius: the radius of the pupil for this array
        ------------------------------------------------------------------- '''
        if name == "SCExAO":
            exec 'res = pupil.subaru((%d,%d), %d, spiders=True)' % (size,size, radius)
        elif name == "NICMOS":
            exec 'res = pupil.HST((%d,%d), %d, spiders=True)' % (size,size, radius)
        else:
            print("Should just be an unobstructed circular aperture by default")
            exec 'res = pupil.subaru((%d,%d), %d, spiders=False)' % (size,size, radius)
        return(res)

    # ==================================================
    def image(self, phscreen=None, dmmap=None):
        ''' For test purposes only?

        Produce an image, given a certain number of phase screens
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

        wf = (1+0j)*np.ones((self.sz, self.sz)) # full wavefront array

        if dmmap != None: # a DM map was provided
            dms = dmmap.shape[0]
            zoom = self.prad0 / (dms/2.0) # scaling factor for DM 2 WF array
            rwf = int(np.round(zoom*dms)) # resized wavefront
        
            x0 = (self.sz-rwf)/2
            x1 = x0 + rwf

            xx,yy  = np.meshgrid(np.arange(rwf)-rwf/2, np.arange(rwf)-rwf/2)
            mydist = np.hypot(yy,xx)

            phs0 = Image.fromarray(mu2phase * dmmap)   # phase map
            phs1 = phs0.resize((rwf, rwf), resample=1) # resampled phase map
            swf  = np.cos(phs1) + 1j * np.sin(phs1)    # small array for wft
            wf[x0:x1,x0:x1] = swf

            
        #pdb.set_trace()

        if phscreen != None: # a phase screen was provided
            wf[x0:x1,x0:x1] += phscreen * nm2phase

        wf[self.pupil == False] = 0+0j # re-apply the pupil map

        img = shift(np.abs(fft(shift(wf)))**2)
        frm = img[self.py0:self.py0+self.ys, self.px0:self.px0+self.xs]
        frm  *= self.signal / frm.sum()

        if self.phot_noise: # need to be recast to fit original format
            frm = np.random.poisson(lam=frm, size=None).astype(self.shm_cam.ddtype)

        self.shm_cam.set_data(frm) # push the image to shared memory


    # ==================================================
    def start(self, delay=0.1, dm_shm=None, atmo_shm=None):
        ''' ----------------------------------------
        Starts an independent thread that looks for
        changes on the DM, atmo and qstatic and
        updates the image

        Parameters:
        ----------
        - delay     : time (in seconds) between exposures
        - dm_shm    : shared mem file for DM
        - atmo_shm  : shared mem file for atmosphere
        ---------------------------------------- '''
        if not self.keepgoing:
            self.keepgoing = True
            t = threading.Thread(target=self.__loop__, 
                                 args=(delay,dm_shm, atmo_shm))
            t.start()
            print("camera server started!")
        else:
            print("camera server already running.")

    # ==================================================
    def stop(self,):
        ''' ----------------------------------------
        Simple high-level accessor to interrupt the
        thread of the camera server infinite loop
        ---------------------------------------- '''
        if self.keepgoing:
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
        if dm_shm != None:
            try:
                dm_map = shm(dm_shm)
            except:
                print("SHM file %s is not valid?" % (dm_shm,))

        if atmo_shm != None:
            try:
                atm_map = shm(atmo_shm)
            except:
                print("SHM file %s is not valid?" % (atm_shm,))
                

        # 2. enter the loop
        # ----------------------------------------------------
        while self.keepgoing:
            cmd_args = "" # commands to be sent to self.image()

            if dm_map != None:
                test = dm_map.get_counter()
                if test != dm_cntr:
                    cmd_args += "dmmap = dm_map.get_data(),"

            if atm_map != None:
                test = atm_map.get_counter()
                if test != atm_cntr:
                    cmd_args += "phscreen = atm_map.get_data(),"

            exec "self.image(%s)" % (cmd_args,)

            time.sleep(delay)

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
            print("DM server started!")
        else:
            print("DM server already running.")

    # ==================================================
    def stop(self,):
        ''' ----------------------------------------
        Simple high-level accessor to interrupt the
        thread of the DM server infinite loop
        ---------------------------------------- '''
        if self.keepgoing:
            self.keepgoing = False
        else:
            print("The *DM* server was not running")

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
                combi = np.zeros_like(self.disp0.get_data())
                for i in xrange(self.nch):
                    exec "combi += self.disp%d.get_data()" % (i,)
                self.dmd = combi
                self.disp.set_data(combi)
                #print("DM shape updated!")
            time.sleep(delay)

