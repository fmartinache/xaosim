#!/usr/bin/env python3

import functools

from weakref import WeakValueDictionary

import numpy as np
import time
from . import pupil

from .camera import Cam, SHCam, CoroCam
from .DM import DM, HexDM
from .atmosphere import Phscreen

import matplotlib.pyplot as plt
plt.ion()
plt.show()
plt.rcParams['image.origin'] = "lower"

# ===========================================================
# ===========================================================


class Singleton(type):
    _instances = WeakValueDictionary()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super(Singleton, cls).__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


def singleton(cls):
    instance = [None]

    @functools.wraps(cls)
    def wrapper(*args, **kwargs):
        print("instance[0] = ", instance[0])
        if instance[0] is None:
            instance[0] = cls(*args, **kwargs)
        else:
            print("already instantiated!")
        return instance[0]
    return wrapper


# ===========================================================
# ===========================================================


class Telescope(object):
    ''' Telecope class
    ---------------------------------------------------------------------------
    This may be overengineering at this point but there may be something to
    gained from gathering information that is relevant to a telescope into a
    separate object.

    One interest is that several cameras fed by the same telescope can have
    their own aperture mask, that can add on top of that of the telescope?

    Information stored here:
    Telescope diameter, pupil geometry, and orientation, ...

    And since I am in the process of rewriting this code, why not?
    ---------------------------------------------------------------------------

    '''
    # ==================================================
    def __init__(self, name="", size=256, radius=128, rebin=5):
        self.tname = "EMPTY"    # telescope name
        self.iname = name       # instrument name
        self.pdiam = 1.0        # telescope diameter (meters)
        self.PA = 0.0           # pupil angle (degrees)
        self.rebin = rebin      # rebin for "grey model"
        self.size = size
        self.radius = radius
        self.update_pupil()     # update pupil array

    # ==================================================
    def get_pupil(self,):
        return self.pupil

    # ==================================================
    def update_pupil(self,):
        ''' Update telescope properties based on provided name
        ---------------------------------------------------------------------------

        Default config: an unbostructed  1-meter diameter telescope
        ---------------------------------------------------------------------------
        '''
        rsz = self.size * self.rebin
        rrad = self.radius * self.rebin

        if "scexao" in self.iname.lower():
            self.tname = "Subaru"
            self.iname = "SCExAO"
            self.pdiam = 7.92  # telescope pupil diameter
            self.PA = 0.0
            pup = 1.0 * pupil.subaru(
                rsz, rsz, rrad, spiders=True, between_pix=True)

        elif "hst" in self.iname.lower():
            self.tname = "HST"
            self.iname = "NICMOS1"
            self.pdiam = 2.4
            self.PA = 45.0
            tmp = pupil.HST_NIC1(2*rsz, rrad, ang=self.PA)
            pup = tmp[rsz//2:rsz//2+rsz, rsz//2:rsz//2+rsz]

        elif "ciao" in self.iname.lower():
            self.tname = "C2PU"
            self.iname = "CIAO"
            self.pdiam = 1.0
            self.PA = 0.0
            pup = pupil.uniform_disk(rsz, rsz, rrad)

        elif "gravity" in self.iname.lower():
            self.tname = "VLT"
            self.iname = "GRAVITY+"
            self.pdiam = 8.0
            self.PA = 0.0
            pup = pupil.VLT(rsz, rsz, rrad)

        elif "pharo" in self.iname.lower():
            self.tname = "Hale"
            self.iname = "PHARO"
            self.pdiam = 4.978  # PHARO standard cross diameter
            self.PA = 0.0
            pup = pupil.PHARO(
                rsz, rrad, mask="std", between_pix=True, ang=0)
            if "med" in self.iname.lower():
                pup = pupil.PHARO(
                    rsz, rrad, mask="med", between_pix=True, ang=-2)

        elif "nirc2" in self.iname.lower():
            self.tname = "Keck2"
            self.iname = "NIRC2"
            self.pdiam = 10.2
            self.PA = -20.5
            th0 = self.PA*np.pi/180.0  # pupil angle
            pup = pupil.segmented_aperture(rsz, 3, int(rrad/3), rot=th0)

        elif "jwst" in self.iname.lower():
            self.tname = "JWST"
            self.iname = "NIRISS"
            self.pdiam = 6.5
            self.PA = 0.0
            pup = pupil.JWST(rsz, pscale=self.pdiam/rsz, aperture="CLEARP")
            if "nrm" in self.iname.lower():
                pup = pupil.JWST_NRM(rsz, pscale=self.pdiam/rsz)

        elif "kernel" in self.iname.lower():
            self.tname = "BENCH"
            self.iname = "KERNEL"
            self.pdiam = 9.75  # BMC DM largest dimension (in mm)
            self.PA = 0.0
            pup = pupil.KBENCH(rsz, pscale=self.pdiam/rsz)

        else:
            print("Default: unbstructed circular aperture")
            pup = pupil.uniform_disk(rsz, rsz, rrad)

        self.pupil = pup.reshape(
            self.size, self.rebin, self.size, self.rebin).mean(3).mean(1)

# =============================================================================
# =============================================================================


class instrument(metaclass=Singleton):

    # ==================================================
    def __init__(self, name="SCExAO", shdir='/dev/shm/', csz=320):
        print("calling constructor")
        self.name = name
        self.shdir = shdir
        self.delay = 0.1  # sets the default simulation frame rate
        self.csz = csz    # computation size for wavefronts

        self.tel = Telescope(
            name=self.name, size=self.csz, radius=self.csz//2, rebin=5)

        self.cam = None
        self.cam2 = None
        self.DM = None
        self.atmo = None

        # ---------------------------------------------------------------------
        # SCExAO template: DM, atmo and IR camera
        # ---------------------------------------------------------------------
        if "scexao" in self.name.lower():
            print("Creating %s" % (self.name,))
            if "coro" in self.name.lower():
                lstop = pupil.subaru_lstop(self.csz)
                self.cam = CoroCam(
                    "SCExAO_coro", csz=self.csz, ysz=256, xsz=320,
                    pupil=self.tel.pupil, lstop=lstop,
                    pdiam=self.tel.pdiam, pscale=16.7, wl=1.6e-6,
                    shdir=shdir, shmf="scexao_corocam.im.shm")
            else:
                self.cam = Cam(
                    name="SCExAO_chuck", csz=self.csz, ysz=256, xsz=320,
                    pupil=self.tel.pupil,
                    pdiam=self.tel.pdiam, pscale=16.7, wl=1.6e-6,
                    shdir=shdir, shmf="scexao_ircam.im.shm")

            self.DM = DM(instrument="SCExAO", dms=50, nch=8,
                         shdir=shdir, csz=self.csz, na0=49, iftype="cosine")

            self.atmo = Phscreen(name="MaunaKea", csz=self.csz,
                                 lsz=self.tel.pdiam, r0=0.5, L0=10.0,
                                 fc=24.5, correc=10.0,
                                 shdir=shdir, shmf='phscreen.wf.shm')

        # ---------------------------------------------------------------------
        # CIAO template: DM, atmosphere and 2 cameras (vis SH + imager IR)
        # ---------------------------------------------------------------------
        elif "ciao" in self.name.lower():
            print("Creating %s" % (self.name,))
            self.cam = SHCam(name="CIAO_ASO", csz=self.csz, dsz=128,
                             pupil=self.tel.pupil, wl=1.6e-6,
                             shdir=shdir, shmf="ciao_shcam.im.shm")

            self.DM = DM(instrument="CIAO", dms=11, nch=8,
                         shdir=shdir, csz=self.csz, na0=10, iftype="cone")

            self.cam2 = Cam(name="HIPiC", csz=self.csz, ysz=256, xsz=256,
                            pupil=self.tel.pupil,
                            pdiam=self.tel.pdiam, pscale=100, wl=1.6e-6,
                            shdir=shdir, shmf="hipic.im.shm")

            self.atmo = Phscreen(name="Calern", csz=self.csz,
                                 lsz=self.tel.pdiam, r0=0.2, L0=10.0,
                                 fc=5, correc=1.0,
                                 shdir=shdir, shmf='calern.wf.shm')

        # ---------------------------------------------------------------------
        # GRAVITY+ template: DM, atmosphere and 2 cameras (vis SH + imager IR)
        # ---------------------------------------------------------------------
        elif "gravity" in self.name.lower():
            print("Creating %s" % (self.name,))
            self.cam = SHCam(name="GRAVITY_WFS", csz=self.csz, dsz=240,
                             mls=40, pupil=self.tel.pupil, wl=0.7e-6,
                             shdir=shdir, shmf="gravity_shcam.im.shm")

            self.DM = DM(instrument="GRAVITY_WFS", dms=40, nch=8,
                         shdir=shdir, csz=self.csz, na0=39, iftype="cone")

            self.cam2 = Cam(name="VLT_IR", csz=self.csz, ysz=256, xsz=256,
                            pupil=self.tel.pupil,
                            pdiam=self.tel.pdiam, pscale=10, wl=1.6e-6,
                            shdir=shdir, shmf="vlt_ircam.im.shm")

            self.atmo = Phscreen(name="Paranal", csz=self.csz,
                                 lsz=self.tel.pdiam, r0=0.5, L0=10.0,
                                 fc=20, correc=1.0,
                                 shdir=shdir, shmf='paranal.wf.shm')

        # ---------------------------------------------------------------------
        # HST NICMOS 1 template: one camera (no atmo, no DM) !
        # ---------------------------------------------------------------------
        elif "hst" in self.name.lower():
            print("Creating %s" % (self.name,))
            self.cam = Cam(name="HST_NIC1", csz=self.csz, ysz=84, xsz=84,
                           pupil=self.tel.pupil,
                           pdiam=self.tel.pdiam, pscale=43, wl=1.9e-6,
                           shdir=shdir, shmf="hst_nic1.im.shm")

            self.DM = None    # no DM onboard
            self.atmo = None  # no atmosphere in space!

        # ---------------------------------------------------------------------
        # KERNEL bench template: HexDM, and IR camera
        # ---------------------------------------------------------------------
        elif "kernel" in self.name.lower():
            print("Creating %s" % (self.name,))
            self.cam = Cam(name="KBENCH", csz=self.csz, ysz=256, xsz=320,
                           pupil=self.tel.pupil,
                           pdiam=self.tel.pdiam, pscale=10., wl=1.6e-6,
                           shdir=shdir, shmf="kbench_ircam.im.shm")
            self.DM = HexDM(self.name, nr=7, nch=4,
                            shdir=shdir, shm_root="hex_disp",
                            csz=self.csz, na0=15)

            self.atmo = Phscreen(name="lab", csz=self.csz,
                                 lsz=self.tel.pdiam, r0=0.2, L0=10.0,
                                 fc=5, correc=10.0,
                                 shdir=shdir, shmf='phscreen.wf.shm')

        # ---------------------------------------------------------------------
        # NO template? Go manual.
        # ---------------------------------------------------------------------
        else:
            print("""No template for '%s':
            check your spelling or...
            specify characteristics by hand!""" % (self.name))

    # ==================================================
    def snap(self):
        ''' image snap produced without starting servers.

        Mode useful for simulations not requiring the semi real-time
        features of xaosim.
        ------------------------------------------------------------------- '''

        dmmap = None
        phscreen = None

        if self.DM is not None:
            self.DM.update(verbose=False)
            dmmap = self.DM.wft.get_data()

        if self.atmo is not None:
            phscreen = self.atmo.shm_phs.get_data()

        self.cam.make_image(dmmap=dmmap, phscreen=phscreen)
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

        ------------------------------------------------------------------- '''

        self.delay = delay

        dm_shmf = None
        atmo_shmf = None

        if self.DM is not None:
            self.DM.start(delay)
            dm_shmf = self.shdir+self.DM.shmf

        if self.atmo is not None:
            self.atmo.start(delay)
            atmo_shmf = self.shdir+self.atmo.shmf

        self.cam.start(delay=delay,
                       dm_shmf=dm_shmf,
                       atmo_shmf=atmo_shmf)

        if self.cam2 is not None:
            self.cam2.start(delay=delay,
                            dm_shmf=dm_shmf,
                            atmo_shmf=atmo_shmf)

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
        try:
            _ = self.atmo
            if self.atmo is not None:
                self.atmo.stop()
        except AttributeError:
            pass

        try:
            _ = self.DM
            if self.DM is not None:
                self.DM.stop()
        except AttributeError:
            pass

        try:
            _ = self.cam
            if self.cam is not None:
                self.cam.stop()
        except AttributeError:
            pass

        try:
            _ = self.cam2
            if self.cam2 is not None:
                self.cam2.stop()
        except AttributeError:
            pass

    # ==================================================
    def __del__(self,):
        self.close()
        del self
        # print(dict(Singleton._instances))
        # print("instance of %s destroyed" % (dict(Singleton._instances),))

    # ==================================================
    def close(self,):
        ''' A function to call after the work with the servers is over
        -------------------------------------------------------------------
        To properly release all the file descriptors that point toward
        the shared memory data structures.
        ------------------------------------------------------------------- '''
        # --- just in case ---
        self.stop()

        # --- the atmospheric phase screen ---
        try:
            test = self.atmo.shm_phs.fd
            if (self.atmo.shm_phs.fd != 0):
                self.atmo.shm_phs.close()
                del self.atmo

        except AttributeError:
            print("No atmo to shut down")

        time.sleep(self.delay)

        # --- the DM ---
        try:
            self.DM.close()
            del self.DM
        except AttributeError:
            print("No DM to shut down")

        try:
            test = self.DM.volt.fd
            if (test != 0):
                self.DM.volt.close()
                del self.DM
        except AttributeError:
            pass

        time.sleep(self.delay)
        # --- the camera itself ---
        try:
            _ = self.cam.shm_cam.fd
            if (_ != 0):
                self.cam.shm_cam.close()
                del self.cam
        except AttributeError:
            print("no primary camera to shut down")
            pass

        try:
            _ = self.cam2.shm_cam.fd
            if (_ != 0):
                self.cam2.shm_cam.close()
                del self.cam2
        except AttributeError:
            print("no secondary camera to shut down")
            pass
