#!/usr/bin/env python3

import functools

from weakref import WeakValueDictionary

import numpy as np
import time
from . import pupil

from .camera import Cam, SHCam, CoroCam
from .DM import DM, HexDM
from .atmosphere import Phscreen
from .pupil import uniform_disk as ud

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
        ''' Instantiation of a Telescope:

        Parameters:
        ----------
        - name   : a string descriving the telescope
        - size   : size of the array describing the telescope pupil (pixels)
        - radius : pupil radius (in pixels) within the array
        - rebin  : ovsersampling parameter for finer pupil model
        ------------------------------------------------------------------- '''
        self.tname = "EMPTY"    # telescope name
        self.iname = name       # instrument name
        self.pdiam = 1.0        # telescope diameter (meters)
        self.PA = 0.0           # pupil angle (degrees)
        self.rebin = rebin      # rebin for "grey model"
        self.size = size
        self.radius = radius
        self.update_pupil()     # update pupil array

    # ==================================================
    def __str__(self):
        msg = f"Telescope : {self.tname} - diameter : {self.pdiam:.1f} meters\n"
        msg += f"Instrument : - {self.iname}\n"
        return msg

    # ==================================================
    def get_pupil(self,):
        return self.pupil

    # ==================================================
    def update_pupil(self,):
        ''' Update telescope properties based on provided name
        -----------------------------------------------------------------------

        Default config: an unobstructed  1-meter diameter telescope
        -----------------------------------------------------------------------
        '''
        rsz = self.size * self.rebin
        rrad = self.radius * self.rebin

        if "scexao" in self.iname.lower():
            self.tname = "Subaru"
            self.iname = "SCExAO"
            self.pdiam = 7.92  # telescope pupil diameter
            pup = 1.0 * pupil.subaru(
                rsz, rsz, rrad, spiders=True, between_pix=True)

        elif "elt" in self.iname.lower():
            self.tname = "ELT"
            self.iname = "petalometer"
            self.pdiam = 39  # telescope "diameter"
            pup = pupil.ELT(rsz, pscale=self.pdiam/rsz, spiders=True)

        elif "hst" in self.iname.lower():
            self.tname = "HST"
            self.iname = "NICMOS1"
            self.pdiam = 2.4
            self.PA = 45.0
            tmp = pupil.HST_NIC1(2*rsz, rrad, ang=self.PA)
            pup = tmp[rsz//2:rsz//2+rsz, rsz//2:rsz//2+rsz]

        elif "aoc" in self.iname.lower():
            self.tname = "C2PU"
            self.iname = "AOC"
            self.pdiam = 1.0
            pup = pupil.uniform_disk(rsz, rsz, rrad)

        elif "baldr" in self.iname.lower():
            self.tname = "VLTI"
            if "ut" in self.iname.lower():
                self.pdiam = 8.0
                pup = pupil.VLT(rsz, rsz, rrad)
                self.iname = "BALDR - UT"
            else:
                self.pdiam = 1.8
                pup = pupil.VLT(rsz, rsz, rrad)
                self.iname = "BALDR - AT"

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
            self.pdiam = 6.6
            self.PA = 0.0
            if "nrm" in self.iname.lower():
                pup = pupil.JWST_NRM(rsz, pscale=self.pdiam/rsz)
            else:
                pup = pupil.JWST(rsz, pscale=self.pdiam/rsz, aperture="CLEARP")
            self.iname = "NIRISS"

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
                self.add_corono_camera(
                    name="SCExAO_coro", ysz=256, xsz=320, lstop=lstop,
                    pscale=16.7, wl=1.6e-6, slot=1)
            else:
                self.add_imaging_camera(
                    name="SCExAO_chuck", ysz=256, xsz=320,
                    pscale=16.7, wl=1.6e-6, slot=1)

            self.add_membrane_DM(
                dms=50, nch=8, na0=49, iftype="cosine",
                ifr0=1.0, dtype=np.float32)

            self.add_phscreen(
                name="MaunaKea", r0=0.5, L0=10.0, fc=24.5, correc=10.0)

        # ---------------------------------------------------------------------
        # BALDR template: DM, atmosphere & pupil viewing camera
        # ---------------------------------------------------------------------
        elif "baldr" in self.name.lower():
            print(f"Creating {self.name}")
            self.add_corono_camera(name="BALDR", ysz=12, xsz=12, pview=12,
                                   pscale=20, wl=1.6e-6, slot=1)
            # Zernike phase mask
            self.cam.fpm = 1 - (1 - 1j) * ud(csz, csz, self.cam.ld0,
                                             between_pix=True)

            self.add_membrane_DM(
                dms=12, nch=8, na0=10, iftype="cosine", ifr0=1.0)

            self.add_phscreen(
                name="Paranal", r0=0.5, L0=10.0, fc=24.5, correc=1.0)

        # ---------------------------------------------------------------------
        # AOC template: DM, atmosphere and 2 cameras (vis SH + imager IR)
        # ---------------------------------------------------------------------
        elif "aoc" in self.name.lower():
            print("Creating %s" % (self.name,))
            self.add_SH_camera(
                name="AOC_ASO", dsz=128, mls=10, wl=0.8e-6, slot=1)

            self.add_membrane_DM(
                dms=11, nch=8, na0=10, iftype="cone", ifr0=1.0)

            self.add_imaging_camera(
                name="HiPiC", ysz=256, xsz=256, pscale=100, wl=1.6e-6, slot=2)

            self.add_phscreen(
                name="Calern", r0=0.2, L0=10.0, fc=5, correc=1.0)

        # ---------------------------------------------------------------------
        # GRAVITY+ template: DM, atmosphere and 2 cameras (vis SH + imager IR)
        # ---------------------------------------------------------------------
        elif "gravity" in self.name.lower():
            print("Creating %s" % (self.name,))
            self.add_SH_camera(
                name="GRAVITY_WFS", dsz=240, mls=40, wl=0.7e-6, slot=1)

            self.add_membrane_DM(
                dms=40, nch=8, na0=39, iftype="cone", ifr0=1.0)

            self.add_imaging_camera(
                name="VLT_IR", ysz=320, xsz=320, pscale=10.0, wl=1.6e-6,
                slot=2)

            self.add_phscreen(
                name="Paranal", r0=0.5, L0=10.0, fc=20, correc=10.0)

        # ---------------------------------------------------------------------
        # HST NICMOS 1 template: one camera (no atmo, no DM) !
        # ---------------------------------------------------------------------
        elif "hst" in self.name.lower():
            print("Creating %s" % (self.name,))
            self.add_imaging_camera(
                name="HST_NIC1", ysz=84, xsz=84, pscale=43.0, wl=1.9e-6,
                slot=1)

        # ---------------------------------------------------------------------
        # JWST NIRISS template: one camera for now (no DM control yet?)
        # ---------------------------------------------------------------------
        elif "jwst" in self.name.lower():
            print("Creating %s" % (self.name,))
            iname = "NIRISS"
            if "nrm" in self.name.lower():
                iname += "_NRM"
            self.add_imaging_camera(
                name=iname, ysz=80, xsz=80, pscale=65.6, wl=4.8e-6,
                slot=1)

        # ---------------------------------------------------------------------
        # KERNEL bench template: HexDM, and IR camera
        # ---------------------------------------------------------------------
        elif "kernel" in self.name.lower():
            print("Creating %s" % (self.name,))
            self.add_imaging_camera(
                name="kbench", ysz=256, xsz=320, pscale=10.0, wl=1.6e-6,
                slot=1)

            self.add_hex_DM(nr=7, nch=8, na0=15, srad=323.75)

            self.add_phscreen(
                name="Valrose", r0=1.0, L0=10.0, fc=7.5, correc=10.0)

        # ---------------------------------------------------------------------
        # ELT petalometer template:
        # ---------------------------------------------------------------------
        elif "elt" in self.name.lower():
            print("Creating %s" % (self.name,))
            self.add_imaging_camera(
                name="petalometer2.1", ysz=256, xsz=256, pscale=5.0, wl=2.1e-6,
                slot=1)

            self.add_imaging_camera(
                name="petalometer2.3", ysz=256, xsz=256, pscale=5.0, wl=2.3e-6,
                slot=2)

            self.add_membrane_DM(dms=100, nch=4, na0=99,
                                 iftype="cosine", ifr0=1.0)
            self.add_phscreen(
                name="Armazones", r0=0.144, L0=25.0, fc=15, wl=0.5e-6, correc=1.0)

        # ---------------------------------------------------------------------
        # NO template? Go manual.
        # ---------------------------------------------------------------------
        else:
            print("""No template available for '%s':
            assuming that you want a custom configuration...""" % (self.name))

    # ==================================================
    def __str__(self):
        msg = "XAOSIM virtual instrument:\n" + 26*"-" + "\n"
        msg += self.tel.__str__()
        msg += f"\n"

        if self.cam is not None:
            msg += self.cam.__str__()
            return msg

    # ==================================================
    def _install_camera(self, cam, slot=1):
        ''' -------------------------------------------------------------------
        Attempts to integrate a camera to the instrument.

        Checks for the availability of the requested slot.
        Parameters:
        ----------
        - cam: an instance of camera
        - slot: marks the "position" of the camera (1 or 2, default = 1)

        Notes:
        -----
        At the moment, I only anticipate a user to want to have two distinct
        cameras as part of one instrument and only have two slots with hard
        coded names. I could change that and offer the possibility of having
        an ever extensible list of cameras?
        ------------------------------------------------------------------- '''
        if slot == 1:
            if self.cam is None:
                self.cam = cam
                print("Camera installed in slot 1. Handle is xx.cam")
            else:
                print("Camera already in place in slot 1")
        else:
            if self.cam2 is None:
                self.cam2 = cam
                print("Camera installed in slot 2. Handle is xx.cam2")
            else:
                print("Camera already in place in slot 2")

    # ==================================================
    def _get_handle(self, slot=1):
        ''' -------------------------------------------------------------------
        Notes: clearly, it would be better to have a list of slots rather
        than two hardcoded handles... but can I afford this change without
        making some of my current users really angry at me?
        ------------------------------------------------------------------- '''
        if slot == 1:
            try:
                _ = self.cam
                return self.cam
            except AttributeError:
                return None
        else:
            try:
                _ = self.cam2
                return self.cam2
            except AttributeError:
                return None

    # ==================================================
    def _delete_handle(self, slot=1):
        ''' -------------------------------------------------------------------
        Notes: clearly, it would be better to have a list of slots rather
        than two hardcoded handles... but can I afford this change without
        making some of my current users really angry at me?
        ------------------------------------------------------------------- '''
        if slot == 1:
            try:
                del self.cam
                self.cam = None
                print("Handle for slot #%d now free" % (slot,))
            except AttributeError:
                print("Handle for slot #%d already free" % (slot,))
                pass
        else:
            try:
                del self.cam2
                self.cam2 = None
                print("Handle for slot #%d now free" % (slot,))
            except AttributeError:
                print("Handle for slot #%d already free" % (slot,))
                pass

    # ==================================================
    def clear_camera_slot(self, slot=1):
        ''' -------------------------------------------------------------------
        Removes a camera from the instrument.

        Removal is required prior to installing a new camera on a given slot.
        ------------------------------------------------------------------- '''
        tmp = self._get_handle(slot=slot)

        try:
            tmp.stop()
            _ = tmp.shm_cam.fd
            if _ != 0:
                tmp.shm_cam.close()
                self._delete_handle(slot=slot)
        except AttributeError:
            print("Camera slot #%d already clear" % (slot,))
            pass
        del tmp

    # ==================================================
    def add_imaging_camera(self, name="ircam", ysz=256, xsz=320,
                           pscale=16.7, wl=1.6e-6, slot=1):
        ''' -------------------------------------------------------------------
        Adds a conventional imaging camera to the instrument.

        Parameters:
        ----------
        - name   : string describing the camera (used for shm file name)
        - ysz    : vertical size of the detector in pixels (default = 256)
        - xsz    : horizontal size of the detector in pixels (default = 320)
        - pscale : detector plate scale in mas/pixels (default = 16.7)
        - wl     : wavelength of observation in meters (default = 1.6e-6)
        - slot   : marks the "position" of the camera (1 or 2, default = 1)

        ------------------------------------------------------------------- '''
        shmf = name.lower() + ".im.shm"
        tmp = Cam(name=name, csz=self.csz, ysz=ysz, xsz=xsz,
                  pupil=self.tel.pupil, pdiam=self.tel.pdiam,
                  pscale=pscale, wl=wl, shdir=self.shdir, shmf=shmf)

        self._install_camera(tmp, slot=slot)

    # ==================================================
    def add_corono_camera(self, name="corocam", ysz=256, xsz=320,
                          lstop=None, fpm=None, pview=False,
                          pscale=16.7, wl=1.6e-6, slot=1):
        ''' -------------------------------------------------------------------
        Adds a coronagraphic camera to the instrument.

        Parameters:
        ----------
        - name   : string describing the camera (used for shm file name)
        - ysz    : vertical size of the detector in pixels (default = 256)
        - xsz    : horizontal size of the detector in pixels (default = 320)
        - lstop  : 2D array describing the Lyot-stop (default = None)
        - fpm    : 2D array describing the focal plane mask (default = None)
        - pview  : is this a pupil viewing camera, like ZELDA (default = False)
        - pscale : detector plate scale in mas/pixels (default = 16.7)
        - wl     : wavelength of observation in meters (default = 1.6e-6)
        - slot   : marks the "position" of the camera (1 or 2, default = 1)

        ------------------------------------------------------------------- '''
        shmf = name.lower() + ".im.shm"
        tmp = CoroCam(name=name, csz=self.csz, ysz=ysz, xsz=xsz,
                      pupil=self.tel.pupil, pdiam=self.tel.pdiam,
                      fpm=fpm, lstop=lstop, pview=pview,
                      pscale=pscale, wl=wl, shdir=self.shdir, shmf=shmf)

        self._install_camera(tmp, slot=slot)

    # ==================================================
    def add_SH_camera(self, name="SHcam", dsz=240, mls=40,
                      wl=0.7e-6, slot=1):
        ''' -------------------------------------------------------------------
        Adds a Shack-Hartman WFS camera to the instrument.

        Parameters:
        ----------
        - name   : string describing the camera (used for shm file name)
        - dsz    : detector size (square) in pixels (default = 240)
        - mls    : micro-lens array size (integer, default = 40)
        - wl     : wavelength of operation (float, default = 0.7e-6)
        - slot   : marks the "position" of the camera (1 or 2, default = 1)

        Notes:
        -----
        I realize that there is no plate scale parameter here: the assumption
        is that the SH images are always just nyquist sampled... which may
        not always be desirable...
        ------------------------------------------------------------------- '''
        shmf = name.lower() + ".im.shm"
        tmp = SHCam(name=name, csz=self.csz, dsz=dsz, mls=mls,
                    pupil=self.tel.pupil, wl=wl, shdir=self.shdir, shmf=shmf)

        self._install_camera(tmp, slot=slot)

    # ==================================================
    def add_membrane_DM(
            self, dms=50, nch=4, na0=49, iftype="cosine",
            ifr0=1.0, dtype=np.float64):
        ''' -------------------------------------------------------------------
        Adds a continuous membrane DM made of a regular grid of actuators
        to an instrument.

        Parameters:
        ----------
        - dms    : linear size of the DM in actuators (default = 50)
        - nch    : number of virtual DM channels created (default = 4)
        - na0    : number of actuators across a pupil diameter (default = 49)
        - iftype : influence function type (default = "cosine"
        - ifr0   : influence function radius in actuator size (default = 1.0)
        - dtype  : the data type written to shared memory (default np.float64)
        ------------------------------------------------------------------- '''
        if self.DM is None:
            self.DM = DM(dms=dms, nch=nch, shdir=self.shdir,
                         csz=self.csz, na0=na0, iftype=iftype, dtype=dtype)
        else:
            print("A DM is already in place. Remove to replace it!")

    # ==================================================
    def add_hex_DM(self, nr=7, nch=4, na0=15, srad=323.75):
        ''' -------------------------------------------------------------------
        Adds a segmented mirror with hexagonal geometry to an instrument.

        Parameters:
        ----------
        - nr   : number of rings of segments (default = 7)
        - nch  : number of virtual DM channels created (default = 4)
        - na0  : number of segments across a pupil diameter (default = 15)
        - srad : physical radius of a segment in microns (default = 323.75)

        Note:
        ----

        srad only matters when applying tip-tilt to segments, to convert the
        tip-tilt values (in mrad) into actuator vertical displacements.
        Assumed actuator layout behind segments is that of the BMC Hex-507.
        ------------------------------------------------------------------- '''
        if self.DM is None:
            self.DM = HexDM(nr=nr, nch=nch, csz=self.csz, na0=na0, srad=srad)
        else:
            print("A DM is already in place. Remove to replace it!")

    # ==================================================
    def add_phscreen(self, name="MaunaKea", r0=0.5, L0=10.0, wl=0.5e-6,
                     fc=24.5, correc=10.0):
        ''' -------------------------------------------------------------------
        Adds an atmospheric phase screen above the instrument.
        Phase structure is Von-Karman with partial correction by some factor
        up to a cut-off spatial frequency.

        Parameters:
        ----------
        - name   : string describing the phase screen (used by shm file name)
        - r0     : Fried parameter in meters (default = 0.5)
        - wl     : wavelength where r0 is specified (in meters)
        - L0     : outer scale parameter in meters (default = 10)
        - fc     : cut-off frequency of correction in l/D (default = 24.5)
        - correc : uniform correction factor up until fc (default = 10)

        Note:
        ----
        correc = 10 means that up to spatial frequency fc, the PSD of the
        phase screen is attenuated by a factor 10.
        ------------------------------------------------------------------- '''
        shmf = name.lower() + ".im.shm"
        if self.atmo is None:
            self.atmo = Phscreen(name=name, csz=self.csz, lsz=self.tel.pdiam,
                                 r0=r0, wl=wl, L0=L0, fc=fc, correc=correc,
                                 shdir=self.shdir, shmf=shmf)
        else:
            print("""An atmospheric phase screen is already in place.
            You can update its properties or remove it from the simulation""")

    # ==================================================
    def clear_DM(self):
        ''' -------------------------------------------------------------------
        Removes the DM from the instrument.

        Removal is required prior to installing a new DM
        ------------------------------------------------------------------- '''
        try:
            self.DM.close()
            del self.DM
            self.DM = None
        except AttributeError:
            print("No DM to remove from instrument")

    # ==================================================
    def clear_atmo(self):
        ''' -------------------------------------------------------------------
        Removes the atmospheric phase screen from the instrument.
        ------------------------------------------------------------------- '''
        try:
            _ = self.atmo.shm_phs.fd
            if (self.atmo.shm_phs.fd != 0):
                self.atmo.shm_phs.close()
                del self.atmo
                self.atmo = None

        except AttributeError:
            print("No atmo to remove from instrument")

    # ==================================================
    def snap(self):
        ''' -------------------------------------------------------------------
        Image snap produced without starting servers.

        Mode useful for simulations not requiring the semi real-time
        features of xaosim.

        Returns the image acquired by camera on slot #1 but computes
        images by multiple cameras if relevant.
        ------------------------------------------------------------------- '''

        dmmap = None
        phscreen = None

        if self.DM is not None:
            self.DM.update(verbose=False)
            dmmap = self.DM.wft.get_data()

        if self.atmo is not None:
            phscreen = self.atmo.shm_phs.get_data()

        if self.cam is not None:
            self.cam.make_image(dmmap=dmmap, opdmap=phscreen)
        if self.cam2 is not None:
            self.cam2.make_image(dmmap=dmmap, opdmap=phscreen)
        return(self.cam.get_image())

    # ==================================================
    def start(self, delay=0.1):
        ''' -------------------------------------------------------------------
        A function that starts all the components *servers*

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

        >> myinstrument.start(delay=0.2)

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

        for dev in [self.cam, self.cam2]:
            if dev is not None:
                dev.start(delay=delay, dm_shmf=dm_shmf, atmo_shmf=atmo_shmf)
                del dev

    # ==================================================
    def stop(self,):
        ''' -------------------------------------------------------------------
        Stops all the threads of the different devices part of the instrument.
        ------------------------------------------------------------------- '''
        for dev in [self.atmo, self.DM, self.cam, self.cam2]:
            try:
                _ = dev
                if dev is not None:
                    dev.stop()
                    del dev
            except AttributeError:
                pass

    # ==================================================
    def __del__(self,):
        self.close()
        print("instrument was effectively destroyed!")

    # ==================================================
    def close(self,):
        ''' -------------------------------------------------------------------
        Closes shared memory data structures and release file descriptors.
        ------------------------------------------------------------------- '''
        self.stop()  # just in case!
        time.sleep(self.delay)

        self.clear_atmo()
        self.clear_DM()
        self.clear_camera_slot(slot=1)
        self.clear_camera_slot(slot=2)

        try:
            _ = self.tel
            del self.tel
        except AttributeError:
            pass
