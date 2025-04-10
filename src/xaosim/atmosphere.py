#!/usr/bin/env python3

''' ====================================================================
This is the atmospheric simulation module of XAOSIM.

It defines a generic atmospheric phase screen class that is to be used
in dynamic simulations.
==================================================================== '''

import numpy as np
import threading
import time
from . import wavefront as wft
from .shmlib import shm


# ===========================================================
class Phscreen(object):
    '''Atmospheric Kolmogorov-type phase screen.

    ====================================================================

    Class Attributes:
    ----------------
    - csz     : size (csz x csz) of the phase screen       (in pixels)
    - pdiam   : diameter of the aperture within this array (in pixels)
    - rndarr  : uniformly distributed random array         (csz x csz)
    - kolm    : the original phase screen                  (csz x csz)
    - kolm2   : the oversized phase screen             (2*csz x 2*csz)
    - qstatic : an optional quasi static aberration    (pdiam x pdiam)
    - rms     : total phase screen rms value           (in nanometers)
    - rms_i   : instant rms inside the pupil           (in nanometers)

    Comment:
    -------
    While the attributes are documented here for reference, the prefered
    way of interacting with them is via the functions defined within the
    class.

    Kolm & kolm2 were for a long time computed and kept in radians. The
    need for polychromatic simulations that include the atmosphere forced
    this to change. As of November 2021, the atmospheric phase screen is
    converted into optical path displacement (in meters).

    As a consequence, to create an atmospheric phase screen, in addition
    to r0 and L0, one also needs to specify for what wavelength these
    are given.
    ====================================================================

    '''
    # ==================================================
    def __init__(self, name="MaunaKea", csz=512,
                 lsz=8.0, r0=0.2, wl=1.6e-6, L0=10.0,
                 fc=24.5, correc=1.0,
                 shmf='phscreen.wf.shm', shdir='/dev/shm/'):

        ''' Kolmogorov type atmosphere + qstatic error

        -----------------------------------------------------
        Parameters:
        ----------
        - name  : a string describing the instrument
        - csz   : the size of the Fourier array
        - lsz   : the screen linear size (in meters)
        - r0    : the Fried parameter (in meters)
        - wl    : wavelength where r0 is specified (in meters)
        - L0    : the outer scale parameter (in meters)
        - shmf  : file name to point to shared memory
        - shdir : location of the shm "files"
        -----------------------------------------------------
        '''
        self.shmf = shmf
        self.shdir = shdir
        self.csz = csz
        self.lsz = lsz
        self.r0 = r0
        self.wl = wl
        self.L0 = L0
        self.split_mode = False  # ASGARD special mode

        self.rms_i = 0.0
        self.correc = correc
        self.fc = fc
        phase = wft.atmo_screen(csz, lsz, r0, L0, fc, correc).real
        self.kolm = phase * self.wl / (2 * np.pi)

        self.qstatic = np.zeros((self.csz, self.csz))
        self.shm_phs = shm(
            shdir + shmf, data=self.qstatic, verbose=False)

        self.kolm2 = np.tile(self.kolm, (2, 2))

        self.keepgoing = False

        self.offx = 0  # x-offset on the "large" phase screen array
        self.offy = 0  # y-offset on the "large" phase screen array

        self.ttc = False   # Tip-tilt correction flag

        # auxilliary array (for tip-tilt correction)
        self.xx, self.yy = np.meshgrid(np.arange(self.csz)-self.csz//2,
                                       np.arange(self.csz)-self.csz//2)
        self.xxnorm2 = np.sum(self.xx**2)
        self.yynorm2 = np.sum(self.yy**2)

    # ==============================================================
    def set_split_mode(self, nn=4, ssz=48):
        ''' --------------------------------------------------------
        Special mode developed for ASGARD sim.

        The phase screen is split into n sub-arrays simultaneously
        refreshed.

        Parameters:
        ----------
        - nn: the number of splits (should be 4 for ASGARD)
        - ssz: the subarray size (keep it < self.csz / nn)
        -------------------------------------------------------- '''
        self.split_mode = True
        self.nsplit = nn
        self.ssz = ssz
        sstep = self.csz // nn  # split step
        self.split_ii0 = (sstep - ssz) // 2 + sstep * np.arange(nn)
        self.split_ii1 = self.split_ii0 + ssz

        tmp = np.zeros((ssz, ssz))
        self.split_shm_phs = []
        for ii in range(self.nsplit):
            self.split_shm_phs.append(
                shm(self.shdir + f'split_{ii}_' + self.shmf,
                    data=tmp, verbose=False))
        return

    # ==============================================================
    def update_piston(self, piston=np.zeros(4)):
        ''' --------------------------------------------------------
        Another ASGARD specific tidbit: add piston to the atmosphere

        Parameters:
        ----------
        - piston: a 1D vector of pistons (in meters)
        -------------------------------------------------------- '''
        tmp = np.zeros_like(self.qstatic)
        ii0 = self.split_ii0
        ii1 = self.split_ii1
        for ii in range(self.nsplit):
            tmp[ii0[ii]:ii1[ii], 0:self.ssz] = piston[ii]
        self.set_qstatic(tmp)
        return

    # ==============================================================
    def start(self, delay=0.1, dx=2, dy=1):
        ''' ----------------------------------------
        High-level accessor to start the thread of
        the phase screen server infinite loop.

        Parameters:
        ----------
        - delay: the time delay between refresh  (0.1 sec)
        - dx: the screen slide increment per iteration
        - dy: the screen slide increment per iteration
        ---------------------------------------- '''
        if not self.keepgoing:

            self.kolm2 = np.tile(self.kolm, (2, 2))

            self.keepgoing = True
            t = threading.Thread(target=self.__loop__, args=(delay, dx, dy))
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
    def set_qstatic(self, qstatic=None):
        if qstatic is not None:
            if qstatic.shape == (self.csz, self.csz):
                self.qstatic = qstatic
                print("quasi-static phase screen updated!")
            else:
                print("could not update quasi-static phase screen")
                print("array should be %d x %d (dtype=%s)" % (
                    self.csz, self.csz, str(self.qstatic.dtype)))

            # if simulation is not already active, update phase screen!
            if not self.keepgoing:
                self.shm_phs.set_data(self.qstatic)

        else:
            print("no new quasi-static screen was provided!")

    # ==============================================================
    def update_screen(self, correc=None, fc=None, r0=None, L0=None, wl=None):
        ''' ------------------------------------------------
        Generic update of the properties of the phase-screen
        ------------------------------------------------ '''
        if r0 is not None:
            self.r0 = r0

        if L0 is not None:
            self.L0 = L0

        if correc is not None:
            self.correc = correc

        if fc is not None:
            self.fc = fc

        if wl is not None:
            self.wl = wl

        phase = wft.atmo_screen(
            self.csz, self.lsz, self.r0, self.L0, self.fc, self.correc).real
        self.kolm = phase * self.wl / (2 * np.pi)
        self.kolm2 = np.tile(self.kolm, (2, 2))

        if self.keepgoing is False:
            # case that must be adressed:
            # amplitude changed when atmo is frozen!
            subk = self.kolm2[self.offx:self.offx+self.csz,
                              self.offy:self.offy+self.csz].copy()

            if self.ttc is True:
                ttx = np.sum(subk*self.xx) / self.xxnorm2
                tty = np.sum(subk*self.yy) / self.yynorm2
                subk -= ttx * self.xx + tty * self.yy

            self.rms_i = subk.std()
            self.shm_phs.set_data(subk + self.qstatic)

    # ==============================================================
    def __loop__(self, delay=0.1, dx=2, dy=1):
        ''' ------------------------------------------
        Main loop: frozen screen slid over the aperture

        Options:
        ---------
        - delay: the time delay between refresh  (0.1 sec)
        - dx: the screen slide increment per iteration
        - dy: the screen slide increment per iteration
        -----------------------------------------  '''

        while self.keepgoing:
            self.offx += dx
            self.offy += dy
            self.offx %= self.csz
            self.offy %= self.csz

            subk = self.kolm2[self.offx:self.offx+self.csz,
                              self.offy:self.offy+self.csz].copy()

            if self.ttc is True:
                ttx = np.sum(subk*self.xx) / self.xxnorm2
                tty = np.sum(subk*self.yy) / self.yynorm2
                subk -= ttx * self.xx + tty * self.yy

            self.rms_i = subk.std()

            tmp = subk + self.qstatic
            self.shm_phs.set_data(tmp)

            if self.split_mode is True:
                ii0 = self.split_ii0
                ii1 = self.split_ii1
                for ii in range(self.nsplit):
                    self.split_shm_phs[ii].set_data(
                        tmp[ii0[ii]:ii1[ii], 0:self.ssz])
            time.sleep(delay)

    # =========================================================================
    def close(self,):
        ''' ----------------------------------------
        Closes the linked shared memory structure
        ---------------------------------------- '''
        self.shm_phs.close()
        if self.split_mode:
            for ii in range(self.nsplit):
                self.split_shm_phs[ii].close()
            self.split_shm_phs = []
        return
