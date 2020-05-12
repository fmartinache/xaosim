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
    - kolm2   : the oversized phase screen    ((csz + pdiam) x (csz + pdiam))
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
    def __init__(self, name="MaunaKea", csz = 512,
                 lsz=8.0, r0=0.2, L0=10.0,
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
        - L0    : the outer scale parameter (in meters)
        - shmf  : file name to point to shared memory
        - shdir : location of the shm "files"
        -----------------------------------------------------
        '''
        self.shmf    = shmf
        self.shdir   = shdir
        self.csz     = csz
        self.lsz     = lsz
        self.r0      = r0
        self.L0      = L0
        
        self.rms_i   = 0.0
        self.correc  = correc
        self.fc      = fc
        self.kolm    = wft.atmo_screen(csz, lsz, r0, L0, fc, correc).real
        
        self.qstatic = np.zeros((self.csz, self.csz))
        self.shm_phs = shm(shdir + shmf,
                           data = self.qstatic, verbose=False)

        self.kolm2   = np.tile(self.kolm, (2,2))
        #self.kolm2   = self.kolm2[:self.sz+self.pdiam,:self.sz+self.pdiam]

        self.keepgoing = False

        self.offx = 0 # x-offset on the "large" phase screen array
        self.offy = 0 # y-offset on the "large" phase screen array

        self.ttc     = False   # Tip-tilt correction flag
        
        # auxilliary array (for tip-tilt correction)
        self.xx, self.yy  = np.meshgrid(np.arange(self.csz)-self.csz//2,
                                        np.arange(self.csz)-self.csz//2)
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
            #self.kolm2   = self.kolm2[:self.sz+self.pdiam,:self.sz+self.pdiam]

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
            if self.keepgoing == False:
                self.shm_phs.set_data(self.qstatic)

        else:
            print("no new quasi-static screen was provided!")

    # ==============================================================
    def update_screen(self, correc=None, fc=None, r0=None, L0=None):
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
            
        self.kolm    = wft.atmo_screen(
            self.csz, self.lsz, self.r0, self.L0, self.fc, self.correc).real

        self.kolm2   = np.tile(self.kolm, (2,2))

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
            self.offx = self.offx % self.csz
            self.offy = self.offy % self.csz

            subk = self.kolm2[self.offx:self.offx+self.csz,
                              self.offy:self.offy+self.csz].copy()

            if self.ttc is True:
                ttx = np.sum(subk*self.xx) / self.xxnorm2
                tty = np.sum(subk*self.yy) / self.yynorm2
                subk -= ttx * self.xx + tty * self.yy

            self.rms_i = subk.std()
            self.shm_phs.set_data(subk + self.qstatic)
            time.sleep(delay)

