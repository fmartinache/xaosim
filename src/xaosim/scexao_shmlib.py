#!/usr/bin/env python

'''---------------------------------------------------------------------------
Read and write access to shared memory (SHM) structures used by SCExAO

- Author : Frantz Martinache
- Date   : June 13, 2018

Improved version of the original SHM structure used by SCExAO and friends.

Inherits from the more generic shm class defined in the shmlib module,
and includes additional functions expecting specific keywords to be
implemented, when dealing with camera images.
---------------------------------------------------------------------------

'''

from .shmlib import shm as shm0
import numpy as np
import posix_ipc as ipc
import os


class shm(shm0):
    # =====================================================================
    def __init__(self, fname=None, data=None,
                 verbose=False, nbkw=0):
        ''' --------------------------------------------------------------
        Constructor for a SHM (shared memory) object.

        Parameters:
        ----------
        - fname: name of the shared memory file structure
        - data: some array (1, 2 or 3D of data)
        - verbose: optional boolean
        - nbkw: # of keywords to be appended to the data structure (optional)

        Depending on whether the file already exists, and/or some new
        data is provided, the file will be created or overwritten.

        In addition to the generic shm data structure, semaphores are
        connected to the object.
        -------------------------------------------------------------- '''
        self.nosem = True
        shm0.__init__(self, fname, data, verbose, False, nbkw)
        self.nsem = 10  # number of semaphores to address
        myname = os.path.basename(fname).split('.')[0]

        for ii in range(self.nsem):
            semf = "%s_sem%02d" % (myname, ii)
            exec('self.sem%02d = ipc.Semaphore(semf, ipc.O_RDWR | ipc.O_CREAT)' % (ii,))
            _ = ipc.Semaphore("%s_semlog" % (myname,),
                              ipc.O_RDWR | ipc.O_CREAT)
        self.nosem = False

    # =====================================================================
    def set_data(self, data, check_dt=False):
        ''' --------------------------------------------------------------
        On SCExAO, in addition to updating the actual DM data, on must post
        semaphores to signal the DM to update
        -------------------------------------------------------------- '''
        shm0.set_data(self, data, check_dt)
        if self.nosem is False:
            for ii in range(10):
                semf = "%s_sem%02d" % (self.mtdata['imname'], ii)
                exec('self.sem%02d.release()' % (ii,))
        else:
            print("skip sem post this first time")

    # =====================================================================
    def close(self,):
        shm0.close(self)
        for ii in range(self.nsem):
            semf = "%s_sem%02d" % (self.mtdata['imname'], ii)
            exec('self.sem%02d.close()' % (ii,))

    # =====================================================================
    def get_expt(self,):
        ''' --------------------------------------------------------------
        SCExAO specific: returns the exposure time (from keyword)
        -------------------------------------------------------------- '''
        ii0 = 0  # index of exposure time in keywords
        self.read_keyword(ii0)
        self.expt = self.kwds[ii0]['value']
        return self.expt

    # =====================================================================
    def get_fps(self,):
        ''' --------------------------------------------------------------
        SCExAO specific: returns the frame rate (from keyword)
        -------------------------------------------------------------- '''
        ii0 = 1 # index of exposure time in keywords
        self.read_keyword(ii0)
        self.fps = self.kwds[ii0]['value']
        return self.fps

    # =====================================================================
    def get_ndr(self,):
        ''' --------------------------------------------------------------
        SCExAO specific: returns the frame rate (from keyword)
        -------------------------------------------------------------- '''
        ii0 = 2 # index of exposure time in keywords
        self.read_keyword(ii0)
        self.ndr = self.kwds[ii0]['value']
        return self.ndr

    # =====================================================================
    def get_crop(self,):
        ''' --------------------------------------------------------------
        SCExAO specific: returns the frame rate (from keyword)
        -------------------------------------------------------------- '''
        ii0 = 3 # index of exposure time in keywords
        self.crop = np.zeros(4)
        for i in range(4):
            self.read_keyword(ii0+i)
            self.crop[i] = self.kwds[ii0+i]['value']
        return self.crop

# =================================================================
# =================================================================
