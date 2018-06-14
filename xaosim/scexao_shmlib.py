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

from shmlib import shm as shm0
import numpy as np

class shm(shm0):
    def __init__(self, fname=None, data=None,
                 verbose=False, packed=False, nbkw=0):
        ''' --------------------------------------------------------------
        Constructor for a SHM (shared memory) object.

        Parameters:
        ----------
        - fname: name of the shared memory file structure
        - data: some array (1, 2 or 3D of data)
        - verbose: optional boolean
        - packed: True -> packed / False -> aligned data format
        - nbkw: # of keywords to be appended to the data structure (optional)

        Depending on whether the file already exists, and/or some new
        data is provided, the file will be created or overwritten.
        -------------------------------------------------------------- '''
        shm0.__init__(self, fname, data, verbose, packed, nbkw)

    def get_expt(self,):
        ''' --------------------------------------------------------------
        SCExAO specific: returns the exposure time (from keyword)
        -------------------------------------------------------------- '''
        ii0 = 0 # index of exposure time in keywords
        self.read_keyword(ii0)
        self.expt = self.kwds[ii0]['value']
        return self.expt

    def get_fps(self,):
        ''' --------------------------------------------------------------
        SCExAO specific: returns the frame rate (from keyword)
        -------------------------------------------------------------- '''
        ii0 = 1 # index of exposure time in keywords
        self.read_keyword(ii0)
        self.fps = self.kwds[ii0]['value']
        return self.fps

    def get_ndr(self,):
        ''' --------------------------------------------------------------
        SCExAO specific: returns the frame rate (from keyword)
        -------------------------------------------------------------- '''
        ii0 = 2 # index of exposure time in keywords
        self.read_keyword(ii0)
        self.ndr = self.kwds[ii0]['value']
        return self.ndr

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
