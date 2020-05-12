#!/usr/bin/env python3

''' 
===============================================================================
This is the deformable mirror simulation module of XAOSIM. 

It defines two classes of deformable mirrors:
- square grid membrane mirror (ALPAO or BMC membrane)
- segmented hexagonal mirror (BMC Hex)
===============================================================================
'''

import numpy as np
import threading
from .shmlib import shm
from . import pupil
import time


# ===========================================================
# ===========================================================
def influ_fun(iftype="cosine", sz=512, ifs=10):
    ''' -------------------------------------------------------------------
    Pre-computes an actuator influence function for a simulated setup.

    Parameters:
    ----------
    - iftype: influence function type (for now, only "cosine")
    - sz: size of the square array to compute (in pixels
    - ifs: influence function size (in pixels)

    Remarks:
    -------
    Anything else than "cos" for now will result in a "cone" shaped
    influence function. There is something that still needs to be adjusted
    here: the intent is to have the max of that influence function = 1,
    but when we are centered in between pixels, it is not the case.

    NOTE: after playing with the CIAO control and plugging in an imaging 
    camera, I realized there was a problem with the cosine functions. In
    the end, it seems that the best choice is the cone!
    ------------------------------------------------------------------- '''

    xl = 1.0 * (np.arange(sz) - sz // 2) # profile pixel coordinates
    # if int(sz % 2) == 0:
    #     xl += 0.5
            
    if "cos" in iftype.lower():
        px = 0.5 * (1.0 + np.cos(np.pi*np.abs(xl)/ifs))
        px[np.abs(xl) > ifs] = 0.0
        res = np.outer(px, px)
    else:
        px = 1.0 - np.abs(xl) / ifs
        px[np.abs(xl) > ifs] = 0.0
        res = np.outer(px, px)
    return res

class DM(object):
    ''' -------------------------------------------------------------------
    Deformable mirror class

    The displacement maps *self.dmd* are in microns
    ------------------------------------------------------------------- '''

    # ==================================================
    def __init__(self, instrument="SCExAO", dms=50, nch=8, 
                 shm_root="dmdisp", shdir="/dev/shm/",
                 csz=256, na0=50.0, dx=0.0, dy=0.0,
                 iftype="", ifr0=1.0):
        ''' -----------------------------------------
        Constructor for instance of deformable mirror
        Parameters:
        ----------
        - instrument : a string
        - dms: an integer (linear size of the DM)
        - nch: number of channels
        - shm_root: the root name for shared mem files
        - shdir: directory where shared mem files will be
        
        Additions:
        ---------
        - csz: array size for computations
        - na0: number of actuators across 
        - dx: DM L/R misalignment (in actuators)
        - dy: DM U/D misalignment (in actuators)

        Experimental:
        ------------
        - iftype: inf. function type ("cone", "cos", ...)
        - ifr0:   inf. function characteristic size (in actuator steps)
        ----------------------------------------- '''
        self.keepgoing = False
        self.dms = dms # DM size of (dms x dms) actuators
        self.nch = nch # numbers of channels to drive the DM
        self.dmtype = "square" # square grid of actuators
        self.iftype = iftype   # type of influence function
        self.ifr0   = 1        # infl. function characteristic size (in actuators)

        self.dmd0   = np.zeros((dms, dms), dtype=np.float32)
        self.dmd    = self.dmd0.copy()
        self.shm_cntr = np.zeros(nch) - 1

        # --------------------------------------------------------------
        # create or overwrite shared memory data structures for the DM
        # --------------------------------------------------------------
        self.disp = shm(fname='%s%s.im.shm' % (shdir, shm_root), 
                        data=self.dmd0, verbose=False)
        
        for i in range(nch):
            exec('''self.disp%d = shm(fname='%s%s%d.im.shm', 
                 data=self.dmd0, verbose=False)''' % (i,shdir, shm_root,i))

        # -----------------------------------------------------
        # DM-induced wavefront shared memory data structure
        # -----------------------------------------------------
        self.wft = shm(fname='%s%s.wf.shm' % (shdir, shm_root), 
                       data=np.zeros((csz,csz)), verbose=False)
        
        self.dx = dx
        self.dy = dy
        self.na0 = na0
        self.csz = csz
        self.astep = csz // na0
        self._if_psz = self.astep*self.ifr0  # infl. func. charac size (pixels)
        self._if_asz = int(np.round(2*self._if_psz)) # infl. func. array size (pixels)
        self.shm_root = shm_root
        rwf = int(np.round(self.astep*self.dms))
        self.infun = influ_fun(iftype=self.iftype,
                               sz=self._if_asz, ifs=self._if_psz)
        
    # ==================================================
    def get_counter_channel(self, chn):
        ''' ----------------------------------------
        Return the current channel counter value.
        Reads from the already-opened shared memory
        data structure.
        ---------------------------------------- '''
        cnt = 0
        if chn < self.nch:
            cnt = eval("self.disp%d.get_counter()" % (chn,))
        else:
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
        for i in range(self.nch):
            exec("combi += self.disp%d.get_data()" % (i,))
        self.dmd = combi
        self.disp.set_data(combi)
        self.wft.set_data(self.map2D(self.csz, self.astep))
        if verbose:
            print("DM shape updated!")
        
    # ==================================================
    def __loop__(self, delay=0.1):
        ''' ----------------------------------------
        Thread (infinite loop) that updates the DM
        shape until told to stop.

        Do not use directly: call start()
        and stop() instead.
        ---------------------------------------- '''
        updt = True
        while self.keepgoing:
            for i in range(self.nch):
                test = self.get_counter_channel(i)
                if test != self.shm_cntr[i]:
                    self.shm_cntr[i] = test
                    updt = True
            if updt:
                updt = False
                self.update()
            time.sleep(delay)

    # ==================================================
    def map2D(self, msz, astep):
        ''' -----------------------------------------
        Returns a 2D displacement map of the DM.
        assuming a square grid of actuators

        Parameters:
        ----------
        - msz  : the size of the map (in pixels)
        - astep: actuator grid size (in pixels)
        ----------------------------------------- '''
        dmmap = np.zeros((msz, msz), dtype=np.float64)
        dms = self.dms

        if self.iftype is not "": # influence function specified!
            amap = self.dmd.copy()

            dsz = int(self._if_asz//2 + astep) # extra space to accomodate infl. func.
            map1 = np.zeros((msz+dsz, msz+dsz), dtype=np.float64) # oversized temp map!

            y0, x0 = int(np.round((dsz - astep)/2)), int(np.round((dsz - astep)/2))
            
            for jj in range(dms*dms):
                iy, ix = jj % dms, jj // dms     # actuator indices
                if np.abs(amap[iy, ix]) > 1e-4:  # save some computation time?
                    _y0 = int(np.round(iy*astep))
                    _x0 = int(np.round(ix*astep))
                    map1[_y0:_y0+self._if_asz,
                         _x0:_x0+self._if_asz] += amap[iy, ix] * self.infun

            dmmap = map1[dsz//2:dsz//2+msz, dsz//2:dsz//2+msz]
            
        else:
            rwf = int(np.round(astep*dms)) # resized wavefront
            x0 = (self.csz-rwf) // 2
            x1 = x0 + rwf
            
            map0 = Image.fromarray(self.dmd)
            map1 = map0.resize((rwf, rwf), resample=1)
            dmmap[x0:x1,x0:x1] = map1
            
        return dmmap

# ===========================================================
# ===========================================================

class HexDM(DM):
    ''' -------------------------------------------------------------------
    Hexagonal Segmented Deformable mirror class

    ------------------------------------------------------------------- '''

    # ==================================================
    def __init__(self, instrument="KERNEL", nr=1, nch=8, 
                 shm_root="hex_disp", shdir="/dev/shm/",
                 csz=512, na0=15.0, dx=0.0, dy=0.0):
        ''' -----------------------------------------
        Constructor for instance of deformable mirror
        Parameters:
        ----------
        - instrument : a string
        - dms: an integer (linear size of the DM)
        - nch: number of channels
        - shm_root: the root name for shared mem files

        Additions:
        ---------
        - csz: array size for computations
        - na0: number of segments across pupil diam
        - dx: DM L/R misalignment (in actuators)
        - dy: DM U/D misalignment (in actuators)
        ----------------------------------------- '''
        self.keepgoing = False
        self.dmtype = "hex"
        self.nr = nr            # DM number of rings
        self.ns = 1+3*nr*(nr+1) # DM number of segments
        self.nch = nch          # numbers of DM channels
        self.dmd0 = np.zeros((self.ns, 3), dtype=np.float32)
        self.shm_cntr = np.zeros(nch) - 1
        self.disp = shm(fname='%s%s.im.shm' % (shdir, shm_root), 
                        data=self.dmd0, verbose=False, nbkw=1)

        # share dmtype with instrument via keywords
        self.disp.update_keyword(0, "dmtype", self.dmtype,
                                 "geometry of the Deformable Mirror")

        for ii in range(nch):
            exec('''self.disp%d = shm(fname='%s%s%d.im.shm', 
            data=self.dmd0, verbose=False)''' % (ii,shdir, shm_root,ii))
            
        # additional shared memory data structure for the wavefront
        self.wft = shm(fname='%s%s.wf.shm' % (shdir, shm_root), 
                       data=np.zeros((csz,csz)), verbose=False)
        self.dx = dx
        self.dy = dy
        self.na0 = na0
        self.csz = csz
        self.astep = csz // na0
        self.shm_root = shm_root

    # ==================================================
    def map2D(self, msz, astep):
        ''' -----------------------------------------
        Returns a 2D displacement map of the DM.

        Parameters:
        ----------
        - msz   : the size of the map (in pixels)
        - astep : actuator pitch (in pixels)
        ----------------------------------------- '''
        nr    = self.nr
        arad  = astep/np.sqrt(3)
        dmmap = np.zeros((msz, msz), dtype=np.float64)
        xx,yy = np.meshgrid(np.arange(msz)-msz/2, np.arange(msz)-msz/2)
        
        xy = pupil.hex_grid_coords(nr+1, astep, rot=0)
        xy[0] += self.dx * astep # optional offset of
        xy[1] += self.dy * astep # the DM position
        xy = np.round(xy).astype(np.int)
        
        seg = pupil.uniform_hex(msz, msz, arad).T
                
        for ii in range(self.ns):
            seg1 = seg*(self.dmd[ii,0]*seg + self.dmd[ii,1]*xx + self.dmd[ii,2]*yy)
            dmmap += np.roll(np.roll(seg1, xy[0,ii], axis=0), xy[1,ii], axis=1)

        return dmmap

    # ==================================================
    def map2D_2_TTP(self, dmmap=None):
        ''' -------------------------------------------------------------------
        Turns the provided DM map into tip-tilt-piston commands for the DM
        *in the same unit as the input*

        Parameters:
        ----------
        - dmmap: the DM map
        
        Remarks:
        -------

        Before being sent to the DM, the commands need to be properly scaled:
        if the input is in radians, the commands must be converted into microns
        for the DM, and take into account the x2 factor of the reflection! 

        The scaling parameter will typically be *lambda / (4*PI)*
        ----------------------------------------- '''
        nr    = self.nr
        arad  = self.astep/np.sqrt(3)

        xx,yy = np.meshgrid(np.arange(self.csz)-self.csz/2,
                            np.arange(self.csz)-self.csz/2)
        
        xy = pupil.hex_grid_coords(nr+1, self.astep, rot=0)
        xy[0] += self.dx * self.astep # optional offset of
        xy[1] += self.dy * self.astep # the DM position
        xy = np.round(xy).astype(np.int)

        # centered reference segment
        seg = pupil.uniform_hex(self.csz, self.csz, arad).T

        dmd0 = np.zeros_like(self.dmd0) # empty tip-tilt-piston commands

        xxnorm = xx[seg > 0].dot(xx[seg > 0])
        yynorm = yy[seg > 0].dot(yy[seg > 0])
        
        for ii in range(self.ns):
            tmp = np.roll(np.roll(dmmap, -xy[0,ii], axis=0), -xy[1,ii], axis=1)
            dmd0[ii,0] = tmp[seg > 0].mean()
            dmd0[ii,1] = tmp[seg > 0].dot(xx[seg > 0]) / xxnorm
            dmd0[ii,2] = tmp[seg > 0].dot(yy[seg > 0]) / yynorm

        return dmd0
    
