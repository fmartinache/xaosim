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
from PIL import Image


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
    here.
    ------------------------------------------------------------------- '''
    offset = 0.5 if (sz % 2 == 0) else 0.0

    xl = 1.0 * (np.arange(sz) - sz // 2 + offset)
    px = 0.5 * (1.0 + np.cos(np.pi*np.abs(xl)/ifs))

    dist = pupil._dist(sz, sz, between_pix=(sz % 2 == 0))
    if "cos" in iftype.lower():
        px = 0.5 * (1.0 + np.cos(np.pi*np.abs(xl)/ifs))
        px[np.abs(xl) > ifs] = 0.0
        res = np.outer(px, px)
    else:
        # if not cosine, then cone shaped!
        px = 1.0 - np.abs(xl) / ifs
        px[np.abs(xl) > ifs] = 0.0
        res = np.outer(px, px)
    res[dist > ifs] = 0.0
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
                 iftype="", ifr0=1.0, dtype=np.float64):
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
        self.dms = dms  # DM size of (dms x dms) actuators
        self.nch = nch  # numbers of channels to drive the DM
        self.dmtype = "square"  # square grid of actuators
        self.iftype = iftype    # type of influence function
        self.ifr0 = ifr0        # infl. function size (in actuators)
        self.dtype = dtype      # data type written to shared memory

        self.dmd0 = np.zeros((dms, dms), dtype=dtype)
        self.dmd = self.dmd0.copy()
        self.shm_cntr = np.zeros(nch) - 1

        self.shdir = shdir
        self.shm_root = shm_root
        self.shmf = shm_root + ".wf.shm"

        # --------------------------------------------------------------
        # create or overwrite shared memory data structures for the DM
        # --------------------------------------------------------------
        self.disp = shm(fname='%s%s.im.shm' % (shdir, shm_root),
                        data=self.dmd0, verbose=False)

        for i in range(nch):
            exec('''self.disp%d = shm(fname='%s%s%d.im.shm', 
                 data=self.dmd0, verbose=False)''' % (
                     i, shdir, shm_root, i))

        # -----------------------------------------------------
        # DM-induced wavefront shared memory data structure
        # -----------------------------------------------------
        self.wft = shm(fname='%s%s.wf.shm' % (shdir, shm_root),
                       data=np.zeros((csz, csz)), verbose=False)

        self.dx = dx
        self.dy = dy
        self.na0 = na0
        self.csz = csz
        self.astep = csz // na0
        # influence function characteristic and array sizes (in pixels)
        self._if_psz = self.astep*self.ifr0
        self._if_asz = int(np.round(2*self._if_psz))
        rwf = int(np.round(self.astep*self.dms))
        self.infun = influ_fun(iftype=self.iftype,
                               sz=self._if_asz, ifs=self._if_psz)

    # ==================================================
    def __str__(self):
        msg = f"Continuous Deformable Mirror:\n"
        msg += f"- {self.dms}x{self.dms} {self.dmtype} actuator grid\n"
        msg += f"- {self.nch} control channels\n"
        msg += f"- Computation size = {self.csz} "
        msg += f"({self.astep} pixels / actuators)\n"
        return msg

    # ==================================================
    def update_infun(self, iftype=None, ifr0=None):
        ''' ----------------------------------------
        Updates the DM influence function!

        Parameters:
        ----------
        - iftype: influence function type
        - ifr0: influence function radius (in actuators)
        ---------------------------------------- '''
        change = False
        if ifr0 is not None:
            self._if_psz = self.astep*self.ifr0
            self._if_asz = int(np.round(2*self._if_psz))
            change = True

        if iftype is not None:
            self.iftype = str(iftype)
            change = True

        if change is True:
            self.infun = influ_fun(
                iftype=self.iftype,
                sz=self._if_asz, ifs=self._if_psz)

    # ==================================================
    def close(self,):
        ''' ----------------------------------------
        Closes all the shared memory data structures
        for the DM (channels, global disp and wft)
        ---------------------------------------- '''
        self.disp.close()
        for ii in range(self.nch):
            exec('self.disp%d.close()' % (ii))
        self.wft.close()

    # ==================================================
    def get_counter_channel(self, chn=0):
        ''' ----------------------------------------
        Return the current channel counter value.
        Reads from the already-opened shared memory
        data structure.

        Parameters:
        ----------
        - chn  : the channel (integer)
        ---------------------------------------- '''
        cnt = 0
        if chn < self.nch:
            cnt = eval("self.disp%d.get_counter()" % (chn,))
        else:
            cnt = self.disp.get_counter()
        return(cnt)

    # ==================================================
    def reset_channel(self, chn=0):
        ''' ----------------------------------------
        Resets the state of the channel to zeros

        Parameters:
        ----------
        - chn  : the channel (integer)
        ---------------------------------------- '''
        if 0 <= chn < self.nch:
            eval("self.disp%d.set_data(self.dmd0)" % (chn,))

    # ==================================================
    def set_data_channel(self, dmap=None, chn=0):
        ''' ----------------------------------------
        Sends the dmap to the requested channel

        Parameters:
        ----------
        - dmap : float ndarray (size dms x dms)
        - chn  : the channel (integer)
        ---------------------------------------- '''
        if dmap is not None:
            if 0 <= chn < self.nch:
                eval("self.disp%d.set_data(dmap)" % (chn,))

    # ==================================================
    def get_data_channel(self, chn=0):
        ''' ----------------------------------------
        Returns the data present on a channel

        Parameters:
        ----------
        - chn: the channel (integer)
        ---------------------------------------- '''
        res = None
        if 0 <= chn < self.nch:
            res = eval("self.disp%d.get_data()" % (chn,))
        return res

    # ==================================================
    def start(self, delay=0.1):
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
        self.wft.set_data(self.map2D())
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
    def map2D(self):
        ''' -----------------------------------------
        Returns a 2D displacement map of the DM.
        assuming a square grid of actuators

        Parameters:
        ----------
        - astep: actuator grid size (in pixels)
        ----------------------------------------- '''
        dmmap = np.zeros((self.csz, self.csz), dtype=np.float64)
        dms = self.dms
        dsz = self._if_asz  # padding for influence function
        astep = dsz // 2    # actuator step on map

        if self.iftype != "":  # influence function specified!
            amap = self.dmd.copy()

            na_off = (dms - self.na0) // 2  # offset actuators
            wsz = (dms+1) * astep          # work-map size
            wmap = np.zeros((wsz, wsz), dtype=np.float64)
            for jj in range(dms*dms):
                iy, ix = jj % dms, jj // dms     # actuator indices
                _y0 = int(np.round(iy*astep))
                _x0 = int(np.round(ix*astep))
                wmap[_y0:_y0+self._if_asz,
                     _x0:_x0+self._if_asz] += amap[iy, ix] * self.infun

            # keep useful part only
            y0 = int((na_off + 0.5 + self.dy) * astep)
            x0 = int((na_off + 0.5 + self.dx) * astep)
            dmmap = wmap[y0:y0+self.csz, x0:x0+self.csz]

        else:
            # interpolation using PIL!
            x0 = int(np.round((dsz - astep)/2))

            rwf = int(np.round(astep*dms))  # resized wavefront
            x0 = (self.csz-rwf) // 2
            x1 = x0 + rwf

            map0 = Image.fromarray(self.dmd)
            map1 = map0.resize((rwf, rwf), resample=1)
            dmmap[x0:x1, x0:x1] = map1

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
                 csz=512, na0=15.0, dx=0.0, dy=0.0,
                 srad=323.75):
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
        - srad: segment radius (in microns)

        Note:
        ----
        Segment radius is defined as that of the circle
        that would be inscribed inside one segment.
        ----------------------------------------- '''
        self.keepgoing = False
        self.dmtype = "hex"
        self.nr = nr             # DM number of rings
        self.ns = 1+3*nr*(nr+1)  # DM number of segments
        self.nch = nch           # numbers of DM channels
        self.dmd0 = np.zeros((self.ns, 3), dtype=np.float32)
        self.shm_cntr = np.zeros(nch) - 1
        self.disp = shm(fname='%s%s.im.shm' % (shdir, shm_root),
                        data=self.dmd0, verbose=False, nbkw=1)

        self.shdir = shdir
        self.shm_root = shm_root
        self.shmf = shm_root + ".wf.shm"

        # share dmtype with instrument via keywords
        self.disp.update_keyword(0, "dmtype", self.dmtype,
                                 "geometry of the Deformable Mirror")

        for ii in range(nch):
            exec('''self.disp%d = shm(fname='%s%s%d.im.shm', 
            data=self.dmd0, verbose=False)''' % (
                ii, shdir, shm_root, ii))

        # additional shared memory data structure for the wavefront
        self.wft = shm(fname='%s%s.wf.shm' % (shdir, shm_root),
                       data=np.zeros((csz, csz)), verbose=False)
        self.dx = dx
        self.dy = dy
        self.na0 = na0
        self.csz = csz
        self.astep = csz // na0
        self.srad = srad

    # ==================================================
    def map2D(self):
        ''' -----------------------------------------
        Returns a 2D displacement map of the Hex DM.

        Parameters:
        ----------
        - astep : actuator pitch (in pixels)
        ----------------------------------------- '''
        nr = self.nr
        msz = self.csz
        arad = self.astep/np.sqrt(3)
        dmmap = np.zeros((msz, msz), dtype=np.float64)
        xx, yy = np.meshgrid(np.arange(msz)-msz/2, np.arange(msz)-msz/2)

        scoeff = 2*self.srad*1e-3/self.astep  # slope coefficient

        ttx = self.dmd[:, 1] / scoeff
        tty = self.dmd[:, 2] / scoeff

        xy = pupil.hex_grid_coords(nr+1, astep, rot=2*np.pi/3)
        xy[0] += self.dx * astep  # optional offset of
        xy[1] += self.dy * astep  # the DM position
        xy = np.round(xy).astype(np.int)

        seg = pupil.uniform_hex(msz, msz, arad).T

        for ii in range(self.ns):
            seg1 = seg*(self.dmd[ii, 0]*seg + ttx[ii]*xx + tty[ii]*yy)
            seg1 = np.roll(np.roll(seg1, xy[0, ii], axis=0), xy[1, ii], axis=1)
            dmmap += seg1

        return dmmap

    # ==================================================
    def disp_2_PTT(self, dmap=None):
        ''' -------------------------------------------------------------------
        Turns the provided DM displacement map into piston-tip-tilt commands
        for the HexDM

        Parameters:
        ----------
        - dmap: displacement map the HexDM will attempt to approximate
                 expressed in microns

        Returns: a (nseg x 3) numpy array that contains:
        -------
        - the piston in the first column (in microns)
        - the x-tip  in the second column (in mrad)
        - the y-tilt in the third column (in mrad)

        Remarks:
        -------

        To convert a desired wavefront (in radians), into displacement
        map on the DM (in microns), things must be scaled by the wavelength.
        Just make sure to also take the x2 effect the DM displacement has
        on the wavefront.

        The required scaling parameter will typically be *lambda / (4*PI)*
        ------------------------------------------------------------------- '''
        nr = self.nr
        arad = self.astep/np.sqrt(3)

        xx, yy = np.meshgrid(np.arange(self.csz)-self.csz/2,
                             np.arange(self.csz)-self.csz/2)

        xy = pupil.hex_grid_coords(nr+1, self.astep, rot=2*np.pi/3)
        xy[0] += self.dx * self.astep  # optional offset of
        xy[1] += self.dy * self.astep  # the DM position
        xy = np.round(xy).astype(np.int)

        scoeff = 2*self.srad*1e-3/self.astep  # slope coefficient

        # centered reference segment
        seg = pupil.uniform_hex(self.csz, self.csz, arad).T

        dmd0 = np.zeros_like(self.dmd0)  # empty tip-tilt-piston commands

        xxnorm = xx[seg > 0].dot(xx[seg > 0])
        yynorm = yy[seg > 0].dot(yy[seg > 0])

        for ii in range(self.ns):
            tmp = np.roll(np.roll(dmap, -xy[0, ii], axis=0),
                          -xy[1, ii], axis=1)
            dmd0[ii, 0] = tmp[seg > 0].mean()
            dmd0[ii, 1] = scoeff * tmp[seg > 0].dot(xx[seg > 0]) / xxnorm
            dmd0[ii, 2] = scoeff * tmp[seg > 0].dot(yy[seg > 0]) / yynorm

        return dmd0

    # ========================================================================
    def ptt_2_actuator(self, ptt, piston_only=True):
        ''' -------------------------------------------------------------------
        converts an array of piston+tip-tilt segment states and converts them
        into actuator commands for the Hex BMC mirror.

        Parameters:
        ----------
        - ptt:         the array of piston+tip-tilt command for all segments
        - piston_only: option to apply piston correction only (easier scenario)

        Remarks:
        -------
        - #1 Assumes that the ptt commands are expressed in:
        - microns (for the piston)
        - mrad (for the tip-tilt)
        - #2 Commands sent to the DM are floating point numbers 0 <= cmd <= 1.
        ------------------------------------------------------------------- '''
        csz = 1024   # command size
        nseg = 169   # number of segments
        again = 4.0  # actuator gain: 4 um per ADU
        cmd = np.zeros(csz)

        a0 = self.srad  # 323.75 um for the BMC Hex DM
        mat = np.array([[1,                0,  a0],
                        [1, -np.sqrt(3)/2*a0, -a0/2],
                        [1,  np.sqrt(3)/2*a0, -a0/2]])

        # for the piston only scenario
        if piston_only:
            for ii in range(nseg):
                cmd[ii*3] = ptt[ii, 0]
                cmd[ii*3+1] = ptt[ii, 0]
                cmd[ii*3+2] = ptt[ii, 0]

        else:
            print("TBD!!")

        cmd /= again  # command converted into BMC HexDM ADU

        return cmd
