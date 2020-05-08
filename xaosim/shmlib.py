#!/usr/bin/env python3

'''---------------------------------------------------------------------------
Read and write access to shared memory (SHM) structures used by SCExAO

- Author : Frantz Martinache
- Date   : June 13, 2018
- Revised: May 7, 2019

Improved version of the original SHM structure used by SCExAO and friends.
---------------------------------------------------------------------------

Semaphores required by SCExAO call for the specialized scexao_shm module.
This module relies on the posix_ipc library available on PyPi

---------------------------------------------------------------------------
Note on data alignment (refer to section 7.3.2.1 of python documentation)

By default, C types are represented in the machine's native format and byte
order, and properly aligned by skipping pad bytes if necessary (according to
the rules used by the C compiler).

To request no alignment, while using native byte-order, the first character 
of the format string must be "="! This is used for keywords.

Between 2017 and 2018, SCExAO began adopting an aligned data-structure
again. This class was modified accordingly and includes a "packed" 
constructor option to choose between the two styles.
---------------------------------------------------------------------------

Note on the order of axes for the data:
The convention for the shared memory structure follows that of the fits file:
for a 3D array, the axes are in the x,y,z order.

Internally in this library, the axes are reversed: z,y,x.
---------------------------------------------------------------------------
'''

import os, sys, mmap, struct
import numpy as np
import time

# ------------------------------------------------------
#          list of available data types
# ------------------------------------------------------
all_dtypes = [np.uint8,     np.int8,    np.uint16,    np.int16, 
              np.uint32,    np.int32,   np.uint64,    np.int64,
              np.float32,   np.float64, np.complex64, np.complex128]

# # ------------------------------------------------------
# #    string used to decode the binary shm structure
# # ------------------------------------------------------
# hdr_fmt_pck = '32s 80s B   3I Q B   Q 2Q 2Q 2Q Q 2Q Q B b B   Q Q B   H   Q Q Q Q B   H 64s' # packed style
# hdr_fmt_aln = '32s 80s B3x 3I Q B7x Q 2Q 2Q 2Q Q 2Q Q B b B5x Q Q B1x H4x Q Q Q Q B1x H 64s4x' # aligned style
# c0_hdr_pos  = 20 # position of the counter #0 in the header. Used to speed up access later

# # ------------------------------------------------------
# # list of metadata keys for the shm structure (global)
# # these keys have to match the header introduced above!
# # ------------------------------------------------------
# mtkeys = ['bversion', 'bimname',
#           'naxis', 'x', 'y', 'z', 'nel', 'atype', 'imtype',
#           'crtime_sec', 'crtime_ns',
#           'latime_sec', 'latime_ns',
#           'atime_sec', 'atime_ns', 'atimearr',
#           'wtime_sec', 'wtime_ns', 'wtimearr',
#           'shared', 'loc', 'status',
#           'flag', 'flagarr', 'logflag', 'sem',
#           'cnt0', 'cnt1', 'cnt2', 'cntarr',
#           'write', 'nbkw', 'bcudamem']

# # ------------------------------------------------------
# #    string used to decode the binary shm structure
# # ------------------------------------------------------
hdr_fmt_pck = '32s 80s B   3I Q B   Q 2Q 2Q 2Q 2Q   B b B   Q   B   H   Q Q Q B   H 64s' # packed style
hdr_fmt_aln = '32s 80s B3x 3I Q B7x Q 2Q 2Q 2Q 2Q   B b B5x Q   B1x H4x Q Q Q B1x H 64s4x' # aligned style
c0_hdr_pos  = 17 # position of the counter #0 in the header. Used to speed up access later

# ------------------------------------------------------
# list of metadata keys for the shm structure (global)
# these keys have to match the header introduced above!
# ------------------------------------------------------
mtkeys = ['bversion', 'bimname',
          'naxis', 'x', 'y', 'z', 'nel', 'atype', 'imtype',
          'crtime_sec', 'crtime_ns',
          'latime_sec', 'latime_ns',
          'atime_sec', 'atime_ns',
          'wtime_sec', 'wtime_ns',
          'shared', 'loc', 'status',
          'flag', 'logflag', 'sem',
          'cnt0', 'cnt1', 'cnt2',
          'write', 'nbkw', 'bcudamem']

''' 

One of the most important changes when moving from python2
to python3 for this library was that the string C type 
described by the format code "s" went from python data 
type "str" to "bytes", likely to  acommodate unicode.

---------------------------------------------------------
Table taken from Python 3 documentation, section 7.1.2.2.
---------------------------------------------------------

|--------+--------------------+---------------+----------|
| Format | C Type             | Python type   | Std size |
|--------+--------------------+---------------+----------|
| x      | pad byte           | no value      |          |
| c      | char               | bytes (len=1) |        1 |
| b      | signed char        | integer       |        1 |
| B      | unsigned char      | integer       |        1 |
| ?      | _Bool              | bool          |        1 |
| h      | short              | integer       |        2 |
| H      | unsigned short     | integer       |        2 |
| i      | int                | integer       |        4 |
| I      | unsigned int       | integer       |        4 |
| l      | long               | integer       |        4 |
| L      | unsigned long      | integer       |        4 |
| q      | long long          | integer       |        8 |
| Q      | unsigned long long | integer       |        8 |
| n      | ssize_t            | integer       |          |
| N      | size_t             | integer       |          |
| e      | (7)                | float         |        2 |
| f      | float              | float         |        4 |
| d      | double             | float         |        8 |
| s      | char[]             | bytes         |          |
| p      | char[]             | bytes         |          |
| P      | void *             | integer       |          |
|--------+--------------------+---------------+----------|

'''

class shm:
    def __init__(self, fname=None, data=None, verbose=False, packed=True, nbkw=0):
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
        self.packed = packed
        
        if self.packed:
            self.hdr_fmt = hdr_fmt_pck # packed shm structure
            self.kwfmt0 = "16s s"      # packed keyword structure
        else:
            self.hdr_fmt = hdr_fmt_aln # aligned shm structure
            self.kwfmt0 = "16s s7x"    # aligned keyword structure

        self.c0_offset = 0        # fast-offset for counter #0 (updated later)
        self.kwsz      = struct.calcsize('16s 80s'+' '+self.kwfmt0) # keyword SHM size

        # --------------------------------------------------------------------
        #                dictionary containing the metadata
        # --------------------------------------------------------------------

        self.mtdata = {'version': '', 'bversion': b'',
                       'imname': '', 'bimname': b'',
                       'naxis': 0, 'x' : 0, 'y': 0, 'z': 0,
                       'size': (0,0,0), 'nel': 0, 'atype': 0, 'imtype': 0,
                       'crtime_sec' : 0, 'crtime_ns' : 0, 'crtime': (0,0),
                       'latime_sec' : 0, 'latime_ns' : 0, 'latime': (0,0),
                       'atime_sec' : 0,  'atime_ns' : 0,  'atime': (0,0), 'atimearr': 0, 
                       'wtime_sec' : 0,  'wtime_ns' : 0,  'wtime': (0,0), 'wtimearr': 0,
                       'shared': 0, 'loc': 0, 'status': 0, 'flag': 0,
                       'flagarr': 0, 'logflag': 0, 'sem': 0,
                       'cnt0': 0, 'cnt1': 0, 'cnt2': 0, 'cntarr': 0,
                       'write': 0, 'nbkw': 0,
                       'cudamem': '', 'bcudamem': b''}

        # --------------------------------------------------------------------
        #          dictionary describing the content of a keyword
        # --------------------------------------------------------------------
        self.kwd = {'name': '', 'type': 'N', 'value': '', 'comment': ''}

        fmt     = self.hdr_fmt
        self.c0_offset = struct.calcsize(' '.join(fmt.split()[:c0_hdr_pos]))
        self.im_offset = struct.calcsize(fmt)

        # ---------------
        if fname is None:
            print("No SHM file name provided")
            return(None)

        # ---------------
        self.fname = fname
        if ((not os.path.exists(fname)) or (data is not None)):
            print("%s will be created or overwritten" % (fname,))
            self.create(fname, data, nbkw)

        # ---------------
        else:
            if verbose:
                print("reading from existing %s" % (fname,))
            self.fd      = os.open(fname, os.O_RDWR)
            self.stats   = os.fstat(self.fd)
            self.buf_len = self.stats.st_size
            self.buf     = mmap.mmap(self.fd, self.buf_len, mmap.MAP_SHARED)
            self.read_meta_data(verbose=verbose)

            self.select_dtype()        # identify main data-type
            self.get_data()            # read the main data
            self.create_keyword_list() # create empty list of keywords
            self.read_keywords()       # populate the keywords with data
            
    def create(self, fname, data, nbkw=0):
        ''' --------------------------------------------------------------
        Create a shared memory data structure

        Parameters:
        ----------
        - fname: name of the shared memory file structure
        - data: some array (1, 2 or 3D of data)
        
        Called by the constructor if the provided file-name does not
        exist: a new structure needs to be created, and will be populated
        with information based on the provided data.
        -------------------------------------------------------------- '''
        
        if data is None:
            print("No data (ndarray) provided! Nothing happens here")
            return

        # ---------------------------------------------------------
        # feed the relevant dictionary entries with available data
        # ---------------------------------------------------------
        self.npdtype            = data.dtype
        self.mtdata['imname']   = fname.ljust(80, ' ')
        self.mtdata['bimname']  = bytes(self.mtdata['imname'], 'ascii')

        self.mtdata['version']  = "xaosim".ljust(32, ' ')
        self.mtdata['bversion'] = bytes(self.mtdata['version'], 'ascii')
        
        self.mtdata['naxis']    = data.ndim
        self.mtdata['size']     = data.shape[:data.ndim][::-1]
        self.mtdata['nel']      = data.size
        self.mtdata['atype']    = self.select_atype()
        self.mtdata['shared']   = 1
        self.mtdata['nbkw']     = nbkw
        
        if data.ndim == 3:
            self.mtdata['x']    = self.mtdata['size'][0]
            self.mtdata['y']    = self.mtdata['size'][1]
            self.mtdata['z']    = self.mtdata['size'][2]
        if data.ndim == 2:
            self.mtdata['x']    = self.mtdata['size'][0]
            self.mtdata['y']    = self.mtdata['size'][1]
            self.mtdata['size'] = self.mtdata['size'] + (0,)

        self.select_dtype()

        # ---------------------------------------------------------
        #          reconstruct a SHM metadata buffer
        # ---------------------------------------------------------
        fmt     = self.hdr_fmt
        temp    = [self.mtdata[mtkeys[ii]] for ii in range(len(mtkeys))]
        minibuf = struct.pack(fmt, *temp)

        
        # ---------------------------------------------------------
        #             allocate the file and mmap it
        # ---------------------------------------------------------
        kwspace = self.kwsz * nbkw                    # kword space
        fsz = self.im_offset + self.img_len + kwspace # file size
        npg = fsz // mmap.PAGESIZE + 1                 # nb pages

        self.fd = os.open(fname, os.O_CREAT | os.O_TRUNC | os.O_RDWR)
        os.fchmod(self.fd, 0o777) # give RWX access to all users
        os.write(self.fd, b'\x00' * npg * mmap.PAGESIZE)
        self.buf = mmap.mmap(self.fd, npg * mmap.PAGESIZE, 
                             mmap.MAP_SHARED, mmap.PROT_WRITE)

        # ---------------------------------------------------------
        #              write the information to SHM
        # ---------------------------------------------------------
        self.buf[:self.im_offset] = minibuf # the metadata
        
        self.set_data(data)
        self.create_keyword_list()
        self.write_keywords()

    def rename_img(self, newname):
        ''' --------------------------------------------------------------
        Gives the user a chance to rename the image.

        Parameter:
        ---------
        - newname: a string (< 80 char) with the name
        -------------------------------------------------------------- '''
        
        self.mtdata['imname']  = newname.ljust(80, ' ')
        self.mtdata['bimname'] = bytes(self.mtdata['imname'], 'ascii')
        self.buf[0:80]        = struct.pack('80s', self.mtdata['bimname'])

    def close(self,):
        ''' --------------------------------------------------------------
        Clean close of a SHM data structure link

        Clean close of buffer, release the file descriptor.
        -------------------------------------------------------------- '''
        c0   = self.c0_offset                     # counter offset
        self.buf[c0:c0+8]   = struct.pack('Q', 0) # set counter to zero
        self.buf.close()
        os.close(self.fd)
        self.fd = 0
        return(0)

    def read_meta_data(self, verbose=True):
        ''' --------------------------------------------------------------
        Read the metadata fraction of the SHM file.
        Populate the shm object mtdata dictionary.

        Parameters:
        ----------
        - verbose: (boolean, default: True), prints its findings
        -------------------------------------------------------------- '''
        fmt = self.hdr_fmt
        hlen = struct.calcsize(fmt)
        temp = struct.unpack(fmt, self.buf[:hlen])
        
        for ii in range(len(mtkeys)):
            self.mtdata[mtkeys[ii]] = temp[ii]

        # special repackaging: image name (string) and size (tuple)
        self.mtdata['imname']  = self.mtdata['bimname'].decode('ascii').strip('\x00')
        self.mtdata['version'] = self.mtdata['bversion'].decode('ascii').strip('\x00')
        self.mtdata['cudamem'] = self.mtdata['bcudamem'].decode('ascii').strip('\x00')
        self.mtdata['size']    = self.mtdata['z'], self.mtdata['y'], self.mtdata['x']
        self.mtdata['crtime'] = self.mtdata['crtime_sec'], self.mtdata['crtime_ns']
        self.mtdata['latime'] = self.mtdata['latime_sec'], self.mtdata['latime_ns']
        self.mtdata['atime']  = self.mtdata['atime_sec'], self.mtdata['atime_ns']
        self.mtdata['wtime']  = self.mtdata['wtime_sec'], self.mtdata['wtime_ns']

        if verbose:
            self.print_meta_data()

    def create_keyword_list(self):
        ''' --------------------------------------------------------------
        Place-holder. The name should be sufficiently explicit.
        -------------------------------------------------------------- '''
        nbkw = self.mtdata['nbkw']     # how many keywords
        self.kwds = []                 # prepare an empty list 
        for ii in range(nbkw):         # fill with empty dictionaries
            self.kwds.append(self.kwd.copy())
            
    def read_keywords(self):
        ''' --------------------------------------------------------------
        Read all keywords from SHM file
        -------------------------------------------------------------- '''        
        for ii in range(self.mtdata['nbkw']):
            self.read_keyword(ii)

    def write_keywords(self):
        ''' --------------------------------------------------------------
        Writes all keyword data to SHM file
        -------------------------------------------------------------- '''
        for ii in range(self.mtdata['nbkw']):
            self.write_keyword(ii)

    def read_keyword(self, ii):
        ''' --------------------------------------------------------------
        Read the content of keyword of given index.

        Parameters:
        ----------
        - ii: index of the keyword to read
        -------------------------------------------------------------- '''
        kwsz = self.kwsz              # keyword SHM data structure size
        k0   = self.im_offset + self.img_len + ii * kwsz # kword offset

        # ------------------------------------------
        #             read from SHM
        # ------------------------------------------
        kwlen = struct.calcsize(self.kwfmt0)
        kname, ktype = struct.unpack(self.kwfmt0, self.buf[k0:k0+kwlen]) 

        # ------------------------------------------
        # depending on type, select parsing strategy
        # ------------------------------------------
        kwfmt = '16s 80s'
        
        if ktype == b'L':   # keyword value is int64
            kwfmt = 'q 8x 80s'
        elif ktype == b'D': # keyword value is double
            kwfmt = 'd 8x 80s'
        elif ktype == b'S': # keyword value is string
            kwfmt = '16s 80s'
        elif ktype == b'N': # keyword is unused
            kwfmt = '16s 80s'
        
        kval, kcomm = struct.unpack(kwfmt, self.buf[k0+kwlen:k0+kwsz])

        if kwfmt == '16s 80s':
            kval = str(kval).strip('\x00')

        # ------------------------------------------
        #    fill in the dictionary of keywords
        # ------------------------------------------
        if (ktype == b'L'):
            self.kwds[ii]['value'] = np.long(kval)
        elif (ktype == b'D'):
            self.kwds[ii]['value'] = np.double(kval)
        else:
            self.kwds[ii]['value'] = ktype.decode('ascii').strip('\x00')
            
        self.kwds[ii]['name']    = kname.decode('ascii').strip('\x00')
        self.kwds[ii]['type']    = ktype.decode('ascii')
        self.kwds[ii]['comment'] = kcomm.decode('ascii').strip('\x00')
        
    def update_keyword(self, ii, name, value, comment):
        ''' --------------------------------------------------------------
        Update keyword data in dictionary and writes it to SHM file

        Parameters:
        ----------
        - ii      : index of the keyword to write (integer)
        - name    : the new keyword name 
        -------------------------------------------------------------- '''

        if (ii >= self.mtdata['nbkw']):
            print("Keyword index %d is not allocated and cannot be written")
            return

        # ------------------------------------------
        #    update relevant keyword dictionary
        # ------------------------------------------
        try:
            self.kwds[ii]['name'] = str(name).ljust(16, ' ')
        except:
            print('Keyword name not compatible (< 16 char)')

        if isinstance(value, int):
            self.kwds[ii]['type'] = 'L'
            self.kwds[ii]['value'] = np.long(value)
            
        elif isinstance(value, float):
            self.kwds[ii]['type'] = 'D'
            self.kwds[ii]['value'] = np.double(value)
            
        elif isinstance(value, str):
            self.kwds[ii]['type'] = 'S'
            self.kwds[ii]['value'] = str(value)
        else:
            self.kwds[ii]['type'] = 'N'
            self.kwds[ii]['value'] = str(value)

        try:
            self.kwds[ii]['comment'] = str(comment).ljust(80, ' ')
        except:
            print('Keyword comment not compatible (< 80 char)')

        # ------------------------------------------
        #          write keyword to SHM
        # ------------------------------------------
        self.write_keyword(ii)
        
    def write_keyword(self, ii):
        ''' --------------------------------------------------------------
        Write keyword data to shared memory.

        Parameters:
        ----------
        - ii      : index of the keyword to write (integer)
        -------------------------------------------------------------- '''

        if (ii >= self.mtdata['nbkw']):
            print("Keyword index %d is not allocated and cannot be written")
            return

        kwsz = self.kwsz
        k0   = self.im_offset + self.img_len + ii * kwsz # kword offset
        
        # ------------------------------------------
        #    read the keyword dictionary
        # ------------------------------------------
        kname = self.kwds[ii]['name']
        ktype = self.kwds[ii]['type']
        kval  = self.kwds[ii]['value']
        kcomm = self.kwds[ii]['comment']

        if ktype == 'L' or ktype == b'L':
            kwfmt = '='+self.kwfmt0+' q 8x 80s'
            tmp = (bytes(kname, "ascii"),
                   bytes(ktype, "ascii"),
                   kval,
                   bytes(kcomm, "ascii"))
        elif ktype == 'D' or ktype == b'D':
            kwfmt = '='+self.kwfmt0+' d 8x 80s'
            tmp = (bytes(kname, "ascii"),
                   bytes(ktype, "ascii"),
                   kval,
                   bytes(kcomm, "ascii"))
        else: # 'S' or 'N'
            kwfmt = '='+self.kwfmt0+' 16s 80s'
            tmp = (bytes(kname, "ascii"),
                   bytes(ktype, "ascii"),
                   bytes(kval, "ascii"),
                   bytes(kcomm, "ascii"))

        self.buf[k0:k0+kwsz] = struct.pack(kwfmt,  *tmp) 

    def print_meta_data(self):
        ''' --------------------------------------------------------------
        Basic printout of the content of the mtdata dictionary.
        -------------------------------------------------------------- '''
        for ii in range(len(mtkeys)):
            print(mtkeys[ii], self.mtdata[mtkeys[ii]])
        
    def select_dtype(self):
        ''' --------------------------------------------------------------
        Based on the value of the 'atype' code used in SHM, determines
        which numpy data format to use.
        -------------------------------------------------------------- '''
        atype        = self.mtdata['atype']
        self.npdtype = all_dtypes[atype-1]
        self.img_len = self.mtdata['nel'] * self.npdtype().itemsize

    def select_atype(self):
        ''' --------------------------------------------------------------
        Based on the type of numpy data provided, sets the appropriate
        'atype' value in the metadata of the SHM file.
        -------------------------------------------------------------- '''
        for i, mydt in enumerate(all_dtypes):
            if mydt == self.npdtype:
                self.mtdata['atype'] = i+1
        return(self.mtdata['atype'])

    def get_counter(self,):
        ''' --------------------------------------------------------------
        Read the image counter from SHM
        -------------------------------------------------------------- '''
        c0   = self.c0_offset                           # counter offset
        cntr = struct.unpack('Q', self.buf[c0:c0+8])[0] # read from SHM
        self.mtdata['cnt0'] = cntr                      # update object mtdata
        return(cntr)

    def increment_counter(self,):
        ''' --------------------------------------------------------------
        Increment the image counter. Called when writing new data to SHM
        -------------------------------------------------------------- '''
        c0                  = self.c0_offset         # counter offset
        cntr                = self.get_counter() + 1 # increment counter
        self.buf[c0:c0+8]   = struct.pack('Q', cntr) # update SHM file
        self.mtdata['cnt0'] = cntr                   # update object mtdata
        return(cntr)

    def get_data(self, check=False, reform=True, sleepT=0.001, timeout=5):
        ''' --------------------------------------------------------------
        Reads and returns the data part of the SHM file

        Parameters:
        ----------
        - check: integer (last index) if not False, waits image update
        - reform: boolean, if True, reshapes the array in a 2-3D format
        - sleepT: time increment (in seconds) when waiting for new data
        - timeout: timeout in seconds
        -------------------------------------------------------------- '''
        i0 = self.im_offset                                  # image offset
        i1 = i0 + self.img_len                               # image end

        time0 = time.time()
        if check is not False:
            timen = time.time()
            
            while ((self.get_counter() <= check) and (timen-time0 < timeout)):
                time.sleep(sleepT)
                timen = time.time()

        data = np.fromstring(self.buf[i0:i1],dtype=self.npdtype) # read img
        
        if reform:
            if self.mtdata['naxis'] == 2:
                rsz = self.mtdata['y'], self.mtdata['x']
            else:
                rsz = self.mtdata['z'], self.mtdata['y'], self.mtdata['x']
            data = np.reshape(data, rsz)
        return(data)

    def set_data(self, data, check_dt=False):
        ''' --------------------------------------------------------------
        Upload new data to the SHM file.

        Parameters:
        ----------
        - data: the array to upload to SHM
        - check_dt: boolean (default: false) recasts data

        Note:
        ----

        The check_dt is available here for comfort. For the sake of
        performance, data should be properly cast to start with, and
        this option not used!
        -------------------------------------------------------------- '''
        i0 = self.im_offset                                      # image offset
        i1 = i0 + self.img_len                                   # image end

        if check_dt is True:
            self.buf[i0:i1] = data.astype(self.npdtype()).tostring()
        else:
            try:
                self.buf[i0:i1] = data.tostring()
            except:
                print("Warning: writing wrong data-type to shared memory")
                return
        self.increment_counter()

        return

    def save_as_fits(self, fitsname):
        ''' --------------------------------------------------------------
        Convenient sometimes, to be able to export the data as a fits file.
        
        Parameters:
        ----------
        - fitsname: a filename (clobber=True)
        -------------------------------------------------------------- '''
        try:
            import astropy.io.fits as pf
            pf.writeto(fitsname, self.get_data(), overwrite=True)
        except:
            import pyfits as pf
            pf.writeto(fitsname, self.get_data(), clobber=True)
        return(0)

# =================================================================
# =================================================================
