''' xaosim: diffractive simulation tool-kit
    ===============================================================

    Example use:
    -----------

    >> python
    # import xaosim as xs
    # mysetup = xs.instrument("SCExAO")
    # mysetup.start()

    In a distinct shell:

    >> shmview /dev/shm/maunakea.im.shm &
    >> shmview /dev/shm/scexao_chuck.im.shm &

    Opens two external GUIs that show the live image and phase screen.
    --------------------------------------------------------------------- '''

# from .instrument import *

# import numpy.fft as fftmod

__version__ = "1.0.11"
# print("XAOSIM version ", __version__)
