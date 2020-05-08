XAOSIM: a series of tools used to simulate an XAO-like system
===============================================================

XAOSIM is a simulation package developed for theoretical and practical work in high angular resolution astronomy.

The specificity of XAOSIM is that it was built around shared memory data structures defined in the context of the SCExAO instrument by Olivier Guyon and Frantz Martinache. This approach makes it possible to design real-time AO control software using XAOSIM's simulation environment data and deploy it on an actual instrument (assuming that it uses the same exchange format) in a transparent manner.

In addition to the simulation library package, a shared memory data viewer GUI
(shmview) is now part of the distribution.

Acknowledgement
---------------

The development and maintenance of XAOSIM receives support from the KERNEL project. KERNEL has received funding from the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation program (grand agreement CoG - 683029). For more information about the KERNEL project, visit: http://frantzmartinache.eu/index.php/category/kernel/

Recommandation for installation:
-------------------------------

>> python setup.py install --user

This will create a .config/xaosim/ configuration directory in the user home folder.

It is also recommended to add ~/.local/bin/ to the path:

>> export PATH=$HOME/.local/bin/:$PATH


Example use:
-----------

The package makes it possible to create an "instrument", that is a system made up of a telescope, an atmosphere, a deformable mirror that feed one or more cameras. To facilitate the use, it comes with a series of preset templates, such as the SCExAO instrument at the Subaru Telescope, the NIRC2 camera on Keck II, the NICMOS1 camera of HST, and a Shack-Hartman based AO system built at OCA for one of the C2PU telescopes. New templates can be added as required.

>> python

# import xaosim as xs

# scexao = xs.instrument("SCExAO", csz=245)

# scexao.start()

In a distinct shell:

>> shmview /dev/shm/phscreen.im.shm &

>> shmview /dev/shm/scexao_ircam.im.shm &

Will open two display utilities that show the live image and phase screen.

The package includes:
--------------------

- a deformable mirror (DM) module that includes a multi-channel communication
  system, assumed to be located in a pupil plane. The DM class simulates a
  continuous membrane mirror with actuators laid out on a regualar grid.

- an atmospheric phase screen (Phscreen) module that simulates a Kolmogorov +
  Von Karman frozen screen drifting over the aperture in a predefined
  direction.
  
- a camera (Cam) module that produces images of a point source affected after
  the wavefront has undergone the transformation induced by the DM. One special
  case of camera is the Shack-Hartman camera (SHcam) used for wavefront sensing.


The code is reasonably well documented and if you are experienced with
diffractive optics simulations, you should quickly feel at home, and change the
parameters of the turbulence, simulate partial AO correction and even use a
perfect coronagraph.

It relies on a series of auxilliary tools:
-----------------------------------------

- pupil: used to generate models of the aperture of several telescopes.

- zernike: used to generate Zernike wavefront modes on the DM.

- shmlib: a library that uses shared memory data structures like the ones
  designed to control the SCExAO instrument at the Subaru Telescope.

- shmview: a python GUI using the shared memory library to visualize the 2D
  data structures. The current version of shmview is no longer based on the
  pygame framework but on Qt5. This program benefits from an additional module
  designed by Eric Jeschke (eric@naoj.org) to make it multi-thread safe.
  
Release Notes:
-------------

- March 2019: XAOSIM is now fully Python 3 compliant.
- May 2020: XAOSIM was rewritten during the COVID19 lockdown to accomodate emerging needs: segmented mirrors, higher fidelity DM simulation for fine focal plane-based metrology, Shack-Hartman camera, the ability to change the filter of the camera without altering the rest of the system.
  
