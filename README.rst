xaosim: a series of tools used to simulate an XAO-like system
===============================================================

The specificity of XAOSIM is that it was built around shared memory data
structures defined in the context of the SCExAO instrument by Olivier Guyon and
Frantz Martinache. This approach makes it possible to design real-time AO
control software using XAOSIM's simulation environment data and deploy it on an
actual instrument (assuming that it uses the same exchange format) in a
transparent manner.

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

>> python

# import xaosim

# mysetup = xaosim.instrument("SCExAO")

# mysetup.start()

In a distinct shell:

>> shmview /dev/shm/phscreen.im.shm &

>> shmview /dev/shm/ircam.im.shm &

Will open two pygame displays that show the live image and phase screen.

The package includes:
--------------------

- a deformable mirror (DM) module that includes a multi-channel communication
  system, assumed to be located in a pupil plane.

- an atmospheric phase screen (phscreen) module that simulates a Kolmogorov
  frozen screen drifting over the aperture in a predefined direction.
  
- a camera (CAM) module that produces images of a point source affected after
  the wavefront has undergone the transformation induced by the DM.

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

As of March 2019, XAOSIM is now fully Python 3 compliant.
