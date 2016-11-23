xaosim: a series of tools used to simulate an XAO-like system
===============================================================

Example use:
-----------

>> python

# import xaosim

# mysetup = xaosim.instrument("CIAO")

# mysetup.start()

In a distinct shell:

>> shmview /tmp/phscreen.im.shm &

>> shmview /tmp/ciao_cam.im.shm &

Will open two pygame displays that show the live image and phase screen.

The package includes:
--------------------

- a deformable mirror (DM) module that includes a multi-channel communication
  system, assumed to be located in a pupil plane.

- an atmospheric phase screen (phscreen) module that simulates a Kolmogorov
  frozen screen drifting over the aperture in a predefined direction.
  
- a camera (CAM) module that produces images of a point source affected after
  the wavefront has undergone the transformation induced by the DM.

It relies on a series of auxilliary tools:
-----------------------------------------

- pupil: used to generate models of the aperture of several telescopes.

- zernike: used to generate Zernike wavefront modes on the DM.

- shmlib: a library that uses shared memory data structures like the ones
  designed to control the SCExAO instrument at the Subaru Telescope.

- shmview: a python program using the shared memory library to visualize the 2D
  data structures.
  
Release Notes:
-------------

- On some systems, after installation is complete, the call for "shmview" may
  fail due to apparent permission limitations. The exact error message
  encountered is: "IOError: [Errno 13] Permission denied:
  '/usr/local/lib/python2.7/dist-packages/xaosim-0.1-py2.7.egg/EGG-INFO/requires.txt"

  The current fix is to set the permissions right on this requirement file:

  "sudo chmod a+r /usr/local/lib/python2.7/dist-packages/\
   xaosim-0.1-py2.7.egg/EGG-INFO/requires.txt"

   
