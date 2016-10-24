xaosim: a series of tools used to simulate an XAO-like system
===============================================================

The package includes:
--------------------

- a deformable mirror (DM) module that includes a multi-channel communication
  system, assumed to be located in a pupil plane

- a camera (CAM) module that produces images of a point source affected after
  the wavefront has undergone the transformation induced by the DM

It relies on a series of auxilliary tools:
-----------------------------------------

- pupil: used to generate models of the aperture of several telescopes.

- zernike: used to generate Zernike wavefront modes on the DM.

- shmlib: a library that uses shared memory data structures like the ones
  designed to control the SCExAO instrument at the Subaru Telescope.

