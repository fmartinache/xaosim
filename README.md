# XAOSIM: a XAO-like system simulation and interaction environment

**XAOSIM** is a simulation package developed for theoretical and practical work in high angular resolution astronomy.

## Why yet another XAO simulation environment?

The specificity of **XAOSIM** is that it was built around shared memory data structures defined in the context of the SCExAO instrument by Olivier Guyon and Frantz Martinache. This approach makes it possible to design real-time AO control software using XAOSIM's simulation environment data and deploy it on an actual instrument (assuming that it uses the same exchange format) in a transparent manner.

## Acknowledgement

The development and maintenance of **XAOSIM** has been supported by the KERNEL project funded by the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation program (grant agreement CoG - 683029).

## Installation

> pip install xaosim

## The package includes:

- a deformable mirror (DM) module that includes a multi-channel communication system, assumed to be located in a pupil plane. The DM class simulates a continuous membrane mirror with actuators laid out on a regualar grid.
- an atmospheric phase screen (Phscreen) module that simulates a Kolmogorov + Von Karman frozen screen drifting over the aperture in a predefined direction.
- a camera (Cam) module that produces images of a point source affected after the wavefront has undergone the transformation induced by the DM. One special case of camera is the Shack-Hartman camera (SHcam) used for wavefront sensing.  Another special case is the coronagraphic camera (CoroCam) that can be used to simulate a wide range of coronagraphs (no PIAA-like coronagraphs though).

The code is reasonably well documented and if you are experienced with diffractive optics simulations, you should quickly feel at home, and change the parameters of the turbulence, simulate partial AO correction and even use a perfect coronagraph.
