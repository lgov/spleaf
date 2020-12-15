
S+LEAF documentation
====================

S+LEAF provides a flexible noise model with fast and scalable methods.
It is largely inspired by the
`celerite <https://github.com/dfm/celerite>`_ / `celerite2 <https://github.com/exoplanet-dev/celerite2>`_
model proposed by [1]_.
In particular the modeling of gaussian processes is similar,
and uses the same semiseparable matrices representation as celerite.
S+LEAF extends the celerite model by allowing to account
for close to diagonal (LEAF) noises such as instrument calibration errors
(see [2]_ for more details).

Installation
------------

The S+LEAF package can be installed using the pip utility

``pip install --extra-index-url https://obswww.unige.ch/~delisle spleaf``

and upgraded with

``pip install --extra-index-url https://obswww.unige.ch/~delisle spleaf --upgrade``

Usage
-----

S+LEAF covariance matrices are generated using the
:doc:`classes/spleaf.cov.Cov` class.
The covariance matrix is modeled as the sum of different components (or terms),
which split into two categories:
noise terms and kernel terms (gaussian processes).
See the :ref:`API reference<api_ref>` for a list of available terms.

The typical way to use S+LEAF is:

.. code-block:: python

   # import
   from spleaf.cov import Cov
   from spleaf.term import *

   # Generate the covariance matrix
   cov = Cov(t,
      err = Error(yerr),
      caliberr = CalibrationError(calib_id, calib_err),
      sho = SHOKernel(sig, P0, Q))

   # Use it
   print(cov.loglike(y))

See the :ref:`API reference<api_ref>` for more details.

The low level implementation of
S+LEAF matrices as defined by [2]_
is available as the :doc:`classes/spleaf.Spleaf` class,
but one typically does not need to directly deal with it.

.. _api_ref:

API Reference
-------------

.. autosummary::
   :toctree: classes
   :template: autosummary/class.rst
   :nosignatures:

   spleaf.cov.Cov
   spleaf.term.Term
   spleaf.term.Noise
   spleaf.term.Kernel
   spleaf.term.Error
   spleaf.term.Jitter
   spleaf.term.InstrumentJitter
   spleaf.term.CalibrationError
   spleaf.term.CalibrationJitter
   spleaf.term.ExponentialKernel
   spleaf.term.QuasiperiodicKernel
   spleaf.term.Matern32Kernel
   spleaf.term.USHOKernel
   spleaf.term.OSHOKernel
   spleaf.term.SHOKernel
   spleaf.Spleaf

References
----------

.. [1] `Foreman-Mackey et al., "Fast and Scalable Gaussian Process Modeling with Applications to Astronomical Time Series", 2017 <http://adsabs.harvard.edu/abs/2017AJ....154..220F>`_.
.. [2] `Delisle, J.-B., Hara, N., and Ségransan, D., "Efficient modeling of correlated noise. II. A flexible noise model with fast and scalable methods", 2020 <https://ui.adsabs.harvard.edu/abs/2020A\&A...638A..95D>`_.
