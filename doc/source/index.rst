
S+LEAF documentation
====================

Installation
------------

The S+LEAF package can be installed using the pip utility

``pip install --extra-index-url https://obswww.unige.ch/~delisle spleaf``

and upgraded with

``pip install --extra-index-url https://obswww.unige.ch/~delisle spleaf --upgrade``

Usage
-----

The S+LEAF package provides two main classes:
:doc:`classes/spleaf.Spleaf` and :doc:`classes/spleaf.rv.Cov`.

The :doc:`classes/spleaf.Spleaf` class is
a generic low level implementation of S+LEAF (semiseparable + leaf) matrices
as defined by [1]_.

The :doc:`classes/spleaf.rv.Cov` class provides a higher level API
to model correlated noise in a radial velocity timeseries
using S+LEAF matrices (see also [1]_).

Examples of the use of the S+LEAF package are available at
`<https://gitlab.unige.ch/jean-baptiste.delisle/spleaf/tree/master/examples/>`_.

API Reference
-------------

.. autosummary::
   :toctree: classes
   :template: autosummary/class.rst
   :nosignatures:

   spleaf.Spleaf
   spleaf.rv.Cov

References
----------

.. [1] Delisle, J.-B., Hara, N., and Ségransan, D.,
   "Efficient modeling of correlated noise.
   Application to the analysis of radial velocity timeseries",
   2019
