# -*- coding: utf-8 -*-

# Copyright 2020 Jean-Baptiste Delisle
#
# This file is part of spleaf.
#
# spleaf is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# spleaf is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with spleaf.  If not, see <http://www.gnu.org/licenses/>.

__all__ = ['Cov']

import numpy as np
from . import Spleaf, libspleaf
from .term import Noise, Kernel

class Cov(Spleaf):
  r"""
  Covariance matrix class.

  The covariance is modeled as the sum of different components (or terms),
  which split into two categories:
  noise terms and kernel terms (gaussian processes).
  See the :ref:`API reference<api_ref>` for a list of available terms.

  Each component is provided in \*\*kwargs, where the key is user-defined
  and is used to refer to the term parameters.

  The covariance matrix is represented internally as
  a S+LEAF matrix (see :class:`spleaf.Spleaf`).

  Parameters
  ----------
  t : (n,) ndarray
    Time of the measurements.
    This array must be in increasing order.
  **kwargs :
    Components of the covariance matrix.

  Attributes
  ----------
  t : (n,) ndarray
    See parameters.
  n : int
    Number of measurements.
  dt : (n-1,) ndarray
    time between two measurements.
  param : list,
    List of all covariance parameters.
  term : dict,
    Dictionary of all covariance terms.
  noise : dict,
    Dictionary of noise terms.
  kernel : dict,
    Dictionary of kernel terms.
  r, A, U, V, phi, offsetrow, b, F, D, W, G :
    S+LEAF representation of the covariance matrix
    (see :class:`spleaf.Spleaf`).
  """

  def __init__(self, t, **kwargs):
    self.t = t
    if not isinstance(t, np.ndarray):
      raise Exception('Cov: t is not an array.')
    self.n = t.size
    self.dt = t[1:] - t[:-1]
    if np.min(self.dt) < 0:
      raise Exception('Cov: the timeseries must be provided'
        ' in increasing order.')

    # Read kwargs
    self.noise = {}
    self.b = np.zeros(self.n, dtype=int)
    self.kernel = {}
    self.r = 0
    self.term = {}
    self.param = []
    self._param_dict = {}
    for key in kwargs:
      if isinstance(kwargs[key], Noise):
        self.noise[key] = kwargs[key]
        kwargs[key]._link(self)
        self.b = np.maximum(self.b, kwargs[key]._b)
      elif isinstance(kwargs[key], Kernel):
        self.kernel[key] = kwargs[key]
        kwargs[key]._link(self, self.r)
        self.r += kwargs[key]._r
      else:
        raise Exception(
          'The provided argument is not of type Noise or Kernel.')
      self.term[key] = kwargs[key]
      self.param += [f'{key}.{param}' for param in kwargs[key]._param]
      self._param_dict.update({f'{key}.{param}':(key, param)
        for param in kwargs[key]._param})

    # Compute S+LEAF representation
    self.A = np.zeros(self.n)
    self.U = np.empty((self.n, self.r))
    self.V = np.empty((self.n, self.r))
    self.phi = np.empty((self.n-1, self.r))
    self.offsetrow = np.cumsum(self.b-1) + 1
    self.F = np.zeros(np.sum(self.b))
    for key in self.term:
      self.term[key]._compute()

    super().__init__(self.A, self.U, self.V, self.phi,
      self.offsetrow, self.b, self.F)

  def get_param(self, paramlist=None):
    r"""
    Get the values of the parameters.

    Parameters
    ----------
    paramlist : list or str or None
      List of parameters (or single parameter).
      If None, all the covariance parameters are provided.

    Returns
    -------
    value : (p,) ndarray or float
      Values of the parameters.
    """

    single = False
    if paramlist is None:
      paramlist = self.param
    if isinstance(paramlist, str):
      paramlist = [paramlist]
      single = True
    value = np.empty(len(paramlist))
    for k, param in enumerate(paramlist):
      key, par = self._param_dict[param]
      value[k] = getattr(self.term[key], f'_{par}')
    return(value[0] if single else value)

  def set_param(self, value, paramlist=None):
    r"""
    Set the values of the parameters.

    Parameters
    ----------
    value : (p,) ndarray or float
      Values of the parameters.

    paramlist : list or str or None
      List of parameters (or single parameter).
      If None, all the covariance parameters are set.
    """

    if paramlist is None:
      paramlist = self.param
    if isinstance(paramlist, str):
      paramlist = [paramlist]
      value = np.array([value])
    param_split = {key:{} for key in self.term}
    for param, value in zip(paramlist, value):
      key, par = self._param_dict[param]
      param_split[key][par] = value
    self.A[:] = 0
    self.F[:] = 0
    for key in self.term:
      if param_split[key] != {}:
        self.term[key]._set_param(**param_split[key])
      self.term[key]._recompute()

    super().set_param(self.A, self.U, self.V, self.phi, self.F)

  def grad_param(self, paramlist=None):
    r"""
    Gradient of a function with respect to
    the parameters, after a call to :func:`cholesky_back`.

    Parameters
    ----------
    paramlist : list or str or None
      List of parameters (or single parameter).
      If None, all the covariance parameters are provided.

    Returns
    -------
    grad_param : (p,) ndarray or float
      Gradient of the function with respect to the parameters.
    """
    single = False
    if paramlist is None:
      paramlist = self.param
      term = self.term
    else:
      if isinstance(paramlist, str):
        paramlist = [paramlist]
        single = True
      term = {self._param_dict[param][0] for param in paramlist}
    self._sum_grad_A = np.sum(self._grad_A)
    grad = {}
    for key in term:
      grad_key = self.term[key]._grad_param()
      for param in grad_key:
        grad[f'{key}.{param}'] = grad_key[param]
    if single:
      return(grad[paramlist[0]])
    else:
      return(np.array([grad[param] for param in paramlist]))

  def _kernel_index(self, kernel=None):
    r"""
    List of indices corresponding to the requested kernel terms
    in the semiseparable representation of the matrix.

    Parameters
    ----------
    kernel : list or None
      List of kernel identifiers.

    Returns
    -------
    index : (r',) ndarray
      Indices of the corresponding semiseparable terms.
    """

    if kernel is None:
      return(None)
    else:
      return(
        np.array([s for key in kernel
          for s in range(self.kernel[key]._offset,
            self.kernel[key]._offset+self.kernel[key]._r)
        ], dtype=int)
      )

  def self_conditional(self, y, calc_cov=False, kernel=None):
    r"""
    Conditional mean and covariance
    of the kernel part, or a subset of kernel terms,
    knowning the observed values :math:`y`.

    Parameters
    ----------
    y : (n,) ndarray
      The vector of observed values :math:`y`.
    calc_cov : False (default), True, or 'diag'
      Whether to output only the conditional mean (False),
      the mean and full covariance matrix (True),
      or the mean and main diagonal of the covariance matrix ('diag').
    kernel : list or None
      List of kernel identifiers
      that should be considered for the Gaussian process.
      Other terms (kernel or noise) are considered as noise.
      If kernel is None, all kernel terms are considered for the
      Gaussian process.

    Returns
    -------
    mu : (n,) ndarray
      The vector of conditional mean values.
    cov : (n, n) ndarray
      Full covariance matrix (if calc_cov is True).
    var : (n,) ndarray
      Main diagonal of the covariance matrix (if calc_cov is 'diag').

    Warnings
    --------
    While the computational cost of the conditional mean scales as
    :math:`\mathcal{O}(n)`,
    the computational cost of the variance scales as
    :math:`\mathcal{O}(n^2)`,
    and the computational cost of the full covariance scales as
    :math:`\mathcal{O}(n^3)`.
    """

    return(super().self_conditional(y, calc_cov, self._kernel_index(kernel)))

  def conditional(self, y, t2, calc_cov=False, kernel=None):
    r"""
    Conditional mean and covariance
    of the kernel part, or a subset of kernel terms,
    at new times :math:`t_2`,
    knowning the observed values :math:`y`.

    Parameters
    ----------
    y : (n,) ndarray
      The vector of observed values :math:`y`.
    t2 : (n2,) ndarrays
      The vector of new times.
    calc_cov : False (default), True, or 'diag'
      Whether to output only the conditional mean (False),
      the mean and full covariance matrix (True),
      or the mean and main diagonal of the covariance matrix ('diag').
    kernel : list or None
      List of kernel identifiers
      that should be considered for the Gaussian process.
      Other terms (kernel or noise) are considered as noise.
      If kernel is None, all kernel terms are considered for the
      Gaussian process.

    Returns
    -------
    mu : (n2,) ndarray
      The vector of conditional mean values.
    cov : (n2, n2) ndarray
      Full covariance matrix (if calc_cov is True).
    var : (n2,) ndarray
      Main diagonal of the covariance matrix (if calc_cov is 'diag').

    Warnings
    --------
    While the computational cost of the conditional mean scales as
    :math:`\mathcal{O}(n+n_2)`,
    the computational cost of the variance scales as
    :math:`\mathcal{O}(n n_2)`,
    and the computational cost of the full covariance scales as
    :math:`\mathcal{O}(n n_2^2)`.
    """

    n2 = t2.size
    dt2 = t2[1:] - t2[:-1]
    U2 = np.empty((n2, self.r))
    V2 = np.empty((n2, self.r))
    phi2 = np.empty((n2-1, self.r))
    phi2left = np.empty((n2, self.r))
    phi2right = np.empty((n2, self.r))

    ref2left = np.searchsorted(self.t, t2) - 1
    dt2left = t2 - self.t[ref2left]
    dt2right = self.t[np.minimum(ref2left+1,self.n-1)] - t2
    dt2left[ref2left==-1] = 0 # useless but avoid overflow warning
    dt2right[ref2left==self.n-1] = 0 # useless but avoid overflow warning

    kernel_list = self.kernel if kernel is None else kernel
    for key in kernel_list:
      self.kernel[key]._compute_t2(
        t2, dt2, U2, V2, phi2,
        ref2left, dt2left, dt2right, phi2left, phi2right)

    return(super().conditional(y, U2, V2, phi2, ref2left, phi2left, phi2right,
      calc_cov, self._kernel_index(kernel)))

  def self_conditional_derivative(self, y, calc_cov=False, kernel=None):
    r"""
    Conditional mean and covariance
    of the derivative of the kernel part, or a subset of kernel terms,
    knowning the observed values :math:`y`.

    Parameters
    ----------
    y : (n,) ndarray
      The vector of observed values :math:`y`.
    calc_cov : False (default), True, or 'diag'
      Whether to output only the derivative conditional mean (False),
      the mean and full covariance matrix (True),
      or the mean and main diagonal of the covariance matrix ('diag').
    kernel : list or None
      List of kernel identifiers
      that should be considered for the Gaussian process.
      Other terms (kernel or noise) are considered as noise.
      If kernel is None, all kernel terms are considered for the
      Gaussian process.

    Returns
    -------
    mu : (n,) ndarray
      The vector of derivative conditional mean values.
    cov : (n, n) ndarray
      Full covariance matrix (if calc_cov is True).
    var : (n,) ndarray
      Main diagonal of the covariance matrix (if calc_cov is 'diag').

    Warnings
    --------
    This method should only be used with differentiable kernels.
    The ExponentialKernel and QuasiperiodicKernel are
    in the general case non-differentiable.
    All other kernels are differentiable.

    While the computational cost of the derivative conditional mean scales as
    :math:`\mathcal{O}(n)`,
    the computational cost of the variance scales as
    :math:`\mathcal{O}(n^2)`,
    and the computational cost of the full covariance scales as
    :math:`\mathcal{O}(n^3)`.
    """

    dU = np.empty((self.n,self.r))
    if calc_cov:
      d2U = np.empty((self.n,self.r))
    else:
      d2U = None

    kernel_list = self.kernel if kernel is None else kernel
    for key in kernel_list:
      self.kernel[key]._deriv(dU, d2U)

    return(super().self_conditional_derivative(y, dU, d2U,
      calc_cov, self._kernel_index(kernel)))

  def conditional_derivative(self, y, t2, calc_cov=False, kernel=None):
    r"""
    Conditional mean and covariance
    of the derivative of the kernel part, or a subset of kernel terms,
    at new times :math:`t_2`,
    knowning the observed values :math:`y`.

    Parameters
    ----------
    y : (n,) ndarray
      The vector of observed values :math:`y`.
    t2 : (n2,) ndarrays
      The vector of new times.
    calc_cov : False (default), True, or 'diag'
      Whether to output only the derivative conditional mean (False),
      the mean and full covariance matrix (True),
      or the mean and main diagonal of the covariance matrix ('diag').

    Returns
    -------
    mu : (n2,) ndarray
      The vector of derivative conditional mean values.
    cov : (n2, n2) ndarray
      Full covariance matrix (if calc_cov is True).
    var : (n2,) ndarray
      Main diagonal of the covariance matrix (if calc_cov is 'diag').
    kernel : list or None
      List of kernel identifiers
      that should be considered for the Gaussian process.
      Other terms (kernel or noise) are considered as noise.
      If kernel is None, all kernel terms are considered for the
      Gaussian process.

    Warnings
    --------
    This method should only be used with differentiable kernels.
    The ExponentialKernel and QuasiperiodicKernel are
    in the general case non-differentiable.
    All other kernels are differentiable.

    While the computational cost of the derivative conditional mean scales as
    :math:`\mathcal{O}(n+n_2)`,
    the computational cost of the variance scales as
    :math:`\mathcal{O}(n n_2)`,
    and the computational cost of the full covariance scales as
    :math:`\mathcal{O}(n n_2^2)`.
    """

    n2 = t2.size
    dt2 = t2[1:] - t2[:-1]
    dU = np.empty((self.n,self.r))
    dU2 = np.empty((n2,self.r))
    V2 = np.empty((n2, self.r))
    phi2 = np.empty((n2-1, self.r))
    phi2left = np.empty((n2, self.r))
    phi2right = np.empty((n2, self.r))

    ref2left = np.searchsorted(self.t, t2) - 1
    dt2left = t2 - self.t[ref2left]
    dt2right = self.t[np.minimum(ref2left+1,self.n-1)] - t2
    dt2left[ref2left==-1] = 0 # useless but avoid overflow warning
    dt2right[ref2left==self.n-1] = 0 # useless but avoid overflow warning

    if calc_cov:
      d2U2 = np.empty((n2,self.r))
    else:
      d2U2 = None

    kernel_list = self.kernel if kernel is None else kernel
    for key in kernel_list:
      self.kernel[key]._deriv(dU)
      self.kernel[key]._deriv_t2(t2, dt2, dU2, V2, phi2,
        ref2left, dt2left, dt2right, phi2left, phi2right,
        d2U2)

    return(super().conditional_derivative(
      y, dU, dU2, d2U2, V2, phi2, ref2left, phi2left, phi2right,
      calc_cov, self._kernel_index(kernel)))

  def eval(self, dt, kernel=None):
    r"""
    Direct evaluation of the kernel part at lag :math:`\delta t`.

    Parameters
    ----------
    dt : ndarray or float
      lag.
    kernel : list or None
      List of kernel identifiers
      that should be considered for the evaluation.
      If kernel is None, all kernel terms are taken into account.

    Returns
    -------
    K : ndarray or float
      Kernel part evaluated at lag dt.

    Warnings
    --------
    The cost scales as the size of the lag ndarray.
    """

    if kernel is None:
      kernel = self.kernel
    return(sum(self.kernel[key].eval(dt) for key in kernel))
