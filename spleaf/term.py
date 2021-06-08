# -*- coding: utf-8 -*-

# Copyright 2020-2021 Jean-Baptiste Delisle
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

__all__ = [
  'Error', 'Jitter', 'InstrumentJitter', 'CalibrationError',
  'CalibrationJitter', 'ExponentialKernel', 'QuasiperiodicKernel',
  'Matern32Kernel', 'Matern52Kernel', 'USHOKernel', 'OSHOKernel', 'SHOKernel'
]

import numpy as np


class Term:
  r"""
  Generic class for covariance terms.
  """

  def __init__(self):
    self._linked = False
    self._param = []

  def _link(self, cov):
    r"""
    Link the term to a covariance matrix.
    """
    if self._linked:
      raise Exception(
        'This term has already been linked to a covariance matrix.')
    self._cov = cov
    self._linked = True

  def _compute(self):
    r"""
    Compute the S+LEAF representation of the term.
    """
    pass

  def _recompute(self):
    r"""
    Recompute the S+LEAF representation of the term.
    """
    self._compute()

  def _set_param(self):
    r"""
    Update the term parameters.
    """
    pass

  def _get_param(self, par):
    r"""
    Get the term parameters.
    """
    return (self.__dict__[f'_{par}'])

  def _grad_param(self):
    r"""
    Gradient of a function with respect to the term parameters
    (listed in self._param).
    """
    return ({})


class Noise(Term):
  r"""
  Generic class for covariance noise terms.
  """

  def __init__(self):
    super().__init__()
    self._b = 0


class Kernel(Term):
  r"""
  Generic class for covariance kernel (Gaussian process) terms.
  """

  def __init__(self):
    super().__init__()
    self._r = 0

  def _link(self, cov, offset):
    super()._link(cov)
    self._offset = offset

  def _compute_t2(self, t2, dt2, U2, V2, phi2, ref2left, dt2left, dt2right,
    phi2left, phi2right):
    r"""
    Compute the S+LEAF representation of the covariance for a new calendar t2.
    """

    pass

  def _deriv(self, calc_d2=False):
    r"""
    Compute the S+LEAF representation of the derivative of the GP.
    """

    pass

  def _deriv_t2(self,
    t2,
    dt2,
    dU2,
    V2,
    phi2,
    ref2left,
    dt2left,
    dt2right,
    phi2left,
    phi2right,
    d2U2=None):
    r"""
    Compute the S+LEAF representation of the derivative of the GP
    for a new calendar t2.
    """

    pass

  def eval(self, dt):
    r"""
    Evaluate the kernel at lag dt.
    """

    raise NotImplementedError(
      'The eval method should be implemented in Kernel childs.')


class Error(Noise):
  r"""
  Uncorrelated measurement errors.

  Parameters
  ----------
  sig : (n,) ndarray
    Vector of measurements errobars (std).
  """

  def __init__(self, sig):
    super().__init__()
    self._sig = sig

  def _compute(self):
    self._cov.A += self._sig**2


class Jitter(Noise):
  r"""
  Uncorrelated global jitter.

  Parameters
  ----------
  sig : float
    Jitter (std).
  """

  def __init__(self, sig):
    super().__init__()
    self._sig = sig
    self._param = ['sig']

  def _compute(self):
    self._cov.A += self._sig**2

  def _set_param(self, sig=None):
    if sig is not None:
      self._sig = sig

  def _grad_param(self):
    grad = {}
    grad['sig'] = 2 * self._sig * self._cov._sum_grad_A
    return (grad)


class InstrumentJitter(Noise):
  r"""
  Uncorrelated instrument jitter.

  Parameters
  ----------
  indices : (n,) ndarray or list
    Mask or list of indices affected by this jitter.
  sig : float
    Jitter (std).
  """

  def __init__(self, indices, sig):
    super().__init__()
    self._indices = indices
    self._sig = sig
    self._param = ['sig']

  def _compute(self):
    self._cov.A[self._indices] += self._sig**2

  def _set_param(self, sig=None):
    if sig is not None:
      self._sig = sig

  def _grad_param(self):
    grad = {}
    grad['sig'] = 2 * self._sig * np.sum(self._cov._grad_A[self._indices])
    return (grad)


class CalibrationError(Noise):
  r"""
  Correlated calibration error.

  The calibration error is shared by blocks of measurements
  using the same calibration.

  Parameters
  ----------
  calib_id : (n,) ndarray
    Identifier of the calibration used for each measurement.
  sig : (n,) ndarray
    Calibration error for each measurement (std).
    Measurements having the same calib_id should have the same sig.
  """

  def __init__(self, calib_id, sig):
    super().__init__()
    self._calib_id = calib_id
    self._sig = sig
    n = calib_id.size
    self._b = np.empty(n, dtype=int)
    # Find groups of points using same calibration
    self._groups = {}
    for k in range(n):
      if calib_id[k] not in self._groups:
        self._groups[calib_id[k]] = [k]
      else:
        self._groups[calib_id[k]].append(k)
      self._b[k] = k - self._groups[calib_id[k]][0]

  def _compute(self):
    var = self._sig**2
    self._cov.A += var
    for group in self._groups.values():
      for i in range(1, len(group)):
        for j in range(i):
          self._cov.F[self._cov.offsetrow[group[i]] +
            group[j]] += var[group[0]]


class CalibrationJitter(Noise):
  r"""
  Correlated calibration jitter.

  The calibration jitter is shared by blocks of measurements
  using the same calibration.

  Parameters
  ----------
  indices : (n,) ndarray or list
    Mask or list of indices affected by this jitter.
  calib_id : (n,) ndarray
    Identifier of the calibration used for each measurement.
  sig : float
    Calibration jitter (std).
  """

  def __init__(self, indices, calib_id, sig):
    super().__init__()
    self._calib_id = calib_id
    self._indices = indices
    self._sig = sig
    n = calib_id.size
    self._b = np.zeros(n, dtype=int)
    # Find groups of points using same calibration
    self._groups = {}
    for k in np.arange(n)[indices]:
      if calib_id[k] not in self._groups:
        self._groups[calib_id[k]] = [k]
      else:
        self._groups[calib_id[k]].append(k)
      self._b[k] = k - self._groups[calib_id[k]][0]
    self._param = ['sig']

  def _compute(self):
    var = self._sig**2
    self._cov.A[self._indices] += var
    self._Fmask = []
    for group in self._groups.values():
      for i in range(1, len(group)):
        for j in range(i):
          self._Fmask.append(self._cov.offsetrow[group[i]] + group[j])
    self._cov.F[self._Fmask] += var

  def _recompute(self):
    var = self._sig**2
    self._cov.A[self._indices] += var
    self._cov.F[self._Fmask] += var

  def _set_param(self, sig=None):
    if sig is not None:
      self._sig = sig

  def _grad_param(self):
    grad = {}
    grad['sig'] = 2 * self._sig * (np.sum(self._cov._grad_A[self._indices]) +
      np.sum(self._cov._grad_F[self._Fmask]))
    return (grad)


class ExponentialKernel(Kernel):
  r"""
  Exponential decay kernel.

  This kernel follows:

  .. math:: K(\delta t) = a \mathrm{e}^{-\lambda \delta t}

  Parameters
  ----------
  a : float
    Amplitude (variance).
  la : float
    Decay rate.
  """

  def __init__(self, a, la):
    super().__init__()
    self._a = a
    self._la = la
    self._r = 1
    self._param = ['a', 'la']

  def _compute(self):
    self._cov.A += self._a
    self._cov.U[:, self._offset] = self._a
    self._cov.V[:, self._offset] = 1.0
    self._cov.phi[:, self._offset] = np.exp(-self._la * self._cov.dt)

  def _set_param(self, a=None, la=None):
    if a is not None:
      self._a = a
    if la is not None:
      self._la = la

  def _grad_param(self, grad_dU=None, grad_d2U=None):
    grad = {}
    grad['a'] = self._cov._sum_grad_A + np.sum(self._cov._grad_U[:,
      self._offset])
    grad['la'] = -np.sum(self._cov.dt * self._cov.phi[:, self._offset] *
      self._cov._grad_phi[:, self._offset])

    if grad_dU is not None:
      # self._cov._dU[:, self._offset] = -self._la * self._a
      sum_grad_dU = np.sum(grad_dU[:, self._offset])
      grad['a'] -= self._la * sum_grad_dU
      grad['la'] -= self._a * sum_grad_dU

    if grad_d2U is not None:
      # self._cov._d2U[:, self._offset] = -self._la**2 * self._a
      sum_grad_d2U = np.sum(grad_d2U[:, self._offset])
      grad['a'] -= self._la**2 * sum_grad_d2U
      grad['la'] -= 2 * self._la * self._a * sum_grad_d2U

    return (grad)

  def _compute_t2(self, t2, dt2, U2, V2, phi2, ref2left, dt2left, dt2right,
    phi2left, phi2right):
    U2[:, self._offset] = self._a
    V2[:, self._offset] = 1.0
    phi2[:, self._offset] = np.exp(-self._la * dt2)
    phi2left[:, self._offset] = np.exp(-self._la * dt2left)
    phi2right[:, self._offset] = np.exp(-self._la * dt2right)

  def _deriv(self, calc_d2=False):
    self._cov._dU[:, self._offset] = -self._la * self._a
    if calc_d2:
      self._cov._d2U[:, self._offset] = -self._la**2 * self._a

  def _deriv_t2(self,
    t2,
    dt2,
    dU2,
    V2,
    phi2,
    ref2left,
    dt2left,
    dt2right,
    phi2left,
    phi2right,
    d2U2=None):
    dU2[:, self._offset] = -self._la * self._a
    V2[:, self._offset] = 1.0
    phi2[:, self._offset] = np.exp(-self._la * dt2)
    phi2left[:, self._offset] = np.exp(-self._la * dt2left)
    phi2right[:, self._offset] = np.exp(-self._la * dt2right)
    if d2U2 is not None:
      d2U2[:, self._offset] = -self._la**2 * self._a

  def eval(self, dt):
    return (self._a * np.exp(-self._la * np.abs(dt)))


class QuasiperiodicKernel(Kernel):
  r"""
  Quasiperiodic kernel.

  This kernel follows:

  .. math:: K(\delta t) = \mathrm{e}^{-\lambda \delta t}
    \left(a \cos(\nu \delta t) + b \sin(\nu \delta t)\right)

  Parameters
  ----------
  a, b : float
    Amplitudes (variance) of the cos/sin terms.
  la : float
    Decay rate.
  nu : float
    Angular frequency.
  """

  def __init__(self, a, b, la, nu):
    super().__init__()
    self._a = a
    self._b = b
    self._la = la
    self._nu = nu
    self._r = 2
    self._param = ['a', 'b', 'la', 'nu']

  def _compute(self):
    self._cov.A += self._a
    self._cnut = np.cos(self._nu * self._cov.t)
    self._snut = np.sin(self._nu * self._cov.t)
    self._cov.U[:, self._offset] = self._a * self._cnut + self._b * self._snut
    self._cov.V[:, self._offset] = self._cnut
    self._cov.U[:,
      self._offset + 1] = self._a * self._snut - self._b * self._cnut
    self._cov.V[:, self._offset + 1] = self._snut
    self._cov.phi[:,
      self._offset:self._offset + 2] = np.exp(-self._la * self._cov.dt)[:,
      None]

  def _set_param(self, a=None, b=None, la=None, nu=None):
    if a is not None:
      self._a = a
    if b is not None:
      self._b = b
    if la is not None:
      self._la = la
    if nu is not None:
      self._nu = nu

  def _grad_param(self, grad_dU=None, grad_d2U=None):
    grad = {}
    grad['a'] = self._cov._sum_grad_A + np.sum(self._cov.V[:, self._offset] *
      self._cov._grad_U[:, self._offset] + self._cov.V[:, self._offset + 1] *
      self._cov._grad_U[:, self._offset + 1])
    grad['b'] = np.sum(self._cov.V[:, self._offset + 1] *
      self._cov._grad_U[:, self._offset] -
      self._cov.V[:, self._offset] * self._cov._grad_U[:, self._offset + 1])
    grad['la'] = -np.sum(self._cov.dt * self._cov.phi[:, self._offset] *
      (self._cov._grad_phi[:, self._offset] +
      self._cov._grad_phi[:, self._offset + 1]))
    grad['nu'] = np.sum(self._cov.t *
      (self._cov.U[:, self._offset] * self._cov._grad_U[:, self._offset + 1] -
      self._cov.U[:, self._offset + 1] * self._cov._grad_U[:, self._offset] +
      self._cov.V[:, self._offset] * self._cov._grad_V[:, self._offset + 1] -
      self._cov.V[:, self._offset + 1] * self._cov._grad_V[:, self._offset]))

    if grad_dU is not None:
      da = -self._la * self._a + self._nu * self._b
      db = -self._la * self._b - self._nu * self._a
      # self._cov._dU[:, self._offset] = da * self._cnut + db * self._snut
      # self._cov._dU[:, self._offset + 1] = da * self._snut - db * self._cnut
      grad_da = np.sum(self._cnut * grad_dU[:, self._offset] +
        self._snut * grad_dU[:, self._offset + 1])
      grad_db = np.sum(self._snut * grad_dU[:, self._offset] -
        self._cnut * grad_dU[:, self._offset + 1])
      grad['nu'] += np.sum(self._cov.t *
        (self._cov._dU[:, self._offset] * grad_dU[:, self._offset + 1] -
        self._cov._dU[:, self._offset + 1] * grad_dU[:, self._offset]))
      # da = -self._la * self._a + self._nu * self._b
      # db = -self._la * self._b - self._nu * self._a
      grad['a'] -= self._la * grad_da + self._nu * grad_db
      grad['b'] += self._nu * grad_da - self._la * grad_db
      grad['la'] -= self._a * grad_da + self._b * grad_db
      grad['nu'] += self._b * grad_da - self._a * grad_db

    if grad_d2U is not None:
      d2a = (self._nu**2 -
        self._la**2) * self._a + 2 * self._la * self._nu * self._b
      d2b = (self._nu**2 -
        self._la**2) * self._b - 2 * self._la * self._nu * self._a
      # self._cov._d2U[:, self._offset] = d2a * self._cnut + d2b * self._snut
      # self._cov._d2U[:, self._offset + 1] = d2a * self._snut - d2b * self._cnut
      grad_d2a = np.sum(self._cnut * grad_d2U[:, self._offset] +
        self._snut * grad_d2U[:, self._offset + 1])
      grad_d2b = np.sum(self._snut * grad_d2U[:, self._offset] -
        self._cnut * grad_d2U[:, self._offset + 1])
      grad['nu'] += np.sum(self._cov.t *
        (self._cov._d2U[:, self._offset] * grad_d2U[:, self._offset + 1] -
        self._cov._d2U[:, self._offset + 1] * grad_d2U[:, self._offset]))
      # d2a = (self._nu**2 -
      #   self._la**2) * self._a + 2 * self._la * self._nu * self._b
      # d2b = (self._nu**2 -
      #   self._la**2) * self._b - 2 * self._la * self._nu * self._a
      grad['a'] += (self._nu**2 -
        self._la**2) * grad_d2a - 2 * self._la * self._nu * grad_d2b
      grad['b'] += 2 * self._la * self._nu * grad_d2a + (self._nu**2 -
        self._la**2) * grad_d2b
      grad['la'] += 2 * ((self._nu * self._b - self._la * self._a) * grad_d2a -
        (self._nu * self._a + self._la * self._b) * grad_d2b)
      grad['nu'] += 2 * ((self._nu * self._a + self._la * self._b) * grad_d2a +
        (self._nu * self._b - self._la * self._a) * grad_d2b)

    return (grad)

  def _compute_t2(self, t2, dt2, U2, V2, phi2, ref2left, dt2left, dt2right,
    phi2left, phi2right):
    cnut2 = np.cos(self._nu * t2)
    snut2 = np.sin(self._nu * t2)
    U2[:, self._offset] = self._a * cnut2 + self._b * snut2
    V2[:, self._offset] = cnut2
    U2[:, self._offset + 1] = self._a * snut2 - self._b * cnut2
    V2[:, self._offset + 1] = snut2
    phi2[:, self._offset:self._offset + 2] = np.exp(-self._la * dt2)[:, None]
    phi2left[:, self._offset:self._offset + 2] = np.exp(-self._la * dt2left)[:,
      None]
    phi2right[:,
      self._offset:self._offset + 2] = np.exp(-self._la * dt2right)[:, None]

  def _deriv(self, calc_d2=False):
    da = -self._la * self._a + self._nu * self._b
    db = -self._la * self._b - self._nu * self._a
    self._cov._dU[:, self._offset] = da * self._cnut + db * self._snut
    self._cov._dU[:, self._offset + 1] = da * self._snut - db * self._cnut
    if calc_d2:
      d2a = (self._nu**2 -
        self._la**2) * self._a + 2 * self._la * self._nu * self._b
      d2b = (self._nu**2 -
        self._la**2) * self._b - 2 * self._la * self._nu * self._a
      self._cov._d2U[:, self._offset] = d2a * self._cnut + d2b * self._snut
      self._cov._d2U[:, self._offset + 1] = d2a * self._snut - d2b * self._cnut

  def _deriv_t2(self,
    t2,
    dt2,
    dU2,
    V2,
    phi2,
    ref2left,
    dt2left,
    dt2right,
    phi2left,
    phi2right,
    d2U2=None):
    da = -self._la * self._a + self._nu * self._b
    db = -self._la * self._b - self._nu * self._a
    cnut2 = np.cos(self._nu * t2)
    snut2 = np.sin(self._nu * t2)
    dU2[:, self._offset] = da * cnut2 + db * snut2
    V2[:, self._offset] = cnut2
    dU2[:, self._offset + 1] = da * snut2 - db * cnut2
    V2[:, self._offset + 1] = snut2
    phi2[:, self._offset:self._offset + 2] = np.exp(-self._la * dt2)[:, None]
    phi2left[:, self._offset:self._offset + 2] = np.exp(-self._la * dt2left)[:,
      None]
    phi2right[:,
      self._offset:self._offset + 2] = np.exp(-self._la * dt2right)[:, None]
    if d2U2 is not None:
      d2a = (self._nu**2 -
        self._la**2) * self._a + 2 * self._la * self._nu * self._b
      d2b = (self._nu**2 -
        self._la**2) * self._b - 2 * self._la * self._nu * self._a
      d2U2[:, self._offset] = d2a * cnut2 + d2b * snut2
      d2U2[:, self._offset + 1] = d2a * snut2 - d2b * cnut2

  def eval(self, dt):
    adt = np.abs(dt)
    return (np.exp(-self._la * adt) *
      (self._a * np.cos(self._nu * adt) + self._b * np.sin(self._nu * adt)))


class Matern32Kernel(QuasiperiodicKernel):
  r"""
  Approximate Matérn 3/2 kernel.

  This kernel approximates the Matérn 3/2 kernel:

  .. math:: K(\delta t) = \sigma^2 \mathrm{e}^{-\sqrt{3}\frac{\delta t}{\rho}}
    \left(1 + \sqrt{3}\frac{\delta t}{\rho}\right)

  See `Foreman-Mackey et al. 2017 <http://adsabs.harvard.edu/abs/2017AJ....154..220F>`_
  for more details.

  Parameters
  ----------
  sig : float
    Amplitude (std).
  rho : float
    Scale.
  eps : float
    Precision of the approximation (0.01 by default).
  """

  def __init__(self, sig, rho, eps=0.01):
    self._sig = sig
    self._rho = rho
    self._eps = eps
    super().__init__(*self._getcoefs())
    self._param = ['sig', 'rho']

  def _getcoefs(self):
    a = self._sig**2
    nu = self._eps
    la = np.sqrt(3) / self._rho
    b = a * la / self._eps
    return (a, b, la, nu)

  def _set_param(self, sig=None, rho=None):
    if sig is not None:
      self._sig = sig
    if rho is not None:
      self._rho = rho
    a, b, la, nu = self._getcoefs()
    super()._set_param(a, b, la, nu)

  def _grad_param(self, grad_dU=None, grad_d2U=None):
    gradQP = super()._grad_param(grad_dU, grad_d2U)
    grad = {}
    grad['sig'] = 2 * self._sig * (gradQP['a'] +
      gradQP['b'] * self._la / self._eps)
    grad['rho'] = -np.sqrt(3) / self._rho**2 * (gradQP['la'] +
      gradQP['b'] * self._a / self._eps)
    return (grad)


class Matern52Kernel(Kernel):
  r"""
  Approximate Matérn 5/2 kernel.

  This kernel approximates the Matérn 5/2 kernel:

  .. math:: K(\delta t) = \sigma^2 \mathrm{e}^{-\sqrt{5}\frac{\delta t}{\rho}}
    \left(1 + \sqrt{5}\frac{\delta t}{\rho} + \frac{5}{3}\left(\frac{\delta t}{\rho}\right)^2\right)

  Parameters
  ----------
  sig : float
    Amplitude (std).
  rho : float
    Scale.
  eps : float
    Precision of the approximation (0.01 by default).
  """

  def __init__(self, sig, rho, eps=0.01):
    super().__init__()
    self._sig = sig
    self._rho = rho
    self._eps = eps
    aexp, a, b, la, nu = self._getcoefs()
    self._exp = ExponentialKernel(aexp, la)
    self._qp = QuasiperiodicKernel(a, b, la, nu)
    self._r = 3
    self._param = ['sig', 'rho']

  def _getcoefs(self):
    var = self._sig**2
    la = np.sqrt(5) / self._rho
    nu = self._eps
    alpha = la / self._eps
    b = var * alpha
    a = -2 / 3 * alpha * b
    aexp = var - a
    return (aexp, a, b, la, nu)

  def _link(self, cov, offset):
    super()._link(cov, offset)
    self._exp._link(cov, offset)
    self._qp._link(cov, offset + 1)

  def _compute(self):
    self._exp._compute()
    self._qp._compute()

  def _set_param(self, sig=None, rho=None):
    if sig is not None:
      self._sig = sig
    if rho is not None:
      self._rho = rho
    aexp, a, b, la, nu = self._getcoefs()
    self._exp._set_param(aexp, la)
    self._qp._set_param(a, b, la, nu)

  def _grad_param(self, grad_dU=None, grad_d2U=None):
    gradExp = self._exp._grad_param(grad_dU, grad_d2U)
    gradQP = self._qp._grad_param(grad_dU, grad_d2U)
    grad = {}
    grad['sig'] = 2 / self._sig * (gradExp['a'] * self._exp._a +
      gradQP['a'] * self._qp._a + gradQP['b'] * self._qp._b)
    grad['rho'] = -1 / self._rho * (
      (gradExp['la'] + gradQP['la']) * self._exp._la + 2 * gradExp['a'] *
      (self._exp._a - self._sig**2) + 2 * gradQP['a'] * self._qp._a +
      gradQP['b'] * self._qp._b)
    return (grad)

  def _compute_t2(self, t2, dt2, U2, V2, phi2, ref2left, dt2left, dt2right,
    phi2left, phi2right):
    self._exp._compute_t2(t2, dt2, U2, V2, phi2, ref2left, dt2left, dt2right,
      phi2left, phi2right)
    self._qp._compute_t2(t2, dt2, U2, V2, phi2, ref2left, dt2left, dt2right,
      phi2left, phi2right)

  def _deriv(self, calc_d2=False):
    self._exp._deriv(calc_d2)
    self._qp._deriv(calc_d2)

  def _deriv_t2(self,
    t2,
    dt2,
    dU2,
    V2,
    phi2,
    ref2left,
    dt2left,
    dt2right,
    phi2left,
    phi2right,
    d2U2=None):
    self._exp._deriv_t2(t2, dt2, dU2, V2, phi2, ref2left, dt2left, dt2right,
      phi2left, phi2right, d2U2)
    self._qp._deriv_t2(t2, dt2, dU2, V2, phi2, ref2left, dt2left, dt2right,
      phi2left, phi2right, d2U2)

  def eval(self, dt):
    return (self._exp.eval(dt) + self._qp.eval(dt))


class USHOKernel(QuasiperiodicKernel):
  r"""
  Under-damped SHO Kernel.

  This kernel follows the differential equation of
  a stochastically-driven harmonic oscillator (SHO)
  in the under-damped case (:math:`Q>0.5`).
  See `Foreman-Mackey et al. 2017 <http://adsabs.harvard.edu/abs/2017AJ....154..220F>`_)
  for more details.

  Parameters
  ----------
  sig : float
    Amplitude (std).
  P0 : float
    Undamped period.
  Q : float
    Quality factor.
  eps : float
    Regularization parameter (1e-5 by default).
  """

  def __init__(self, sig, P0, Q, eps=1e-5):
    self._sig = sig
    self._P0 = P0
    self._Q = Q
    self._eps = eps
    super().__init__(*self._getcoefs())
    self._param = ['sig', 'P0', 'Q']

  def _getcoefs(self):
    self._sqQ = np.sqrt(max(4 * self._Q**2 - 1, self._eps))
    a = self._sig**2
    la = np.pi / (self._P0 * self._Q)
    b = a / self._sqQ
    nu = la * self._sqQ
    return (a, b, la, nu)

  def _set_param(self, sig=None, P0=None, Q=None):
    if sig is not None:
      self._sig = sig
    if P0 is not None:
      self._P0 = P0
    if Q is not None:
      self._Q = Q
    a, b, la, nu = self._getcoefs()
    super()._set_param(a, b, la, nu)

  def _grad_param(self, grad_dU=None, grad_d2U=None):
    gradQP = super()._grad_param(grad_dU, grad_d2U)
    grad = {}
    grad['sig'] = 2 * self._sig * (gradQP['a'] + gradQP['b'] / self._sqQ)
    grad['P0'] = -np.pi / (self._P0**2 * self._Q) * (gradQP['la'] +
      gradQP['nu'] * self._sqQ)
    grad['Q'] = -np.pi / (self._P0 * self._Q**2) * (gradQP['la'] +
      gradQP['nu'] * self._sqQ)
    if 4 * self._Q**2 - 1 > self._eps:
      grad['Q'] += 4 * self._Q / self._sqQ * (gradQP['nu'] * self._la -
        gradQP['b'] * self._a / self._sqQ**2)
    return (grad)


class OSHOKernel(Kernel):
  r"""
  Over-damped SHO Kernel.

  This kernel follows the differential equation of
  a stochastically-driven harmonic oscillator (SHO)
  in the over-damped case (:math:`Q<0.5`).
  See `Foreman-Mackey et al. 2017 <http://adsabs.harvard.edu/abs/2017AJ....154..220F>`_)
  for more details.

  Parameters
  ----------
  sig : float
    Amplitude (std).
  P0 : float
    Undamped period.
  Q : float
    Quality factor.
  eps : float
    Regularization parameter (1e-5 by default).
  """

  def __init__(self, sig, P0, Q, eps=1e-5):
    super().__init__()
    self._sig = sig
    self._P0 = P0
    self._Q = Q
    self._eps = eps
    a1, la1, a2, la2 = self._getcoefs()
    self._exp1 = ExponentialKernel(a1, la1)
    self._exp2 = ExponentialKernel(a2, la2)
    self._r = 2
    self._param = ['sig', 'P0', 'Q']

  def _getcoefs(self):
    self._sqQ = np.sqrt(max(1 - 4 * self._Q**2, self._eps))
    self._a = self._sig**2
    self._la = np.pi / (self._P0 * self._Q)
    return (self._a * (1 + 1 / self._sqQ) / 2, self._la * (1 - self._sqQ),
      self._a * (1 - 1 / self._sqQ) / 2, self._la * (1 + self._sqQ))

  def _link(self, cov, offset):
    super()._link(cov, offset)
    self._exp1._link(cov, offset)
    self._exp2._link(cov, offset + 1)

  def _compute(self):
    self._exp1._compute()
    self._exp2._compute()

  def _set_param(self, sig=None, P0=None, Q=None):
    if sig is not None:
      self._sig = sig
    if P0 is not None:
      self._P0 = P0
    if Q is not None:
      self._Q = Q
    a1, la1, a2, la2 = self._getcoefs()
    self._exp1._set_param(a1, la1)
    self._exp2._set_param(a2, la2)

  def _grad_param(self, grad_dU=None, grad_d2U=None):
    gradExp1 = self._exp1._grad_param(grad_dU, grad_d2U)
    gradExp2 = self._exp2._grad_param(grad_dU, grad_d2U)
    grad = {}
    grad['sig'] = 2 * self._sig * (gradExp1['a'] *
      (1 + 1 / self._sqQ) / 2 + gradExp2['a'] * (1 - 1 / self._sqQ) / 2)
    grad['P0'] = -np.pi / (self._P0**2 * self._Q) * (gradExp1['la'] *
      (1 - self._sqQ) + gradExp2['la'] * (1 + self._sqQ))
    grad['Q'] = -np.pi / (self._P0 * self._Q**2) * (gradExp1['la'] *
      (1 - self._sqQ) + gradExp2['la'] * (1 + self._sqQ))
    if 1 - 4 * self._Q**2 > self._eps:
      grad['Q'] -= 4 * self._Q / self._sqQ * (
        (gradExp2['a'] - gradExp1['a']) * self._a / (2 * self._sqQ**2) +
        (gradExp2['la'] - gradExp1['la']) * self._la)
    return (grad)

  def _compute_t2(self, t2, dt2, U2, V2, phi2, ref2left, dt2left, dt2right,
    phi2left, phi2right):
    self._exp1._compute_t2(t2, dt2, U2, V2, phi2, ref2left, dt2left, dt2right,
      phi2left, phi2right)
    self._exp2._compute_t2(t2, dt2, U2, V2, phi2, ref2left, dt2left, dt2right,
      phi2left, phi2right)

  def _deriv(self, calc_d2=False):
    self._exp1._deriv(calc_d2)
    self._exp2._deriv(calc_d2)

  def _deriv_t2(self,
    t2,
    dt2,
    dU2,
    V2,
    phi2,
    ref2left,
    dt2left,
    dt2right,
    phi2left,
    phi2right,
    d2U2=None):
    self._exp1._deriv_t2(t2, dt2, dU2, V2, phi2, ref2left, dt2left, dt2right,
      phi2left, phi2right, d2U2)
    self._exp2._deriv_t2(t2, dt2, dU2, V2, phi2, ref2left, dt2left, dt2right,
      phi2left, phi2right, d2U2)

  def eval(self, dt):
    return (self._exp1.eval(dt) + self._exp2.eval(dt))


class SHOKernel(Kernel):
  r"""
  SHO Kernel.

  This kernel follows the differential equation of
  a stochastically-driven harmonic oscillator (SHO).
  It merges the under-damped (:math:`Q>0.5`) USHOKernel
  and the over-damped (:math:`Q<0.5`) OSHOKernel.
  See `Foreman-Mackey et al. 2017 <http://adsabs.harvard.edu/abs/2017AJ....154..220F>`_)
  for more details.

  Parameters
  ----------
  sig : float
    Amplitude (std).
  P0 : float
    Undamped period.
  Q : float
    Quality factor.
  eps : float
    Regularization parameter (1e-5 by default).
  """

  def __init__(self, sig, P0, Q, eps=1e-5):
    super().__init__()
    self._sig = sig
    self._P0 = P0
    self._Q = Q
    self._eps = eps
    self._usho = USHOKernel(sig, P0, Q, eps)
    self._osho = OSHOKernel(sig, P0, Q, eps)
    self._r = 2
    self._param = ['sig', 'P0', 'Q']

  def _link(self, cov, offset):
    super()._link(cov, offset)
    self._usho._link(cov, offset)
    self._osho._link(cov, offset)

  def _compute(self):
    if self._Q > 0.5:
      self._usho._compute()
    else:
      self._osho._compute()

  def _set_param(self, sig=None, P0=None, Q=None):
    if sig is not None:
      self._sig = sig
    if P0 is not None:
      self._P0 = P0
    if Q is not None:
      self._Q = Q
    if self._Q > 0.5:
      self._usho._set_param(self._sig, self._P0, self._Q)
    else:
      self._osho._set_param(self._sig, self._P0, self._Q)

  def _grad_param(self, grad_dU=None, grad_d2U=None):
    if self._Q > 0.5:
      return (self._usho._grad_param(grad_dU, grad_d2U))
    else:
      return (self._osho._grad_param(grad_dU, grad_d2U))

  def _compute_t2(self, t2, dt2, U2, V2, phi2, ref2left, dt2left, dt2right,
    phi2left, phi2right):
    if self._Q > 0.5:
      self._usho._compute_t2(t2, dt2, U2, V2, phi2, ref2left, dt2left,
        dt2right, phi2left, phi2right)
    else:
      self._osho._compute_t2(t2, dt2, U2, V2, phi2, ref2left, dt2left,
        dt2right, phi2left, phi2right)

  def _deriv(self, calc_d2=False):
    if self._Q > 0.5:
      self._usho._deriv(calc_d2)
    else:
      self._osho._deriv(calc_d2)

  def _deriv_t2(self,
    t2,
    dt2,
    dU2,
    V2,
    phi2,
    ref2left,
    dt2left,
    dt2right,
    phi2left,
    phi2right,
    d2U2=None):
    if self._Q > 0.5:
      self._usho._deriv_t2(t2, dt2, dU2, V2, phi2, ref2left, dt2left, dt2right,
        phi2left, phi2right, d2U2)
    else:
      self._osho._deriv_t2(t2, dt2, dU2, V2, phi2, ref2left, dt2left, dt2right,
        phi2left, phi2right, d2U2)

  def eval(self, dt):
    if self._Q > 0.5:
      return (self._usho.eval(dt))
    else:
      return (self._osho.eval(dt))


class _FakeCov:

  def __init__(self, t, dt, r):
    self.t = t
    self.dt = dt
    self.n = t.size
    self.r = r
    self.A = np.zeros(self.n)
    self.U = np.empty((self.n, r))
    self.V = np.empty((self.n, r))
    self.phi = np.empty((self.n - 1, r))

    self._dU = np.empty((self.n, r))
    self._d2U = np.empty((self.n, r))
    self._d2A = None

    self._grad_A = np.empty(self.n)
    self._grad_U = np.empty((self.n, r))
    self._grad_V = np.empty((self.n, r))
    self._grad_phi = np.empty((self.n - 1, r))
    self._sum_grad_A = None

    self._grad_dU = np.empty((self.n, r))
    self._grad_d2U = np.empty((self.n, r))
    self._grad_d2A = np.empty(self.n)


class MultiSeriesKernel(Kernel):
  r"""
  Linear combination of a Kernel and its derivative
  applied to heterogenous time series.

  This kernel allows to model efficiently
  several (heterogeneous) time series (:math`y_i`)
  which depend on different linear combinations
  of the same GP (:math:`G`)
  and its derivative (:math:`G'`):

  .. math:: y_{i,j} = \alpha_i G(t_{i,j}) + \beta_i G'(t_{i,j}).

  The times of measurements need not be the same for each time series
  (i.e. we may have :math:`t_{i,.} \neq t_{j,.}`).

  This allows to define models similar to
  `Rajpaul et al. (2015) <https://ui.adsabs.harvard.edu/abs/2015MNRAS.452.2269R>`_
  but with fast and scalable algorithms.

  Parameters
  ----------
  kernel : Kernel
    Kernel of the GP (:math:`G`).
  series_index : list of ndarrays
    Indices corresponding to each original time series in the merged time series.
  alpha : list
    Coefficients in front of the GP for each original time series.
  beta : list or None
    Coefficients in front of the GP derivative for each original time series.
    If None, the derivative is ignored.
  """

  def __init__(self, kernel, series_index, alpha, beta=None):
    super().__init__()
    self._kernel = kernel
    self._series_index = series_index
    self._nseries = len(series_index)
    self._param = kernel._param + [f'alpha_{k}' for k in range(self._nseries)]
    self._alpha = alpha
    self._beta = beta
    if beta is None:
      self._with_derivative = False
      self._coef_r = 1
    else:
      self._with_derivative = True
      self._coef_r = 4
      self._param += [f'beta_{k}' for k in range(self._nseries)]
    self._r = self._coef_r * kernel._r

    self._cond_alpha = 1
    self._cond_beta = 0
    self._cond_series_id = None

  def _link(self, cov, offset):
    super()._link(cov, offset)
    self._kernel._link(_FakeCov(cov.t, cov.dt, self._kernel._r), 0)

  def _compute(self):
    self._kernel._cov.A[:] = 0
    self._kernel._compute()
    if self._with_derivative:
      self._kernel._deriv(True)
      self._kernel._cov._d2A = np.sum(self._kernel._cov._d2U *
        self._kernel._cov.V,
        axis=1)
    for k in range(self._nseries):
      ik = self._series_index[k]
      # cov(GP, GP)
      self._cov.A[ik] += self._alpha[k]**2 * self._kernel._cov.A[ik]
      self._cov.U[ik, self._offset:self._offset +
        self._kernel._r] = self._alpha[k] * self._kernel._cov.U[ik]
      self._cov.V[ik, self._offset:self._offset +
        self._kernel._r] = self._alpha[k] * self._kernel._cov.V[ik]
      if self._with_derivative:
        # cov(GP, dGP)
        self._cov.U[ik, self._offset + self._kernel._r:self._offset +
          2 * self._kernel._r] = -self._alpha[k] * self._kernel._cov._dU[ik]
        self._cov.V[ik, self._offset + self._kernel._r:self._offset +
          2 * self._kernel._r] = self._beta[k] * self._kernel._cov.V[ik]
        # cov(dGP, GP)
        self._cov.U[ik, self._offset + 2 * self._kernel._r:self._offset +
          3 * self._kernel._r] = self._beta[k] * self._kernel._cov._dU[ik]
        self._cov.V[ik, self._offset + 2 * self._kernel._r:self._offset +
          3 * self._kernel._r] = self._alpha[k] * self._kernel._cov.V[ik]
        # cov(dGP, dGP)
        self._cov.A[ik] += self._beta[k]**2 * self._kernel._cov._d2A[ik]
        self._cov.U[ik, self._offset + 3 * self._kernel._r:self._offset +
          4 * self._kernel._r] = self._beta[k] * self._kernel._cov._d2U[ik]
        self._cov.V[ik, self._offset + 3 * self._kernel._r:self._offset +
          4 * self._kernel._r] = self._beta[k] * self._kernel._cov.V[ik]
    for k in range(self._coef_r):
      self._cov.phi[:, self._offset + k * self._kernel._r:self._offset +
        (k + 1) * self._kernel._r] = self._kernel._cov.phi

  def _set_param(self, *args, **kwargs):
    for karg, arg in enumerate(args):
      par = self._param[karg]
      if par in kwargs:
        raise Exception(
          f'MultiSeriesKernel._set_param: parameter {par} multiply defined.')
      kwargs[par] = arg
    kernel_kwargs = {}
    for par in kwargs:
      if par.startswith('alpha_'):
        self._alpha[int(par.replace('alpha_', ''))] = kwargs[par]
      elif par.startswith('beta_'):
        self._beta[int(par.replace('beta_', ''))] = kwargs[par]
      else:
        kernel_kwargs[par] = kwargs[par]
    self._kernel._set_param(**kernel_kwargs)

  def _get_param(self, par):
    if par.startswith('alpha_'):
      return (self._alpha[int(par.replace('alpha_', ''))])
    elif par.startswith('beta_'):
      return (self._beta[int(par.replace('beta_', ''))])
    else:
      return (self._kernel._get_param(par))

  def _grad_param(self):
    grad = {}
    for k in range(self._nseries):
      ik = self._series_index[k]
      # cov(GP, GP)
      grad[f'alpha_{k}'] = 2 * self._alpha[k] * np.sum(
        self._cov._grad_A[ik] * self._kernel._cov.A[ik])
      grad[f'alpha_{k}'] += np.sum(
        self._cov._grad_U[ik, self._offset:self._offset + self._kernel._r] *
        self._kernel._cov.U[ik] +
        self._cov._grad_V[ik, self._offset:self._offset + self._kernel._r] *
        self._kernel._cov.V[ik])
      self._kernel._cov._grad_A[ik] = self._alpha[k]**2 * self._cov._grad_A[ik]
      self._kernel._cov._grad_U[ik] = self._alpha[k] * self._cov._grad_U[ik,
        self._offset:self._offset + self._kernel._r]
      self._kernel._cov._grad_V[ik] = self._alpha[k] * self._cov._grad_V[ik,
        self._offset:self._offset + self._kernel._r]
      if self._with_derivative:
        # cov(GP, dGP)
        grad[f'alpha_{k}'] -= np.sum(self._cov._grad_U[ik,
          self._offset + self._kernel._r:self._offset + 2 * self._kernel._r] *
          self._kernel._cov._dU[ik])
        grad[f'beta_{k}'] = np.sum(self._cov._grad_V[ik,
          self._offset + self._kernel._r:self._offset + 2 * self._kernel._r] *
          self._kernel._cov.V[ik])
        self._kernel._cov._grad_dU[ik] = -self._alpha[k] * self._cov._grad_U[
          ik,
          self._offset + self._kernel._r:self._offset + 2 * self._kernel._r]
        self._kernel._cov._grad_V[ik] += self._beta[k] * self._cov._grad_V[ik,
          self._offset + self._kernel._r:self._offset + 2 * self._kernel._r]
        # cov(dGP, GP)
        grad[f'beta_{k}'] += np.sum(self._cov._grad_U[ik, self._offset +
          2 * self._kernel._r:self._offset + 3 * self._kernel._r] *
          self._kernel._cov._dU[ik])
        grad[f'alpha_{k}'] += np.sum(self._cov._grad_V[ik, self._offset +
          2 * self._kernel._r:self._offset + 3 * self._kernel._r] *
          self._kernel._cov.V[ik])
        self._kernel._cov._grad_dU[ik] += self._beta[k] * self._cov._grad_U[ik,
          self._offset + 2 * self._kernel._r:self._offset +
          3 * self._kernel._r]
        self._kernel._cov._grad_V[ik] += self._alpha[k] * self._cov._grad_V[ik,
          self._offset + 2 * self._kernel._r:self._offset +
          3 * self._kernel._r]
        # cov(dGP, dGP)
        grad[f'beta_{k}'] += 2 * self._beta[k] * np.sum(
          self._cov._grad_A[ik] * self._kernel._cov._d2A[ik])
        grad[f'beta_{k}'] += np.sum(self._cov._grad_U[ik, self._offset +
          3 * self._kernel._r:self._offset + 4 * self._kernel._r] *
          self._kernel._cov._d2U[ik] + self._cov._grad_V[ik, self._offset +
          3 * self._kernel._r:self._offset + 4 * self._kernel._r] *
          self._kernel._cov.V[ik])
        self._kernel._cov._grad_d2A[
          ik] = self._beta[k]**2 * self._cov._grad_A[ik]
        self._kernel._cov._grad_d2U[ik] = self._beta[k] * self._cov._grad_U[ik,
          self._offset + 3 * self._kernel._r:self._offset +
          4 * self._kernel._r]
        self._kernel._cov._grad_V[ik] += self._beta[k] * self._cov._grad_V[ik,
          self._offset + 3 * self._kernel._r:self._offset +
          4 * self._kernel._r]

    self._kernel._cov._grad_phi = sum(self._cov._grad_phi[:, self._offset +
      k * self._kernel._r:self._offset + (k + 1) * self._kernel._r]
      for k in range(self._coef_r))
    self._kernel._cov._sum_grad_A = np.sum(self._kernel._cov._grad_A)

    if self._with_derivative:
      self._kernel._cov._grad_d2U += self._kernel._cov.V * self._kernel._cov._grad_d2A[:,
        None]
      self._kernel._cov._grad_V += self._kernel._cov._d2U * self._kernel._cov._grad_d2A[:,
        None]
      grad.update(
        self._kernel._grad_param(self._kernel._cov._grad_dU,
        self._kernel._cov._grad_d2U))
    else:
      grad.update(self._kernel._grad_param())

    return (grad)

  def set_conditionnal_coef(self, alpha=1, beta=0, series_id=None):
    r"""
    Set the coefficients used for the conditional computations.

    Parameters
    ----------
    alpha : float
      Amplitude in front of the GP.
      This is only used if series_id is None.
    beta : float
      Amplitude in front of the GP derivative.
      This is only used if series_id is None.
    series_id : int
      Use the coefficents corresponding to a given time series.
    """

    self._cond_series_id = series_id
    if series_id is None:
      self._cond_alpha = alpha
      self._cond_beta = beta
    else:
      self._cond_alpha = None
      self._cond_beta = None

  def _compute_t2(self, t2, dt2, U2, V2, phi2, ref2left, dt2left, dt2right,
    phi2left, phi2right):

    if self._cond_series_id is None:
      alpha = self._cond_alpha
      if self._with_derivative:
        beta = self._cond_beta
    else:
      alpha = self._alpha[self._cond_series_id]
      if self._with_derivative:
        beta = self._beta[self._cond_series_id]

    kernel_U2 = np.empty((t2.size, self._kernel._r))
    kernel_V2 = np.empty((t2.size, self._kernel._r))
    kernel_phi2 = np.empty((t2.size - 1, self._kernel._r))
    kernel_phi2left = np.empty((t2.size, self._kernel._r))
    kernel_phi2right = np.empty((t2.size, self._kernel._r))
    self._kernel._compute_t2(t2, dt2, kernel_U2, kernel_V2, kernel_phi2,
      ref2left, dt2left, dt2right, kernel_phi2left, kernel_phi2right)
    # cov(GP, GP)
    U2[:, self._offset:self._offset + self._kernel._r] = alpha * kernel_U2
    V2[:, self._offset:self._offset + self._kernel._r] = alpha * kernel_V2
    if self._with_derivative:
      kernel_dU2 = np.empty((t2.size, self._kernel._r))
      if beta == 0:
        kernel_d2U2 = 0
        self._kernel._deriv_t2(t2, dt2, kernel_dU2, kernel_V2, kernel_phi2,
          ref2left, dt2left, dt2right, kernel_phi2left, kernel_phi2right, None)
      else:
        kernel_d2U2 = np.empty((t2.size, self._kernel._r))
        self._kernel._deriv_t2(t2, dt2, kernel_dU2, kernel_V2, kernel_phi2,
          ref2left, dt2left, dt2right, kernel_phi2left, kernel_phi2right,
          kernel_d2U2)
      # cov(GP, dGP)
      U2[:, self._offset + self._kernel._r:self._offset +
        2 * self._kernel._r] = -alpha * kernel_dU2
      V2[:, self._offset + self._kernel._r:self._offset +
        2 * self._kernel._r] = beta * kernel_V2
      # cov(dGP, GP)
      U2[:, self._offset + 2 * self._kernel._r:self._offset +
        3 * self._kernel._r] = beta * kernel_dU2
      V2[:, self._offset + 2 * self._kernel._r:self._offset +
        3 * self._kernel._r] = alpha * kernel_V2
      # cov(dGP, dGP)
      U2[:, self._offset + 3 * self._kernel._r:self._offset +
        4 * self._kernel._r] = beta * kernel_d2U2
      V2[:, self._offset + 3 * self._kernel._r:self._offset +
        4 * self._kernel._r] = beta * kernel_V2

    for k in range(self._coef_r):
      phi2[:, self._offset + k * self._kernel._r:self._offset +
        (k + 1) * self._kernel._r] = kernel_phi2
      phi2left[:, self._offset + k * self._kernel._r:self._offset +
        (k + 1) * self._kernel._r] = kernel_phi2left
      phi2right[:, self._offset + k * self._kernel._r:self._offset +
        (k + 1) * self._kernel._r] = kernel_phi2right

  def _deriv(self, calc_d2=False):
    if self._with_derivative:
      raise NotImplementedError
    else:
      self._kernel._deriv(calc_d2)
      for k in range(self._nseries):
        ik = self._series_index[k]
        self._cov._dU[ik, self._offset:self._offset +
          self._kernel._r] = self._alpha[k] * self._kernel._cov._dU
        if calc_d2:
          self._cov._d2U[ik, self._offset:self._offset +
            self._kernel._r] = self._alpha[k] * self._kernel._cov._d2U

  def _deriv_t2(self,
    t2,
    dt2,
    dU2,
    V2,
    phi2,
    ref2left,
    dt2left,
    dt2right,
    phi2left,
    phi2right,
    d2U2=None):

    if self._with_derivative:
      raise NotImplementedError
    else:
      if self._cond_series_id is None:
        alpha = self._cond_alpha
      else:
        alpha = self._alpha[self._cond_series_id]
      kernel_dU2 = np.empty((t2.size, self._kernel._r))
      kernel_V2 = np.empty((t2.size, self._kernel._r))
      kernel_phi2 = np.empty((t2.size - 1, self._kernel._r))
      kernel_phi2left = np.empty((t2.size, self._kernel._r))
      kernel_phi2right = np.empty((t2.size, self._kernel._r))
      if d2U2 is None:
        kernel_d2U2 = None
      else:
        kernel_d2U2 = np.empty((t2.size, self._kernel._r))
      self._kernel._deriv_t2(t2, dt2, kernel_dU2, kernel_V2, kernel_phi2,
        ref2left, dt2left, dt2right, kernel_phi2left, kernel_phi2right,
        kernel_d2U2)
      dU2[:, self._offset:self._offset + self._kernel._r] = alpha * kernel_dU2
      V2[:, self._offset:self._offset + self._kernel._r] = alpha * kernel_V2
      if d2U2 is not None:
        d2U2[:,
          self._offset:self._offset + self._kernel._r] = alpha * kernel_d2U2

      phi2[:, self._offset:self._offset + self._kernel._r] = kernel_phi2
      phi2left[:,
        self._offset:self._offset + self._kernel._r] = kernel_phi2left
      phi2right[:,
        self._offset:self._offset + self._kernel._r] = kernel_phi2right

  def eval(self, dt):
    return (self._kernel.eval(dt))
