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
  'Matern32Kernel', 'Matern52Kernel', 'ApproxGaussianKernel', 'USHOKernel',
  'OSHOKernel', 'SHOKernel', 'ApproxAigrainKernel', 'MultiSeriesKernel'
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
    dV2=None):
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

  def _grad_param(self, grad_dU=None, grad_dV=None):
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

    if grad_dV is not None:
      # self._cov._dV[:, self._offset] = self._la
      grad['la'] += np.sum(grad_dV[:, self._offset])

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
      self._cov._dV[:, self._offset] = self._la

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
    dV2=None):
    dU2[:, self._offset] = -self._la * self._a
    V2[:, self._offset] = 1.0
    phi2[:, self._offset] = np.exp(-self._la * dt2)
    phi2left[:, self._offset] = np.exp(-self._la * dt2left)
    phi2right[:, self._offset] = np.exp(-self._la * dt2right)
    if dV2 is not None:
      dV2[:, self._offset] = self._la

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
    self._nut = self._nu * self._cov.t
    self._cnut = np.cos(self._nut)
    self._snut = np.sin(self._nut)
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

  def _grad_param(self, grad_dU=None, grad_dV=None):
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

    if grad_dV is not None:
      # self._cov._dV[:, self._offset] = self._la * self._cnut - self._nu * self._snut
      # self._cov._dV[:, self._offset + 1] = self._la * self._snut + self._nu * self._cnut
      grad['la'] += np.sum(self._cnut * grad_dV[:, self._offset] +
        self._snut * grad_dV[:, self._offset + 1])
      latp1 = self._la * self._cov.t + 1
      grad['nu'] += np.sum((latp1 * self._cnut - self._nut * self._snut) *
        grad_dV[:, self._offset + 1] -
        (self._nut * self._cnut + latp1 * self._snut) *
        grad_dV[:, self._offset])

    return (grad)

  def _compute_t2(self, t2, dt2, U2, V2, phi2, ref2left, dt2left, dt2right,
    phi2left, phi2right):
    nut2 = self._nu * t2
    cnut2 = np.cos(nut2)
    snut2 = np.sin(nut2)
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
      self._cov._dV[:,
        self._offset] = self._la * self._cnut - self._nu * self._snut
      self._cov._dV[:,
        self._offset + 1] = self._la * self._snut + self._nu * self._cnut

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
    dV2=None):
    da = -self._la * self._a + self._nu * self._b
    db = -self._la * self._b - self._nu * self._a
    nut2 = self._nu * t2
    cnut2 = np.cos(nut2)
    snut2 = np.sin(nut2)
    dU2[:, self._offset] = da * cnut2 + db * snut2
    V2[:, self._offset] = cnut2
    dU2[:, self._offset + 1] = da * snut2 - db * cnut2
    V2[:, self._offset + 1] = snut2
    phi2[:, self._offset:self._offset + 2] = np.exp(-self._la * dt2)[:, None]
    phi2left[:, self._offset:self._offset + 2] = np.exp(-self._la * dt2left)[:,
      None]
    phi2right[:,
      self._offset:self._offset + 2] = np.exp(-self._la * dt2right)[:, None]
    if dV2 is not None:
      dV2[:, self._offset] = self._la * cnut2 - self._nu * snut2
      dV2[:, self._offset + 1] = self._la * snut2 + self._nu * cnut2

  def eval(self, dt):
    adt = np.abs(dt)
    return (np.exp(-self._la * adt) *
      (self._a * np.cos(self._nu * adt) + self._b * np.sin(self._nu * adt)))


class Matern32Kernel(Kernel):
  r"""
  Matérn 3/2 kernel.

  .. math:: K(\delta t) = \sigma^2 \mathrm{e}^{-\sqrt{3}\frac{\delta t}{\rho}}
    \left(1 + \sqrt{3}\frac{\delta t}{\rho}\right)

  Parameters
  ----------
  sig : float
    Amplitude (std).
  rho : float
    Scale.
  """

  def __init__(self, sig, rho):
    super().__init__()
    self._sig = sig
    self._rho = rho
    self._r = 2
    self._param = ['sig', 'rho']

  def _link(self, cov, offset):
    super()._link(cov, offset)
    self._t0 = (self._cov.t[0] + self._cov.t[-1]) / 2
    self._dt0 = self._cov.t - self._t0

  def _compute(self):
    self._a = self._sig**2
    self._la = np.sqrt(3) / self._rho
    self._la2 = self._la**2
    self._x = self._la * self._dt0
    self._1mx = 1 - self._x
    self._cov.A += self._a
    self._cov.U[:, self._offset] = self._a * self._x
    self._cov.V[:, self._offset] = 1.0
    self._cov.U[:, self._offset + 1] = self._a
    self._cov.V[:, self._offset + 1] = self._1mx
    self._cov.phi[:,
      self._offset:self._offset + 2] = np.exp(-self._la * self._cov.dt)[:,
      None]

  def _set_param(self, sig=None, rho=None):
    if sig is not None:
      self._sig = sig
    if rho is not None:
      self._rho = rho

  def _grad_param(self, grad_dU=None, grad_dV=None):
    grad = {}
    grad['sig'] = 2 * self._sig * (self._cov._sum_grad_A +
      np.sum(self._x * self._cov._grad_U[:, self._offset] +
      self._cov._grad_U[:, self._offset + 1]))
    grad['rho'] = -1 / self._rho * (np.sum(self._x *
      (self._a * self._cov._grad_U[:, self._offset] -
      self._cov._grad_V[:, self._offset + 1])) -
      self._la * np.sum(self._cov.dt * self._cov.phi[:, self._offset] *
      (self._cov._grad_phi[:, self._offset] +
      self._cov._grad_phi[:, self._offset + 1])))

    if grad_dU is not None:
      # self._cov._dU[:, self._offset] = self._la * self._a * self._1mx
      # self._cov._dU[:, self._offset + 1] = -self._la * self._a
      sum_grad_dU = np.sum(self._1mx * grad_dU[:, self._offset] -
        grad_dU[:, self._offset + 1])
      grad['sig'] += 2 * self._sig * self._la * sum_grad_dU
      grad['rho'] -= self._la / self._rho * self._a * (sum_grad_dU -
        np.sum(self._x * grad_dU[:, self._offset]))

    if grad_dV is not None:
      # self._cov._dV[:, self._offset] = self._la
      # self._cov._dV[:, self._offset + 1] = -self._la * self._x
      grad['rho'] -= self._la / self._rho * np.sum(grad_dV[:, self._offset] -
        2 * self._x * grad_dV[:, self._offset + 1])

    return (grad)

  def _compute_t2(self, t2, dt2, U2, V2, phi2, ref2left, dt2left, dt2right,
    phi2left, phi2right):
    x2 = self._la * (t2 - self._t0)
    U2[:, self._offset] = self._a * x2
    V2[:, self._offset] = 1.0
    U2[:, self._offset + 1] = self._a
    V2[:, self._offset + 1] = 1 - x2
    phi2[:, self._offset:self._offset + 2] = np.exp(-self._la * dt2)[:, None]
    phi2left[:, self._offset:self._offset + 2] = np.exp(-self._la * dt2left)[:,
      None]
    phi2right[:,
      self._offset:self._offset + 2] = np.exp(-self._la * dt2right)[:, None]

  def _deriv(self, calc_d2=False):
    self._cov._dU[:, self._offset] = self._la * self._a * self._1mx
    self._cov._dU[:, self._offset + 1] = -self._la * self._a
    if calc_d2:
      self._cov._dV[:, self._offset] = self._la
      self._cov._dV[:, self._offset + 1] = -self._la * self._x

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
    dV2=None):
    x2 = self._la * (t2 - self._t0)
    onemx2 = 1 - x2
    dU2[:, self._offset] = self._la * self._a * onemx2
    V2[:, self._offset] = 1.0
    dU2[:, self._offset + 1] = -self._la * self._a
    V2[:, self._offset + 1] = onemx2
    phi2[:, self._offset:self._offset + 2] = np.exp(-self._la * dt2)[:, None]
    phi2left[:, self._offset:self._offset + 2] = np.exp(-self._la * dt2left)[:,
      None]
    phi2right[:,
      self._offset:self._offset + 2] = np.exp(-self._la * dt2right)[:, None]
    if dV2 is not None:
      dV2[:, self._offset] = self._la
      dV2[:, self._offset + 1] = -self._la * x2

  def eval(self, dt):
    dx = self._la * np.abs(dt)
    return (self._a * np.exp(-dx) * (1 + dx))


class Matern52Kernel(Kernel):
  r"""
  Matérn 5/2 kernel.

  .. math:: K(\delta t) = \sigma^2 \mathrm{e}^{-\sqrt{5}\frac{\delta t}{\rho}}
    \left(1 + \sqrt{5}\frac{\delta t}{\rho} + \frac{5}{3}\left(\frac{\delta t}{\rho}\right)^2\right)

  Parameters
  ----------
  sig : float
    Amplitude (std).
  rho : float
    Scale.
  """

  def __init__(self, sig, rho):
    super().__init__()
    self._sig = sig
    self._rho = rho
    self._r = 3
    self._param = ['sig', 'rho']

  def _link(self, cov, offset):
    super()._link(cov, offset)
    self._t0 = (self._cov.t[0] + self._cov.t[-1]) / 2
    self._dt0 = self._cov.t - self._t0

  def _compute(self):
    self._a = self._sig**2
    self._la = np.sqrt(5) / self._rho
    self._la2 = self._la**2
    self._x = self._la * self._dt0
    self._x2_3 = self._x * self._x / 3
    self._1mx = 1 - self._x
    self._cov.A += self._a
    self._cov.U[:, self._offset] = self._a * (self._x + self._x2_3)
    self._cov.V[:, self._offset] = 1.0
    self._cov.U[:, self._offset + 1] = self._a
    self._cov.V[:, self._offset + 1] = self._1mx + self._x2_3
    self._cov.U[:, self._offset + 2] = self._a * self._x
    self._cov.V[:, self._offset + 2] = -2 / 3 * self._x
    self._cov.phi[:,
      self._offset:self._offset + 3] = np.exp(-self._la * self._cov.dt)[:,
      None]

  def _set_param(self, sig=None, rho=None):
    if sig is not None:
      self._sig = sig
    if rho is not None:
      self._rho = rho

  def _grad_param(self, grad_dU=None, grad_dV=None):
    grad = {}
    grad['sig'] = 2 * self._sig * (self._cov._sum_grad_A +
      np.sum((self._x + self._x2_3) * self._cov._grad_U[:, self._offset] +
      self._cov._grad_U[:, self._offset + 1] +
      self._x * self._cov._grad_U[:, self._offset + 2]))
    grad['rho'] = -1 / self._rho * (np.sum(self._a *
      (self._x + 2 * self._x2_3) * self._cov._grad_U[:, self._offset] +
      (2 * self._x2_3 - self._x) * self._cov._grad_V[:, self._offset + 1] +
      self._x * (self._a * self._cov._grad_U[:, self._offset + 2] -
      2 / 3 * self._cov._grad_V[:, self._offset + 2])) -
      self._la * np.sum(self._cov.dt * self._cov.phi[:, self._offset] *
      (self._cov._grad_phi[:, self._offset] + self._cov._grad_phi[:,
      self._offset + 1] + self._cov._grad_phi[:, self._offset + 2])))

    if grad_dU is not None:
      # self._cov._dU[:,
      #   self._offset] = self._la * self._a * (1 - self._x / 3 - self._x2_3)
      # self._cov._dU[:, self._offset + 1] = -self._la * self._a
      # self._cov._dU[:, self._offset + 2] = self._la * self._1mx * self._a
      sum_grad_dU = np.sum((1 - self._x / 3 - self._x2_3) *
        grad_dU[:, self._offset] - grad_dU[:, self._offset + 1] +
        self._1mx * grad_dU[:, self._offset + 2])
      grad['sig'] += 2 * self._sig * self._la * sum_grad_dU
      grad['rho'] -= self._la / self._rho * self._a * (sum_grad_dU -
        np.sum((self._x / 3 + 2 * self._x2_3) * grad_dU[:, self._offset] +
        self._x * grad_dU[:, self._offset + 2]))

    if grad_dV is not None:
      # self._cov._dV[:, self._offset] = self._la
      # self._cov._dV[:, self._offset + 1] = -self._la * self._x / 3 * self._1mx
      # self._cov._dV[:, self._offset + 2] = -2 / 3 * self._la * (1 + self._x)
      grad['rho'] -= self._la / self._rho * np.sum(grad_dV[:, self._offset] +
        (3 * self._x2_3 - 2 / 3 * self._x) * grad_dV[:, self._offset + 1] -
        2 / 3 * (1 + 2 * self._x) * grad_dV[:, self._offset + 2])

    return (grad)

  def _compute_t2(self, t2, dt2, U2, V2, phi2, ref2left, dt2left, dt2right,
    phi2left, phi2right):
    x2 = self._la * (t2 - self._t0)
    x22_3 = x2 * x2 / 3
    U2[:, self._offset] = self._a * (x2 + x22_3)
    V2[:, self._offset] = 1.0
    U2[:, self._offset + 1] = self._a
    V2[:, self._offset + 1] = 1 - x2 + x22_3
    U2[:, self._offset + 2] = self._a * x2
    V2[:, self._offset + 2] = -2 / 3 * x2
    phi2[:, self._offset:self._offset + 3] = np.exp(-self._la * dt2)[:, None]
    phi2left[:, self._offset:self._offset + 3] = np.exp(-self._la * dt2left)[:,
      None]
    phi2right[:,
      self._offset:self._offset + 3] = np.exp(-self._la * dt2right)[:, None]

  def _deriv(self, calc_d2=False):
    self._cov._dU[:,
      self._offset] = self._la * self._a * (1 - self._x / 3 - self._x2_3)
    self._cov._dU[:, self._offset + 1] = -self._la * self._a
    self._cov._dU[:, self._offset + 2] = self._la * self._a * self._1mx
    if calc_d2:
      self._cov._dV[:, self._offset] = self._la
      self._cov._dV[:, self._offset + 1] = -self._la * self._x / 3 * self._1mx
      self._cov._dV[:, self._offset + 2] = -2 / 3 * self._la * (1 + self._x)

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
    dV2=None):
    x2 = self._la * (t2 - self._t0)
    onemx2 = 1 - x2
    x22_3 = x2 * x2 / 3
    dU2[:, self._offset] = self._la * self._a * (1 - x2 / 3 - x22_3)
    V2[:, self._offset] = 1.0
    dU2[:, self._offset + 1] = -self._la * self._a
    V2[:, self._offset + 1] = onemx2 + x22_3
    dU2[:, self._offset + 2] = self._la * onemx2 * self._a
    V2[:, self._offset + 2] = -2 / 3 * x2
    phi2[:, self._offset:self._offset + 3] = np.exp(-self._la * dt2)[:, None]
    phi2left[:, self._offset:self._offset + 3] = np.exp(-self._la * dt2left)[:,
      None]
    phi2right[:,
      self._offset:self._offset + 3] = np.exp(-self._la * dt2right)[:, None]
    if dV2 is not None:
      dV2[:, self._offset] = self._la
      dV2[:, self._offset + 1] = -self._la * x2 / 3 * onemx2
      dV2[:, self._offset + 2] = -2 / 3 * self._la * (1 + x2)

  def eval(self, dt):
    dx = self._la * np.abs(dt)
    return (self._a * np.exp(-dx) * (1 + dx + dx * dx / 3))


class ApproxGaussianKernel(Kernel):
  r"""
  Kernel roughly approximating a Gaussian kernel.

  The exact Gaussian kernel is written as

  .. math:: K(\delta t) = \sigma^2 \exp\left(-\frac{1}{2}\left(\frac{\delta t}{\rho}\right)^2\right)

  while this kernel approximates it with

  .. math:: K(\delta t) = \sigma^2 \exp\left(-\lambda\delta t\right)\left(1 + \frac{\lambda}{\nu}\sin\left(\nu\delta t\right)\right),

  with :math:`\lambda = 1.06/\rho` and :math:`\nu = \lambda/0.74`.


  Parameters
  ----------
  sig : float
    Amplitude (std).
  rho : float
    Scale.

  Warnings
  --------
  This is not an exact Gaussian kernel but only an approximation.
  The deviation from the Gaussian kernel is below 1%.
  """

  def __init__(self, sig, rho):
    super().__init__()
    self._coef_larho = 1.06
    self._coef_la_nu = 0.74
    self._sig = sig
    self._rho = rho
    aexp, b, la, nu = self._getcoefs()
    self._exp = ExponentialKernel(aexp, la)
    self._qp = QuasiperiodicKernel(0, b, la, nu)
    self._r = self._exp._r + self._qp._r
    self._param = ['sig', 'rho']

  def _getcoefs(self):
    la = self._coef_larho / self._rho
    aexp = self._sig**2
    b = self._coef_la_nu * aexp
    nu = la / self._coef_la_nu
    return (aexp, b, la, nu)

  def _link(self, cov, offset):
    super()._link(cov, offset)
    self._exp._link(cov, offset)
    self._qp._link(cov, offset + self._exp._r)

  def _compute(self):
    self._exp._compute()
    self._qp._compute()

  def _set_param(self, sig=None, rho=None):
    if sig is not None:
      self._sig = sig
    if rho is not None:
      self._rho = rho
    aexp, b, la, nu = self._getcoefs()
    self._exp._set_param(aexp, la)
    self._qp._set_param(0, b, la, nu)

  def _grad_param(self, grad_dU=None, grad_dV=None):
    gradExp = self._exp._grad_param(grad_dU, grad_dV)
    gradQP = self._qp._grad_param(grad_dU, grad_dV)
    grad = {}
    grad['sig'] = 2 / self._sig * (gradExp['a'] * self._exp._a +
      gradQP['b'] * self._qp._b)
    grad['rho'] = -self._exp._la / self._rho * (gradExp['la'] + gradQP['la'] +
      gradQP['nu'] / self._coef_la_nu)
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
    dV2=None):
    self._exp._deriv_t2(t2, dt2, dU2, V2, phi2, ref2left, dt2left, dt2right,
      phi2left, phi2right, dV2)
    self._qp._deriv_t2(t2, dt2, dU2, V2, phi2, ref2left, dt2left, dt2right,
      phi2left, phi2right, dV2)

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

  def _grad_param(self, grad_dU=None, grad_dV=None):
    gradQP = super()._grad_param(grad_dU, grad_dV)
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

  def _grad_param(self, grad_dU=None, grad_dV=None):
    gradExp1 = self._exp1._grad_param(grad_dU, grad_dV)
    gradExp2 = self._exp2._grad_param(grad_dU, grad_dV)
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
    dV2=None):
    self._exp1._deriv_t2(t2, dt2, dU2, V2, phi2, ref2left, dt2left, dt2right,
      phi2left, phi2right, dV2)
    self._exp2._deriv_t2(t2, dt2, dU2, V2, phi2, ref2left, dt2left, dt2right,
      phi2left, phi2right, dV2)

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

  def _grad_param(self, grad_dU=None, grad_dV=None):
    if self._Q > 0.5:
      return (self._usho._grad_param(grad_dU, grad_dV))
    else:
      return (self._osho._grad_param(grad_dU, grad_dV))

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
    dV2=None):
    if self._Q > 0.5:
      self._usho._deriv_t2(t2, dt2, dU2, V2, phi2, ref2left, dt2left, dt2right,
        phi2left, phi2right, dV2)
    else:
      self._osho._deriv_t2(t2, dt2, dU2, V2, phi2, ref2left, dt2left, dt2right,
        phi2left, phi2right, dV2)

  def eval(self, dt):
    if self._Q > 0.5:
      return (self._usho.eval(dt))
    else:
      return (self._osho.eval(dt))


class ApproxAigrainKernel(Kernel):
  r"""
  Kernel roughly approximating Aigrain's kernel.

  Aigrain's exact kernel is written as

  .. math:: K(\delta t) = \sigma^2 \exp \left(- \frac{\delta t^2}{2 \rho^2}
    - \frac{\sin^2 \left( \frac{\pi \delta t }{P}\right) }{2 \eta^2}\right),

  while this kernel approximates it with

  .. math:: K(\delta t) = \sigma^2\frac{K_{3/2}(\Delta t)
    + f K_{\mathrm{SHO,\,fund.}}(\Delta t)
    + \frac{f^2}{4} K_{\mathrm{SHO,\,harm.}}(\Delta t)}{1+f+\frac{f^2}{4}},

  where

  .. math::
    \begin{align}
      \nu &= \frac{2\pi}{P},\\
      f &= \frac{1}{4\eta^2},\\
      K_{3/2}(\Delta t) &= \exp\left(-\frac{\sqrt{3}\Delta t}{\rho}\right)\left(1+\frac{\sqrt{3}\Delta t}{\rho}\right),\\
      K_{\mathrm{SHO,\,fund.}}(\Delta t) &= \exp\left(-\frac{\Delta t}{\rho}\right)\left(\cos\left(\nu \Delta t\right)+\frac{1}{\nu\rho}\sin\left(\nu \Delta t\right)\right),\\
      K_{\mathrm{SHO,\,harm.}}(\Delta t) &= \exp\left(-\frac{\Delta t}{\rho}\right)\left(\cos\left(2\nu \Delta t\right)+\frac{1}{2\nu\rho}\sin\left(2\nu \Delta t\right)\right).
    \end{align}

  Parameters
  ----------
  sig : float
    Amplitude (std).
  P : float
    Period.
  rho : float
    Scale.
  eta : float
    Scale of oscillations.
  """

  def __init__(self, sig, P, rho, eta):
    super().__init__()
    self._sig = sig
    self._P = P
    self._rho = rho
    self._eta = eta
    sig0, a1, a2, b1, b2, la, nu = self._getcoefs()
    self._mat = Matern32Kernel(sig0, self._rho)
    self._qp1 = QuasiperiodicKernel(a1, b1, la, nu)
    self._qp2 = QuasiperiodicKernel(a2, b2, la, 2 * nu)
    self._kernels = [self._mat, self._qp1, self._qp2]
    self._r = sum(kernel._r for kernel in self._kernels)
    self._param = ['sig', 'P', 'rho', 'eta']

  def _getcoefs(self):
    la = 1 / self._rho
    self._var = self._sig * self._sig
    self._eta2 = self._eta * self._eta
    self._f = 1 / (4 * self._eta2)
    self._f2 = self._f * self._f
    self._f2_4 = self._f2 / 4
    self._deno = 1 + self._f + self._f2_4
    a0 = self._var / self._deno
    sig0 = np.sqrt(a0)
    a1 = self._f * a0
    a2 = self._f2_4 * a0
    nu = 2 * np.pi / self._P
    la_nu = la / nu
    b1 = a1 * la_nu
    b2 = a2 * la_nu / 2
    return (sig0, a1, a2, b1, b2, la, nu)

  def _link(self, cov, offset):
    super()._link(cov, offset)
    off = offset
    for kernel in self._kernels:
      kernel._link(cov, off)
      off += kernel._r

  def _compute(self):
    for kernel in self._kernels:
      kernel._compute()

  def _set_param(self, sig=None, P=None, rho=None, eta=None):
    if sig is not None:
      self._sig = sig
    if P is not None:
      self._P = P
    if rho is not None:
      self._rho = rho
    if eta is not None:
      self._eta = eta
    sig0, a1, a2, b1, b2, la, nu = self._getcoefs()
    self._mat._set_param(sig0, self._rho)
    self._qp1._set_param(a1, b1, la, nu)
    self._qp2._set_param(a2, b2, la, 2 * nu)

  def _grad_param(self, grad_dU=None, grad_dV=None):
    gradMat = self._mat._grad_param(grad_dU, grad_dV)
    gradQP1 = self._qp1._grad_param(grad_dU, grad_dV)
    gradQP2 = self._qp2._grad_param(grad_dU, grad_dV)

    sgs0 = self._mat._sig * gradMat['sig']
    aga1 = self._qp1._a * gradQP1['a']
    aga2 = self._qp2._a * gradQP2['a']
    bgb1 = self._qp1._b * gradQP1['b']
    bgb2 = self._qp2._b * gradQP2['b']
    bgb = bgb1 + bgb2
    sgs = sgs0 + 2 * (aga1 + aga2 + bgb)

    grad = {}
    grad['sig'] = sgs / self._sig
    grad['P'] = (bgb - self._qp1._nu * gradQP1['nu'] -
      self._qp2._nu * gradQP2['nu']) / self._P
    grad['rho'] = gradMat['rho'] - (bgb + self._qp1._la *
      (gradQP1['la'] + gradQP2['la'])) / self._rho
    grad['eta'] = (sgs * (self._f + self._f2 / 2) / self._deno - 2 *
      (aga1 + bgb1 + 2 * (aga2 + bgb2))) / self._eta
    return (grad)

  def _compute_t2(self, t2, dt2, U2, V2, phi2, ref2left, dt2left, dt2right,
    phi2left, phi2right):
    for kernel in self._kernels:
      kernel._compute_t2(t2, dt2, U2, V2, phi2, ref2left, dt2left, dt2right,
        phi2left, phi2right)

  def _deriv(self, calc_d2=False):
    for kernel in self._kernels:
      kernel._deriv(calc_d2)

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
    dV2=None):
    for kernel in self._kernels:
      kernel._deriv_t2(t2, dt2, dU2, V2, phi2, ref2left, dt2left, dt2right,
        phi2left, phi2right, dV2)

  def eval(self, dt):
    return (sum(kernel.eval(dt) for kernel in self._kernels))


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
    self._dV = np.empty((self.n, r))
    self._B = None

    self._grad_A = np.empty(self.n)
    self._grad_U = np.empty((self.n, r))
    self._grad_V = np.empty((self.n, r))
    self._grad_phi = np.empty((self.n - 1, r))
    self._sum_grad_A = None

    self._grad_dU = np.empty((self.n, r))
    self._grad_dV = np.empty((self.n, r))
    self._grad_B = np.empty(self.n)


class MultiSeriesKernel(Kernel):
  r"""
  Linear combination of a Kernel and its derivative
  applied to heterogenous time series.

  This kernel allows to model efficiently
  several (heterogeneous) time series (:math:`y_i`)
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
    else:
      self._with_derivative = True
      self._param += [f'beta_{k}' for k in range(self._nseries)]
    self._r = kernel._r

    self._cond_alpha = 1
    self._cond_beta = 0
    self._cond_series_id = None

  def _link(self, cov, offset):
    super()._link(cov, offset)
    self._kernel._link(_FakeCov(cov.t, cov.dt, self._r), 0)

  def _compute(self):
    self._kernel._cov.A[:] = 0
    self._kernel._compute()
    if self._with_derivative:
      self._kernel._deriv(True)
      self._kernel._cov._B = np.sum(self._kernel._cov._dU *
        self._kernel._cov._dV,
        axis=1)
    for k in range(self._nseries):
      ik = self._series_index[k]
      # cov(GP, GP)
      self._cov.A[ik] += self._alpha[k]**2 * self._kernel._cov.A[ik]
      self._cov.U[ik, self._offset:self._offset +
        self._r] = self._alpha[k] * self._kernel._cov.U[ik]
      self._cov.V[ik, self._offset:self._offset +
        self._r] = self._alpha[k] * self._kernel._cov.V[ik]
      if self._with_derivative:
        # cov(GP, dGP), cov(dGP, GP), cov(dGP, dGP)
        self._cov.A[ik] += self._beta[k]**2 * self._kernel._cov._B[ik]
        self._cov.U[ik, self._offset:self._offset +
          self._r] += self._beta[k] * self._kernel._cov._dU[ik]
        self._cov.V[ik, self._offset:self._offset +
          self._r] += self._beta[k] * self._kernel._cov._dV[ik]
    self._cov.phi[:,
      self._offset:self._offset + self._r] = self._kernel._cov.phi

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
      grad[f'alpha_{k}'] += np.sum(self._cov._grad_U[ik,
        self._offset:self._offset + self._r] * self._kernel._cov.U[ik] +
        self._cov._grad_V[ik, self._offset:self._offset + self._r] *
        self._kernel._cov.V[ik])
      self._kernel._cov._grad_A[ik] = self._alpha[k]**2 * self._cov._grad_A[ik]
      self._kernel._cov._grad_U[ik] = self._alpha[k] * self._cov._grad_U[ik,
        self._offset:self._offset + self._r]
      self._kernel._cov._grad_V[ik] = self._alpha[k] * self._cov._grad_V[ik,
        self._offset:self._offset + self._r]
      if self._with_derivative:
        # cov(GP, dGP), cov(dGP, GP), cov(dGP, dGP)
        grad[f'beta_{k}'] = 2 * self._beta[k] * np.sum(
          self._cov._grad_A[ik] * self._kernel._cov._B[ik])
        grad[f'beta_{k}'] += np.sum(self._cov._grad_U[ik,
          self._offset:self._offset + self._r] * self._kernel._cov._dU[ik] +
          self._cov._grad_V[ik, self._offset:self._offset + self._r] *
          self._kernel._cov._dV[ik])
        self._kernel._cov._grad_B[
          ik] = self._beta[k]**2 * self._cov._grad_A[ik]
        self._kernel._cov._grad_dU[ik] = self._beta[k] * self._cov._grad_U[ik,
          self._offset:self._offset + self._r]
        self._kernel._cov._grad_dV[ik] = self._beta[k] * self._cov._grad_V[ik,
          self._offset:self._offset + self._r]
    self._kernel._cov._grad_phi = self._cov._grad_phi[:,
      self._offset:self._offset + self._r]

    self._kernel._cov._sum_grad_A = np.sum(self._kernel._cov._grad_A)
    if self._with_derivative:
      self._kernel._cov._grad_dU += self._kernel._cov._dV * self._kernel._cov._grad_B[:,
        None]
      self._kernel._cov._grad_dV += self._kernel._cov._dU * self._kernel._cov._grad_B[:,
        None]
      grad.update(
        self._kernel._grad_param(self._kernel._cov._grad_dU,
        self._kernel._cov._grad_dV))
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

    kernel_U2 = np.empty((t2.size, self._r))
    kernel_V2 = np.empty((t2.size, self._r))
    kernel_phi2 = np.empty((t2.size - 1, self._r))
    kernel_phi2left = np.empty((t2.size, self._r))
    kernel_phi2right = np.empty((t2.size, self._r))
    self._kernel._compute_t2(t2, dt2, kernel_U2, kernel_V2, kernel_phi2,
      ref2left, dt2left, dt2right, kernel_phi2left, kernel_phi2right)
    # cov(GP, GP)
    U2[:, self._offset:self._offset + self._r] = alpha * kernel_U2
    V2[:, self._offset:self._offset + self._r] = alpha * kernel_V2
    phi2[:, self._offset:self._offset + self._r] = kernel_phi2
    phi2left[:, self._offset:self._offset + self._r] = kernel_phi2left
    phi2right[:, self._offset:self._offset + self._r] = kernel_phi2right

    if self._with_derivative and beta != 0:
      kernel_dU2 = np.empty((t2.size, self._r))
      kernel_dV2 = np.empty((t2.size, self._r))
      self._kernel._deriv_t2(t2, dt2, kernel_dU2, kernel_V2, kernel_phi2,
        ref2left, dt2left, dt2right, kernel_phi2left, kernel_phi2right,
        kernel_dV2)
      # cov(GP, dGP), cov(dGP, GP), cov(dGP, dGP)
      U2[:, self._offset:self._offset + self._r] += beta * kernel_dU2
      V2[:, self._offset:self._offset + self._r] += beta * kernel_dV2

  def _deriv(self, calc_d2=False):
    if self._with_derivative:
      raise NotImplementedError
    else:
      self._kernel._deriv(calc_d2)
      for k in range(self._nseries):
        ik = self._series_index[k]
        self._cov._dU[ik, self._offset:self._offset +
          self._r] = self._alpha[k] * self._kernel._cov._dU
        if calc_d2:
          self._cov._dV[ik, self._offset:self._offset +
            self._r] = self._alpha[k] * self._kernel._cov._dV

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
    dV2=None):

    if self._with_derivative:
      raise NotImplementedError
    else:
      if self._cond_series_id is None:
        alpha = self._cond_alpha
      else:
        alpha = self._alpha[self._cond_series_id]
      kernel_dU2 = np.empty((t2.size, self._r))
      kernel_V2 = np.empty((t2.size, self._r))
      kernel_phi2 = np.empty((t2.size - 1, self._r))
      kernel_phi2left = np.empty((t2.size, self._r))
      kernel_phi2right = np.empty((t2.size, self._r))
      if dV2 is None:
        kernel_dV2 = None
      else:
        kernel_dV2 = np.empty((t2.size, self._r))
      self._kernel._deriv_t2(t2, dt2, kernel_dU2, kernel_V2, kernel_phi2,
        ref2left, dt2left, dt2right, kernel_phi2left, kernel_phi2right,
        kernel_dV2)
      dU2[:, self._offset:self._offset + self._r] = alpha * kernel_dU2
      V2[:, self._offset:self._offset + self._r] = alpha * kernel_V2
      if dV2 is not None:
        dV2[:, self._offset:self._offset + self._r] = alpha * kernel_dV2

      phi2[:, self._offset:self._offset + self._r] = kernel_phi2
      phi2left[:, self._offset:self._offset + self._r] = kernel_phi2left
      phi2right[:, self._offset:self._offset + self._r] = kernel_phi2right

  def eval(self, dt):
    return (self._kernel.eval(dt))
