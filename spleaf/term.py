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

__all__ = [
  'Error', 'Jitter', 'InstrumentJitter',
  'CalibrationError', 'CalibrationJitter',
  'ExponentialKernel', 'QuasiperiodicKernel', 'Matern32Kernel',
  'USHOKernel', 'OSHOKernel', 'SHOKernel']

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
      raise Exception('This term has already been linked to a covariance matrix.')
    self._cov = cov
    self._linked = True

  def _compute(self):
    r"""
    Compute the S+LEAF representation of the term.
    """
    pass

  def _set_param(self):
    r"""
    Update the term parameters.
    """
    pass

  def _grad_param(self):
    r"""
    Gradient of a function with respect to the term parameters
    (listed in self._param).
    """
    return({})

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

  def _compute_t2(self, t2, dt2, U2, V2, phi2,
    ref2left, dt2left, dt2right, phi2left, phi2right):
    r"""
    Compute the S+LEAF representation of the covariance for a new calendar t2.
    """

    pass

  def _deriv(self, dU, d2U=None):
    r"""
    Compute the S+LEAF representation of the derivative of the GP.
    """

    pass

  def _deriv_t2(self, t2, dt2, dU2, V2, phi2,
    ref2left, dt2left, dt2right, phi2left, phi2right,
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

    raise NotImplementedError('The eval method should be implemented in Kernel childs.')

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
      self._cov.A += sig**2 - self._sig**2
      self._sig = sig

  def _grad_param(self):
    grad = {}
    grad['sig'] = 2*self._sig*self._cov._sum_grad_A
    return(grad)

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
      self._cov.A[self._indices] += sig**2 - self._sig**2
      self._sig = sig

  def _grad_param(self):
    grad = {}
    grad['sig'] = 2*self._sig*np.sum(self._cov._grad_A[self._indices])
    return(grad)

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
          self._cov.F[self._cov.offsetrow[group[i]]+group[j]] += var[group[0]]

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
          self._Fmask.append(self._cov.offsetrow[group[i]]+group[j])
    self._cov.F[self._Fmask] += var

  def _set_param(self, sig=None):
    if sig is not None:
      shift = sig**2 - self._sig**2
      self._cov.A[self._indices] += shift
      self._cov.F[self._Fmask] += shift
      self._sig = sig

  def _grad_param(self):
    grad = {}
    grad['sig'] = 2*self._sig * (
      np.sum(self._cov._grad_A[self._indices])
      + np.sum(self._cov._grad_F[self._Fmask]))
    return(grad)

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
    self._cov.phi[:, self._offset] = np.exp(-self._la*self._cov.dt)

  def _set_param(self, a=None, la=None):
    if a is not None:
      self._cov.A += a - self._a
      self._cov.U[:, self._offset] = a
      self._a = a
    if la is not None:
      self._cov.phi[:, self._offset] = np.exp(-la*self._cov.dt)
      self._la = la

  def _grad_param(self):
    grad = {}
    grad['a'] = self._cov._sum_grad_A + np.sum(self._cov._grad_U[:, self._offset])
    grad['la'] = -np.sum(self._cov.dt * self._cov.phi[:, self._offset] *
      self._cov._grad_phi[:, self._offset])
    return(grad)

  def _compute_t2(self, t2, dt2, U2, V2, phi2,
    ref2left, dt2left, dt2right, phi2left, phi2right):
    U2[:, self._offset] = self._a
    V2[:, self._offset] = 1.0
    phi2[:, self._offset] = np.exp(-self._la*dt2)
    phi2left[:, self._offset] = np.exp(-self._la*dt2left)
    phi2right[:, self._offset] = np.exp(-self._la*dt2right)

  def _deriv(self, dU, d2U=None):
    dU[:, self._offset] = -self._la * self._a
    if d2U is not None:
      d2U[:, self._offset] = -self._la**2 * self._a

  def _deriv_t2(self, t2, dt2, dU2, V2, phi2,
    ref2left, dt2left, dt2right, phi2left, phi2right,
    d2U2=None):
    dU2[:, self._offset] = -self._la * self._a
    V2[:, self._offset] = 1.0
    phi2[:, self._offset] = np.exp(-self._la*dt2)
    phi2left[:, self._offset] = np.exp(-self._la*dt2left)
    phi2right[:, self._offset] = np.exp(-self._la*dt2right)
    if d2U2 is not None:
      d2U2[:, self._offset] = -self._la**2 * self._a

  def eval(self, dt):
    return(self._a*np.exp(-self._la*np.abs(dt)))

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
    self._cnut = np.cos(self._nu*self._cov.t)
    self._snut = np.sin(self._nu*self._cov.t)
    self._cov.U[:, self._offset] = self._a * self._cnut + self._b * self._snut
    self._cov.V[:, self._offset] = self._cnut
    self._cov.U[:, self._offset+1] = self._a * self._snut - self._b * self._cnut
    self._cov.V[:, self._offset+1] = self._snut
    self._cov.phi[:, self._offset:self._offset+2] = np.exp(-self._la*self._cov.dt)[:, None]

  def _set_param(self, a=None, b=None, la=None, nu=None):
    updateU = False
    if a is not None:
      self._cov.A += a-self._a
      self._a = a
      updateU = True
    if b is not None:
      self._b = b
      updateU = True
    if la is not None:
      self._cov.phi[:, self._offset:self._offset+2] = np.exp(-la*self._cov.dt)[:, None]
      self._la = la
    if nu is not None:
      self._cnut = np.cos(nu*self._cov.t)
      self._snut = np.sin(nu*self._cov.t)
      self._cov.V[:, self._offset] = self._cnut
      self._cov.V[:, self._offset+1] = self._snut
      self._nu = nu
      updateU = True
    if updateU:
      self._cov.U[:, self._offset] = self._a * self._cnut + self._b * self._snut
      self._cov.U[:, self._offset+1] = self._a * self._snut - self._b * self._cnut

  def _grad_param(self):
    grad = {}
    grad['a'] = self._cov._sum_grad_A + np.sum(
      self._cov.V[:,self._offset] * self._cov._grad_U[:, self._offset]
      + self._cov.V[:,self._offset+1] * self._cov._grad_U[:, self._offset+1])
    grad['b'] = np.sum(
      self._cov.V[:,self._offset+1] * self._cov._grad_U[:, self._offset]
      - self._cov.V[:,self._offset] * self._cov._grad_U[:, self._offset+1])
    grad['la'] = -np.sum(self._cov.dt * self._cov.phi[:, self._offset] *
      (self._cov._grad_phi[:, self._offset] + self._cov._grad_phi[:, self._offset+1]))
    grad['nu'] = np.sum(self._cov.t * (
        self._cov.U[:,self._offset] * self._cov._grad_U[:,self._offset+1]
        - self._cov.U[:,self._offset+1] * self._cov._grad_U[:,self._offset]
        + self._cov.V[:,self._offset] * self._cov._grad_V[:,self._offset+1]
        - self._cov.V[:,self._offset+1] * self._cov._grad_V[:,self._offset]))
    return(grad)

  def _compute_t2(self, t2, dt2, U2, V2, phi2,
    ref2left, dt2left, dt2right, phi2left, phi2right):
    cnut2 = np.cos(self._nu*t2)
    snut2 = np.sin(self._nu*t2)
    U2[:, self._offset] = self._a * cnut2 + self._b * snut2
    V2[:, self._offset] = cnut2
    U2[:, self._offset+1] = self._a * snut2 - self._b * cnut2
    V2[:, self._offset+1] = snut2
    phi2[:, self._offset:self._offset+2] = np.exp(-self._la*dt2)[:, None]
    phi2left[:, self._offset:self._offset+2] = np.exp(-self._la*dt2left)[:, None]
    phi2right[:, self._offset:self._offset+2] = np.exp(-self._la*dt2right)[:, None]

  def _deriv(self, dU, d2U=None):
    da = -self._la*self._a + self._nu*self._b
    db = -self._la*self._b - self._nu*self._a
    dU[:, self._offset] = da * self._cnut + db * self._snut
    dU[:, self._offset+1] = da * self._snut - db * self._cnut
    if d2U is not None:
      d2a = (self._nu**2-self._la**2)*self._a + 2*self._la*self._nu*self._b
      d2b = (self._nu**2-self._la**2)*self._b - 2*self._la*self._nu*self._a
      d2U[:, self._offset] = d2a * self._cnut + d2b * self._snut
      d2U[:, self._offset+1] = d2a * self._snut - d2b * self._cnut

  def _deriv_t2(self, t2, dt2, dU2, V2, phi2,
    ref2left, dt2left, dt2right, phi2left, phi2right,
    d2U2=None):
    da = -self._la*self._a + self._nu*self._b
    db = -self._la*self._b - self._nu*self._a
    cnut2 = np.cos(self._nu*t2)
    snut2 = np.sin(self._nu*t2)
    dU2[:, self._offset] = da*cnut2 + db*snut2
    V2[:, self._offset] = cnut2
    dU2[:, self._offset+1] = da*snut2 - db*cnut2
    V2[:, self._offset+1] = snut2
    phi2[:, self._offset:self._offset+2] = np.exp(-self._la*dt2)[:, None]
    phi2left[:, self._offset:self._offset+2] = np.exp(-self._la*dt2left)[:, None]
    phi2right[:, self._offset:self._offset+2] = np.exp(-self._la*dt2right)[:, None]
    if d2U2 is not None:
      d2a = (self._nu**2-self._la**2)*self._a + 2*self._la*self._nu*self._b
      d2b = (self._nu**2-self._la**2)*self._b - 2*self._la*self._nu*self._a
      d2U2[:, self._offset] = d2a*cnut2 + d2b*snut2
      d2U2[:, self._offset+1] = d2a*snut2 - d2b*cnut2

  def eval(self, dt):
    adt = np.abs(dt)
    return(np.exp(-self._la*adt)*(self._a*np.cos(self._nu*adt) + self._b*np.sin(self._nu*adt)))

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
    la = np.sqrt(3)/self._rho
    b = a*la/self._eps
    return(a, b, la, nu)

  def _set_param(self, sig=None, rho=None):
    if sig is not None:
      self._sig = sig
    if rho is not None:
      self._rho = rho
    a, b, la, nu = self._getcoefs()
    super()._set_param(a, b, la, nu)

  def _grad_param(self):
    gradQP = super()._grad_param()
    grad = {}
    grad['sig'] = 2*self._sig*(gradQP['a'] + gradQP['b']*self._la/self._eps)
    grad['rho'] = -np.sqrt(3)/self._rho**2 * (gradQP['la'] + gradQP['b']*self._a/self._eps)
    return(grad)

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
    self._sqQ = np.sqrt(max(4*self._Q**2-1, self._eps))
    a = self._sig**2
    la = np.pi/(self._P0*self._Q)
    b = a/self._sqQ
    nu = la*self._sqQ
    return(a, b, la, nu)

  def _set_param(self, sig=None, P0=None, Q=None):
    if sig is not None:
      self._sig = sig
    if P0 is not None:
      self._P0 = P0
    if Q is not None:
      self._Q = Q
    a, b, la, nu = self._getcoefs()
    super()._set_param(a, b, la, nu)

  def _grad_param(self):
    gradQP = super()._grad_param()
    grad = {}
    grad['sig'] = 2*self._sig*(gradQP['a'] + gradQP['b']/self._sqQ)
    grad['P0'] = -np.pi/(self._P0**2*self._Q)*(gradQP['la'] + gradQP['nu']*self._sqQ)
    grad['Q'] = -np.pi/(self._P0*self._Q**2) * (gradQP['la'] + gradQP['nu']*self._sqQ)
    if 4*self._Q**2-1 > self._eps:
      grad['Q'] += 4*self._Q/self._sqQ*(gradQP['nu']*self._la - gradQP['b']*self._a/self._sqQ**2)
    return(grad)

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
    self._sqQ = np.sqrt(max(1-4*self._Q**2, self._eps))
    self._a = self._sig**2
    self._la = np.pi/(self._P0*self._Q)
    return(
      self._a*(1+1/self._sqQ)/2, self._la*(1-self._sqQ),
      self._a*(1-1/self._sqQ)/2, self._la*(1+self._sqQ))

  def _link(self, cov, offset):
    super()._link(cov, offset)
    self._exp1._link(cov, offset)
    self._exp2._link(cov, offset+1)

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

  def _grad_param(self):
    gradExp1 = self._exp1._grad_param()
    gradExp2 = self._exp2._grad_param()
    grad = {}
    grad['sig'] = 2*self._sig*(gradExp1['a']*(1+1/self._sqQ)/2 + gradExp2['a']*(1-1/self._sqQ)/2)
    grad['P0'] = -np.pi/(self._P0**2*self._Q)*(gradExp1['la']*(1-self._sqQ) + gradExp2['la']*(1+self._sqQ))
    grad['Q'] = -np.pi/(self._P0*self._Q**2)*(gradExp1['la']*(1-self._sqQ) + gradExp2['la']*(1+self._sqQ))
    if 1-4*self._Q**2 > self._eps:
      grad['Q'] -= 4*self._Q/self._sqQ*(
        (gradExp2['a']-gradExp1['a'])*self._a/(2*self._sqQ**2)
        +(gradExp2['la']-gradExp1['la'])*self._la)
    return(grad)

  def _compute_t2(self, t2, dt2, U2, V2, phi2,
    ref2left, dt2left, dt2right, phi2left, phi2right):
    self._exp1._compute_t2(t2, dt2, U2, V2, phi2,
      ref2left, dt2left, dt2right, phi2left, phi2right)
    self._exp2._compute_t2(t2, dt2, U2, V2, phi2,
      ref2left, dt2left, dt2right, phi2left, phi2right)

  def _deriv(self, dU, d2U=None):
    self._exp1._deriv(dU, d2U)
    self._exp2._deriv(dU, d2U)

  def _deriv_t2(self, t2, dt2, dU2, V2, phi2,
    ref2left, dt2left, dt2right, phi2left, phi2right,
    d2U2=None):
    self._exp1._deriv_t2(t2, dt2, dU2, V2, phi2,
      ref2left, dt2left, dt2right, phi2left, phi2right,
      d2U2)
    self._exp2._deriv_t2(t2, dt2, dU2, V2, phi2,
      ref2left, dt2left, dt2right, phi2left, phi2right,
      d2U2)

  def eval(self, dt):
    return(self._exp1.eval(dt) + self._exp2.eval(dt))

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
      if self._Q > 0.5 and Q <= 0.5:
        # USHO -> OSHO
        self._cov.A += self._osho._a - self._usho._a
        self._cov.V[:, self._offset:self._offset+2] = 1.0
        sig = self._sig
        P0 = self._P0
      elif self._Q <= 0.5 and Q > 0.5:
        # OSHO -> USHO
        self._cov.A += self._usho._a - self._osho._a
        sig = self._sig
        P0 = self._P0
      self._Q = Q

    if self._Q > 0.5:
      self._usho._set_param(sig, P0, Q)
    else:
      self._osho._set_param(sig, P0, Q)

  def _grad_param(self):
    if self._Q > 0.5:
      return(self._usho._grad_param())
    else:
      return(self._osho._grad_param())

  def _compute_t2(self, t2, dt2, U2, V2, phi2,
    ref2left, dt2left, dt2right, phi2left, phi2right):
    if self._Q > 0.5:
      self._usho._compute_t2(t2, dt2, U2, V2, phi2,
        ref2left, dt2left, dt2right, phi2left, phi2right)
    else:
      self._osho._compute_t2(t2, dt2, U2, V2, phi2,
        ref2left, dt2left, dt2right, phi2left, phi2right)

  def _deriv(self, dU, d2U=None):
    if self._Q > 0.5:
      self._usho._deriv(dU, d2U)
    else:
      self._osho._deriv(dU, d2U)

  def _deriv_t2(self, t2, dt2, dU2, V2, phi2,
    ref2left, dt2left, dt2right, phi2left, phi2right,
    d2U2=None):
    if self._Q > 0.5:
      self._usho._deriv_t2(t2, dt2, dU2, V2, phi2,
        ref2left, dt2left, dt2right, phi2left, phi2right,
        d2U2)
    else:
      self._osho._deriv_t2(t2, dt2, dU2, V2, phi2,
        ref2left, dt2left, dt2right, phi2left, phi2right,
        d2U2)

  def eval(self, dt):
    if self._Q > 0.5:
      return(self._usho.eval(dt))
    else:
      return(self._osho.eval(dt))
