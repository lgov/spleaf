# -*- coding: utf-8 -*-

# Copyright 2019 Jean-Baptiste Delisle
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
from . import Spleaf

class Cov(Spleaf):
  r"""
  Covariance matrix of a radial velocity (rv) timeseries.

  The covariance matrix (:math:`C`) can take into account
  several noise components:

  - photon noise
  - jitter
  - calibration noise
  - correlated noise with an exponential kernel
  - correlated noise with a quasiperiodic kernel.

  The complete noise model reads (for :math:`i,j<n`)

  .. math::
    :nowrap:

    \begin{align*}
    C_{i,j} &= \delta_{i,j} \left(
      \sigma^2_{\mathrm{photon}, i} + \sigma^2_\mathrm{jitter} +
      \sum_{k<n_\mathrm{inst}} \delta_{\mathrm{inst}_i, k} \sigma^2_{\mathrm{jitter\,inst}, k}
    \right)\\
    &+ \delta_{\mathrm{calib}_i,\mathrm{calib}_j} \left(
      \sigma^2_{\mathrm{calib\,meas}, i} +
      \sum_{k<n_\mathrm{inst}} \delta_{\mathrm{inst}_i, k} \sigma^2_{\mathrm{calib\,inst}, k}
    \right)\\
    &+ \sum_{k<n_\mathrm{exp}} \sigma^2_{\mathrm{exp}, k}
      \mathrm{e}^{-\lambda_{\mathrm{exp}, k}|t_j-t_i|}\\
    &+ \sum_{k<n_\mathrm{qper}} \left(
        \sigma^2_{\mathrm{cos}, k} \cos(\nu_k |t_j-t_i|)
        + \sigma^2_{\mathrm{sin}, k} \sin(\nu_k |t_j-t_i|)
      \right) \mathrm{e}^{-\lambda_{\mathrm{qper}, k}|t_j-t_i|}
    \end{align*}

  The covariance matrix :math:`C` is represented internally as
  a S+LEAF matrix (see :class:`spleaf.Spleaf`).

  Parameters
  ----------
  t : (n,) ndarray
    Time of the measurements.
    This array must be in increasing order.
  var_photon : (n,) ndarray
    Observational photon noise
    (:math:`\sigma^2_\mathrm{photon}`).
  var_jitter : float
    Stellar jitter
    (:math:`\sigma^2_\mathrm{jitter}`)
    that is added to the diagonal of the covariance matrix.
  inst_id: (n,) ndarray
    Instrument ids (in the range [0,ninst])
    identifying the instrument used for each measurement.
    This array must be of dtype int.
  var_jitter_inst: (ninst,) ndarray
    Instrumental jitter
    (:math:`\sigma^2_\mathrm{jitter\,inst}`).
  calib_file: (n,) ndarray
    Name of the calibration file used for each measurement.
    This is used to identify measurements that share the same calibration
    (and thus the same calibration noise).
    An empty string means that
    the measurement does not share its calibration with any other point
    (or that no calibration was used).
  var_calib_meas: (n,) ndarray
    Calibration error for each measurement
    (:math:`\sigma^2_\mathrm{calib\,meas}`).
  var_calib_inst: (ninst,) ndarray
    Additional calibration error for each instrument
    (:math:`\sigma^2_\mathrm{calib\,inst}`).
  var_exp: (nexp,) ndarray
    Amplitude of exponential terms
    (:math:`\sigma^2_\mathrm{exp}`).
  lambda_exp: (nexp,) ndarray
    Decay rate of exponential terms
    (:math:`\lambda_\mathrm{exp}`).
  var_cos_qper: (nqper,) ndarray
    Amplitude of the cosine of quasiperiodic terms
    (:math:`\sigma^2_\mathrm{cos}`).
  var_sin_qper: (nqper,) ndarray
    Amplitude of the sine of quasiperiodic terms
    (:math:`\sigma^2_\mathrm{sin}`).
  lambda_qper: (nqper,) ndarray
    Decay rate of quasiperiodic terms
    (:math:`\lambda_\mathrm{qper}`).
  nu_qper: (nqper,) ndarray
    Angular frequency of quasiperiodic terms
    (:math:`\nu_\mathrm{qper}`).
  copy : bool
    Whether to copy arrays.

  Attributes
  ----------
  t, var_photon,
  var_jitter,
  inst_id, var_jitter_inst,
  calib_file, var_calib_meas, var_calib_inst,
  var_exp, lambda_exp,
  var_cos_qper, var_sin_qper, lambda_qper, nu_qper :
    See parameters.
  n : int
    Number of measurements.
  ninst : int
    Number of instruments.
  nexp : int
    Number of exponential terms.
  nqper : int
    Number of quasiperiodic terms.
  r, A, U, V, phi, offsetrow, b, F, D, W, G :
    S+LEAF representation of the covariance matrix
    (see :class:`spleaf.Spleaf`).
  """

  def __init__(self, t, var_photon,
    var_jitter=0.0,
    inst_id=None, var_jitter_inst=None,
    calib_file=None, var_calib_meas=None, var_calib_inst=None,
    var_exp=None, lambda_exp=None,
    var_cos_qper=None, var_sin_qper=None,
    lambda_qper=None, nu_qper=None,
    copy=False):

    self.t = t
    self.var_photon = self._copy(var_photon, copy)

    self.var_jitter = self._copy(var_jitter, copy)

    self.inst_id = self._copy(inst_id, copy)
    self.var_jitter_inst = self._copy(var_jitter_inst, copy)
    self.calib_file = self._copy(calib_file, copy)
    self.var_calib_meas = self._copy(var_calib_meas, copy)
    self.var_calib_inst = self._copy(var_calib_inst, copy)

    self.var_exp = self._copy(var_exp, copy)
    self.lambda_exp = self._copy(lambda_exp, copy)

    self.var_cos_qper = self._copy(var_cos_qper, copy)
    self.var_sin_qper = self._copy(var_sin_qper, copy)
    self.lambda_qper = self._copy(lambda_qper, copy)
    self.nu_qper = self._copy(nu_qper, copy)

    if not isinstance(t, np.ndarray):
      raise Exception('Cov: t is not an array.')
    n = t.size
    self._dt = t[1:] - t[:-1]
    if np.min(self._dt) < 0:
      raise Exception('Cov: the timeseries must be provided'
        ' in increasing order.')

    # Photon noise and stellar jitter
    if not isinstance(var_photon, np.ndarray):
      raise Exception('Cov: var_photon is not an array.')
    elif var_photon.size != n:
      raise Exception('Cov: t and var_photon have incompatible sizes.')
    if not isinstance(var_jitter, float):
      raise Exception('Cov: var_jitter is not a float.')
    A = var_photon + var_jitter

    # Instruments
    if inst_id is None:
      self.ninst = 1
      self.inst_id = np.zeros(n, dtype=int)
    elif not isinstance(inst_id, np.ndarray) or inst_id.dtype != int:
      raise Exception('Cov: inst_id is not an array of int.')
    else:
      self.ninst = inst_id.max() + 1
    self._dAdvar_inst = np.empty((self.ninst, n))
    for i in range(self.ninst):
      self._dAdvar_inst[i] = self.inst_id == i

    # Instruments jitter
    if var_jitter_inst is None:
      self.var_jitter_inst = np.zeros(self.ninst)
    elif not isinstance(var_jitter_inst, np.ndarray):
      raise Exception('Cov: var_jitter_inst is not an array.')
    elif var_jitter_inst.size != self.ninst:
        raise Exception('Cov: the size of var_jitter_inst'
          ' should be the number of instruments.')

    # Instruments calibration error
    if var_calib_meas is None:
      self.var_calib_meas = np.zeros(n)
    elif not isinstance(var_calib_meas, np.ndarray):
      raise Exception('Cov: var_calib_meas is not an array.')
    elif var_calib_meas.size != n:
      raise Exception('Cov: t and var_calib_meas have incompatible sizes.')

    if var_calib_inst is None:
      self.var_calib_inst = np.zeros(self.ninst)
    elif not isinstance(var_calib_inst, np.ndarray):
      raise Exception('Cov: var_calib_inst is not an array.')
    elif var_calib_inst.size != self.ninst:
      raise Exception('Cov: the size of var_calib_inst'
        ' should be the number of instruments.')

    b = np.zeros(n, dtype=int)
    if calib_file is None:
      F = np.empty(0)
      self._dFdvar_calib_inst = np.empty((self.ninst,0))
    elif not isinstance(calib_file, np.ndarray):
      raise Exception('Cov: calib_file is not an array.')
    elif calib_file.size != n:
      raise Exception('Cov: t and calib_file have incompatible sizes.')
    else:
      # Find groups of points using same calibration
      lastcalibinst = self.ninst*['']
      groups = [[] for _ in range(self.ninst)]
      for k in range(n):
        i = self.inst_id[k]
        if lastcalibinst[i] != '' and calib_file[k] == lastcalibinst[i]:
          groups[i][-1].append(k)
        else:
          groups[i].append([k])
          lastcalibinst[i] = calib_file[k]
      # Compute b, F, dF
      listF = [[] for _ in range(n)]
      listdF = [[[] for _ in range(n)] for _ in range(self.ninst)]
      for i in range(self.ninst):
        for group in groups[i]:
          start = group[0]
          end = group[-1]
          deltagroup = np.array(group, dtype=int)-start
          Fgroup = np.zeros(end-start+1)
          Fgroup[deltagroup] = (self.var_calib_meas[start]
            + self.var_calib_inst[i])
          dFgroup = np.zeros(end-start+1)
          dFgroup[deltagroup] = 1.0
          for k in group[1:]:
            b[k] = k-start
            listF[k] = Fgroup[:b[k]]
            for ib in range(self.ninst):
              if ib == i:
                listdF[i][k] = dFgroup[:b[k]]
              else:
                listdF[ib][k] = np.zeros(b[k])
      F = np.concatenate(listF)
      self._dFdvar_calib_inst = np.array([np.concatenate(listdF[i])
        for i in range(self.ninst)])

    # Instruments jitter/calib contributions to diagonal part
    A += (self.var_calib_meas
      + (self.var_calib_inst + self.var_jitter_inst).dot(self._dAdvar_inst))

    # Exponential terms
    if var_exp is None:
      self.nexp = 0
    elif not isinstance(var_exp, np.ndarray):
      raise Exception('Cov: var_exp is not an array.')
    else:
      self.nexp = var_exp.size
      if not isinstance(lambda_exp, np.ndarray) or lambda_exp.size != self.nexp:
        raise Exception('Cov: incompatible sizes'
          ' in the exponential part (var_exp, lambda_exp).')
      A += np.sum(var_exp)

    # Quasiperiodic terms
    if var_cos_qper is None:
      self.nqper = 0
    elif not isinstance(var_cos_qper, np.ndarray):
      raise Exception('Cov: var_cos_qper is not an array.')
    else:
      self.nqper = var_cos_qper.size
      for x_qper in [var_sin_qper, lambda_qper, nu_qper]:
        if not isinstance(x_qper, np.ndarray) or x_qper.size != self.nqper:
          raise Exception('Cov: incompatible sizes'
            ' in the quasiperiodic part'
            ' (var_cos_qper, var_sin_qper, lambda_qper, nu_qper).')
      A += np.sum(var_cos_qper)

    # Semiseparable part (U, V, phi) for exp and qper terms
    r = self.nexp + 2*self.nqper
    U = np.empty((n,r))
    V = np.empty((n,r))
    phi = np.empty((n-1,r))
    U[:,:self.nexp] = var_exp
    V[:,:self.nexp] = 1.0
    for k in range(self.nexp):
      phi[:,k] = np.exp(-lambda_exp[k]*self._dt)
    for k in range(self.nqper):
      cnut = np.cos(nu_qper[k]*t)
      snut = np.sin(nu_qper[k]*t)
      U[:,self.nexp+k] = var_cos_qper[k]*cnut + var_sin_qper[k]*snut
      U[:,k-self.nqper] = var_cos_qper[k]*snut - var_sin_qper[k]*cnut
      V[:,self.nexp+k] = cnut
      V[:,k-self.nqper] = snut
      phi[:,self.nexp+k] = np.exp(-lambda_qper[k]*self._dt)
      phi[:,k-self.nqper] = phi[:,self.nexp+k]

    # Spleaf initialization (Cholesky decomposition)
    offsetrow = np.cumsum(b-1) + 1
    super().__init__(A, U, V, phi, offsetrow, b, F)

  def update_param(self, var_jitter=None,
    var_jitter_inst=None, var_calib_inst=None,
    var_exp=None, lambda_exp=None,
    var_cos_qper=None, var_sin_qper=None,
    lambda_qper=None, nu_qper=None,
    copy=False):
    r"""
    Update the initial parameters
    (`var_jitter`, `var_jitter_inst`, `var_calib_inst`,
    `var_exp`, `lambda_exp`,
    `var_cos_qper`, `var_sin_qper`, `lambda_qper`, `nu_qper`)
    and recompute the Cholesky decomposition of the matrix.
    """

    A, U, V, phi, F = self.A, self.U, self.V, self.phi, self.F

    if var_jitter is not None:
      A += var_jitter - self.var_jitter
      self.var_jitter = self._copy(var_jitter, copy)

    if var_jitter_inst is not None:
      A += (var_jitter_inst-self.var_jitter_inst).dot(self._dAdvar_inst)
      self.var_jitter_inst = self._copy(var_jitter_inst, copy)

    if var_calib_inst is not None:
      F += (var_calib_inst-self.var_calib_inst).dot(self._dFdvar_calib_inst)
      A += (var_calib_inst-self.var_calib_inst).dot(self._dAdvar_inst)
      self.var_calib_inst = self._copy(var_calib_inst, copy)

    if var_exp is not None:
      A += np.sum(var_exp - self.var_exp)
      self.var_exp = self._copy(var_exp, copy)
      U[:,:self.nexp] = self.var_exp

    if lambda_exp is not None:
      for k in range(self.nexp):
        phi[:,k] = np.exp(-lambda_exp[k]*self._dt)
      self.lambda_exp = self._copy(lambda_exp, copy)

    recompute_U_qper = False
    if var_cos_qper is not None:
      A += np.sum(var_cos_qper - self.var_cos_qper)
      self.var_cos_qper = self._copy(var_cos_qper, copy)
      recompute_U_qper = True

    if var_sin_qper is not None:
      self.var_sin_qper = self._copy(var_sin_qper, copy)
      recompute_U_qper = True

    if nu_qper is not None:
      self.nu_qper = self._copy(nu_qper, copy)
      recompute_U_qper = True

    if recompute_U_qper:
      for k in range(self.nqper):
        cnut = np.cos(self.nu_qper[k]*self.t)
        snut = np.sin(self.nu_qper[k]*self.t)
        U[:,self.nexp+k] = (self.var_cos_qper[k]*cnut
          + self.var_sin_qper[k]*snut)
        U[:,k-self.nqper] = (self.var_cos_qper[k]*snut
          - self.var_sin_qper[k]*cnut)
        if nu_qper is not None:
          V[:,self.nexp+k] = cnut
          V[:,k-self.nqper] = snut

    if lambda_qper is not None:
      self.lambda_qper = self._copy(lambda_qper, copy)
      for k in range(self.nqper):
        phi[:,self.nexp+k] = np.exp(-self.lambda_qper[k]*self._dt)
        phi[:,k-self.nqper] = phi[:,self.nexp+k]

    super().update_param(A, U, V, phi, F)

  def grad_param(self):
    r"""
    Gradient of a function with respect to
    the initial parameters
    (`var_jitter`, `var_jitter_inst`, `var_calib_inst`,
    `var_exp`, `lambda_exp`,
    `var_cos_qper`, `var_sin_qper`, `lambda_qper`, `nu_qper`)
    after a call to :func:`cholesky_back`.

    Returns
    -------
    grad_var_jitter : float
      Gradient of the function with respect to `var_jitter`.
    grad_var_jitter_inst : (ninst,) ndarray
      Gradient of the function with respect to `var_jitter_inst`.
    grad_var_calib_inst : (ninst,) ndarray
      Gradient of the function with respect to `var_calib_inst`.
    grad_var_exp : (nexp,) ndarray
      Gradient of the function with respect to `var_exp`.
    grad_lambda_exp : (nexp,) ndarray
      Gradient of the function with respect to `lambda_exp`.
    grad_var_cos_qper : (nqper,) ndarray
      Gradient of the function with respect to `var_cos_qper`.
    grad_var_sin_qper : (nqper,) ndarray
      Gradient of the function with respect to `var_sin_qper`.
    grad_lambda_qper : (nqper,) ndarray
      Gradient of the function with respect to `lambda_qper`.
    grad_nu_qper : (nqper,) ndarray
      Gradient of the function with respect to `nu_qper`.
    """

    grad_var_jitter = np.sum(self._grad_A)

    grad_var_jitter_inst = self._dAdvar_inst.dot(self._grad_A)
    grad_var_calib_inst = (self._dFdvar_calib_inst.dot(self._grad_F)
      + self._dAdvar_inst.dot(self._grad_A))

    grad_var_exp = grad_var_jitter + np.sum(self._grad_U[:,:self.nexp], axis=0)
    grad_lambda_exp = np.array([
      -np.sum(self._dt * self.phi[:,k] * self._grad_phi[:,k])
      for k in range(self.nexp)])

    grad_var_cos_qper = grad_var_jitter + np.sum(
      self.V[:,self.nexp:self.nexp+self.nqper] *
      self._grad_U[:,self.nexp:self.nexp+self.nqper] +
      self.V[:,self.nexp+self.nqper:] *
      self._grad_U[:,self.nexp+self.nqper:],
      axis=0)
    grad_var_sin_qper = np.sum(
      self.V[:,self.nexp+self.nqper:] *
      self._grad_U[:,self.nexp:self.nexp+self.nqper] -
      self.V[:,self.nexp:self.nexp+self.nqper] *
      self._grad_U[:,self.nexp+self.nqper:],
      axis=0)
    grad_lambda_qper = np.array([
      -np.sum(self._dt * self.phi[:,self.nexp+k] *
        (self._grad_phi[:,self.nexp+k]
          + self._grad_phi[:,k-self.nqper]))
      for k in range(self.nqper)])
    grad_nu_qper = np.array([
      np.sum(self.t * (
        self.U[:,self.nexp+k] * self._grad_U[:,k-self.nqper]
        - self.U[:,k-self.nqper] * self._grad_U[:,self.nexp+k]
        + self.V[:,self.nexp+k] * self._grad_V[:,k-self.nqper]
        - self.V[:,k-self.nqper] * self._grad_V[:,self.nexp+k]))
      for k in range(self.nqper)])

    return(grad_var_jitter,
      grad_var_jitter_inst, grad_var_calib_inst,
      grad_var_exp, grad_lambda_exp,
      grad_var_cos_qper, grad_var_sin_qper, grad_lambda_qper, grad_nu_qper)
