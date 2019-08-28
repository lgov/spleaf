import numpy as np
from scipy.optimize import minimize

def read_rdb(filename):
  r"""
  Read a rdb file
  and return the dictionnary of its columns (as numpy arrays).
  """

  with open(filename, 'r') as rdbfile:
    header = rdbfile.readline()
  data = np.genfromtxt(filename, delimiter='\t', skip_header=2,
    comments='#', dtype=None, encoding=None)
  datadic = {}
  for colid, colname in enumerate(header.split()):
    datadic[colname] = np.array([line[colid] for line in data],
      dtype=data.dtype[colid])
  return(datadic)

def periodogram(rv, cov, M0, nu0_rad_d, dnu_rad_d, nfreq):
  r"""
  Compute the periodogram of the rv timeseries, with covariance cov,
  linear base model M0.
  A total of nfreq frequencies are sampled
  starting from nu0_rad_d and with a spacing of dnu_rad_d.
  """

  N0t = np.array([cov.solveL(M0k)/cov.sqD() for M0k in M0])
  u = cov.solveL(rv)/cov.sqD()
  u2 = np.sum(u*u)
  N0tu = N0t@u
  chi20 = u2 - N0tu.T@np.linalg.inv(N0t@N0t.T)@N0tu
  nu_rad_d = nu0_rad_d + np.arange(nfreq)*dnu_rad_d
  chi2 = np.empty(nfreq)
  dnut_rad = dnu_rad_d*cov.t
  cosdnut = np.cos(dnut_rad)
  sindnut = np.sin(dnut_rad)
  nu0t_rad = nu0_rad_d*cov.t
  cosnut = np.cos(nu0t_rad)
  sinnut = np.sin(nu0t_rad)
  Nt = np.vstack(([cov.solveL(cosnut)/cov.sqD(),
    cov.solveL(sinnut)/cov.sqD()], N0t))
  Ntu = Nt@u
  chi2[0] = u2 - Ntu.T@np.linalg.inv(Nt@Nt.T)@Ntu
  for kfreq in range(1, nfreq):
    cosnut, sinnut = (cosnut*cosdnut-sinnut*sindnut,
      sinnut*cosdnut+cosnut*sindnut)
    Nt[0] = cov.solveL(cosnut)/cov.sqD()
    Nt[1] = cov.solveL(sinnut)/cov.sqD()
    Ntu[0] = Nt[0]@u
    Ntu[1] = Nt[1]@u
    chi2[kfreq] = u2 - Ntu.T@np.linalg.inv(Nt@Nt.T)@Ntu
  power = 1.0 - chi2/chi20
  return(nu_rad_d, power)

def calc_Teff(cov, numax_rad_d):
  r"""
  Compute the effective timeseries length for a covariance cov
  and a maximum frequency numax_rad_d.
  """

  W = cov.expandInv()
  sinc = np.array([np.sinc(numax_rad_d*(cov.t-ti)) for ti in cov.t])
  Wsinc = W*sinc
  Wsinct = Wsinc@cov.t
  q = np.sum(Wsinc)
  s = np.sum(Wsinct)
  r = cov.t@Wsinct
  return(2.0*np.sqrt(np.pi*(r/q-(s/q)**2)))

def fap(zmax, cov, M0, numax_rad_d, Teff=None):
  r"""
  Compute the FAP for normalized power zmax, a covariance cov,
  a linear base model M0, and a maximum frequency numax_rad_d.
  If Teff is not provided, calc_Teff is called to obtain its value.
  """

  Nh = cov.n - M0.shape[0]
  Nk = Nh - 2
  fmax = numax_rad_d/(2.0*np.pi)
  if Teff is None:
    Teff = calc_Teff(cov, numax_rad_d)
  W = fmax * Teff
  chi2ratio = 1.0 - zmax
  FapSingle = chi2ratio**(Nk/2.0)
  tau = W * FapSingle * np.sqrt(Nh*zmax/(2.0*chi2ratio))
  Fap = FapSingle + tau
  if Fap > 1e-5:
    Fap = 1.0 - (1.0 - FapSingle) * np.exp(-tau)
  return(Fap)

def fit(rv, cov, covparamlist, M, xlin0):
  r"""
  Maximize the likelihood by adjusting the linear parameters (model matrix M)
  and the noise parameters listed in covparamlist.
  The linear parameters are initialized at xlin0,
  and the noise parameters at their current value in cov.
  The L-BFGS-B method is used to maximize the likelihood.
  """

  nlin = xlin0.size
  covdic = {param: np.array([getattr(cov, param)]).flatten()
    for param in covparamlist}
  covdic['copy'] = True
  covsize = {param: covdic[param].size for param in covparamlist}
  x0 = np.concatenate((xlin0, *(covdic[param] for param in covparamlist)))
  # Define function to minimize (-loglikelihood)
  def func(x):
    xlin = x[:nlin]
    k0 = nlin
    for param in covparamlist:
      covdic[param] = x[k0:k0+covsize[param]]
      k0 += covsize[param]
    cov.update_param(**covdic)
    res = rv - xlin.dot(M)
    ll = cov.loglike(res)
    return(-ll)
  # Define Jacobian of func
  def jac(x):
    xlin = x[:nlin]
    k0 = nlin
    for param in covparamlist:
      covdic[param] = x[k0:k0+covsize[param]]
      k0 += covsize[param]
    cov.update_param(**covdic)
    res = rv - xlin.dot(M)
    cov.loglike(res)
    ( grad_res,
      grad_var_jitter,
      grad_var_jitter_inst, grad_var_calib_inst,
      grad_var_exp, grad_lambda_exp,
      grad_var_cos_qper, grad_var_sin_qper,
      grad_lambda_qper, grad_nu_qper ) = cov.loglike_grad()
    grad_noise = {
      'var_jitter': np.array([grad_var_jitter]),
      'var_jitter_inst': grad_var_jitter_inst,
      'var_calib_inst': grad_var_calib_inst,
      'var_exp': grad_var_exp,
      'lambda_exp': grad_lambda_exp,
      'var_cos_qper': grad_var_cos_qper,
      'var_sin_qper': grad_var_sin_qper,
      'lambda_qper': grad_lambda_qper,
      'nu_qper': grad_nu_qper
    }
    J = -M.dot(grad_res)
    for param in covparamlist:
      J = np.concatenate((J, grad_noise[param]))
    return(-J)
  # Set bounds for each parameter
  bounds = [(None,None)]*nlin
  for param in covparamlist:
    if param.startswith('var_'):
      bounds += [(0,None)]*covsize[param]
    else:
      bounds += [(None,None)]*covsize[param]
  # Call minimize (with L-BFGS-B method)
  result = minimize(func, x0, jac=jac, method='L-BFGS-B', bounds=bounds)
  # Update cov with the results
  if result.success:
    x = result.x
  else:
    print('WARNING: fit did not converge.')
    x = x0
  k0 = nlin
  for param in covparamlist:
    covdic[param] = x[k0:k0+covsize[param]]
    k0 += covsize[param]
  cov.update_param(**covdic)
  return(x[:nlin])
