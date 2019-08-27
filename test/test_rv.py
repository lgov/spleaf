import pytest
import numpy as np
from spleaf.rv import Cov

prec = 1e-12
n = 143
ninst = 3
calibmax = 12
calibprob = 0.8
nexp = 2
nqper = 4
delta = 1e-5

def _generate_random_C(seed=0):

  np.random.seed(seed)

  k = np.arange(n)

  t = np.cumsum(10**np.random.uniform(-2,2,n))
  var_photon = np.random.uniform(0.5, 1.5, n)
  var_jitter = np.random.uniform(0.5, 1.5)
  inst_id = np.random.randint(0,ninst,n)
  var_jitter_inst = np.random.uniform(0.5, 1.5, ninst)
  calib_file = np.empty(n, dtype=str)
  var_calib_meas = np.empty(n)
  lastfileinst = ["" for _ in range(ninst)]
  lastvarinst = [0 for _ in range(ninst)]
  nlastinst = [0 for _ in range(ninst)]
  for k in range(n):
    i = inst_id[k]
    if lastfileinst[i] == "" or nlastinst[i] == calibmax or np.random.rand() > calibprob:
      calib_file[k] = '{}'.format(k)
      var_calib_meas[k] = np.random.uniform(0.5, 1.5)
      lastfileinst[i] = calib_file[k]
      lastvarinst[i] = var_calib_meas[k]
      nlastinst[i] = 1
    else:
      calib_file[k] = lastfileinst[i]
      var_calib_meas[k] = lastvarinst[i]
      nlastinst[i] += 1
  var_calib_inst = np.random.uniform(0.5, 1.5, ninst)
  var_exp = np.random.uniform(0.5, 1.5, nexp)
  lambda_exp = 10**np.random.uniform(-2, 2, nexp)
  var_cos_qper = np.random.uniform(0.5, 1.5, nqper)
  var_sin_qper = np.random.uniform(0.05, 0.15, nqper)
  lambda_qper = 10**np.random.uniform(-2, 2, nqper)
  nu_qper = 10**np.random.uniform(-2, 2, nqper)

  return(Cov(t, var_photon, var_jitter,
    inst_id, var_jitter_inst, calib_file, var_calib_meas, var_calib_inst,
    var_exp, lambda_exp, var_cos_qper, var_sin_qper, lambda_qper, nu_qper))

def _generate_random_param(seed=1):

  np.random.seed(seed)

  k = np.arange(n)

  var_jitter = np.random.uniform(0.5, 1.5)
  var_jitter_inst = np.random.uniform(0.5, 1.5, ninst)
  var_calib_inst = np.random.uniform(0.5, 1.5, ninst)
  var_exp = np.random.uniform(0.5, 1.5, nexp)
  lambda_exp = 10**np.random.uniform(-2, 2, nexp)
  var_cos_qper = np.random.uniform(0.5, 1.5, nqper)
  var_sin_qper = np.random.uniform(0.05, 0.15, nqper)
  lambda_qper = 10**np.random.uniform(-2, 2, nqper)
  nu_qper = 10**np.random.uniform(-2, 2, nqper)

  return(var_jitter, var_jitter_inst, var_calib_inst,
    var_exp, lambda_exp, var_cos_qper, var_sin_qper, lambda_qper, nu_qper)

def test_Cov():
  C = _generate_random_C()

  C_full = C.expand()
  L_full = C.expandL()
  D_full = np.diag(C.D)

  LDLt_full = L_full@D_full@L_full.T
  err = np.max(np.abs(C_full-LDLt_full))
  assert err < prec, ('Cholesky decomposition not working'
    ' at required precision ({} > {})').format(err, prec)

def test_update_param():
  C = _generate_random_C()
  param = list(_generate_random_param())
  Cb = Cov(C.t, C.var_photon, param[0],
    C.inst_id, param[1],
    C.calib_file, C.var_calib_meas, param[2],
    *param[3:])
  C.update_param(*param)

  C_full = C.expand()
  Cb_full = Cb.expand()
  L_full = C.expandL()
  Lb_full = Cb.expandL()

  err = np.max(np.abs(C_full-Cb_full))
  err = max(err, np.max(np.abs(L_full-Lb_full)))
  err = max(err, np.max(np.abs(C.D-Cb.D)))

  assert err < prec, ('update_param not working'
    ' at required precision ({} > {})').format(err, prec)

def test_inv():
  C = _generate_random_C()

  C_full = C.expand()
  invC_full = C.expandInv()

  CinvC_full = C_full@invC_full
  err = np.max(np.abs(CinvC_full-np.identity(n)))
  assert err < prec, ('Inversion not working'
    ' at required precision ({} > {})').format(err, prec)

def test_invL():
  C = _generate_random_C()

  L_full = C.expandL()
  invL_full = C.expandInvL()

  LinvL_full = L_full@invL_full
  err = np.max(np.abs(LinvL_full-np.identity(n)))
  assert err < prec, ('Cholesky inversion not working'
    ' at required precision ({} > {})').format(err, prec)

def test_logdet():
  C = _generate_random_C()

  logdet = C.logdet()

  C_full = C.expand()
  sign_full, logdet_full = np.linalg.slogdet(C_full)

  err = abs(logdet/logdet_full-1)

  assert sign_full > 0, 'logdet is not positive'
  assert err < prec, ('logdet not working'
    ' at required precision ({} > {})').format(err, prec)

def test_dotL():
  C = _generate_random_C()
  x = np.random.normal(0.0, 1.0, C.n)

  y = C.dotL(x)

  L_full = C.expandL()
  y_full = L_full.dot(x)

  err = np.max(np.abs(y-y_full))

  assert err < prec, ('dotL not working'
    ' at required precision ({} > {})').format(err, prec)

def test_solveL():
  C = _generate_random_C()
  y = np.random.normal(0.0, 5.0, C.n)

  x = C.solveL(y)

  L_full = C.expandL()
  x_full = np.linalg.solve(L_full, y)

  err = np.max(np.abs(x-x_full))

  assert err < prec, ('solveL not working'
    ' at required precision ({} > {})').format(err, prec)

def test_chi2():
  C = _generate_random_C()
  y = np.random.normal(0.0, 5.0, C.n)

  chi2 = C.chi2(y)

  C_full = C.expand()
  invC_full = np.linalg.inv(C_full)
  chi2_full = y.T@invC_full@y

  err = abs(chi2-chi2_full)

  assert err < prec, ('chi2 not working'
    ' at required precision ({} > {})').format(err, prec)

def test_loglike():
  C = _generate_random_C()
  y = np.random.normal(0.0, 5.0, C.n)

  loglike = C.loglike(y)

  C_full = C.expand()
  invC_full = np.linalg.inv(C_full)
  chi2_full = y.T@invC_full@y
  _, logdet_full = np.linalg.slogdet(C_full)
  loglike_full = -0.5*(chi2_full + logdet_full + C.n*np.log(2.0*np.pi))

  err = abs(loglike-loglike_full)
  assert err < prec, ('loglike not working'
    ' at required precision ({} > {})').format(err, prec)

def _test_method_back(method):
  """
  Common code for testing dotL_back, solveL_back, dotLT_back, solveLT_back
  """
  C = _generate_random_C()
  a = np.random.normal(0.0, 5.0, C.n)
  grad_b = np.random.normal(0.0, 1.0, C.n)

  func = getattr(C, method)
  b = func(a)
  C.init_grad()
  grad_a = getattr(C, method+'_back')(grad_b)
  C.cholesky_back()
  grad_param = C.grad_param()

  # grad_a
  grad_a_num = []
  for dx in [delta, -delta]:
    grad_a_num_dx = []
    for k in range(C.n):
      a[k] += dx
      db = func(a) - b
      grad_a_num_dx.append(db@grad_b/dx)
      a[k] -= dx
    grad_a_num.append(grad_a_num_dx)
  grad_a_num = np.array(grad_a_num)
  err = np.max(np.abs(grad_a-np.mean(grad_a_num, axis=0)))
  num_err = np.max(np.abs(grad_a_num[1]-grad_a_num[0]))
  err = max(0.0, err-num_err)
  assert err < prec, ('{}_back (a) not working'
    ' at required precision ({} > {})').format(method, err, prec)

  # grad_param
  for kparam, param in enumerate([
    'var_jitter', 'var_jitter_inst', 'var_calib_inst',
    'var_exp', 'lambda_exp',
    'var_cos_qper', 'var_sin_qper', 'lambda_qper', 'nu_qper']):
    grad_param_num = []
    for dx in [delta, -delta]:
      grad_param_num_dx = []
      Cparam = getattr(C, param)
      if isinstance(Cparam, np.ndarray):
        Cparam = Cparam.copy()
        for k in range(Cparam.size):
          Cparam.flat[k] += dx
          kwargs = {param: Cparam.copy()}
          C.update_param(**kwargs)
          db = getattr(C, method)(a) - b
          grad_param_num_dx.append(db@grad_b/dx)
          Cparam.flat[k] -= dx
      else:
        Cparam += dx
        kwargs = {param: Cparam}
        C.update_param(**kwargs)
        db = getattr(C, method)(a) - b
        grad_param_num_dx.append(db@grad_b/dx)
        Cparam -= dx
      kwargs = {param: Cparam}
      C.update_param(**kwargs)
      grad_param_num.append(grad_param_num_dx)
    grad_param_num = np.array(grad_param_num)
    err = np.max(np.abs(grad_param[kparam].flat-np.mean(grad_param_num, axis=0)))
    num_err = np.max(np.abs(grad_param_num[1]-grad_param_num[0]))
    err = max(0.0, err-num_err)
    assert err < prec, ('{}_back ({}) not working'
      ' at required precision ({} > {})').format(method, param, err, prec)

def test_dotL_back():
  _test_method_back('dotL')

def test_solveL_back():
  _test_method_back('solveL')

def test_dotLT_back():
  _test_method_back('dotLT')

def test_solveLT_back():
  _test_method_back('solveLT')

def _test_method_grad(method):
  """
  Common code for testing chi2_grad, loglike_grad
  """
  C = _generate_random_C()
  y = np.random.normal(0.0, 5.0, C.n)

  func = getattr(C, method)
  f = func(y)
  f_grad = getattr(C, method+'_grad')()

  # grad_y
  f_grad_num = []
  for dx in [delta, -delta]:
    f_grad_num_dx = []
    for k in range(C.n):
      y[k] += dx
      df = func(y) - f
      f_grad_num_dx.append(df/dx)
      y[k] -= dx
    f_grad_num.append(f_grad_num_dx)
  f_grad_num = np.array(f_grad_num)
  err = np.max(np.abs(f_grad[0]-np.mean(f_grad_num, axis=0)))
  num_err = np.max(np.abs(f_grad_num[1]-f_grad_num[0]))
  err = max(0.0, err-num_err)
  assert err < prec, ('{}_grad (y) not working'
    ' at required precision ({} > {})').format(method, err, prec)

  # grad_param
  for kparam, param in enumerate([
    'var_jitter', 'var_jitter_inst', 'var_calib_inst',
    'var_exp', 'lambda_exp',
    'var_cos_qper', 'var_sin_qper', 'lambda_qper', 'nu_qper']):
    f_grad_num = []
    for dx in [delta, -delta]:
      f_grad_num_dx = []
      Cparam = getattr(C, param)
      if isinstance(Cparam, np.ndarray):
        Cparam = Cparam.copy()
        for k in range(Cparam.size):
          Cparam.flat[k] += dx
          kwargs = {param: Cparam.copy()}
          C.update_param(**kwargs)
          df = getattr(C, method)(y) - f
          f_grad_num_dx.append(df/dx)
          Cparam.flat[k] -= dx
      else:
        Cparam += dx
        kwargs = {param: Cparam}
        C.update_param(**kwargs)
        df = getattr(C, method)(y) - f
        f_grad_num_dx.append(df/dx)
        Cparam -= dx
      kwargs = {param: Cparam}
      C.update_param(**kwargs)
      f_grad_num.append(f_grad_num_dx)
    f_grad_num = np.array(f_grad_num)
    err = np.max(np.abs(f_grad[kparam+1].flat-np.mean(f_grad_num, axis=0)))
    num_err = np.max(np.abs(f_grad_num[1]-f_grad_num[0]))
    err = max(0.0, err-num_err)
    assert err < prec, ('{}_grad ({}) not working'
      ' at required precision ({} > {})').format(method, param, err, prec)

def test_chi2_grad():
  _test_method_grad('chi2')

def test_loglike_grad():
  _test_method_grad('loglike')
