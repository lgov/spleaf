import pytest
import numpy as np
from spleaf import Spleaf

prec = 1e-12
n = 143
r = 11
bmax = 11
delta = 1e-7

def _generate_random_C(seed=0):
  np.random.seed(seed)

  k = np.arange(n)

  U = np.random.uniform(0.5, 1.5, (n, r))
  V = np.random.uniform(0.5, 1.5, (n, r))
  log10phi = np.random.uniform(-4,0,(n-1,r))
  phi = 10**log10phi

  b = np.minimum(k, np.random.randint(bmax+1))
  offsetrow = np.cumsum(b-1)+1
  nF = offsetrow[-1]+n-1
  F = np.random.uniform(0.25, 0.5, nF)**2

  A = np.random.uniform(1.5, 2.5, n)**2 + np.sum(U*V,axis=1)

  return(Spleaf(A, U, V, phi, offsetrow, b, F))

def _generate_random_param(C, seed=1):
  np.random.seed(seed)

  k = np.arange(n)

  U = np.random.uniform(0.5, 1.5, (n, r))
  V = np.random.uniform(0.5, 1.5, (n, r))
  log10phi = np.random.uniform(-4,0,(n-1,r))
  phi = 10**log10phi

  nF = C.offsetrow[-1]+n-1
  F = np.random.uniform(0.25, 0.5, nF)**2

  A = np.random.uniform(1.5, 2.5, n)**2 + np.sum(U*V,axis=1)

  return(A, U, V, phi, F)

def test_Spleaf():
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
  A, U, V, phi, F = _generate_random_param(C)
  Cb = Spleaf(A, U, V, phi, C.offsetrow, C.b, F)
  C.update_param(A, U, V, phi, F)

  C_full = C.expand()
  Cb_full = Cb.expand()
  L_full = C.expandL()
  Lb_full = Cb.expandL()

  err = np.max(np.abs(C_full-Cb_full))
  err = max(err, np.max(np.abs(L_full-Lb_full)))
  err = max(err, np.max(np.abs(C.D-Cb.D)))

  assert err < prec, ('update_param not working'
    ' at required precision ({} > {})').format(err, prec)

def test_expandInv():
  C = _generate_random_C()

  C_full = C.expand()
  invC_full = C.expandInv()

  CinvC_full = C_full@invC_full
  err = np.max(np.abs(CinvC_full-np.identity(n)))
  assert err < prec, ('Inversion not working'
    ' at required precision ({} > {})').format(err, prec)

def test_expandInvL():
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

def test_dotLT():
  C = _generate_random_C()
  x = np.random.normal(0.0, 1.0, C.n)

  y = C.dotLT(x)

  L_full = C.expandL()
  y_full = L_full.T.dot(x)

  err = np.max(np.abs(y-y_full))

  assert err < prec, ('dotL not working'
    ' at required precision ({} > {})').format(err, prec)

def test_solveLT():
  C = _generate_random_C()
  y = np.random.normal(0.0, 5.0, C.n)

  x = C.solveLT(y)

  L_full = C.expandL()
  x_full = np.linalg.solve(L_full.T, y)

  err = np.max(np.abs(x-x_full))

  assert err < prec, ('solveL not working'
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
  kwargs = {'A': C.A, 'U': C.U, 'V': C.V, 'phi': C.phi, 'F': C.F}
  for kparam, param in enumerate(['A', 'U', 'V', 'phi', 'F']):
    grad_param_num = []
    for dx in [delta, -delta]:
      grad_param_num_dx = []
      Cparam = getattr(C, param).copy()
      for k in range(Cparam.size):
        Cparam.flat[k] += dx
        kwargs[param] = Cparam.copy()
        C.update_param(**kwargs)
        db = getattr(C, method)(a) - b
        grad_param_num_dx.append(db@grad_b/dx)
        Cparam.flat[k] -= dx
      kwargs[param] = Cparam
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
  kwargs = {'A': C.A, 'U': C.U, 'V': C.V, 'phi': C.phi, 'F': C.F}
  for kparam, param in enumerate(['A', 'U', 'V', 'phi', 'F']):
    f_grad_num = []
    for dx in [delta, -delta]:
      f_grad_num_dx = []
      Cparam = getattr(C, param).copy()
      for k in range(Cparam.size):
        Cparam.flat[k] += dx
        kwargs[param] = Cparam.copy()
        C.update_param(**kwargs)
        df = getattr(C, method)(y) - f
        f_grad_num_dx.append(df/dx)
        Cparam.flat[k] -= dx
      kwargs[param] = Cparam
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
