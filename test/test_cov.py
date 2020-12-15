import pytest
import numpy as np
from spleaf.cov import Cov
from spleaf.term import *

prec = 1e-10
n = 143
ninst = 3
calibmax = 12
calibprob = 0.8
nexp = 1
nqper = 1
nmat32 = 1
nusho = 1
nosho = 1
nsho = 1
delta = 1e-5

def _generate_random_C(seed=0, deriv=False):
  np.random.seed(seed)
  t = np.cumsum(10**np.random.uniform(-2,2,n))
  sig_err = np.random.uniform(0.5, 1.5, n)
  sig_jitter = np.random.uniform(0.5, 1.5)
  inst_id = np.random.randint(0,ninst,n)
  sig_jitter_inst = np.random.uniform(0.5, 1.5, ninst)
  calib_file = np.empty(n, dtype=object)
  sig_calib_meas = np.empty(n)
  lastfileinst = ["" for _ in range(ninst)]
  lastvarinst = [0 for _ in range(ninst)]
  nlastinst = [0 for _ in range(ninst)]
  for k in range(n):
    i = inst_id[k]
    if lastfileinst[i] == "" or nlastinst[i] == calibmax or np.random.rand() > calibprob:
      calib_file[k] = '{}'.format(k)
      sig_calib_meas[k] = np.random.uniform(0.5, 1.5)
      lastfileinst[i] = calib_file[k]
      lastvarinst[i] = sig_calib_meas[k]
      nlastinst[i] = 1
    else:
      calib_file[k] = lastfileinst[i]
      sig_calib_meas[k] = lastvarinst[i]
      nlastinst[i] += 1
  sig_calib_inst = np.random.uniform(0.5, 1.5, ninst)
  if not deriv:
    a_exp = np.random.uniform(0.5, 1.5, nexp)
    la_exp = 10**np.random.uniform(-2, 2, nexp)
    a_qper = np.random.uniform(0.5, 1.5, nqper)
    b_qper = np.random.uniform(0.05, 0.15, nqper)
    la_qper = 10**np.random.uniform(-2, 2, nqper)
    nu_qper = 10**np.random.uniform(-2, 2, nqper)
  sig_mat32 = np.random.uniform(0.5, 1.5, nmat32)
  rho_mat32 = 10**np.random.uniform(-2, 2, nmat32)
  sig_usho = np.random.uniform(0.5, 1.5, nusho)
  P0_usho = 10**np.random.uniform(-2, 2, nusho)
  Q_usho = np.random.uniform(0.5, 20.0, nusho)
  sig_osho = np.random.uniform(0.5, 1.5, nosho)
  P0_osho = 10**np.random.uniform(-2, 2, nosho)
  Q_osho = np.random.uniform(0.01, 0.5, nosho)
  sig_sho = np.random.uniform(0.5, 1.5, nsho)
  P0_sho = 10**np.random.uniform(-2, 2, nsho)
  Q_sho = np.random.uniform(0.01, 2.0, nsho)

  if deriv:
    return(Cov(t,
      err=Error(sig_err),
      jit=Jitter(sig_jitter),
      **{f'insjit_{k}':InstrumentJitter(inst_id==k, sig_jitter_inst[k]) for k in range(ninst)},
      calerr=CalibrationError(calib_file, sig_calib_meas),
      **{f'caljit_{k}':CalibrationJitter(inst_id==k, calib_file, sig_calib_inst[k]) for k in range(ninst)},
      **{f'mat32_{k}':Matern32Kernel(sig_mat32[k], rho_mat32[k]) for k in range(nmat32)},
      **{f'usho_{k}':USHOKernel(sig_usho[k], P0_usho[k], Q_usho[k]) for k in range(nusho)},
      **{f'osho_{k}':OSHOKernel(sig_osho[k], P0_osho[k], Q_osho[k]) for k in range(nosho)},
      **{f'sho_{k}':SHOKernel(sig_sho[k], P0_sho[k], Q_sho[k]) for k in range(nsho)}))
  else:
    return(Cov(t,
      err=Error(sig_err),
      jit=Jitter(sig_jitter),
      **{f'insjit_{k}':InstrumentJitter(inst_id==k, sig_jitter_inst[k]) for k in range(ninst)},
      calerr=CalibrationError(calib_file, sig_calib_meas),
      **{f'caljit_{k}':CalibrationJitter(inst_id==k, calib_file, sig_calib_inst[k]) for k in range(ninst)},
      **{f'exp_{k}':ExponentialKernel(a_exp[k], la_exp[k]) for k in range(nexp)},
      **{f'qper_{k}':QuasiperiodicKernel(a_qper[k], b_qper[k], la_qper[k], nu_qper[k]) for k in range(nqper)},
      **{f'mat32_{k}':Matern32Kernel(sig_mat32[k], rho_mat32[k]) for k in range(nmat32)},
      **{f'usho_{k}':USHOKernel(sig_usho[k], P0_usho[k], Q_usho[k]) for k in range(nusho)},
      **{f'osho_{k}':OSHOKernel(sig_osho[k], P0_osho[k], Q_osho[k]) for k in range(nosho)},
      **{f'sho_{k}':SHOKernel(sig_sho[k], P0_sho[k], Q_sho[k]) for k in range(nsho)}))

def _generate_random_param(seed=1):
  np.random.seed(seed)
  sig_jitter = np.random.uniform(0.5, 1.5, 1)
  sig_jitter_inst = np.random.uniform(0.5, 1.5, ninst)
  sig_calib_inst = np.random.uniform(0.5, 1.5, ninst)
  a_exp = np.random.uniform(0.5, 1.5, nexp)
  la_exp = 10**np.random.uniform(-2, 2, nexp)
  a_qper = np.random.uniform(0.5, 1.5, nqper)
  b_qper = np.random.uniform(0.05, 0.15, nqper)
  la_qper = 10**np.random.uniform(-2, 2, nqper)
  nu_qper = 10**np.random.uniform(-2, 2, nqper)
  sig_mat32 = np.random.uniform(0.5, 1.5, nmat32)
  rho_mat32 = 10**np.random.uniform(-2, 2, nmat32)
  sig_usho = np.random.uniform(0.5, 1.5, nusho)
  P0_usho = 10**np.random.uniform(-2, 2, nusho)
  Q_usho = np.random.uniform(0.5, 20.0, nusho)
  sig_osho = np.random.uniform(0.5, 1.5, nosho)
  P0_osho = 10**np.random.uniform(-2, 2, nosho)
  Q_osho = np.random.uniform(0.01, 0.5, nosho)
  sig_sho = np.random.uniform(0.5, 1.5, nsho)
  P0_sho = 10**np.random.uniform(-2, 2, nsho)
  Q_sho = np.random.uniform(0.01, 2.0, nsho)

  return(sig_jitter, sig_jitter_inst, sig_calib_inst,
    a_exp, la_exp, a_qper, b_qper, la_qper, nu_qper,
    sig_mat32, rho_mat32, sig_usho, P0_usho, Q_usho,
    sig_osho, P0_osho, Q_osho, sig_sho, P0_sho, Q_sho)

def test_Cov():
  C = _generate_random_C()

  C_full = C.expand()
  L_full = C.expandL()
  D_full = np.diag(C.D)

  LDLt_full = L_full@D_full@L_full.T
  err = np.max(np.abs(C_full-LDLt_full))
  assert err < prec, ('Cholesky decomposition not working'
    ' at required precision ({} > {})').format(err, prec)

def test_set_param():
  C = _generate_random_C()
  param = list(_generate_random_param())
  Cb = Cov(C.t,
    err=Error(C.term['err']._sig),
    jit=Jitter(param[0][0]),
    **{f'insjit_{k}':InstrumentJitter(C.term[f'insjit_{k}']._indices, param[1][k]) for k in range(ninst)},
    calerr=CalibrationError(C.term['calerr']._calib_id, C.term['calerr']._sig),
    **{f'caljit_{k}':CalibrationJitter(
      C.term[f'insjit_{k}']._indices,
      C.term['calerr']._calib_id,
      param[2][k]) for k in range(ninst)},
    **{f'exp_{k}':ExponentialKernel(param[3][k], param[4][k]) for k in range(nexp)},
    **{f'qper_{k}':QuasiperiodicKernel(param[5][k], param[6][k], param[7][k], param[8][k]) for k in range(nqper)},
    **{f'mat32_{k}':Matern32Kernel(param[9][k], param[10][k]) for k in range(nmat32)},
    **{f'usho_{k}':USHOKernel(param[11][k], param[12][k], param[13][k]) for k in range(nusho)},
    **{f'osho_{k}':OSHOKernel(param[14][k], param[15][k], param[16][k]) for k in range(nosho)},
    **{f'sho_{k}':SHOKernel(param[17][k], param[18][k], param[19][k]) for k in range(nsho)})

  C.set_param(
    np.concatenate(param),
    ['jit.sig']
    + [f'insjit_{k}.sig' for k in range(ninst)]
    + [f'caljit_{k}.sig' for k in range(ninst)]
    + [f'exp_{k}.a' for k in range(nexp)]
    + [f'exp_{k}.la' for k in range(nexp)]
    + [f'qper_{k}.a' for k in range(nqper)]
    + [f'qper_{k}.b' for k in range(nqper)]
    + [f'qper_{k}.la' for k in range(nqper)]
    + [f'qper_{k}.nu' for k in range(nqper)]
    + [f'mat32_{k}.sig' for k in range(nmat32)]
    + [f'mat32_{k}.rho' for k in range(nmat32)]
    + [f'usho_{k}.sig' for k in range(nusho)]
    + [f'usho_{k}.P0' for k in range(nusho)]
    + [f'usho_{k}.Q' for k in range(nusho)]
    + [f'osho_{k}.sig' for k in range(nosho)]
    + [f'osho_{k}.P0' for k in range(nosho)]
    + [f'osho_{k}.Q' for k in range(nosho)]
    + [f'sho_{k}.sig' for k in range(nsho)]
    + [f'sho_{k}.P0' for k in range(nsho)]
    + [f'sho_{k}.Q' for k in range(nsho)]
  )

  C_full = C.expand()
  Cb_full = Cb.expand()
  L_full = C.expandL()
  Lb_full = Cb.expandL()

  err = np.max(np.abs(C_full-Cb_full))
  err = max(err, np.max(np.abs(L_full-Lb_full)))
  err = max(err, np.max(np.abs(C.D-Cb.D)))

  assert err < prec, ('set_param not working'
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
  for kparam, param in enumerate(C.param):
    grad_param_num = []
    Cparam = C.get_param(param)
    deltaparam = delta*max(delta, abs(Cparam))
    for dx in [deltaparam, -deltaparam]:
      C.set_param([Cparam+dx], [param])
      db = getattr(C, method)(a) - b
      grad_param_num.append(db@grad_b/dx)
    C.set_param([Cparam], [param])
    err = np.max(np.abs(grad_param[kparam].flat-np.mean(grad_param_num)))
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
  f_grad_res, f_grad_param = getattr(C, method+'_grad')()

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
  err = np.max(np.abs(f_grad_res-np.mean(f_grad_num, axis=0)))
  num_err = np.max(np.abs(f_grad_num[1]-f_grad_num[0]))
  err = max(0.0, err-num_err)
  assert err < prec, ('{}_grad (y) not working'
    ' at required precision ({} > {})').format(method, err, prec)

  # grad_param
  for kparam, param in enumerate(C.param):
    f_grad_num = []
    Cparam = C.get_param(param)
    deltaparam = delta*max(delta, abs(Cparam))
    for dx in [deltaparam, -deltaparam]:
      C.set_param([Cparam+dx], [param])
      df = getattr(C, method)(y) - f
      f_grad_num.append(df/dx)
    C.set_param([Cparam], [param])
    err = np.max(np.abs(f_grad_param[kparam].flat-np.mean(f_grad_num)))
    num_err = np.max(np.abs(f_grad_num[1]-f_grad_num[0]))
    err = max(0.0, err-num_err)
    assert err < prec, ('{}_grad ({}) not working'
      ' at required precision ({} > {})').format(method, param, err, prec)

def test_chi2_grad():
  _test_method_grad('chi2')

def test_loglike_grad():
  _test_method_grad('loglike')

def test_self_conditional():
  C = _generate_random_C()
  y = C.dotL(np.random.normal(0.0, C.sqD()))

  mu = C.self_conditional(y)
  muv, var = C.self_conditional(y, calc_cov='diag')
  muc, cov = C.self_conditional(y, calc_cov=True)

  invC_full = C.expandInv()
  invCy_full = invC_full.dot(y)
  term = {}
  for key in C.kernel:
    term[key] = C.kernel[key].__class__(*[getattr(C.kernel[key], f'_{param}') for param in C.kernel[key]._param])
  K = Cov(C.t, **term)
  K_full = K.expand()
  mu_full = K_full@invCy_full
  cov_full = K_full - K_full@invC_full@K_full
  var_full = np.diag(cov_full)

  err = np.max(np.abs(mu-mu_full))
  assert err < prec, ('conditional not working'
    ' at required precision ({} > {})').format(err, prec)

  err = np.max(np.abs(muv-mu_full))
  assert err < prec, ('conditional not working'
    ' at required precision ({} > {})').format(err, prec)

  err = np.max(np.abs(muc-mu_full))
  assert err < prec, ('conditional not working'
    ' at required precision ({} > {})').format(err, prec)

  err = np.max(np.abs(var-var_full))
  assert err < prec, ('conditional not working'
    ' at required precision ({} > {})').format(err, prec)

  err = np.max(np.abs(cov-cov_full))
  assert err < prec, ('conditional not working'
    ' at required precision ({} > {})').format(err, prec)

def test_conditional():
  C = _generate_random_C()
  y = C.dotL(np.random.normal(0.0, C.sqD()))

  n2 = 300
  Dt = C.t[-1] - C.t[0]
  margin = Dt/10
  t2 = np.linspace(C.t[0]-margin, C.t[-1]+margin, n2)
  mu = C.conditional(y, t2)
  muv, var = C.conditional(y, t2, calc_cov='diag')
  muc, cov = C.conditional(y, t2, calc_cov=True)

  invC_full = C.expandInv()
  invCy_full = invC_full.dot(y)
  Km_full = C.eval(t2[:, None] - C.t[None, :])
  term = {}
  for key in C.kernel:
    term[key] = C.kernel[key].__class__(*[getattr(C.kernel[key], f'_{param}') for param in C.kernel[key]._param])
  K = Cov(t2, **term)
  K_full = K.expand()
  mu_full = Km_full@invCy_full
  cov_full = K_full - Km_full@invC_full@Km_full.T
  var_full = np.diag(cov_full)

  err = np.max(np.abs(mu-mu_full))
  assert err < prec, ('conditional not working'
    ' at required precision ({} > {})').format(err, prec)

  err = np.max(np.abs(muv-mu_full))
  assert err < prec, ('conditional not working'
    ' at required precision ({} > {})').format(err, prec)

  err = np.max(np.abs(muc-mu_full))
  assert err < prec, ('conditional not working'
    ' at required precision ({} > {})').format(err, prec)

  err = np.max(np.abs(var-var_full))
  assert err < prec, ('conditional not working'
    ' at required precision ({} > {})').format(err, prec)

  err = np.max(np.abs(cov-cov_full))
  assert err < prec, ('conditional not working'
    ' at required precision ({} > {})').format(err, prec)

def test_self_conditional_derivative():
  C = _generate_random_C(deriv=True)
  y = C.dotL(np.random.normal(0.0, C.sqD()))

  dmu = C.self_conditional_derivative(y)
  dmuv, dvar = C.self_conditional_derivative(y, calc_cov='diag')
  dmuc, dcov = C.self_conditional_derivative(y, calc_cov=True)

  num_dmu = []
  num_dcov = []
  for dt in [delta, -2*delta]:
    tfull = np.sort(np.concatenate((C.t, C.t+dt)))
    mu, cov = C.conditional(y, tfull, calc_cov=True)
    num_dmu.append((mu[1::2]-mu[::2])/abs(dt))
    num_dcov.append((cov[1::2,1::2]+cov[::2,::2]-cov[1::2,::2]-cov[::2,1::2])/dt**2)

  num_dmu_mean = (num_dmu[0]+num_dmu[1])/2
  num_dmu_err = np.max(np.abs(num_dmu[0]-num_dmu[1]))

  num_dcov_mean = (num_dcov[0]+num_dcov[1])/2
  num_dcov_err = np.max(np.abs(num_dcov[0]-num_dcov[1]))

  num_dvar_mean = num_dcov_mean.diagonal()
  num_dvar_err = np.max(np.abs(num_dcov[0].diagonal()-num_dcov[1].diagonal()))

  err = np.max(np.abs(dmu-num_dmu_mean))
  err = max(0.0, err-num_dmu_err)
  assert err < prec, ('self_conditional_derivative not working'
    ' at required precision ({} > {})').format(err, prec)

  err = np.max(np.abs(dmuv-num_dmu))
  err = max(0.0, err-num_dmu_err)
  assert err < prec, ('self_conditional_derivative not working'
    ' at required precision ({} > {})').format(err, prec)

  err = np.max(np.abs(dmuc-num_dmu))
  err = max(0.0, err-num_dmu_err)
  assert err < prec, ('self_conditional_derivative not working'
    ' at required precision ({} > {})').format(err, prec)

  err = np.max(np.abs(dvar-num_dvar_mean))
  err = max(0.0, err-num_dvar_err)
  assert err < prec, ('self_conditional_derivative not working'
    ' at required precision ({} > {})').format(err, prec)

  err = np.max(np.abs(dcov-num_dcov_mean))
  err = max(0.0, err-num_dcov_err)
  assert err < prec, ('self_conditional_derivative not working'
    ' at required precision ({} > {})').format(err, prec)

def test_conditional_derivative():
  C = _generate_random_C(deriv=True)
  y = C.dotL(np.random.normal(0.0, C.sqD()))

  n2 = 1001
  Dt = C.t[-1] - C.t[0]
  margin = Dt/10
  t2 = np.linspace(C.t[0]-margin, C.t[-1]+margin, n2)

  dmu = C.conditional_derivative(y, t2)
  dmuv, dvar = C.conditional_derivative(y, t2, calc_cov='diag')
  dmuc, dcov = C.conditional_derivative(y, t2, calc_cov=True)

  num_dmu = []
  num_dcov = []
  for dt in [delta, -2*delta]:
    tfull = np.sort(np.concatenate((t2, t2+dt)))
    mu, cov = C.conditional(y, tfull, calc_cov=True)
    num_dmu.append((mu[1::2]-mu[::2])/abs(dt))
    num_dcov.append((cov[1::2,1::2]+cov[::2,::2]-cov[1::2,::2]-cov[::2,1::2])/dt**2)

  num_dmu_mean = (num_dmu[0]+num_dmu[1])/2
  num_dmu_err = np.max(np.abs(num_dmu[0]-num_dmu[1]))

  num_dcov_mean = (num_dcov[0]+num_dcov[1])/2
  num_dcov_err = np.max(np.abs(num_dcov[0]-num_dcov[1]))

  num_dvar_mean = num_dcov_mean.diagonal()
  num_dvar_err = np.max(np.abs(num_dcov[0].diagonal()-num_dcov[1].diagonal()))

  err = np.max(np.abs(dmu-num_dmu_mean))
  print(err, num_dmu_err)
  err = max(0.0, err-num_dmu_err)
  assert err < prec, ('conditional_derivative not working'
    ' at required precision ({} > {})').format(err, prec)

  err = np.max(np.abs(dmuv-num_dmu))
  err = max(0.0, err-num_dmu_err)
  assert err < prec, ('conditional_derivative not working'
    ' at required precision ({} > {})').format(err, prec)

  err = np.max(np.abs(dmuc-num_dmu))
  err = max(0.0, err-num_dmu_err)
  assert err < prec, ('conditional_derivative not working'
    ' at required precision ({} > {})').format(err, prec)

  err = np.max(np.abs(dvar-num_dvar_mean))
  print(err, num_dvar_err)
  err = max(0.0, err-num_dvar_err)
  assert err < prec, ('conditional_derivative not working'
    ' at required precision ({} > {})').format(err, prec)

  err = np.max(np.abs(dcov-num_dcov_mean))
  print(err, num_dcov_err)
  err = max(0.0, err-num_dcov_err)
  assert err < prec, ('conditional_derivative not working'
    ' at required precision ({} > {})').format(err, prec)