// Copyright 2019 Jean-Baptiste Delisle
//
// This file is part of spleaf.
//
// spleaf is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// spleaf is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with spleaf.  If not, see <http://www.gnu.org/licenses/>.

#define NPY_NO_DEPRECATED_API NPY_1_16_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include "libspleaf.h"

// Module docstrings
static char module_docstring[] =
  "This module provides an interface for the C library libspleaf.";
static char spleaf_cholesky_docstring[] =
  "Cholesky decomposition of the (n x n) symmetric S+LEAF\n"
  "(semiseparable + leaf) matrix C\n"
  "defined as\n"
  "C = A + Sep + F\n"
  "with\n"
  "* A: the diagonal part of C, stored as a vector of size n.\n"
  "* Sep: the symmetric semiseparable part of C.\n"
  "  For i > j,\n"
  "  Sep_{i,j} = Sep_{j,i}\n"
  "            = Sum_{s=0}^{r-1} U_{i,s} V_{j,s} Prod_{k=j}^{i-1} phi_{k,s}\n"
  "  where U, V are (n x r) matrices, and phi is a (n-1 x r) matrix,\n"
  "  all stored in row major order.\n"
  "  By definition Sep_{i,i} = 0.\n"
  "* F: the symmetric leaf part of C,\n"
  "  stored in strictly lower triangular form, and in row major order.\n"
  "  The i-th row of F is of size b[i], i.e., by definition,\n"
  "  F_{i,j} = 0 for j<i-b[i] and for j=i.\n"
  "  For i-b[i] <= j < i,\n"
  "  the non-zero values F_{i,j} is stored at index (offsetrow[i] + j)\n"
  "  (i.e. offsetrow should be defined as offsetrow = cumsum(b-1) + 1).\n"
  "\n"
  "The Cholesky decomposition of C reads\n"
  "C = L D L^T\n"
  "with\n"
  "L = Lsep + G\n"
  "and\n"
  "* D: diagonal part of the decomposition (vector of size n, like A).\n"
  "* Lsep: the strictly lower triangular semiseparable part of L.\n"
  "  For i > j,\n"
  "  Lsep_{i,j} = Sum_{s=0}^{r-1} U_{i,s} W_{j,s} Prod_{k=j}^{i-1} phi_{k,s},\n"
  "  where U and phi are left unchanged and W is a (n x r) matrix (like V).\n"
  "* G: the strictly lower triangular leaf part of L.\n"
  "  G is stored in the same way as F.\n";
static char spleaf_dotL_docstring[] =
  "Compute y = L x,\n"
  "where L comes from the Cholesky decomposition C = L D L^T\n"
  "of a symmetric S+LEAF matrix C using spleaf_cholesky.";
static char spleaf_solveL_docstring[] =
  "Solve for x = L^-1 y,\n"
  "where L comes from the Cholesky decomposition C = L D L^T\n"
  "of a symmetric S+LEAF matrix C using spleaf_cholesky.";
static char spleaf_dotLT_docstring[] =
  "Compute y = L^T x,\n"
  "where L comes from the Cholesky decomposition C = L D L^T\n"
  "of a symmetric S+LEAF matrix C using spleaf_cholesky.";
static char spleaf_solveLT_docstring[] =
  "Solve for x = L^-T y,\n"
  "where L comes from the Cholesky decomposition C = L D L^T\n"
  "of a symmetric S+LEAF matrix C using spleaf_cholesky.";
static char spleaf_cholesky_back_docstring[] =
  "Backward propagation of the gradient for spleaf_cholesky.";
static char spleaf_dotL_back_docstring[] =
  "Backward propagation of the gradient for spleaf_dotL.";
static char spleaf_solveL_back_docstring[] =
  "Backward propagation of the gradient for spleaf_solveL.";
static char spleaf_dotLT_back_docstring[] =
  "Backward propagation of the gradient for spleaf_dotLT.";
static char spleaf_solveLT_back_docstring[] =
  "Backward propagation of the gradient for spleaf_solveLT.";

// Module methods
static PyObject *libspleaf_spleaf_cholesky(PyObject *self, PyObject *args);
static PyObject *libspleaf_spleaf_dotL(PyObject *self, PyObject *args);
static PyObject *libspleaf_spleaf_solveL(PyObject *self, PyObject *args);
static PyObject *libspleaf_spleaf_dotLT(PyObject *self, PyObject *args);
static PyObject *libspleaf_spleaf_solveLT(PyObject *self, PyObject *args);
static PyObject *libspleaf_spleaf_cholesky_back(PyObject *self, PyObject *args);
static PyObject *libspleaf_spleaf_dotL_back(PyObject *self, PyObject *args);
static PyObject *libspleaf_spleaf_solveL_back(PyObject *self, PyObject *args);
static PyObject *libspleaf_spleaf_dotLT_back(PyObject *self, PyObject *args);
static PyObject *libspleaf_spleaf_solveLT_back(PyObject *self, PyObject *args);
static PyMethodDef module_methods[] = {
  {"spleaf_cholesky", libspleaf_spleaf_cholesky, METH_VARARGS, spleaf_cholesky_docstring},
  {"spleaf_dotL", libspleaf_spleaf_dotL, METH_VARARGS, spleaf_dotL_docstring},
  {"spleaf_solveL", libspleaf_spleaf_solveL, METH_VARARGS, spleaf_solveL_docstring},
  {"spleaf_dotLT", libspleaf_spleaf_dotLT, METH_VARARGS, spleaf_dotLT_docstring},
  {"spleaf_solveLT", libspleaf_spleaf_solveLT, METH_VARARGS, spleaf_solveLT_docstring},
  {"spleaf_cholesky_back", libspleaf_spleaf_cholesky_back, METH_VARARGS, spleaf_cholesky_back_docstring},
  {"spleaf_dotL_back", libspleaf_spleaf_dotL_back, METH_VARARGS, spleaf_dotL_back_docstring},
  {"spleaf_solveL_back", libspleaf_spleaf_solveL_back, METH_VARARGS, spleaf_solveL_back_docstring},
  {"spleaf_dotLT_back", libspleaf_spleaf_dotLT_back, METH_VARARGS, spleaf_dotLT_back_docstring},
  {"spleaf_solveLT_back", libspleaf_spleaf_solveLT_back, METH_VARARGS, spleaf_solveLT_back_docstring},
  {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef myModule = {
  PyModuleDef_HEAD_INIT,
  "libspleaf",
  module_docstring,
  -1,
  module_methods
};

// Module initialization
PyMODINIT_FUNC PyInit_libspleaf(void) {
  // import numpy arrays
  import_array();
  return PyModule_Create(&myModule);
}

static PyObject *libspleaf_spleaf_cholesky(PyObject *self, PyObject *args) {
  long n;
  long r;
  PyObject *obj_offsetrow;
  PyObject *obj_b;
  PyObject *obj_A;
  PyObject *obj_U;
  PyObject *obj_V;
  PyObject *obj_phi;
  PyObject *obj_F;
  PyObject *obj_D;
  PyObject *obj_W;
  PyObject *obj_G;
  PyObject *obj_S;
  PyObject *obj_Z;

  // Parse input tuple
  if (!PyArg_ParseTuple(args, "llOOOOOOOOOOOO",
    &n,
    &r,
    &obj_offsetrow,
    &obj_b,
    &obj_A,
    &obj_U,
    &obj_V,
    &obj_phi,
    &obj_F,
    &obj_D,
    &obj_W,
    &obj_G,
    &obj_S,
    &obj_Z))
    return(NULL);

  // Interpret input objects as numpy arrays
  PyObject *arr_offsetrow = PyArray_FROM_OTF(obj_offsetrow, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_b = PyArray_FROM_OTF(obj_b, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_A = PyArray_FROM_OTF(obj_A, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_U = PyArray_FROM_OTF(obj_U, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_V = PyArray_FROM_OTF(obj_V, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_phi = PyArray_FROM_OTF(obj_phi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_F = PyArray_FROM_OTF(obj_F, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_D = PyArray_FROM_OTF(obj_D, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_W = PyArray_FROM_OTF(obj_W, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_G = PyArray_FROM_OTF(obj_G, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_S = PyArray_FROM_OTF(obj_S, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_Z = PyArray_FROM_OTF(obj_Z, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

  // Generate exception in case of failure
  if (
    arr_offsetrow == NULL ||
    arr_b == NULL ||
    arr_A == NULL ||
    arr_U == NULL ||
    arr_V == NULL ||
    arr_phi == NULL ||
    arr_F == NULL ||
    arr_D == NULL ||
    arr_W == NULL ||
    arr_G == NULL ||
    arr_S == NULL ||
    arr_Z == NULL) {
    // Dereference arrays
    Py_XDECREF(arr_offsetrow);
    Py_XDECREF(arr_b);
    Py_XDECREF(arr_A);
    Py_XDECREF(arr_U);
    Py_XDECREF(arr_V);
    Py_XDECREF(arr_phi);
    Py_XDECREF(arr_F);
    Py_XDECREF(arr_D);
    Py_XDECREF(arr_W);
    Py_XDECREF(arr_G);
    Py_XDECREF(arr_S);
    Py_XDECREF(arr_Z);
    return NULL;
  }

  // Get C-types pointers to numpy arrays
  long *offsetrow = (long*)PyArray_DATA(arr_offsetrow);
  long *b = (long*)PyArray_DATA(arr_b);
  double *A = (double*)PyArray_DATA(arr_A);
  double *U = (double*)PyArray_DATA(arr_U);
  double *V = (double*)PyArray_DATA(arr_V);
  double *phi = (double*)PyArray_DATA(arr_phi);
  double *F = (double*)PyArray_DATA(arr_F);
  double *D = (double*)PyArray_DATA(arr_D);
  double *W = (double*)PyArray_DATA(arr_W);
  double *G = (double*)PyArray_DATA(arr_G);
  double *S = (double*)PyArray_DATA(arr_S);
  double *Z = (double*)PyArray_DATA(arr_Z);

  // Call the C function from libspleaf
  spleaf_cholesky(
    n,
    r,
    offsetrow,
    b,
    A,
    U,
    V,
    phi,
    F,
    D,
    W,
    G,
    S,
    Z);

  // Dereference arrays
  Py_XDECREF(arr_offsetrow);
  Py_XDECREF(arr_b);
  Py_XDECREF(arr_A);
  Py_XDECREF(arr_U);
  Py_XDECREF(arr_V);
  Py_XDECREF(arr_phi);
  Py_XDECREF(arr_F);
  Py_XDECREF(arr_D);
  Py_XDECREF(arr_W);
  Py_XDECREF(arr_G);
  Py_XDECREF(arr_S);
  Py_XDECREF(arr_Z);

  Py_RETURN_NONE;
}

static PyObject *libspleaf_spleaf_dotL(PyObject *self, PyObject *args) {
  long n;
  long r;
  PyObject *obj_offsetrow;
  PyObject *obj_b;
  PyObject *obj_U;
  PyObject *obj_W;
  PyObject *obj_phi;
  PyObject *obj_G;
  PyObject *obj_x;
  PyObject *obj_y;
  PyObject *obj_f;

  // Parse input tuple
  if (!PyArg_ParseTuple(args, "llOOOOOOOOO",
    &n,
    &r,
    &obj_offsetrow,
    &obj_b,
    &obj_U,
    &obj_W,
    &obj_phi,
    &obj_G,
    &obj_x,
    &obj_y,
    &obj_f))
    return(NULL);

  // Interpret input objects as numpy arrays
  PyObject *arr_offsetrow = PyArray_FROM_OTF(obj_offsetrow, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_b = PyArray_FROM_OTF(obj_b, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_U = PyArray_FROM_OTF(obj_U, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_W = PyArray_FROM_OTF(obj_W, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_phi = PyArray_FROM_OTF(obj_phi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_G = PyArray_FROM_OTF(obj_G, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_x = PyArray_FROM_OTF(obj_x, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_y = PyArray_FROM_OTF(obj_y, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_f = PyArray_FROM_OTF(obj_f, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

  // Generate exception in case of failure
  if (
    arr_offsetrow == NULL ||
    arr_b == NULL ||
    arr_U == NULL ||
    arr_W == NULL ||
    arr_phi == NULL ||
    arr_G == NULL ||
    arr_x == NULL ||
    arr_y == NULL ||
    arr_f == NULL) {
    // Dereference arrays
    Py_XDECREF(arr_offsetrow);
    Py_XDECREF(arr_b);
    Py_XDECREF(arr_U);
    Py_XDECREF(arr_W);
    Py_XDECREF(arr_phi);
    Py_XDECREF(arr_G);
    Py_XDECREF(arr_x);
    Py_XDECREF(arr_y);
    Py_XDECREF(arr_f);
    return NULL;
  }

  // Get C-types pointers to numpy arrays
  long *offsetrow = (long*)PyArray_DATA(arr_offsetrow);
  long *b = (long*)PyArray_DATA(arr_b);
  double *U = (double*)PyArray_DATA(arr_U);
  double *W = (double*)PyArray_DATA(arr_W);
  double *phi = (double*)PyArray_DATA(arr_phi);
  double *G = (double*)PyArray_DATA(arr_G);
  double *x = (double*)PyArray_DATA(arr_x);
  double *y = (double*)PyArray_DATA(arr_y);
  double *f = (double*)PyArray_DATA(arr_f);

  // Call the C function from libspleaf
  spleaf_dotL(
    n,
    r,
    offsetrow,
    b,
    U,
    W,
    phi,
    G,
    x,
    y,
    f);

  // Dereference arrays
  Py_XDECREF(arr_offsetrow);
  Py_XDECREF(arr_b);
  Py_XDECREF(arr_U);
  Py_XDECREF(arr_W);
  Py_XDECREF(arr_phi);
  Py_XDECREF(arr_G);
  Py_XDECREF(arr_x);
  Py_XDECREF(arr_y);
  Py_XDECREF(arr_f);

  Py_RETURN_NONE;
}

static PyObject *libspleaf_spleaf_solveL(PyObject *self, PyObject *args) {
  long n;
  long r;
  PyObject *obj_offsetrow;
  PyObject *obj_b;
  PyObject *obj_U;
  PyObject *obj_W;
  PyObject *obj_phi;
  PyObject *obj_G;
  PyObject *obj_y;
  PyObject *obj_x;
  PyObject *obj_f;

  // Parse input tuple
  if (!PyArg_ParseTuple(args, "llOOOOOOOOO",
    &n,
    &r,
    &obj_offsetrow,
    &obj_b,
    &obj_U,
    &obj_W,
    &obj_phi,
    &obj_G,
    &obj_y,
    &obj_x,
    &obj_f))
    return(NULL);

  // Interpret input objects as numpy arrays
  PyObject *arr_offsetrow = PyArray_FROM_OTF(obj_offsetrow, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_b = PyArray_FROM_OTF(obj_b, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_U = PyArray_FROM_OTF(obj_U, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_W = PyArray_FROM_OTF(obj_W, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_phi = PyArray_FROM_OTF(obj_phi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_G = PyArray_FROM_OTF(obj_G, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_y = PyArray_FROM_OTF(obj_y, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_x = PyArray_FROM_OTF(obj_x, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_f = PyArray_FROM_OTF(obj_f, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

  // Generate exception in case of failure
  if (
    arr_offsetrow == NULL ||
    arr_b == NULL ||
    arr_U == NULL ||
    arr_W == NULL ||
    arr_phi == NULL ||
    arr_G == NULL ||
    arr_y == NULL ||
    arr_x == NULL ||
    arr_f == NULL) {
    // Dereference arrays
    Py_XDECREF(arr_offsetrow);
    Py_XDECREF(arr_b);
    Py_XDECREF(arr_U);
    Py_XDECREF(arr_W);
    Py_XDECREF(arr_phi);
    Py_XDECREF(arr_G);
    Py_XDECREF(arr_y);
    Py_XDECREF(arr_x);
    Py_XDECREF(arr_f);
    return NULL;
  }

  // Get C-types pointers to numpy arrays
  long *offsetrow = (long*)PyArray_DATA(arr_offsetrow);
  long *b = (long*)PyArray_DATA(arr_b);
  double *U = (double*)PyArray_DATA(arr_U);
  double *W = (double*)PyArray_DATA(arr_W);
  double *phi = (double*)PyArray_DATA(arr_phi);
  double *G = (double*)PyArray_DATA(arr_G);
  double *y = (double*)PyArray_DATA(arr_y);
  double *x = (double*)PyArray_DATA(arr_x);
  double *f = (double*)PyArray_DATA(arr_f);

  // Call the C function from libspleaf
  spleaf_solveL(
    n,
    r,
    offsetrow,
    b,
    U,
    W,
    phi,
    G,
    y,
    x,
    f);

  // Dereference arrays
  Py_XDECREF(arr_offsetrow);
  Py_XDECREF(arr_b);
  Py_XDECREF(arr_U);
  Py_XDECREF(arr_W);
  Py_XDECREF(arr_phi);
  Py_XDECREF(arr_G);
  Py_XDECREF(arr_y);
  Py_XDECREF(arr_x);
  Py_XDECREF(arr_f);

  Py_RETURN_NONE;
}

static PyObject *libspleaf_spleaf_dotLT(PyObject *self, PyObject *args) {
  long n;
  long r;
  PyObject *obj_offsetrow;
  PyObject *obj_b;
  PyObject *obj_U;
  PyObject *obj_W;
  PyObject *obj_phi;
  PyObject *obj_G;
  PyObject *obj_x;
  PyObject *obj_y;
  PyObject *obj_g;

  // Parse input tuple
  if (!PyArg_ParseTuple(args, "llOOOOOOOOO",
    &n,
    &r,
    &obj_offsetrow,
    &obj_b,
    &obj_U,
    &obj_W,
    &obj_phi,
    &obj_G,
    &obj_x,
    &obj_y,
    &obj_g))
    return(NULL);

  // Interpret input objects as numpy arrays
  PyObject *arr_offsetrow = PyArray_FROM_OTF(obj_offsetrow, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_b = PyArray_FROM_OTF(obj_b, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_U = PyArray_FROM_OTF(obj_U, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_W = PyArray_FROM_OTF(obj_W, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_phi = PyArray_FROM_OTF(obj_phi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_G = PyArray_FROM_OTF(obj_G, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_x = PyArray_FROM_OTF(obj_x, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_y = PyArray_FROM_OTF(obj_y, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_g = PyArray_FROM_OTF(obj_g, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

  // Generate exception in case of failure
  if (
    arr_offsetrow == NULL ||
    arr_b == NULL ||
    arr_U == NULL ||
    arr_W == NULL ||
    arr_phi == NULL ||
    arr_G == NULL ||
    arr_x == NULL ||
    arr_y == NULL ||
    arr_g == NULL) {
    // Dereference arrays
    Py_XDECREF(arr_offsetrow);
    Py_XDECREF(arr_b);
    Py_XDECREF(arr_U);
    Py_XDECREF(arr_W);
    Py_XDECREF(arr_phi);
    Py_XDECREF(arr_G);
    Py_XDECREF(arr_x);
    Py_XDECREF(arr_y);
    Py_XDECREF(arr_g);
    return NULL;
  }

  // Get C-types pointers to numpy arrays
  long *offsetrow = (long*)PyArray_DATA(arr_offsetrow);
  long *b = (long*)PyArray_DATA(arr_b);
  double *U = (double*)PyArray_DATA(arr_U);
  double *W = (double*)PyArray_DATA(arr_W);
  double *phi = (double*)PyArray_DATA(arr_phi);
  double *G = (double*)PyArray_DATA(arr_G);
  double *x = (double*)PyArray_DATA(arr_x);
  double *y = (double*)PyArray_DATA(arr_y);
  double *g = (double*)PyArray_DATA(arr_g);

  // Call the C function from libspleaf
  spleaf_dotLT(
    n,
    r,
    offsetrow,
    b,
    U,
    W,
    phi,
    G,
    x,
    y,
    g);

  // Dereference arrays
  Py_XDECREF(arr_offsetrow);
  Py_XDECREF(arr_b);
  Py_XDECREF(arr_U);
  Py_XDECREF(arr_W);
  Py_XDECREF(arr_phi);
  Py_XDECREF(arr_G);
  Py_XDECREF(arr_x);
  Py_XDECREF(arr_y);
  Py_XDECREF(arr_g);

  Py_RETURN_NONE;
}

static PyObject *libspleaf_spleaf_solveLT(PyObject *self, PyObject *args) {
  long n;
  long r;
  PyObject *obj_offsetrow;
  PyObject *obj_b;
  PyObject *obj_U;
  PyObject *obj_W;
  PyObject *obj_phi;
  PyObject *obj_G;
  PyObject *obj_y;
  PyObject *obj_x;
  PyObject *obj_g;

  // Parse input tuple
  if (!PyArg_ParseTuple(args, "llOOOOOOOOO",
    &n,
    &r,
    &obj_offsetrow,
    &obj_b,
    &obj_U,
    &obj_W,
    &obj_phi,
    &obj_G,
    &obj_y,
    &obj_x,
    &obj_g))
    return(NULL);

  // Interpret input objects as numpy arrays
  PyObject *arr_offsetrow = PyArray_FROM_OTF(obj_offsetrow, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_b = PyArray_FROM_OTF(obj_b, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_U = PyArray_FROM_OTF(obj_U, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_W = PyArray_FROM_OTF(obj_W, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_phi = PyArray_FROM_OTF(obj_phi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_G = PyArray_FROM_OTF(obj_G, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_y = PyArray_FROM_OTF(obj_y, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_x = PyArray_FROM_OTF(obj_x, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_g = PyArray_FROM_OTF(obj_g, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

  // Generate exception in case of failure
  if (
    arr_offsetrow == NULL ||
    arr_b == NULL ||
    arr_U == NULL ||
    arr_W == NULL ||
    arr_phi == NULL ||
    arr_G == NULL ||
    arr_y == NULL ||
    arr_x == NULL ||
    arr_g == NULL) {
    // Dereference arrays
    Py_XDECREF(arr_offsetrow);
    Py_XDECREF(arr_b);
    Py_XDECREF(arr_U);
    Py_XDECREF(arr_W);
    Py_XDECREF(arr_phi);
    Py_XDECREF(arr_G);
    Py_XDECREF(arr_y);
    Py_XDECREF(arr_x);
    Py_XDECREF(arr_g);
    return NULL;
  }

  // Get C-types pointers to numpy arrays
  long *offsetrow = (long*)PyArray_DATA(arr_offsetrow);
  long *b = (long*)PyArray_DATA(arr_b);
  double *U = (double*)PyArray_DATA(arr_U);
  double *W = (double*)PyArray_DATA(arr_W);
  double *phi = (double*)PyArray_DATA(arr_phi);
  double *G = (double*)PyArray_DATA(arr_G);
  double *y = (double*)PyArray_DATA(arr_y);
  double *x = (double*)PyArray_DATA(arr_x);
  double *g = (double*)PyArray_DATA(arr_g);

  // Call the C function from libspleaf
  spleaf_solveLT(
    n,
    r,
    offsetrow,
    b,
    U,
    W,
    phi,
    G,
    y,
    x,
    g);

  // Dereference arrays
  Py_XDECREF(arr_offsetrow);
  Py_XDECREF(arr_b);
  Py_XDECREF(arr_U);
  Py_XDECREF(arr_W);
  Py_XDECREF(arr_phi);
  Py_XDECREF(arr_G);
  Py_XDECREF(arr_y);
  Py_XDECREF(arr_x);
  Py_XDECREF(arr_g);

  Py_RETURN_NONE;
}

static PyObject *libspleaf_spleaf_cholesky_back(PyObject *self, PyObject *args) {
  long n;
  long r;
  PyObject *obj_offsetrow;
  PyObject *obj_b;
  PyObject *obj_D;
  PyObject *obj_U;
  PyObject *obj_W;
  PyObject *obj_phi;
  PyObject *obj_G;
  PyObject *obj_grad_D;
  PyObject *obj_grad_Ucho;
  PyObject *obj_grad_W;
  PyObject *obj_grad_phicho;
  PyObject *obj_grad_G;
  PyObject *obj_grad_A;
  PyObject *obj_grad_U;
  PyObject *obj_grad_V;
  PyObject *obj_grad_phi;
  PyObject *obj_grad_F;
  PyObject *obj_S;
  PyObject *obj_Z;

  // Parse input tuple
  if (!PyArg_ParseTuple(args, "llOOOOOOOOOOOOOOOOOOO",
    &n,
    &r,
    &obj_offsetrow,
    &obj_b,
    &obj_D,
    &obj_U,
    &obj_W,
    &obj_phi,
    &obj_G,
    &obj_grad_D,
    &obj_grad_Ucho,
    &obj_grad_W,
    &obj_grad_phicho,
    &obj_grad_G,
    &obj_grad_A,
    &obj_grad_U,
    &obj_grad_V,
    &obj_grad_phi,
    &obj_grad_F,
    &obj_S,
    &obj_Z))
    return(NULL);

  // Interpret input objects as numpy arrays
  PyObject *arr_offsetrow = PyArray_FROM_OTF(obj_offsetrow, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_b = PyArray_FROM_OTF(obj_b, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_D = PyArray_FROM_OTF(obj_D, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_U = PyArray_FROM_OTF(obj_U, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_W = PyArray_FROM_OTF(obj_W, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_phi = PyArray_FROM_OTF(obj_phi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_G = PyArray_FROM_OTF(obj_G, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_grad_D = PyArray_FROM_OTF(obj_grad_D, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_grad_Ucho = PyArray_FROM_OTF(obj_grad_Ucho, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_grad_W = PyArray_FROM_OTF(obj_grad_W, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_grad_phicho = PyArray_FROM_OTF(obj_grad_phicho, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_grad_G = PyArray_FROM_OTF(obj_grad_G, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_grad_A = PyArray_FROM_OTF(obj_grad_A, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_grad_U = PyArray_FROM_OTF(obj_grad_U, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_grad_V = PyArray_FROM_OTF(obj_grad_V, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_grad_phi = PyArray_FROM_OTF(obj_grad_phi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_grad_F = PyArray_FROM_OTF(obj_grad_F, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_S = PyArray_FROM_OTF(obj_S, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_Z = PyArray_FROM_OTF(obj_Z, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

  // Generate exception in case of failure
  if (
    arr_offsetrow == NULL ||
    arr_b == NULL ||
    arr_D == NULL ||
    arr_U == NULL ||
    arr_W == NULL ||
    arr_phi == NULL ||
    arr_G == NULL ||
    arr_grad_D == NULL ||
    arr_grad_Ucho == NULL ||
    arr_grad_W == NULL ||
    arr_grad_phicho == NULL ||
    arr_grad_G == NULL ||
    arr_grad_A == NULL ||
    arr_grad_U == NULL ||
    arr_grad_V == NULL ||
    arr_grad_phi == NULL ||
    arr_grad_F == NULL ||
    arr_S == NULL ||
    arr_Z == NULL) {
    // Dereference arrays
    Py_XDECREF(arr_offsetrow);
    Py_XDECREF(arr_b);
    Py_XDECREF(arr_D);
    Py_XDECREF(arr_U);
    Py_XDECREF(arr_W);
    Py_XDECREF(arr_phi);
    Py_XDECREF(arr_G);
    Py_XDECREF(arr_grad_D);
    Py_XDECREF(arr_grad_Ucho);
    Py_XDECREF(arr_grad_W);
    Py_XDECREF(arr_grad_phicho);
    Py_XDECREF(arr_grad_G);
    Py_XDECREF(arr_grad_A);
    Py_XDECREF(arr_grad_U);
    Py_XDECREF(arr_grad_V);
    Py_XDECREF(arr_grad_phi);
    Py_XDECREF(arr_grad_F);
    Py_XDECREF(arr_S);
    Py_XDECREF(arr_Z);
    return NULL;
  }

  // Get C-types pointers to numpy arrays
  long *offsetrow = (long*)PyArray_DATA(arr_offsetrow);
  long *b = (long*)PyArray_DATA(arr_b);
  double *D = (double*)PyArray_DATA(arr_D);
  double *U = (double*)PyArray_DATA(arr_U);
  double *W = (double*)PyArray_DATA(arr_W);
  double *phi = (double*)PyArray_DATA(arr_phi);
  double *G = (double*)PyArray_DATA(arr_G);
  double *grad_D = (double*)PyArray_DATA(arr_grad_D);
  double *grad_Ucho = (double*)PyArray_DATA(arr_grad_Ucho);
  double *grad_W = (double*)PyArray_DATA(arr_grad_W);
  double *grad_phicho = (double*)PyArray_DATA(arr_grad_phicho);
  double *grad_G = (double*)PyArray_DATA(arr_grad_G);
  double *grad_A = (double*)PyArray_DATA(arr_grad_A);
  double *grad_U = (double*)PyArray_DATA(arr_grad_U);
  double *grad_V = (double*)PyArray_DATA(arr_grad_V);
  double *grad_phi = (double*)PyArray_DATA(arr_grad_phi);
  double *grad_F = (double*)PyArray_DATA(arr_grad_F);
  double *S = (double*)PyArray_DATA(arr_S);
  double *Z = (double*)PyArray_DATA(arr_Z);

  // Call the C function from libspleaf
  spleaf_cholesky_back(
    n,
    r,
    offsetrow,
    b,
    D,
    U,
    W,
    phi,
    G,
    grad_D,
    grad_Ucho,
    grad_W,
    grad_phicho,
    grad_G,
    grad_A,
    grad_U,
    grad_V,
    grad_phi,
    grad_F,
    S,
    Z);

  // Dereference arrays
  Py_XDECREF(arr_offsetrow);
  Py_XDECREF(arr_b);
  Py_XDECREF(arr_D);
  Py_XDECREF(arr_U);
  Py_XDECREF(arr_W);
  Py_XDECREF(arr_phi);
  Py_XDECREF(arr_G);
  Py_XDECREF(arr_grad_D);
  Py_XDECREF(arr_grad_Ucho);
  Py_XDECREF(arr_grad_W);
  Py_XDECREF(arr_grad_phicho);
  Py_XDECREF(arr_grad_G);
  Py_XDECREF(arr_grad_A);
  Py_XDECREF(arr_grad_U);
  Py_XDECREF(arr_grad_V);
  Py_XDECREF(arr_grad_phi);
  Py_XDECREF(arr_grad_F);
  Py_XDECREF(arr_S);
  Py_XDECREF(arr_Z);

  Py_RETURN_NONE;
}

static PyObject *libspleaf_spleaf_dotL_back(PyObject *self, PyObject *args) {
  long n;
  long r;
  PyObject *obj_offsetrow;
  PyObject *obj_b;
  PyObject *obj_U;
  PyObject *obj_W;
  PyObject *obj_phi;
  PyObject *obj_G;
  PyObject *obj_x;
  PyObject *obj_grad_y;
  PyObject *obj_grad_U;
  PyObject *obj_grad_W;
  PyObject *obj_grad_phi;
  PyObject *obj_grad_G;
  PyObject *obj_grad_x;
  PyObject *obj_f;

  // Parse input tuple
  if (!PyArg_ParseTuple(args, "llOOOOOOOOOOOOOO",
    &n,
    &r,
    &obj_offsetrow,
    &obj_b,
    &obj_U,
    &obj_W,
    &obj_phi,
    &obj_G,
    &obj_x,
    &obj_grad_y,
    &obj_grad_U,
    &obj_grad_W,
    &obj_grad_phi,
    &obj_grad_G,
    &obj_grad_x,
    &obj_f))
    return(NULL);

  // Interpret input objects as numpy arrays
  PyObject *arr_offsetrow = PyArray_FROM_OTF(obj_offsetrow, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_b = PyArray_FROM_OTF(obj_b, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_U = PyArray_FROM_OTF(obj_U, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_W = PyArray_FROM_OTF(obj_W, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_phi = PyArray_FROM_OTF(obj_phi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_G = PyArray_FROM_OTF(obj_G, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_x = PyArray_FROM_OTF(obj_x, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_grad_y = PyArray_FROM_OTF(obj_grad_y, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_grad_U = PyArray_FROM_OTF(obj_grad_U, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_grad_W = PyArray_FROM_OTF(obj_grad_W, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_grad_phi = PyArray_FROM_OTF(obj_grad_phi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_grad_G = PyArray_FROM_OTF(obj_grad_G, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_grad_x = PyArray_FROM_OTF(obj_grad_x, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_f = PyArray_FROM_OTF(obj_f, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

  // Generate exception in case of failure
  if (
    arr_offsetrow == NULL ||
    arr_b == NULL ||
    arr_U == NULL ||
    arr_W == NULL ||
    arr_phi == NULL ||
    arr_G == NULL ||
    arr_x == NULL ||
    arr_grad_y == NULL ||
    arr_grad_U == NULL ||
    arr_grad_W == NULL ||
    arr_grad_phi == NULL ||
    arr_grad_G == NULL ||
    arr_grad_x == NULL ||
    arr_f == NULL) {
    // Dereference arrays
    Py_XDECREF(arr_offsetrow);
    Py_XDECREF(arr_b);
    Py_XDECREF(arr_U);
    Py_XDECREF(arr_W);
    Py_XDECREF(arr_phi);
    Py_XDECREF(arr_G);
    Py_XDECREF(arr_x);
    Py_XDECREF(arr_grad_y);
    Py_XDECREF(arr_grad_U);
    Py_XDECREF(arr_grad_W);
    Py_XDECREF(arr_grad_phi);
    Py_XDECREF(arr_grad_G);
    Py_XDECREF(arr_grad_x);
    Py_XDECREF(arr_f);
    return NULL;
  }

  // Get C-types pointers to numpy arrays
  long *offsetrow = (long*)PyArray_DATA(arr_offsetrow);
  long *b = (long*)PyArray_DATA(arr_b);
  double *U = (double*)PyArray_DATA(arr_U);
  double *W = (double*)PyArray_DATA(arr_W);
  double *phi = (double*)PyArray_DATA(arr_phi);
  double *G = (double*)PyArray_DATA(arr_G);
  double *x = (double*)PyArray_DATA(arr_x);
  double *grad_y = (double*)PyArray_DATA(arr_grad_y);
  double *grad_U = (double*)PyArray_DATA(arr_grad_U);
  double *grad_W = (double*)PyArray_DATA(arr_grad_W);
  double *grad_phi = (double*)PyArray_DATA(arr_grad_phi);
  double *grad_G = (double*)PyArray_DATA(arr_grad_G);
  double *grad_x = (double*)PyArray_DATA(arr_grad_x);
  double *f = (double*)PyArray_DATA(arr_f);

  // Call the C function from libspleaf
  spleaf_dotL_back(
    n,
    r,
    offsetrow,
    b,
    U,
    W,
    phi,
    G,
    x,
    grad_y,
    grad_U,
    grad_W,
    grad_phi,
    grad_G,
    grad_x,
    f);

  // Dereference arrays
  Py_XDECREF(arr_offsetrow);
  Py_XDECREF(arr_b);
  Py_XDECREF(arr_U);
  Py_XDECREF(arr_W);
  Py_XDECREF(arr_phi);
  Py_XDECREF(arr_G);
  Py_XDECREF(arr_x);
  Py_XDECREF(arr_grad_y);
  Py_XDECREF(arr_grad_U);
  Py_XDECREF(arr_grad_W);
  Py_XDECREF(arr_grad_phi);
  Py_XDECREF(arr_grad_G);
  Py_XDECREF(arr_grad_x);
  Py_XDECREF(arr_f);

  Py_RETURN_NONE;
}

static PyObject *libspleaf_spleaf_solveL_back(PyObject *self, PyObject *args) {
  long n;
  long r;
  PyObject *obj_offsetrow;
  PyObject *obj_b;
  PyObject *obj_U;
  PyObject *obj_W;
  PyObject *obj_phi;
  PyObject *obj_G;
  PyObject *obj_x;
  PyObject *obj_grad_x;
  PyObject *obj_grad_U;
  PyObject *obj_grad_W;
  PyObject *obj_grad_phi;
  PyObject *obj_grad_G;
  PyObject *obj_grad_y;
  PyObject *obj_f;

  // Parse input tuple
  if (!PyArg_ParseTuple(args, "llOOOOOOOOOOOOOO",
    &n,
    &r,
    &obj_offsetrow,
    &obj_b,
    &obj_U,
    &obj_W,
    &obj_phi,
    &obj_G,
    &obj_x,
    &obj_grad_x,
    &obj_grad_U,
    &obj_grad_W,
    &obj_grad_phi,
    &obj_grad_G,
    &obj_grad_y,
    &obj_f))
    return(NULL);

  // Interpret input objects as numpy arrays
  PyObject *arr_offsetrow = PyArray_FROM_OTF(obj_offsetrow, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_b = PyArray_FROM_OTF(obj_b, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_U = PyArray_FROM_OTF(obj_U, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_W = PyArray_FROM_OTF(obj_W, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_phi = PyArray_FROM_OTF(obj_phi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_G = PyArray_FROM_OTF(obj_G, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_x = PyArray_FROM_OTF(obj_x, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_grad_x = PyArray_FROM_OTF(obj_grad_x, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_grad_U = PyArray_FROM_OTF(obj_grad_U, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_grad_W = PyArray_FROM_OTF(obj_grad_W, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_grad_phi = PyArray_FROM_OTF(obj_grad_phi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_grad_G = PyArray_FROM_OTF(obj_grad_G, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_grad_y = PyArray_FROM_OTF(obj_grad_y, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_f = PyArray_FROM_OTF(obj_f, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

  // Generate exception in case of failure
  if (
    arr_offsetrow == NULL ||
    arr_b == NULL ||
    arr_U == NULL ||
    arr_W == NULL ||
    arr_phi == NULL ||
    arr_G == NULL ||
    arr_x == NULL ||
    arr_grad_x == NULL ||
    arr_grad_U == NULL ||
    arr_grad_W == NULL ||
    arr_grad_phi == NULL ||
    arr_grad_G == NULL ||
    arr_grad_y == NULL ||
    arr_f == NULL) {
    // Dereference arrays
    Py_XDECREF(arr_offsetrow);
    Py_XDECREF(arr_b);
    Py_XDECREF(arr_U);
    Py_XDECREF(arr_W);
    Py_XDECREF(arr_phi);
    Py_XDECREF(arr_G);
    Py_XDECREF(arr_x);
    Py_XDECREF(arr_grad_x);
    Py_XDECREF(arr_grad_U);
    Py_XDECREF(arr_grad_W);
    Py_XDECREF(arr_grad_phi);
    Py_XDECREF(arr_grad_G);
    Py_XDECREF(arr_grad_y);
    Py_XDECREF(arr_f);
    return NULL;
  }

  // Get C-types pointers to numpy arrays
  long *offsetrow = (long*)PyArray_DATA(arr_offsetrow);
  long *b = (long*)PyArray_DATA(arr_b);
  double *U = (double*)PyArray_DATA(arr_U);
  double *W = (double*)PyArray_DATA(arr_W);
  double *phi = (double*)PyArray_DATA(arr_phi);
  double *G = (double*)PyArray_DATA(arr_G);
  double *x = (double*)PyArray_DATA(arr_x);
  double *grad_x = (double*)PyArray_DATA(arr_grad_x);
  double *grad_U = (double*)PyArray_DATA(arr_grad_U);
  double *grad_W = (double*)PyArray_DATA(arr_grad_W);
  double *grad_phi = (double*)PyArray_DATA(arr_grad_phi);
  double *grad_G = (double*)PyArray_DATA(arr_grad_G);
  double *grad_y = (double*)PyArray_DATA(arr_grad_y);
  double *f = (double*)PyArray_DATA(arr_f);

  // Call the C function from libspleaf
  spleaf_solveL_back(
    n,
    r,
    offsetrow,
    b,
    U,
    W,
    phi,
    G,
    x,
    grad_x,
    grad_U,
    grad_W,
    grad_phi,
    grad_G,
    grad_y,
    f);

  // Dereference arrays
  Py_XDECREF(arr_offsetrow);
  Py_XDECREF(arr_b);
  Py_XDECREF(arr_U);
  Py_XDECREF(arr_W);
  Py_XDECREF(arr_phi);
  Py_XDECREF(arr_G);
  Py_XDECREF(arr_x);
  Py_XDECREF(arr_grad_x);
  Py_XDECREF(arr_grad_U);
  Py_XDECREF(arr_grad_W);
  Py_XDECREF(arr_grad_phi);
  Py_XDECREF(arr_grad_G);
  Py_XDECREF(arr_grad_y);
  Py_XDECREF(arr_f);

  Py_RETURN_NONE;
}

static PyObject *libspleaf_spleaf_dotLT_back(PyObject *self, PyObject *args) {
  long n;
  long r;
  PyObject *obj_offsetrow;
  PyObject *obj_b;
  PyObject *obj_U;
  PyObject *obj_W;
  PyObject *obj_phi;
  PyObject *obj_G;
  PyObject *obj_x;
  PyObject *obj_grad_y;
  PyObject *obj_grad_U;
  PyObject *obj_grad_W;
  PyObject *obj_grad_phi;
  PyObject *obj_grad_G;
  PyObject *obj_grad_x;
  PyObject *obj_g;

  // Parse input tuple
  if (!PyArg_ParseTuple(args, "llOOOOOOOOOOOOOO",
    &n,
    &r,
    &obj_offsetrow,
    &obj_b,
    &obj_U,
    &obj_W,
    &obj_phi,
    &obj_G,
    &obj_x,
    &obj_grad_y,
    &obj_grad_U,
    &obj_grad_W,
    &obj_grad_phi,
    &obj_grad_G,
    &obj_grad_x,
    &obj_g))
    return(NULL);

  // Interpret input objects as numpy arrays
  PyObject *arr_offsetrow = PyArray_FROM_OTF(obj_offsetrow, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_b = PyArray_FROM_OTF(obj_b, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_U = PyArray_FROM_OTF(obj_U, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_W = PyArray_FROM_OTF(obj_W, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_phi = PyArray_FROM_OTF(obj_phi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_G = PyArray_FROM_OTF(obj_G, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_x = PyArray_FROM_OTF(obj_x, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_grad_y = PyArray_FROM_OTF(obj_grad_y, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_grad_U = PyArray_FROM_OTF(obj_grad_U, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_grad_W = PyArray_FROM_OTF(obj_grad_W, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_grad_phi = PyArray_FROM_OTF(obj_grad_phi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_grad_G = PyArray_FROM_OTF(obj_grad_G, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_grad_x = PyArray_FROM_OTF(obj_grad_x, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_g = PyArray_FROM_OTF(obj_g, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

  // Generate exception in case of failure
  if (
    arr_offsetrow == NULL ||
    arr_b == NULL ||
    arr_U == NULL ||
    arr_W == NULL ||
    arr_phi == NULL ||
    arr_G == NULL ||
    arr_x == NULL ||
    arr_grad_y == NULL ||
    arr_grad_U == NULL ||
    arr_grad_W == NULL ||
    arr_grad_phi == NULL ||
    arr_grad_G == NULL ||
    arr_grad_x == NULL ||
    arr_g == NULL) {
    // Dereference arrays
    Py_XDECREF(arr_offsetrow);
    Py_XDECREF(arr_b);
    Py_XDECREF(arr_U);
    Py_XDECREF(arr_W);
    Py_XDECREF(arr_phi);
    Py_XDECREF(arr_G);
    Py_XDECREF(arr_x);
    Py_XDECREF(arr_grad_y);
    Py_XDECREF(arr_grad_U);
    Py_XDECREF(arr_grad_W);
    Py_XDECREF(arr_grad_phi);
    Py_XDECREF(arr_grad_G);
    Py_XDECREF(arr_grad_x);
    Py_XDECREF(arr_g);
    return NULL;
  }

  // Get C-types pointers to numpy arrays
  long *offsetrow = (long*)PyArray_DATA(arr_offsetrow);
  long *b = (long*)PyArray_DATA(arr_b);
  double *U = (double*)PyArray_DATA(arr_U);
  double *W = (double*)PyArray_DATA(arr_W);
  double *phi = (double*)PyArray_DATA(arr_phi);
  double *G = (double*)PyArray_DATA(arr_G);
  double *x = (double*)PyArray_DATA(arr_x);
  double *grad_y = (double*)PyArray_DATA(arr_grad_y);
  double *grad_U = (double*)PyArray_DATA(arr_grad_U);
  double *grad_W = (double*)PyArray_DATA(arr_grad_W);
  double *grad_phi = (double*)PyArray_DATA(arr_grad_phi);
  double *grad_G = (double*)PyArray_DATA(arr_grad_G);
  double *grad_x = (double*)PyArray_DATA(arr_grad_x);
  double *g = (double*)PyArray_DATA(arr_g);

  // Call the C function from libspleaf
  spleaf_dotLT_back(
    n,
    r,
    offsetrow,
    b,
    U,
    W,
    phi,
    G,
    x,
    grad_y,
    grad_U,
    grad_W,
    grad_phi,
    grad_G,
    grad_x,
    g);

  // Dereference arrays
  Py_XDECREF(arr_offsetrow);
  Py_XDECREF(arr_b);
  Py_XDECREF(arr_U);
  Py_XDECREF(arr_W);
  Py_XDECREF(arr_phi);
  Py_XDECREF(arr_G);
  Py_XDECREF(arr_x);
  Py_XDECREF(arr_grad_y);
  Py_XDECREF(arr_grad_U);
  Py_XDECREF(arr_grad_W);
  Py_XDECREF(arr_grad_phi);
  Py_XDECREF(arr_grad_G);
  Py_XDECREF(arr_grad_x);
  Py_XDECREF(arr_g);

  Py_RETURN_NONE;
}

static PyObject *libspleaf_spleaf_solveLT_back(PyObject *self, PyObject *args) {
  long n;
  long r;
  PyObject *obj_offsetrow;
  PyObject *obj_b;
  PyObject *obj_U;
  PyObject *obj_W;
  PyObject *obj_phi;
  PyObject *obj_G;
  PyObject *obj_x;
  PyObject *obj_grad_x;
  PyObject *obj_grad_U;
  PyObject *obj_grad_W;
  PyObject *obj_grad_phi;
  PyObject *obj_grad_G;
  PyObject *obj_grad_y;
  PyObject *obj_g;

  // Parse input tuple
  if (!PyArg_ParseTuple(args, "llOOOOOOOOOOOOOO",
    &n,
    &r,
    &obj_offsetrow,
    &obj_b,
    &obj_U,
    &obj_W,
    &obj_phi,
    &obj_G,
    &obj_x,
    &obj_grad_x,
    &obj_grad_U,
    &obj_grad_W,
    &obj_grad_phi,
    &obj_grad_G,
    &obj_grad_y,
    &obj_g))
    return(NULL);

  // Interpret input objects as numpy arrays
  PyObject *arr_offsetrow = PyArray_FROM_OTF(obj_offsetrow, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_b = PyArray_FROM_OTF(obj_b, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_U = PyArray_FROM_OTF(obj_U, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_W = PyArray_FROM_OTF(obj_W, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_phi = PyArray_FROM_OTF(obj_phi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_G = PyArray_FROM_OTF(obj_G, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_x = PyArray_FROM_OTF(obj_x, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_grad_x = PyArray_FROM_OTF(obj_grad_x, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_grad_U = PyArray_FROM_OTF(obj_grad_U, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_grad_W = PyArray_FROM_OTF(obj_grad_W, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_grad_phi = PyArray_FROM_OTF(obj_grad_phi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_grad_G = PyArray_FROM_OTF(obj_grad_G, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_grad_y = PyArray_FROM_OTF(obj_grad_y, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *arr_g = PyArray_FROM_OTF(obj_g, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

  // Generate exception in case of failure
  if (
    arr_offsetrow == NULL ||
    arr_b == NULL ||
    arr_U == NULL ||
    arr_W == NULL ||
    arr_phi == NULL ||
    arr_G == NULL ||
    arr_x == NULL ||
    arr_grad_x == NULL ||
    arr_grad_U == NULL ||
    arr_grad_W == NULL ||
    arr_grad_phi == NULL ||
    arr_grad_G == NULL ||
    arr_grad_y == NULL ||
    arr_g == NULL) {
    // Dereference arrays
    Py_XDECREF(arr_offsetrow);
    Py_XDECREF(arr_b);
    Py_XDECREF(arr_U);
    Py_XDECREF(arr_W);
    Py_XDECREF(arr_phi);
    Py_XDECREF(arr_G);
    Py_XDECREF(arr_x);
    Py_XDECREF(arr_grad_x);
    Py_XDECREF(arr_grad_U);
    Py_XDECREF(arr_grad_W);
    Py_XDECREF(arr_grad_phi);
    Py_XDECREF(arr_grad_G);
    Py_XDECREF(arr_grad_y);
    Py_XDECREF(arr_g);
    return NULL;
  }

  // Get C-types pointers to numpy arrays
  long *offsetrow = (long*)PyArray_DATA(arr_offsetrow);
  long *b = (long*)PyArray_DATA(arr_b);
  double *U = (double*)PyArray_DATA(arr_U);
  double *W = (double*)PyArray_DATA(arr_W);
  double *phi = (double*)PyArray_DATA(arr_phi);
  double *G = (double*)PyArray_DATA(arr_G);
  double *x = (double*)PyArray_DATA(arr_x);
  double *grad_x = (double*)PyArray_DATA(arr_grad_x);
  double *grad_U = (double*)PyArray_DATA(arr_grad_U);
  double *grad_W = (double*)PyArray_DATA(arr_grad_W);
  double *grad_phi = (double*)PyArray_DATA(arr_grad_phi);
  double *grad_G = (double*)PyArray_DATA(arr_grad_G);
  double *grad_y = (double*)PyArray_DATA(arr_grad_y);
  double *g = (double*)PyArray_DATA(arr_g);

  // Call the C function from libspleaf
  spleaf_solveLT_back(
    n,
    r,
    offsetrow,
    b,
    U,
    W,
    phi,
    G,
    x,
    grad_x,
    grad_U,
    grad_W,
    grad_phi,
    grad_G,
    grad_y,
    g);

  // Dereference arrays
  Py_XDECREF(arr_offsetrow);
  Py_XDECREF(arr_b);
  Py_XDECREF(arr_U);
  Py_XDECREF(arr_W);
  Py_XDECREF(arr_phi);
  Py_XDECREF(arr_G);
  Py_XDECREF(arr_x);
  Py_XDECREF(arr_grad_x);
  Py_XDECREF(arr_grad_U);
  Py_XDECREF(arr_grad_W);
  Py_XDECREF(arr_grad_phi);
  Py_XDECREF(arr_grad_G);
  Py_XDECREF(arr_grad_y);
  Py_XDECREF(arr_g);

  Py_RETURN_NONE;
}