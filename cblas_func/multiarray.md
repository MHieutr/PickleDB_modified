#### np.dot 
/*NUMPY_API
 * Numeric.matrixproduct2(a,v,out)
 * just like inner product but does the swapaxes stuff on the fly
 */
 ```c
NPY_NO_EXPORT PyObject *
PyArray_MatrixProduct2(PyObject *op1, PyObject *op2, PyArrayObject* out)
{
    PyArrayObject *ap1, *ap2, *out_buf = NULL, *result = NULL;
    PyArrayIterObject *it1, *it2;
    npy_intp i, j, l;
    int typenum, nd, axis, matchDim;
    npy_intp is1, is2, os;
    char *op;
    npy_intp dimensions[NPY_MAXDIMS];
    PyArray_DotFunc *dot;
    PyArray_Descr *typec = NULL;
    NPY_BEGIN_THREADS_DEF;

    typenum = PyArray_ObjectType(op1, NPY_NOTYPE);
    if (typenum == NPY_NOTYPE) {
        return NULL;
    }
    typenum = PyArray_ObjectType(op2, typenum);
    if (typenum == NPY_NOTYPE) {
        return NULL;
    }

    typec = PyArray_DescrFromType(typenum);
    if (typec == NULL) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError,
                            "Cannot find a common data type.");
        }
        return NULL;
    }

    Py_INCREF(typec);
    ap1 = (PyArrayObject *)PyArray_FromAny(op1, typec, 0, 0,
                                        NPY_ARRAY_ALIGNED, NULL);
    if (ap1 == NULL) {
        Py_DECREF(typec);
        return NULL;
    }
    ap2 = (PyArrayObject *)PyArray_FromAny(op2, typec, 0, 0,
                                        NPY_ARRAY_ALIGNED, NULL);
    if (ap2 == NULL) {
        Py_DECREF(ap1);
        return NULL;
    }

#if defined(HAVE_CBLAS)
    if (PyArray_NDIM(ap1) <= 2 && PyArray_NDIM(ap2) <= 2 &&
            (NPY_DOUBLE == typenum || NPY_CDOUBLE == typenum ||
             NPY_FLOAT == typenum || NPY_CFLOAT == typenum)) {
        return cblas_matrixproduct(typenum, ap1, ap2, out);
    }
#endif

    if (PyArray_NDIM(ap1) == 0 || PyArray_NDIM(ap2) == 0) {
        PyObject *mul_res = PyObject_CallFunctionObjArgs(
                n_ops.multiply, ap1, ap2, out, NULL);
        Py_DECREF(ap1);
        Py_DECREF(ap2);
        return mul_res;
    }
    l = PyArray_DIMS(ap1)[PyArray_NDIM(ap1) - 1];
    if (PyArray_NDIM(ap2) > 1) {
        matchDim = PyArray_NDIM(ap2) - 2;
    }
    else {
        matchDim = 0;
    }
    if (PyArray_DIMS(ap2)[matchDim] != l) {
        dot_alignment_error(ap1, PyArray_NDIM(ap1) - 1, ap2, matchDim);
        goto fail;
    }
    nd = PyArray_NDIM(ap1) + PyArray_NDIM(ap2) - 2;
    if (nd > NPY_MAXDIMS) {
        PyErr_SetString(PyExc_ValueError, "dot: too many dimensions in result");
        goto fail;
    }
    j = 0;
    for (i = 0; i < PyArray_NDIM(ap1) - 1; i++) {
        dimensions[j++] = PyArray_DIMS(ap1)[i];
    }
    for (i = 0; i < PyArray_NDIM(ap2) - 2; i++) {
        dimensions[j++] = PyArray_DIMS(ap2)[i];
    }
    if (PyArray_NDIM(ap2) > 1) {
        dimensions[j++] = PyArray_DIMS(ap2)[PyArray_NDIM(ap2)-1];
    }

    is1 = PyArray_STRIDES(ap1)[PyArray_NDIM(ap1)-1];
    is2 = PyArray_STRIDES(ap2)[matchDim];
    /* Choose which subtype to return */
    out_buf = new_array_for_sum(ap1, ap2, out, nd, dimensions, typenum, &result);
    if (out_buf == NULL) {
        goto fail;
    }
    /* Ensure that multiarray.dot(<Nx0>,<0xM>) -> zeros((N,M)) */
    if (PyArray_SIZE(ap1) == 0 && PyArray_SIZE(ap2) == 0) {
        if (PyArray_AssignZero(out_buf, NULL) < 0) {
            goto fail;
        }
    }

    dot = PyDataType_GetArrFuncs(PyArray_DESCR(out_buf))->dotfunc;
    if (dot == NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "dot not available for this type");
        goto fail;
    }

    op = PyArray_DATA(out_buf);
    os = PyArray_ITEMSIZE(out_buf);
    axis = PyArray_NDIM(ap1)-1;
    it1 = (PyArrayIterObject *)
        PyArray_IterAllButAxis((PyObject *)ap1, &axis);
    if (it1 == NULL) {
        goto fail;
    }
    it2 = (PyArrayIterObject *)
        PyArray_IterAllButAxis((PyObject *)ap2, &matchDim);
    if (it2 == NULL) {
        Py_DECREF(it1);
        goto fail;
    }
    NPY_BEGIN_THREADS_DESCR(PyArray_DESCR(ap2));
    while (it1->index < it1->size) {
        while (it2->index < it2->size) {
            dot(it1->dataptr, is1, it2->dataptr, is2, op, l, NULL);
            op += os;
            PyArray_ITER_NEXT(it2);
        }
        PyArray_ITER_NEXT(it1);
        PyArray_ITER_RESET(it2);
    }
    NPY_END_THREADS_DESCR(PyArray_DESCR(ap2));
    Py_DECREF(it1);
    Py_DECREF(it2);
    if (PyErr_Occurred()) {
        /* only for OBJECT arrays */
        goto fail;
    }
    Py_DECREF(ap1);
    Py_DECREF(ap2);

    /* Trigger possible copy-back into `result` */
    PyArray_ResolveWritebackIfCopy(out_buf);
    Py_DECREF(out_buf);

    return (PyObject *)result;

fail:
    Py_XDECREF(ap1);
    Py_XDECREF(ap2);
    Py_XDECREF(out_buf);
    Py_XDECREF(result);
    return NULL;
}

static PyObject *
array_dot(PyArrayObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *a = (PyObject *)self, *b, *o = NULL;
    PyArrayObject *ret;
    NPY_PREPARE_ARGPARSER;

    if (npy_parse_arguments("dot", args, len_args, kwnames,
            "b", NULL, &b,
            "|out", NULL, &o,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }

    if (o != NULL) {
        if (o == Py_None) {
            o = NULL;
        }
        else if (!PyArray_Check(o)) {
            PyErr_SetString(PyExc_TypeError,
                            "'out' must be an array");
            return NULL;
        }
    }
    ret = (PyArrayObject *)PyArray_MatrixProduct2(a, b, (PyArrayObject *)o);
    return PyArray_Return(ret);
}
```
#### np.sum
```python
def sum(a, axis=None, dtype=None, out=None, keepdims=np._NoValue,
        initial=np._NoValue, where=np._NoValue):
    """
    Sum of array elements over a given axis.

    Parameters
    ----------
    a : array_like
        Elements to sum.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a sum is performed.  The default,
        axis=None, will sum all of the elements of the input array.  If
        axis is negative it counts from the last to the first axis. If
        axis is a tuple of ints, a sum is performed on all of the axes
        specified in the tuple instead of a single axis or all the axes as
        before.
    dtype : dtype, optional
        The type of the returned array and of the accumulator in which the
        elements are summed.  The dtype of `a` is used by default unless `a`
        has an integer dtype of less precision than the default platform
        integer.  In that case, if `a` is signed then the platform integer
        is used while if `a` is unsigned then an unsigned integer of the
        same precision as the platform integer is used.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output, but the type of the output
        values will be cast if necessary.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `sum` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.
    initial : scalar, optional
        Starting value for the sum. See `~numpy.ufunc.reduce` for details.
    where : array_like of bool, optional
        Elements to include in the sum. See `~numpy.ufunc.reduce` for details.

    Returns
    -------
    sum_along_axis : ndarray
        An array with the same shape as `a`, with the specified
        axis removed.   If `a` is a 0-d array, or if `axis` is None, a scalar
        is returned.  If an output array is specified, a reference to
        `out` is returned.

    See Also
    --------
    ndarray.sum : Equivalent method.
    add: ``numpy.add.reduce`` equivalent function.
    cumsum : Cumulative sum of array elements.
    trapezoid : Integration of array values using composite trapezoidal rule.

    mean, average

    Notes
    -----
    Arithmetic is modular when using integer types, and no error is
    raised on overflow.

    The sum of an empty array is the neutral element 0:

    >>> np.sum([])
    0.0

    For floating point numbers the numerical precision of sum (and
    ``np.add.reduce``) is in general limited by directly adding each number
    individually to the result causing rounding errors in every step.
    However, often numpy will use a  numerically better approach (partial
    pairwise summation) leading to improved precision in many use-cases.
    This improved precision is always provided when no ``axis`` is given.
    When ``axis`` is given, it will depend on which axis is summed.
    Technically, to provide the best speed possible, the improved precision
    is only used when the summation is along the fast axis in memory.
    Note that the exact precision may vary depending on other parameters.
    In contrast to NumPy, Python's ``math.fsum`` function uses a slower but
    more precise approach to summation.
    Especially when summing a large number of lower precision floating point
    numbers, such as ``float32``, numerical errors can become significant.
    In such cases it can be advisable to use `dtype="float64"` to use a higher
    precision for the output.

    Examples
    --------
    >>> import numpy as np
    >>> np.sum([0.5, 1.5])
    2.0
    >>> np.sum([0.5, 0.7, 0.2, 1.5], dtype=np.int32)
    np.int32(1)
    >>> np.sum([[0, 1], [0, 5]])
    6
    >>> np.sum([[0, 1], [0, 5]], axis=0)
    array([0, 6])
    >>> np.sum([[0, 1], [0, 5]], axis=1)
    array([1, 5])
    >>> np.sum([[0, 1], [np.nan, 5]], where=[False, True], axis=1)
    array([1., 5.])

    If the accumulator is too small, overflow occurs:

    >>> np.ones(128, dtype=np.int8).sum(dtype=np.int8)
    np.int8(-128)

    You can also start the sum with a value other than zero:

    >>> np.sum([10], initial=5)
    15
    """
    if isinstance(a, _gentype):
        # 2018-02-25, 1.15.0
        warnings.warn(
            "Calling np.sum(generator) is deprecated, and in the future will "
            "give a different result. Use np.sum(np.fromiter(generator)) or "
            "the python sum builtin instead.",
            DeprecationWarning, stacklevel=2
        )

        res = _sum_(a)
        if out is not None:
            out[...] = res
            return out
        return res

    return _wrapreduction(
        a, np.add, 'sum', axis, dtype, out,
        keepdims=keepdims, initial=initial, where=where
    )

def _wrapreduction(obj, ufunc, method, axis, dtype, out, **kwargs):
    passkwargs = {k: v for k, v in kwargs.items()
                  if v is not np._NoValue}

    if type(obj) is not mu.ndarray:
        try:
            reduction = getattr(obj, method)
        except AttributeError:
            pass
        else:
            # This branch is needed for reductions like any which don't
            # support a dtype.
            if dtype is not None:
                return reduction(axis=axis, dtype=dtype, out=out, **passkwargs)
            else:
                return reduction(axis=axis, out=out, **passkwargs)

    return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
```

#### np.sqrt 
```c
static void
nc_sqrt@c@(@ctype@ *x, @ctype@ *r)
{
    *r = npy_csqrt@c@(*x);
    return;
}
```
#### sort
```c
static PyObject *
array_sort(PyArrayObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    int axis = -1;
    int val;
    NPY_SORTKIND sortkind = _NPY_SORT_UNDEFINED;
    PyObject *order = NULL;
    PyArray_Descr *saved = NULL;
    PyArray_Descr *newd;
    int stable = -1;
    NPY_PREPARE_ARGPARSER;

    if (npy_parse_arguments("sort", args, len_args, kwnames,
            "|axis", &PyArray_PythonPyIntFromInt, &axis,
            "|kind", &PyArray_SortkindConverter, &sortkind,
            "|order", NULL, &order,
            "$stable", &PyArray_OptionalBoolConverter, &stable,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }
    if (order == Py_None) {
        order = NULL;
    }
    if (order != NULL) {
        PyObject *new_name;
        PyObject *_numpy_internal;
        saved = PyArray_DESCR(self);
        if (!PyDataType_HASFIELDS(saved)) {
            PyErr_SetString(PyExc_ValueError, "Cannot specify " \
                            "order when the array has no fields.");
            return NULL;
        }
        _numpy_internal = PyImport_ImportModule("numpy._core._internal");
        if (_numpy_internal == NULL) {
            return NULL;
        }
        new_name = PyObject_CallMethod(_numpy_internal, "_newnames",
                                       "OO", saved, order);
        Py_DECREF(_numpy_internal);
        if (new_name == NULL) {
            return NULL;
        }
        newd = PyArray_DescrNew(saved);
        if (newd == NULL) {
            Py_DECREF(new_name);
            return NULL;
        }
        Py_DECREF(((_PyArray_LegacyDescr *)newd)->names);
        ((_PyArray_LegacyDescr *)newd)->names = new_name;
        ((PyArrayObject_fields *)self)->descr = newd;
    }
    if (sortkind != _NPY_SORT_UNDEFINED && stable != -1) {
        PyErr_SetString(PyExc_ValueError,
            "`kind` and `stable` parameters can't be provided at "
            "the same time. Use only one of them.");
        return NULL;
    }
    else if ((sortkind == _NPY_SORT_UNDEFINED && stable == -1) || (stable == 0)) {
        sortkind = NPY_QUICKSORT;
    }
    else if (stable == 1) {
        sortkind = NPY_STABLESORT;
    }

    val = PyArray_Sort(self, axis, sortkind);
    if (order != NULL) {
        Py_XDECREF(PyArray_DESCR(self));
        ((PyArrayObject_fields *)self)->descr = saved;
    }
    if (val < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}
```