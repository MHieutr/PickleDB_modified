```c
NPY_NO_EXPORT PyObject *
cblas_matrixproduct(int typenum, PyArrayObject *ap1, PyArrayObject *ap2,
                    PyArrayObject *out)
{
    PyArrayObject *result = NULL, *out_buf = NULL;
    npy_intp j, lda, ldb;
    npy_intp l;
    int nd;
    npy_intp ap1stride = 0;
    npy_intp dimensions[NPY_MAXDIMS];
    npy_intp numbytes;
    MatrixShape ap1shape, ap2shape;

    if (_bad_strides(ap1)) {
            PyObject *op1 = PyArray_NewCopy(ap1, NPY_ANYORDER);

            Py_DECREF(ap1);
            ap1 = (PyArrayObject *)op1;
            if (ap1 == NULL) {
                goto fail;
            }
    }
    if (_bad_strides(ap2)) {
            PyObject *op2 = PyArray_NewCopy(ap2, NPY_ANYORDER);

            Py_DECREF(ap2);
            ap2 = (PyArrayObject *)op2;
            if (ap2 == NULL) {
                goto fail;
            }
    }
    ap1shape = _select_matrix_shape(ap1);
    ap2shape = _select_matrix_shape(ap2);

    if (ap1shape == _scalar || ap2shape == _scalar) {
        PyArrayObject *oap1, *oap2;
        oap1 = ap1; oap2 = ap2;
        /* One of ap1 or ap2 is a scalar */
        if (ap1shape == _scalar) {
            /* Make ap2 the scalar */
            PyArrayObject *t = ap1;
            ap1 = ap2;
            ap2 = t;
            ap1shape = ap2shape;
            ap2shape = _scalar;
        }

        if (ap1shape == _row) {
            ap1stride = PyArray_STRIDE(ap1, 1);
        }
        else if (PyArray_NDIM(ap1) > 0) {
            ap1stride = PyArray_STRIDE(ap1, 0);
        }

        if (PyArray_NDIM(ap1) == 0 || PyArray_NDIM(ap2) == 0) {
            npy_intp *thisdims;
            if (PyArray_NDIM(ap1) == 0) {
                nd = PyArray_NDIM(ap2);
                thisdims = PyArray_DIMS(ap2);
            }
            else {
                nd = PyArray_NDIM(ap1);
                thisdims = PyArray_DIMS(ap1);
            }
            l = 1;
            for (j = 0; j < nd; j++) {
                dimensions[j] = thisdims[j];
                l *= dimensions[j];
            }
        }
        else {
            l = PyArray_DIM(oap1, PyArray_NDIM(oap1) - 1);

            if (PyArray_DIM(oap2, 0) != l) {
                dot_alignment_error(oap1, PyArray_NDIM(oap1) - 1, oap2, 0);
                goto fail;
            }
            nd = PyArray_NDIM(ap1) + PyArray_NDIM(ap2) - 2;
            /*
             * nd = 0 or 1 or 2. If nd == 0 do nothing ...
             */
            if (nd == 1) {
                /*
                 * Either PyArray_NDIM(ap1) is 1 dim or PyArray_NDIM(ap2) is
                 * 1 dim and the other is 2 dim
                 */
                dimensions[0] = (PyArray_NDIM(oap1) == 2) ?
                                PyArray_DIM(oap1, 0) : PyArray_DIM(oap2, 1);
                l = dimensions[0];
                /*
                 * Fix it so that dot(shape=(N,1), shape=(1,))
                 * and dot(shape=(1,), shape=(1,N)) both return
                 * an (N,) array (but use the fast scalar code)
                 */
            }
            else if (nd == 2) {
                dimensions[0] = PyArray_DIM(oap1, 0);
                dimensions[1] = PyArray_DIM(oap2, 1);
                /*
                 * We need to make sure that dot(shape=(1,1), shape=(1,N))
                 * and dot(shape=(N,1),shape=(1,1)) uses
                 * scalar multiplication appropriately
                 */
                if (ap1shape == _row) {
                    l = dimensions[1];
                }
                else {
                    l = dimensions[0];
                }
            }

            /* Check if the summation dimension is 0-sized */
            if (PyArray_DIM(oap1, PyArray_NDIM(oap1) - 1) == 0) {
                l = 0;
            }
        }
    }
    else {
        /*
         * (PyArray_NDIM(ap1) <= 2 && PyArray_NDIM(ap2) <= 2)
         * Both ap1 and ap2 are vectors or matrices
         */
        l = PyArray_DIM(ap1, PyArray_NDIM(ap1) - 1);

        if (PyArray_DIM(ap2, 0) != l) {
            dot_alignment_error(ap1, PyArray_NDIM(ap1) - 1, ap2, 0);
            goto fail;
        }
        nd = PyArray_NDIM(ap1) + PyArray_NDIM(ap2) - 2;

        if (nd == 1) {
            dimensions[0] = (PyArray_NDIM(ap1) == 2) ?
                            PyArray_DIM(ap1, 0) : PyArray_DIM(ap2, 1);
        }
        else if (nd == 2) {
            dimensions[0] = PyArray_DIM(ap1, 0);
            dimensions[1] = PyArray_DIM(ap2, 1);
        }
    }

    out_buf = new_array_for_sum(ap1, ap2, out, nd, dimensions, typenum, &result);
    if (out_buf == NULL) {
        goto fail;
    }

    numbytes = PyArray_NBYTES(out_buf);
    memset(PyArray_DATA(out_buf), 0, numbytes);
    if (numbytes == 0 || l == 0) {
            Py_DECREF(ap1);
            Py_DECREF(ap2);
            Py_DECREF(out_buf);
            return PyArray_Return(result);
    }

    if (ap2shape == _scalar) {
        /*
         * Multiplication by a scalar -- Level 1 BLAS
         * if ap1shape is a matrix and we are not contiguous, then we can't
         * just blast through the entire array using a single striding factor
         */
        NPY_BEGIN_ALLOW_THREADS;

        if (typenum == NPY_DOUBLE) {
            if (l == 1) {
                *((double *)PyArray_DATA(out_buf)) = *((double *)PyArray_DATA(ap2)) *
                                                 *((double *)PyArray_DATA(ap1));
            }
            else if (ap1shape != _matrix) {
                CBLAS_FUNC(cblas_daxpy)(l,
                            *((double *)PyArray_DATA(ap2)),
                            (double *)PyArray_DATA(ap1),
                            ap1stride/sizeof(double),
                            (double *)PyArray_DATA(out_buf), 1);
            }
            else {
                int maxind, oind;
                npy_intp i, a1s, outs;
                char *ptr, *optr;
                double val;

                maxind = (PyArray_DIM(ap1, 0) >= PyArray_DIM(ap1, 1) ? 0 : 1);
                oind = 1 - maxind;
                ptr = PyArray_DATA(ap1);
                optr = PyArray_DATA(out_buf);
                l = PyArray_DIM(ap1, maxind);
                val = *((double *)PyArray_DATA(ap2));
                a1s = PyArray_STRIDE(ap1, maxind) / sizeof(double);
                outs = PyArray_STRIDE(out_buf, maxind) / sizeof(double);
                for (i = 0; i < PyArray_DIM(ap1, oind); i++) {
                    CBLAS_FUNC(cblas_daxpy)(l, val, (double *)ptr, a1s,
                                (double *)optr, outs);
                    ptr += PyArray_STRIDE(ap1, oind);
                    optr += PyArray_STRIDE(out_buf, oind);
                }
            }
        }
        else if (typenum == NPY_CDOUBLE) {
            if (l == 1) {
                npy_cdouble *ptr1, *ptr2, *res;

                ptr1 = (npy_cdouble *)PyArray_DATA(ap2);
                ptr2 = (npy_cdouble *)PyArray_DATA(ap1);
                res = (npy_cdouble *)PyArray_DATA(out_buf);
                npy_csetreal(res, npy_creal(*ptr1) * npy_creal(*ptr2)
                                            - npy_cimag(*ptr1) * npy_cimag(*ptr2));
                npy_csetimag(res, npy_creal(*ptr1) * npy_cimag(*ptr2)
                                            + npy_cimag(*ptr1) * npy_creal(*ptr2));
            }
            else if (ap1shape != _matrix) {
                CBLAS_FUNC(cblas_zaxpy)(l,
                            (double *)PyArray_DATA(ap2),
                            (double *)PyArray_DATA(ap1),
                            ap1stride/sizeof(npy_cdouble),
                            (double *)PyArray_DATA(out_buf), 1);
            }
            else {
                int maxind, oind;
                npy_intp i, a1s, outs;
                char *ptr, *optr;
                double *pval;

                maxind = (PyArray_DIM(ap1, 0) >= PyArray_DIM(ap1, 1) ? 0 : 1);
                oind = 1 - maxind;
                ptr = PyArray_DATA(ap1);
                optr = PyArray_DATA(out_buf);
                l = PyArray_DIM(ap1, maxind);
                pval = (double *)PyArray_DATA(ap2);
                a1s = PyArray_STRIDE(ap1, maxind) / sizeof(npy_cdouble);
                outs = PyArray_STRIDE(out_buf, maxind) / sizeof(npy_cdouble);
                for (i = 0; i < PyArray_DIM(ap1, oind); i++) {
                    CBLAS_FUNC(cblas_zaxpy)(l, pval, (double *)ptr, a1s,
                                (double *)optr, outs);
                    ptr += PyArray_STRIDE(ap1, oind);
                    optr += PyArray_STRIDE(out_buf, oind);
                }
            }
        }
        else if (typenum == NPY_FLOAT) {
            if (l == 1) {
                *((float *)PyArray_DATA(out_buf)) = *((float *)PyArray_DATA(ap2)) *
                    *((float *)PyArray_DATA(ap1));
            }
            else if (ap1shape != _matrix) {
                CBLAS_FUNC(cblas_saxpy)(l,
                            *((float *)PyArray_DATA(ap2)),
                            (float *)PyArray_DATA(ap1),
                            ap1stride/sizeof(float),
                            (float *)PyArray_DATA(out_buf), 1);
            }
            else {
                int maxind, oind;
                npy_intp i, a1s, outs;
                char *ptr, *optr;
                float val;

                maxind = (PyArray_DIM(ap1, 0) >= PyArray_DIM(ap1, 1) ? 0 : 1);
                oind = 1 - maxind;
                ptr = PyArray_DATA(ap1);
                optr = PyArray_DATA(out_buf);
                l = PyArray_DIM(ap1, maxind);
                val = *((float *)PyArray_DATA(ap2));
                a1s = PyArray_STRIDE(ap1, maxind) / sizeof(float);
                outs = PyArray_STRIDE(out_buf, maxind) / sizeof(float);
                for (i = 0; i < PyArray_DIM(ap1, oind); i++) {
                    CBLAS_FUNC(cblas_saxpy)(l, val, (float *)ptr, a1s,
                                (float *)optr, outs);
                    ptr += PyArray_STRIDE(ap1, oind);
                    optr += PyArray_STRIDE(out_buf, oind);
                }
            }
        }
        else if (typenum == NPY_CFLOAT) {
            if (l == 1) {
                npy_cfloat *ptr1, *ptr2, *res;

                ptr1 = (npy_cfloat *)PyArray_DATA(ap2);
                ptr2 = (npy_cfloat *)PyArray_DATA(ap1);
                res = (npy_cfloat *)PyArray_DATA(out_buf);
                npy_csetrealf(res, npy_crealf(*ptr1) * npy_crealf(*ptr2)
                                           - npy_cimagf(*ptr1) * npy_cimagf(*ptr2));
                npy_csetimagf(res, npy_crealf(*ptr1) * npy_cimagf(*ptr2)
                                           + npy_cimagf(*ptr1) * npy_crealf(*ptr2));
            }
            else if (ap1shape != _matrix) {
                CBLAS_FUNC(cblas_caxpy)(l,
                            (float *)PyArray_DATA(ap2),
                            (float *)PyArray_DATA(ap1),
                            ap1stride/sizeof(npy_cfloat),
                            (float *)PyArray_DATA(out_buf), 1);
            }
            else {
                int maxind, oind;
                npy_intp i, a1s, outs;
                char *ptr, *optr;
                float *pval;

                maxind = (PyArray_DIM(ap1, 0) >= PyArray_DIM(ap1, 1) ? 0 : 1);
                oind = 1 - maxind;
                ptr = PyArray_DATA(ap1);
                optr = PyArray_DATA(out_buf);
                l = PyArray_DIM(ap1, maxind);
                pval = (float *)PyArray_DATA(ap2);
                a1s = PyArray_STRIDE(ap1, maxind) / sizeof(npy_cfloat);
                outs = PyArray_STRIDE(out_buf, maxind) / sizeof(npy_cfloat);
                for (i = 0; i < PyArray_DIM(ap1, oind); i++) {
                    CBLAS_FUNC(cblas_caxpy)(l, pval, (float *)ptr, a1s,
                                (float *)optr, outs);
                    ptr += PyArray_STRIDE(ap1, oind);
                    optr += PyArray_STRIDE(out_buf, oind);
                }
            }
        }
        NPY_END_ALLOW_THREADS;
    }
    else if ((ap2shape == _column) && (ap1shape != _matrix)) {
        NPY_BEGIN_ALLOW_THREADS;

        /* Dot product between two vectors -- Level 1 BLAS */
        PyDataType_GetArrFuncs(PyArray_DESCR(out_buf))->dotfunc(
                 PyArray_DATA(ap1), PyArray_STRIDE(ap1, (ap1shape == _row)),
                 PyArray_DATA(ap2), PyArray_STRIDE(ap2, 0),
                 PyArray_DATA(out_buf), l, NULL);
        NPY_END_ALLOW_THREADS;
    }
    else if (ap1shape == _matrix && ap2shape != _matrix) {
        /* Matrix vector multiplication -- Level 2 BLAS */
        /* lda must be MAX(M,1) */
        enum CBLAS_ORDER Order;
        npy_intp ap2s;

        if (!PyArray_ISONESEGMENT(ap1)) {
            PyObject *new;
            new = PyArray_Copy(ap1);
            Py_DECREF(ap1);
            ap1 = (PyArrayObject *)new;
            if (new == NULL) {
                goto fail;
            }
        }
        NPY_BEGIN_ALLOW_THREADS
        if (PyArray_ISCONTIGUOUS(ap1)) {
            Order = CblasRowMajor;
            lda = (PyArray_DIM(ap1, 1) > 1 ? PyArray_DIM(ap1, 1) : 1);
        }
        else {
            Order = CblasColMajor;
            lda = (PyArray_DIM(ap1, 0) > 1 ? PyArray_DIM(ap1, 0) : 1);
        }
        ap2s = PyArray_STRIDE(ap2, 0) / PyArray_ITEMSIZE(ap2);
        gemv(typenum, Order, CblasNoTrans, ap1, lda, ap2, ap2s, out_buf);
        NPY_END_ALLOW_THREADS;
    }
    else if (ap1shape != _matrix && ap2shape == _matrix) {
        /* Vector matrix multiplication -- Level 2 BLAS */
        enum CBLAS_ORDER Order;
        npy_intp ap1s;

        if (!PyArray_ISONESEGMENT(ap2)) {
            PyObject *new;
            new = PyArray_Copy(ap2);
            Py_DECREF(ap2);
            ap2 = (PyArrayObject *)new;
            if (new == NULL) {
                goto fail;
            }
        }
        NPY_BEGIN_ALLOW_THREADS
        if (PyArray_ISCONTIGUOUS(ap2)) {
            Order = CblasRowMajor;
            lda = (PyArray_DIM(ap2, 1) > 1 ? PyArray_DIM(ap2, 1) : 1);
        }
        else {
            Order = CblasColMajor;
            lda = (PyArray_DIM(ap2, 0) > 1 ? PyArray_DIM(ap2, 0) : 1);
        }
        if (ap1shape == _row) {
            ap1s = PyArray_STRIDE(ap1, 1) / PyArray_ITEMSIZE(ap1);
        }
        else {
            ap1s = PyArray_STRIDE(ap1, 0) / PyArray_ITEMSIZE(ap1);
        }
        gemv(typenum, Order, CblasTrans, ap2, lda, ap1, ap1s, out_buf);
        NPY_END_ALLOW_THREADS;
    }
    else {
        /*
         * (PyArray_NDIM(ap1) == 2 && PyArray_NDIM(ap2) == 2)
         * Matrix matrix multiplication -- Level 3 BLAS
         *  L x M  multiplied by M x N
         */
        enum CBLAS_ORDER Order;
        enum CBLAS_TRANSPOSE Trans1, Trans2;
        npy_intp M, N, L;

        /* Optimization possible: */
        /*
         * We may be able to handle single-segment arrays here
         * using appropriate values of Order, Trans1, and Trans2.
         */
        if (!PyArray_IS_C_CONTIGUOUS(ap2) && !PyArray_IS_F_CONTIGUOUS(ap2)) {
            PyObject *new = PyArray_Copy(ap2);

            Py_DECREF(ap2);
            ap2 = (PyArrayObject *)new;
            if (new == NULL) {
                goto fail;
            }
        }
        if (!PyArray_IS_C_CONTIGUOUS(ap1) && !PyArray_IS_F_CONTIGUOUS(ap1)) {
            PyObject *new = PyArray_Copy(ap1);

            Py_DECREF(ap1);
            ap1 = (PyArrayObject *)new;
            if (new == NULL) {
                goto fail;
            }
        }

        NPY_BEGIN_ALLOW_THREADS;

        Order = CblasRowMajor;
        Trans1 = CblasNoTrans;
        Trans2 = CblasNoTrans;
        L = PyArray_DIM(ap1, 0);
        N = PyArray_DIM(ap2, 1);
        M = PyArray_DIM(ap2, 0);
        lda = (PyArray_DIM(ap1, 1) > 1 ? PyArray_DIM(ap1, 1) : 1);
        ldb = (PyArray_DIM(ap2, 1) > 1 ? PyArray_DIM(ap2, 1) : 1);

        /*
         * Avoid temporary copies for arrays in Fortran order
         */
        if (PyArray_IS_F_CONTIGUOUS(ap1)) {
            Trans1 = CblasTrans;
            lda = (PyArray_DIM(ap1, 0) > 1 ? PyArray_DIM(ap1, 0) : 1);
        }
        if (PyArray_IS_F_CONTIGUOUS(ap2)) {
            Trans2 = CblasTrans;
            ldb = (PyArray_DIM(ap2, 0) > 1 ? PyArray_DIM(ap2, 0) : 1);
        }

        /*
         * Use syrk if we have a case of a matrix times its transpose.
         * Otherwise, use gemm for all other cases.
         */
        if (
            (PyArray_BYTES(ap1) == PyArray_BYTES(ap2)) &&
            (PyArray_DIM(ap1, 0) == PyArray_DIM(ap2, 1)) &&
            (PyArray_DIM(ap1, 1) == PyArray_DIM(ap2, 0)) &&
            (PyArray_STRIDE(ap1, 0) == PyArray_STRIDE(ap2, 1)) &&
            (PyArray_STRIDE(ap1, 1) == PyArray_STRIDE(ap2, 0)) &&
            ((Trans1 == CblasTrans) ^ (Trans2 == CblasTrans)) &&
            ((Trans1 == CblasNoTrans) ^ (Trans2 == CblasNoTrans))
        ) {
            if (Trans1 == CblasNoTrans) {
                syrk(typenum, Order, Trans1, N, M, ap1, lda, out_buf);
            }
            else {
                syrk(typenum, Order, Trans1, N, M, ap2, ldb, out_buf);
            }
        }
        else {
            gemm(typenum, Order, Trans1, Trans2, L, N, M, ap1, lda, ap2, ldb,
                 out_buf);
        }
        NPY_END_ALLOW_THREADS;
    }


    Py_DECREF(ap1);
    Py_DECREF(ap2);

    /* Trigger possible copyback into `result` */
    PyArray_ResolveWritebackIfCopy(out_buf);
    Py_DECREF(out_buf);

    return PyArray_Return(result);

fail:
    Py_XDECREF(ap1);
    Py_XDECREF(ap2);
    Py_XDECREF(out_buf);
    Py_XDECREF(result);
    return NULL;
}
```