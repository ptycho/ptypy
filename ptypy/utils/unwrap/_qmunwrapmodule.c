/*
Python bindings for the quality-map phase unwrapping routines.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
*/

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <limits.h>

#include "_qmunwrap.h"


static int check_array(Py_buffer* b, const char* name)
{
    const char* fmt = b->format;

    if (b->ndim != 2)
    {
        PyErr_Format(PyExc_ValueError, "%s must be a 2D array", name);
        return -1;
    }
    if (fmt && (fmt[0] == '@' || fmt[0] == '='))
        fmt++;
    if (b->itemsize != sizeof(double) || !fmt || fmt[0] != 'd' || fmt[1] != '\0')
    {
        PyErr_Format(PyExc_TypeError, "%s must be a float64 array", name);
        return -1;
    }
    return 0;
}


// Check that both arrays have the same shape and that it fits in an int.
static int check_shapes(Py_buffer* a, Py_buffer* b, int* N0, int* N1)
{
    if (a->shape[0] != b->shape[0] || a->shape[1] != b->shape[1])
    {
        PyErr_SetString(PyExc_ValueError, "both arrays must have the same shape");
        return -1;
    }
    if (a->shape[0] > INT_MAX || a->shape[1] > INT_MAX ||
        (a->shape[1] > 0 && a->shape[0] > INT_MAX/a->shape[1]))
    {
        PyErr_SetString(PyExc_ValueError, "array is too large");
        return -1;
    }
    *N0 = (int)a->shape[0];
    *N1 = (int)a->shape[1];
    return 0;
}


PyDoc_STRVAR(unwrap_doc,
"unwrap(phase, out, num_levels, start0, start1)\n"
"\n"
"Unwrap the 2D float64 array `phase` into the writable float64 array `out`,\n"
"which must be C-contiguous and of the same shape.");

static PyObject* py_unwrap(PyObject* Py_UNUSED(self), PyObject* args)
{
    PyObject *phase_obj, *out_obj;
    int num_levels, start0, start1;
    int N0, N1, status;
    Py_buffer phase, out;

    if (!PyArg_ParseTuple(args, "OOiii", &phase_obj, &out_obj,
                          &num_levels, &start0, &start1))
        return NULL;

    if (PyObject_GetBuffer(phase_obj, &phase, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0)
        return NULL;
    if (PyObject_GetBuffer(out_obj, &out,
                           PyBUF_C_CONTIGUOUS | PyBUF_FORMAT | PyBUF_WRITABLE) < 0)
    {
        PyBuffer_Release(&phase);
        return NULL;
    }

    if (check_array(&phase, "phase") < 0 || check_array(&out, "out") < 0)
        goto fail;
    if (check_shapes(&phase, &out, &N0, &N1) < 0)
        goto fail;

    Py_BEGIN_ALLOW_THREADS
    status = qmunwrap_unwrap((double*)phase.buf, N0, N1, num_levels,
                             start0, start1, (double*)out.buf);
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&phase);
    PyBuffer_Release(&out);

    if (status == QMUNWRAP_ENOMEM)
        return PyErr_NoMemory();
    if (status != QMUNWRAP_OK)
    {
        PyErr_SetString(PyExc_ValueError,
                        "num_levels must be >= 1 and the starting point must lie "
                        "inside the array");
        return NULL;
    }
    Py_RETURN_NONE;

fail:
    PyBuffer_Release(&phase);
    PyBuffer_Release(&out);
    return NULL;
}


PyDoc_STRVAR(qualitymap_doc,
"qualitymap(phase, qmap)\n"
"\n"
"Accumulate the squared wrapped gradient of the 2D float64 array `phase` into\n"
"the writable float64 array `qmap`, which must be C-contiguous, of the same\n"
"shape, and zero-filled by the caller.");

static PyObject* py_qualitymap(PyObject* Py_UNUSED(self), PyObject* args)
{
    PyObject *phase_obj, *qmap_obj;
    int N0, N1;
    Py_buffer phase, qmap;

    if (!PyArg_ParseTuple(args, "OO", &phase_obj, &qmap_obj))
        return NULL;

    if (PyObject_GetBuffer(phase_obj, &phase, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0)
        return NULL;
    if (PyObject_GetBuffer(qmap_obj, &qmap,
                           PyBUF_C_CONTIGUOUS | PyBUF_FORMAT | PyBUF_WRITABLE) < 0)
    {
        PyBuffer_Release(&phase);
        return NULL;
    }

    if (check_array(&phase, "phase") < 0 || check_array(&qmap, "qmap") < 0)
        goto fail;
    if (check_shapes(&phase, &qmap, &N0, &N1) < 0)
        goto fail;

    Py_BEGIN_ALLOW_THREADS
    qmunwrap_qualitymap((double*)phase.buf, N0, N1, (double*)qmap.buf);
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&phase);
    PyBuffer_Release(&qmap);
    Py_RETURN_NONE;

fail:
    PyBuffer_Release(&phase);
    PyBuffer_Release(&qmap);
    return NULL;
}


static PyMethodDef qmunwrap_methods[] = {
    {"unwrap", py_unwrap, METH_VARARGS, unwrap_doc},
    {"qualitymap", py_qualitymap, METH_VARARGS, qualitymap_doc},
    {NULL, NULL, 0, NULL}
};


PyDoc_STRVAR(module_doc, "Quality-map phase unwrapping (C implementation).");

static struct PyModuleDef qmunwrap_module = {
    PyModuleDef_HEAD_INIT,
    "_qmunwrap",
    module_doc,
    -1,
    qmunwrap_methods,
    NULL, NULL, NULL, NULL
};


PyMODINIT_FUNC PyInit__qmunwrap(void)
{
    return PyModule_Create(&qmunwrap_module);
}
