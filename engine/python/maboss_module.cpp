#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include "maboss_sim.cpp"

/*  define functions in module */
static PyMethodDef cMaBoSS[] =
{ 
     {NULL, NULL, 0, NULL}
};

/* module initialization */
/* Python version 3*/
static struct PyModuleDef cMaBoSSDef =
{
    PyModuleDef_HEAD_INIT,
    "maboss_module", 
    "Some documentation",
    -1,
    cMaBoSS
};

PyMODINIT_FUNC
PyInit_maboss_module(void)
{

    PyObject *m;
    if (PyType_Ready(&cMaBoSSSim) < 0){
        return NULL;
    }

    m = PyModule_Create(&cMaBoSSDef);

    Py_INCREF(&cMaBoSSSim);
    if (PyModule_AddObject(m, "MaBoSSSim", (PyObject *) &cMaBoSSSim) < 0) {
        Py_DECREF(&cMaBoSSSim);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}


