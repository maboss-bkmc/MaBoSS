#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include "maboss_sim.cpp"

// I use these to define the name of the library, and the init function
// Not sure why we need this 2 level thingy... Came from https://stackoverflow.com/a/1489971/11713763
#if defined (MAXNODES) && MAXNODES > 64 
#define NAME2(fun,suffix) fun ## suffix
#define NAME1(fun,suffix) NAME2(fun,suffix)
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
#endif
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
#if ! defined (MAXNODES) || MAXNODES <= 64 
    "maboss_module", 
#else
#define MODULE_NODES NAME1(MAXNODES, n)
#define MODULE_NAME NAME1(maboss_module_, MODULE_NODES)
    STR(MODULE_NAME),
#endif
    "Some documentation",
    -1,
    cMaBoSS
};

PyMODINIT_FUNC
#if ! defined (MAXNODES) || MAXNODES <= 64 
PyInit_maboss_module(void)
#else
#define MODULE_NODES NAME1(MAXNODES, n)
#define MODULE_NAME NAME1(maboss_module_, MODULE_NODES)
#define MODULE_INIT_NAME NAME1(PyInit_, MODULE_NAME)
MODULE_INIT_NAME(void)
#endif
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