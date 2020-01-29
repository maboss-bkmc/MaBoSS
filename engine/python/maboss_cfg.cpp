#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <set>
#include "maboss_net.cpp"
#include "src/RunConfig.h"
#include "src/MaBEstEngine.h"

typedef struct {
  PyObject_HEAD
  RunConfig* config;
} cMaBoSSConfigObject;

static void cMaBoSSConfig_dealloc(cMaBoSSConfigObject *self)
{
    free(self->config);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static RunConfig* cMaBoSSConfig_getConfig(cMaBoSSConfigObject* self) 
{
  return self->config;
}

static PyObject * cMaBoSSConfig_new(PyTypeObject* type, PyObject *args, PyObject* kwargs) 
{
  char * config_file;
  PyObject * network;
  static const char *kwargs_list[] = {"network", "config", NULL};
  if (!PyArg_ParseTupleAndKeywords(
    args, kwargs, "Os", const_cast<char **>(kwargs_list), 
    &network, &config_file
  ))
    return NULL;
  
  cMaBoSSConfigObject* pyconfig;
  pyconfig = (cMaBoSSConfigObject *) type->tp_alloc(type, 0);
  pyconfig->config = new RunConfig();
  
  pyconfig->config->parse((((cMaBoSSNetworkObject* )network)->network), config_file);
  return (PyObject*) pyconfig;
}


static PyMethodDef cMaBoSSConfig_methods[] = {
    {"getConfig", (PyCFunction) cMaBoSSConfig_getConfig, METH_NOARGS, "returns the config object"},
    {NULL}  /* Sentinel */
};

static PyTypeObject cMaBoSSConfig = []{
    PyTypeObject net{PyVarObject_HEAD_INIT(NULL, 0)};

    net.tp_name = "cmaboss.cMaBoSSConfigObject";
    net.tp_basicsize = sizeof(cMaBoSSConfigObject);
    net.tp_itemsize = 0;
    net.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    net.tp_doc = "cMaBoSS Network object";
    net.tp_new = cMaBoSSConfig_new;
    net.tp_dealloc = (destructor) cMaBoSSConfig_dealloc;
    net.tp_methods = cMaBoSSConfig_methods;
    return net;
}();
