#ifndef MABOSS_NETWORK
#define MABOSS_NETWORK

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <set>
#include "src/BooleanNetwork.h"
#include "src/MaBEstEngine.h"

typedef struct {
  PyObject_HEAD
  Network* network;
} cMaBoSSNetworkObject;

static void cMaBoSSNetwork_dealloc(cMaBoSSNetworkObject *self)
{
    free(self->network);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static Network* cMaBoSSNetwork_getNetwork(cMaBoSSNetworkObject* self) 
{
  return self->network;
}

static PyObject * cMaBoSSNetwork_new(PyTypeObject* type, PyObject *args, PyObject* kwargs) 
{
  char * network_file;
  static const char *kwargs_list[] = {"network", NULL};
  if (!PyArg_ParseTupleAndKeywords(
    args, kwargs, "s", const_cast<char **>(kwargs_list), 
    &network_file
  ))
    return NULL;
  
  cMaBoSSNetworkObject* pynetwork;
  pynetwork = (cMaBoSSNetworkObject *) type->tp_alloc(type, 0);
  pynetwork->network = new Network();
  pynetwork->network->parse(network_file);
  return (PyObject*) pynetwork;
}


static PyMethodDef cMaBoSSNetwork_methods[] = {
    {"getNetwork", (PyCFunction) cMaBoSSNetwork_getNetwork, METH_NOARGS, "returns the network object"},
    {NULL}  /* Sentinel */
};

static PyTypeObject cMaBoSSNetwork = []{
    PyTypeObject net{PyVarObject_HEAD_INIT(NULL, 0)};

    net.tp_name = "cmaboss.cMaBoSSNetworkObject";
    net.tp_basicsize = sizeof(cMaBoSSNetworkObject);
    net.tp_itemsize = 0;
    net.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    net.tp_doc = "cMaBoSS Network object";
    net.tp_new = cMaBoSSNetwork_new;
    net.tp_dealloc = (destructor) cMaBoSSNetwork_dealloc;
    net.tp_methods = cMaBoSSNetwork_methods;
    return net;
}();
#endif