/*
#############################################################################
#                                                                           #
# BSD 3-Clause License (see https://opensource.org/licenses/BSD-3-Clause)   #
#                                                                           #
# Copyright (c) 2011-2020 Institut Curie, 26 rue d'Ulm, Paris, France       #
# All rights reserved.                                                      #
#                                                                           #
# Redistribution and use in source and binary forms, with or without        #
# modification, are permitted provided that the following conditions are    #
# met:                                                                      #
#                                                                           #
# 1. Redistributions of source code must retain the above copyright notice, #
# this list of conditions and the following disclaimer.                     #
#                                                                           #
# 2. Redistributions in binary form must reproduce the above copyright      #
# notice, this list of conditions and the following disclaimer in the       #
# documentation and/or other materials provided with the distribution.      #
#                                                                           #
# 3. Neither the name of the copyright holder nor the names of its          #
# contributors may be used to endorse or promote products derived from this #
# software without specific prior written permission.                       #
#                                                                           #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED #
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A           #
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER #
# OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,  #
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,       #
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR        #
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    #
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      #
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              #
#                                                                           #
#############################################################################

   Module:
     popmaboss_cfg.cpp

   Authors:
     Vincent Noël <vincent.noel@curie.fr>
 
   Date:
     January-March 2020
*/

#ifndef POPMABOSS_CONFIG
#define POPMABOSS_CONFIG
#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <set>
#include "maboss_net.cpp"
#include "src/RunConfig.h"
#include "src/PopMaBEstEngine.h"

typedef struct {
  PyObject_HEAD
  RunConfig* config;
} cPopMaBoSSConfigObject;

static void cPopMaBoSSConfig_dealloc(cPopMaBoSSConfigObject *self)
{
    delete self->config;
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static RunConfig* cPopMaBoSSConfig_getConfig(cPopMaBoSSConfigObject* self) 
{
  return self->config;
}

static PyObject* cPopMaBoSSConfig_getMaxTime(cPopMaBoSSConfigObject* self)
{
  return PyFloat_FromDouble(self->config->getMaxTime());
}

static PyObject * cPopMaBoSSConfig_new(PyTypeObject* type, PyObject *args, PyObject* kwargs) 
{
  Py_ssize_t nb_args = PyTuple_Size(args);  

  if (nb_args < 2) {
    return NULL;
  }
  
  cPopMaBoSSNetworkObject * network = (cPopMaBoSSNetworkObject*) PyTuple_GetItem(args, 0);

  cPopMaBoSSConfigObject* pyconfig;
  pyconfig = (cPopMaBoSSConfigObject *) type->tp_alloc(type, 0);
  pyconfig->config = new RunConfig();
  
  try {
    for (Py_ssize_t i = 1; i < nb_args; i++) {
      PyObject* bytes = PyUnicode_AsUTF8String(PyTuple_GetItem(args, i));
      pyconfig->config->parse(network->network, PyBytes_AsString(bytes));
      Py_DECREF(bytes);
    }

  } catch (BNException& e) {
    PyErr_SetString(PyBNException, e.getMessage().c_str());
    return NULL;
  }

  return (PyObject*) pyconfig;
}


static PyMethodDef cPopMaBoSSConfig_methods[] = {
    {"getConfig", (PyCFunction) cPopMaBoSSConfig_getConfig, METH_NOARGS, "returns the config object"},
    {"getMaxTime", (PyCFunction) cPopMaBoSSConfig_getMaxTime, METH_NOARGS, "returns the max time"},
    {NULL}  /* Sentinel */
};

static PyTypeObject cPopMaBoSSConfig = []{
    PyTypeObject net{PyVarObject_HEAD_INIT(NULL, 0)};

    net.tp_name = "cmaboss.cPopMaBoSSConfigObject";
    net.tp_basicsize = sizeof(cPopMaBoSSConfigObject);
    net.tp_itemsize = 0;
    net.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    net.tp_doc = "cPopMaBoSS Network object";
    net.tp_new = cPopMaBoSSConfig_new;
    net.tp_dealloc = (destructor) cPopMaBoSSConfig_dealloc;
    net.tp_methods = cPopMaBoSSConfig_methods;
    return net;
}();

#endif