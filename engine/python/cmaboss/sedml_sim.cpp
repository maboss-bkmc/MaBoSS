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
     sedml_sim.cpp

   Authors:
     Vincent Noël <vincent.noel@curie.fr>
 
   Date:
     April 2025
*/

#include "sedml_sim.h"
#include "maboss_commons.h"

#include "src/SEDMLParser.h"

#ifdef __GLIBC__
#include <malloc.h>
#endif

PyMethodDef sedmlSim_methods[] = {
    // {"run", (PyCFunction) sedmlSim_run, METH_VARARGS | METH_KEYWORDS, "runs the simulation"},
    // {"check", (PyCFunction) sedmlSim_check, METH_VARARGS | METH_KEYWORDS, "checks the model"},
    // {"copy", (PyCFunction) sedmlSim_copy, METH_NOARGS, "returns a copy of the simulation"},
    // {"str_bnd", (PyCFunction) sedmlSim_bnd_str, METH_VARARGS | METH_KEYWORDS, "returns the contents of the bnd file"},
    // {"str_cfg", (PyCFunction) sedmlSim_cfg_str, METH_VARARGS | METH_KEYWORDS, "checks the contents of the cfg file"},
    // {"get_logical_rules", (PyCFunction) sedmlSim_get_logical_rules, METH_VARARGS | METH_KEYWORDS, "returns logical formulas"},
    // {"update_parameters", (PyCFunction) sedmlSim_update_parameters, METH_VARARGS | METH_KEYWORDS, "changes the parameters of the simulation"},
    // {"get_nodes", (PyCFunction) sedmlSim_get_nodes, METH_NOARGS, "returns the list of nodes"},
    {NULL}  /* Sentinel */
};

PyMemberDef sedmlSim_members[] = {
    // {"network", T_OBJECT_EX, offsetof(sedmlSimObject, network), READONLY},
    // {"config", T_OBJECT_EX, offsetof(sedmlSimObject, config), READONLY},
    // {"param", T_OBJECT_EX, offsetof(sedmlSimObject, param), READONLY},
    {NULL}  /* Sentinel */
};

PyTypeObject sedmlSim = []{
    PyTypeObject sim{PyVarObject_HEAD_INIT(NULL, 0)};

    sim.tp_name = build_type_name("sedmlSimObject");
    sim.tp_basicsize = sizeof(sedmlSimObject);
    sim.tp_itemsize = 0;
    sim.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    sim.tp_doc = "cMaBoSS SEDML Simulation object";
    sim.tp_init = sedmlSim_init;
    sim.tp_new = sedmlSim_new;
    sim.tp_dealloc = (destructor) sedmlSim_dealloc;
    sim.tp_methods = sedmlSim_methods;
    sim.tp_members = sedmlSim_members;
    return sim;
}();

void sedmlSim_dealloc(sedmlSimObject *self)
{
    Py_TYPE(self)->tp_free((PyObject *) self);
}

int sedmlSim_init(PyObject* self, PyObject *args, PyObject* kwargs)  
{
  PyObject * sedml_file = Py_None;
  const char *kwargs_list[] = {"sedml_file", NULL};
  if (!PyArg_ParseTupleAndKeywords(
    args, kwargs, "|O", const_cast<char **>(kwargs_list), 
    &sedml_file
  )) {
    return -1;
  }
  sedmlSimObject* py_simulation = (sedmlSimObject *) self;

  try {
    SEDMLParser* parser = new SEDMLParser();
    parser->parse(PyUnicode_AsUTF8(sedml_file));
  }
  catch (BNException& e) {
    PyErr_SetString(PyBNException, e.getMessage().c_str());
    return -1;
  }
  
  return 0;
}

PyObject * sedmlSim_new(PyTypeObject* type, PyObject *args, PyObject* kwargs) 
{
  sedmlSimObject* py_simulation = (sedmlSimObject *) type->tp_alloc(type, 0);
  // py_simulation->network = NULL;
  // py_simulation->config = NULL;
  // py_simulation->param = NULL;

  return (PyObject *) py_simulation;
}
