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
     popmaboss_sim.cpp

   Authors:
     Vincent Noël <vincent.noel@curie.fr>
 
   Date:
     March 2021
*/

#include "popmaboss_sim.h"
#include "engines/PopMaBEstEngine.h"
#include "popmaboss_res.h"

#ifdef __GLIBC__
#include <malloc.h>
#endif

PyMethodDef cPopMaBoSSSim_methods[] = {

    {"get_nodes", (PyCFunction) cPopMaBoSSSim_get_nodes, METH_NOARGS, "gets the list of nodes"},
    {"copy", (PyCFunction) cPopMaBoSSSim_copy, METH_NOARGS, "copy the simulation"},
    {"str_bnd", (PyCFunction) cPopMaBoSSSim_bnd_str, METH_VARARGS | METH_KEYWORDS, "returns the contents of the pbnd file"},
    {"str_cfg", (PyCFunction) cPopMaBoSSSim_cfg_str, METH_VARARGS | METH_KEYWORDS, "checks the contents of the cfg file"},
    {"run", (PyCFunction) cPopMaBoSSSim_run, METH_VARARGS | METH_KEYWORDS, "runs the simulation"},
    {"update_parameters", (PyCFunction) cPopMaBoSSSim_update_parameters, METH_VARARGS | METH_KEYWORDS, "changes the parameters of the simulation"},
    {"set_custom_pop_output", (PyCFunction) cPopMaBoSSSim_setCustomPopOutput, METH_VARARGS, "changes the custom pop output"},
    {NULL}  /* Sentinel */
};

PyMemberDef cPopMaBoSSSim_members[] = {
    {"network", T_OBJECT_EX, offsetof(cPopMaBoSSSimObject, network), READONLY},
    {"param", T_OBJECT_EX, offsetof(cPopMaBoSSSimObject, param), READONLY},
    {NULL}  /* Sentinel */
};

PyTypeObject cPopMaBoSSSim = {
  PyVarObject_HEAD_INIT(NULL, 0)
  build_type_name("cPopMaBoSSSimObject"),               /* tp_name */
  sizeof(cPopMaBoSSSimObject),               /* tp_basicsize */
    0,                              /* tp_itemsize */
  (destructor) cPopMaBoSSSim_dealloc,      /* tp_dealloc */
    0,                              /* tp_vectorcall_offset */
    0,                              /* tp_getattr */
    0,                              /* tp_setattr */
    0,                              /* tp_as_async */
    0,                              /* tp_repr */
    0,                              /* tp_as_number */
    0,                              /* tp_as_sequence */
    0,                              /* tp_as_mapping */
    0,                              /* tp_hash */
    0,                              /* tp_call */
    0,                              /* tp_str */
    0,                              /* tp_getattro */
    0,                              /* tp_setattro */
    0,                              /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,                              /* tp_flags */
  "cPopMaBoSS Simulation object",                   /* tp_doc */
    0,                              /* tp_traverse */
    0,                              /* tp_clear */
    0,                              /* tp_richcompare */
    0,                              /* tp_weaklistoffset */
    0,                              /* tp_iter */
    0,                              /* tp_iternext */
  cPopMaBoSSSim_methods,                              /* tp_methods */
  cPopMaBoSSSim_members,                              /* tp_members */
    0,                              /* tp_getset */
    0,                              /* tp_base */
    0,                              /* tp_dict */
    0,                              /* tp_descr_get */
    0,                              /* tp_descr_set */
    0,                              /* tp_dictoffset */
  cPopMaBoSSSim_init,                              /* tp_init */
    0,                              /* tp_alloc */
  cPopMaBoSSSim_new,                      /* tp_new */ 
};

void cPopMaBoSSSim_dealloc(cPopMaBoSSSimObject *self)
{
    Py_TYPE(self)->tp_free((PyObject *) self);
}

PyObject * cPopMaBoSSSim_new(PyTypeObject* type, PyObject *args, PyObject* kwargs) 
{
  cPopMaBoSSSimObject* py_simulation = (cPopMaBoSSSimObject *) type->tp_alloc(type, 0);
  py_simulation->network = NULL;
  py_simulation->config = NULL;
  py_simulation->param = NULL;
  return (PyObject *) py_simulation;
}


int cPopMaBoSSSim_init(PyObject* self, PyObject *args, PyObject* kwargs) 
{
  try {
    PyObject * network_file = Py_None;
    PyObject * config_file = Py_None;
    PyObject * config_files = Py_None;
    PyObject * network_str = Py_None;
    PyObject * config_str = Py_None;
    PyObject* net = Py_None;
    PyObject* cfg = Py_None;
    const char *kwargs_list[] = {"network", "config", "configs", "network_str", "config_str", "net", "cfg", NULL};
    if (!PyArg_ParseTupleAndKeywords(
      args, kwargs, "|OOOOOOO", const_cast<char **>(kwargs_list), 
      &network_file, &config_file, &config_files, &network_str, &config_str, &net, &cfg
    ))
      return -1;
      
    cPopMaBoSSSimObject* py_simulation = (cPopMaBoSSSimObject *) self;
    if (net != Py_None) 
    {  
      py_simulation->network = (cPopMaBoSSNetworkObject*) net;
    
    } else {
      py_simulation->network = (cPopMaBoSSNetworkObject*) PyObject_CallFunction((PyObject *) &cPopMaBoSSNetwork, 
        "OO", network_file, network_str
      );
    }
    
    if (py_simulation->network == NULL)
    {
      return -1;
    }
    
    if (cfg != Py_None)
    {
      py_simulation->config = (cMaBoSSConfigObject*) cfg;
      
    } else {
      py_simulation->config = (cMaBoSSConfigObject*) PyObject_CallFunction((PyObject *) &cMaBoSSConfig, 
        "OOOO", py_simulation->network, config_file, config_files, config_str
      );
    }
    
    if (py_simulation->config == NULL)
    {
      return -1;
    }
    
    py_simulation->param = (cMaBoSSParamObject*) PyObject_CallFunction((PyObject *) &cMaBoSSParam,
      "OO", py_simulation->network, py_simulation->config
    );
    
    if (py_simulation->param == NULL)
    {
      return -1;
    }
      
    // Error checking
    IStateGroup::checkAndComplete(py_simulation->network->network);
    py_simulation->network->network->getSymbolTable()->checkSymbols();
  }
  catch (BNException& e) {
    PyErr_SetString(PyBNException, e.getMessage().c_str());
    return -1;
  }
  
  return 0;
}
PyObject* cPopMaBoSSSim_update_parameters(cPopMaBoSSSimObject* self, PyObject *args, PyObject* kwargs) 
{
  return cMaBoSSParam_update_parameters(self->param, args, kwargs);
}

PyObject* cPopMaBoSSSim_run(cPopMaBoSSSimObject* self, PyObject *args, PyObject* kwargs) {
  
  time_t start_time, end_time;

  RandomGenerator::resetGeneratedNumberCount();
  
  PopMaBEstEngine* simulation = new PopMaBEstEngine(self->network->network, self->config->config);
  time(&start_time);
  simulation->run(NULL);

#ifdef __GLIBC__
  malloc_trim(0);
#endif

  time(&end_time);
  
  cPopMaBoSSResultObject* res = (cPopMaBoSSResultObject*) PyObject_New(cPopMaBoSSResultObject, &cPopMaBoSSResult);
  res->network = self->network->network;
  res->config = self->config->config;
  res->engine = simulation;
  res->start_time = start_time;
  res->end_time = end_time;
  
  return (PyObject*) res;
}

PyObject* cPopMaBoSSSim_get_nodes(cPopMaBoSSSimObject* self) {

  PyObject *list = PyList_New(self->network->network->getNodes().size());

  size_t index = 0;
  for (auto* node: self->network->network->getNodes()) {
    PyList_SetItem(list, index, PyUnicode_FromString(node->getLabel().c_str()));
    index++;
  }

  return list;
}

PyObject* cPopMaBoSSSim_setCustomPopOutput(cPopMaBoSSSimObject* self, PyObject *args) {
  PyObject* custom_output;
  if (!PyArg_ParseTuple(args, "O", &custom_output))
    return NULL;
  
  std::string str_custom_output("custom_pop_output = ");
  str_custom_output.append(PyUnicode_AsUTF8(custom_output));
  str_custom_output.append(";");  
  try{
    self->config->config->parseExpression(self->network->network, str_custom_output.c_str());  
  } catch (BNException& e) {
    PyErr_SetString(PyBNException, e.getMessage().c_str());
    return NULL;
  }
  
  Py_RETURN_NONE;
}


PyObject* cPopMaBoSSSim_bnd_str(cPopMaBoSSSimObject* self, PyObject *args, PyObject* kwargs) {
  std::ostringstream bnd;
  self->network->network->display(bnd);
  return PyUnicode_FromString(bnd.str().c_str());
}


PyObject* cPopMaBoSSSim_cfg_str(cPopMaBoSSSimObject* self, PyObject *args, PyObject* kwargs) {
  std::ostringstream cfg;
  self->config->config->dump(self->network->network, cfg, PopMaBEstEngine::VERSION, false);
  return PyUnicode_FromString(cfg.str().c_str());
}

PyObject* cPopMaBoSSSim_copy(cPopMaBoSSSimObject* self) {
  std::ostringstream bnd;
  self->network->network->display(bnd);
  
  std::ostringstream cfg;
  self->config->config->dump(self->network->network, cfg, PopMaBEstEngine::VERSION, false);
  
  PyObject* network_str = PyUnicode_FromString(bnd.str().c_str());
  Py_INCREF(network_str);
  PyObject* config_str = PyUnicode_FromString(cfg.str().c_str());
  Py_INCREF(config_str);
  
  PyObject *args = PyTuple_New(0);
  PyObject *kwargs = Py_BuildValue("{s:O,s:O}", "network_str", PyUnicode_FromString(bnd.str().c_str()), "config_str", PyUnicode_FromString(cfg.str().c_str()));

  cPopMaBoSSSimObject* simulation = (cPopMaBoSSSimObject *) PyObject_Call(
    (PyObject *) &cPopMaBoSSSim, args, kwargs
  );
  return (PyObject *) simulation;
}
