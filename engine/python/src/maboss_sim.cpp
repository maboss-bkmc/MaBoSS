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
     maboss_sim.cpp

   Authors:
     Vincent Noël <vincent.noel@curie.fr>
 
   Date:
     January-March 2020
*/

#include "maboss_sim.h"
#include "maboss_commons.h"
#include "maboss_res.h"
#include "maboss_resfinal.h"

#include "engines/MaBEstEngine.h"
#include "engines/FinalStateSimulationEngine.h"
#include <sstream>

#ifdef __GLIBC__
#include <malloc.h>
#endif

PyMethodDef cMaBoSSSim_methods[] = {
    {"run", (PyCFunction) cMaBoSSSim_run, METH_VARARGS | METH_KEYWORDS, "runs the simulation"},
    {"check", (PyCFunction) cMaBoSSSim_check, METH_VARARGS | METH_KEYWORDS, "checks the model"},
    {"copy", (PyCFunction) cMaBoSSSim_copy, METH_NOARGS, "returns a copy of the simulation"},
    {"str_bnd", (PyCFunction) cMaBoSSSim_bnd_str, METH_VARARGS | METH_KEYWORDS, "returns the contents of the bnd file"},
    {"str_cfg", (PyCFunction) cMaBoSSSim_cfg_str, METH_VARARGS | METH_KEYWORDS, "checks the contents of the cfg file"},
    {"get_logical_rules", (PyCFunction) cMaBoSSSim_get_logical_rules, METH_VARARGS | METH_KEYWORDS, "returns logical formulas"},
    {"update_parameters", (PyCFunction) cMaBoSSSim_update_parameters, METH_VARARGS | METH_KEYWORDS, "changes the parameters of the simulation"},
    {"get_nodes", (PyCFunction) cMaBoSSSim_get_nodes, METH_NOARGS, "returns the list of nodes"},
    {"mutate", (PyCFunction) cMaBoSSSim_mutate, METH_VARARGS | METH_KEYWORDS, "mutates a node of the network"},
    {NULL}  /* Sentinel */
};

PyMemberDef cMaBoSSSim_members[] = {
    {"network", T_OBJECT_EX, offsetof(cMaBoSSSimObject, network), READONLY},
    {"config", T_OBJECT_EX, offsetof(cMaBoSSSimObject, config), READONLY},
    {"param", T_OBJECT_EX, offsetof(cMaBoSSSimObject, param), READONLY},
    {NULL}  /* Sentinel */
};

PyTypeObject cMaBoSSSim = {
  PyVarObject_HEAD_INIT(NULL, 0)
  build_type_name("cMaBoSSSimObject"),               /* tp_name */
  sizeof(cMaBoSSSimObject),               /* tp_basicsize */
    0,                              /* tp_itemsize */
  (destructor) cMaBoSSSim_dealloc,      /* tp_dealloc */
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
  "cMaBoSS Simulation object",                   /* tp_doc */
    0,                              /* tp_traverse */
    0,                              /* tp_clear */
    0,                              /* tp_richcompare */
    0,                              /* tp_weaklistoffset */
    0,                              /* tp_iter */
    0,                              /* tp_iternext */
  cMaBoSSSim_methods,                              /* tp_methods */
  cMaBoSSSim_members,                              /* tp_members */
    0,                              /* tp_getset */
    0,                              /* tp_base */
    0,                              /* tp_dict */
    0,                              /* tp_descr_get */
    0,                              /* tp_descr_set */
    0,                              /* tp_dictoffset */
  cMaBoSSSim_init,                              /* tp_init */
    0,                              /* tp_alloc */
  cMaBoSSSim_new,                      /* tp_new */    
};

void cMaBoSSSim_dealloc(cMaBoSSSimObject *self)
{
    Py_TYPE(self)->tp_free((PyObject *) self);
}

int cMaBoSSSim_init(PyObject* self, PyObject *args, PyObject* kwargs)  
{
  PyObject * network_file = Py_None;
  PyObject * config_file = Py_None;
  PyObject * config_files = Py_None;
  PyObject * network_str = Py_None;
  PyObject * config_str = Py_None;
  PyObject * net = Py_None;
  PyObject * cfg = Py_None;
  PyObject * use_sbml_names = Py_False;
  const char *kwargs_list[] = {"network", "config", "configs", "network_str", "config_str", "net", "cfg", "use_sbml_names", NULL};
  if (!PyArg_ParseTupleAndKeywords(
    args, kwargs, "|OOOOOOOO", const_cast<char **>(kwargs_list), 
    &network_file, &config_file, &config_files, &network_str, &config_str, &net, &cfg, &use_sbml_names
  )) {
    return -1;
  }
  cMaBoSSSimObject* py_simulation = (cMaBoSSSimObject *) self;

  try {
    if (net != Py_None) 
    {
      py_simulation->network = (cMaBoSSNetworkObject*) net;
      
    } else {
      py_simulation->network = (cMaBoSSNetworkObject*) PyObject_CallFunction((PyObject *) &cMaBoSSNetwork, 
        "OOO", network_file, network_str, use_sbml_names
      );
    }
    
    if (py_simulation->network == NULL) {
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
    
    if (py_simulation->config == NULL) {
      return -1;
    }
    
    py_simulation->param = (cMaBoSSParamObject*) PyObject_CallFunction((PyObject *) &cMaBoSSParam,
      "OO", py_simulation->network, py_simulation->config
    );
    
    if (py_simulation->param == NULL) {
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

PyObject * cMaBoSSSim_new(PyTypeObject* type, PyObject *args, PyObject* kwargs) 
{
  cMaBoSSSimObject* py_simulation = (cMaBoSSSimObject *) type->tp_alloc(type, 0);
  py_simulation->network = NULL;
  py_simulation->config = NULL;
  py_simulation->param = NULL;

  return (PyObject *) py_simulation;
}

PyObject* cMaBoSSSim_run(cMaBoSSSimObject* self, PyObject *args, PyObject* kwargs) {
  
  int only_last_state = 0;
  const char *kwargs_list[] = {"only_last_state", NULL};
  if (!PyArg_ParseTupleAndKeywords(
    args, kwargs, "|i", const_cast<char **>(kwargs_list), 
    &only_last_state
  ))
    return NULL;
    
  bool b_only_last_state = PyObject_IsTrue(PyBool_FromLong(only_last_state));
  time_t start_time, end_time;

  RandomGenerator::resetGeneratedNumberCount();
  if (b_only_last_state) {
  
    FinalStateSimulationEngine* simulation = new FinalStateSimulationEngine(self->network->network, self->config->config);
    time(&start_time);
    simulation->run(NULL);

#ifdef __GLIBC__
  malloc_trim(0);
#endif

    time(&end_time);
    cMaBoSSResultFinalObject* res = (cMaBoSSResultFinalObject*) PyObject_New(cMaBoSSResultFinalObject, &cMaBoSSResultFinal);
    res->network = self->network->network;
    res->runconfig = self->config->config;
    res->engine = simulation;
    res->start_time = start_time;
    res->end_time = end_time;
    res->last_probtraj = Py_None;
    return (PyObject*) res;
  } else {

    MaBEstEngine* simulation = new MaBEstEngine(self->network->network, self->config->config);
    time(&start_time);
    simulation->run(NULL);
    
#ifdef __GLIBC__
  malloc_trim(0);
#endif

    time(&end_time);
    
    cMaBoSSResultObject* res = (cMaBoSSResultObject*) PyObject_New(cMaBoSSResultObject, &cMaBoSSResult);
    res->network = self->network->network;
    res->runconfig = self->config->config;
    res->engine = simulation;
    res->start_time = start_time;
    res->end_time = end_time;
    res->probtraj = Py_None;
    res->last_probtraj = Py_None;
    res->observed_graph = Py_None;
    res->observed_durations = Py_None;
    return (PyObject*) res;
  }
}


PyObject* cMaBoSSSim_check(cMaBoSSSimObject* self, PyObject *args, PyObject* kwargs) {
  try {
    IStateGroup::checkAndComplete(self->network->network);
    self->network->network->getSymbolTable()->checkSymbols();
  } catch (BNException& e) {
    PyErr_SetString(PyBNException, e.getMessage().c_str());
    return NULL;
  }
  Py_RETURN_NONE;
}

PyObject* cMaBoSSSim_get_logical_rules(cMaBoSSSimObject* self, PyObject *args, PyObject* kwargs) {
  
  std::ostringstream ss;
  self->network->network->generateLogicalExpressions(ss);
  return PyUnicode_FromString(ss.str().c_str());
}

PyObject* cMaBoSSSim_bnd_str(cMaBoSSSimObject* self, PyObject *args, PyObject* kwargs) {
  std::ostringstream bnd;
  self->network->network->display(bnd);
  return PyUnicode_FromString(bnd.str().c_str());
}


PyObject* cMaBoSSSim_cfg_str(cMaBoSSSimObject* self, PyObject *args, PyObject* kwargs) {
  std::ostringstream cfg;
  self->config->config->dump(self->network->network, cfg, MaBEstEngine::VERSION, false);
  return PyUnicode_FromString(cfg.str().c_str());
}

PyObject* cMaBoSSSim_update_parameters(cMaBoSSSimObject* self, PyObject *args, PyObject* kwargs) 
{
  return cMaBoSSParam_update_parameters(self->param, args, kwargs);
}

PyObject* cMaBoSSSim_get_nodes(cMaBoSSSimObject* self) {

  PyObject *list = PyList_New(self->network->network->getNodes().size());

  size_t index = 0;
  for (auto* node: self->network->network->getNodes()) {
    PyList_SetItem(list, index, PyUnicode_FromString(node->getLabel().c_str()));
    index++;
  }

  return list;
}

PyObject* cMaBoSSSim_mutate(cMaBoSSSimObject* self, PyObject *args, PyObject* kwargs)
{
  PyObject *node_name = Py_None;
  PyObject *mutation = Py_None;
  PyObject* simple = Py_True;
  const char *kwargs_list[] = {"node_name", "mutation", "simple", NULL};
  if (!PyArg_ParseTupleAndKeywords(
    args, kwargs, "|OOO", const_cast<char **>(kwargs_list), 
    &node_name, &mutation, &simple
  ))
    return NULL;

  std::string node_name_str = PyUnicode_AsUTF8(node_name);
  std::string mutation_str = PyUnicode_AsUTF8(mutation);
  
  Node* node = self->network->network->getNode(node_name_str);
  bool is_activation = (mutation_str.compare("ON") == 0 || mutation_str.compare("on") == 0);
  if (simple == Py_True) {
    node->mutate((is_activation ? 1.0 : 0.0));
  } else {
    
    node->makeMutable(self->network->network);
    SymbolTable* symbol_table = self->network->network->getSymbolTable();
    const Symbol* lowvar = symbol_table->getSymbol("$Low_" + node_name_str);
    const Symbol* highvar = symbol_table->getSymbol("$High_" + node_name_str);
    symbol_table->setSymbolValue(lowvar, (is_activation ? 0.0 : 1.0));
    symbol_table->setSymbolValue(highvar, (is_activation ? 1.0 : 0.0));
  }
  
  Py_RETURN_NONE;
}

PyObject* cMaBoSSSim_copy(cMaBoSSSimObject* self) {
  std::ostringstream bnd;
  self->network->network->display(bnd);
  
  std::ostringstream cfg;
  self->config->config->dump(self->network->network, cfg, MaBEstEngine::VERSION, false);
   
  PyObject* network_str = PyUnicode_FromString(bnd.str().c_str());
  Py_INCREF(network_str);
  PyObject* config_str = PyUnicode_FromString(cfg.str().c_str());
  Py_INCREF(config_str);
  
  
  PyObject *args = PyTuple_New(0);
  PyObject *kwargs = Py_BuildValue("{s:O,s:O}", "network_str", PyUnicode_FromString(bnd.str().c_str()), "config_str", PyUnicode_FromString(cfg.str().c_str()));

  cMaBoSSSimObject* simulation = (cMaBoSSSimObject *) PyObject_Call(
    (PyObject *) &cMaBoSSSim, args, kwargs
  );

  return (PyObject *) simulation;
}