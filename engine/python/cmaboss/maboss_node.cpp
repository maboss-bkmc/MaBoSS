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
     maboss_node.cpp

   Authors:
     Vincent Noël <vincent.noel@curie.fr>
 
   Date:
     January-March 2020
*/

#ifndef MABOSS_NODE
#define MABOSS_NODE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <set>
#include "src/BooleanNetwork.h"
#include "src/MaBEstEngine.h"
#include "maboss_net.h"

typedef struct {
  PyObject_HEAD
  Node* _node;
  Network* _network;
} cMaBoSSNodeObject;

static void cMaBoSSNode_dealloc(cMaBoSSNodeObject *self)
{
    delete self->_node;
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject* cMaBoSSNode_getLabel(cMaBoSSNodeObject* self) 
{
  return PyUnicode_FromString(self->_node->getLabel().c_str());
}

static PyObject* cMaBoSSNode_setLogic(cMaBoSSNodeObject* self, PyObject* args) 
{
  PyObject * logic = NULL;
  if (!PyArg_ParseTuple(args, "O", &logic))
    return NULL;
  
  try{
    std::map<std::string, NodeIndex> nodes_indexes;
    for (auto* node: self->_network->getNodes()) {
      nodes_indexes[node->getLabel()] = node->getIndex();
    }

    if (logic != NULL) {
    //   std::cout << "Setting logical input to NULL" << std::endl;
    //   self->_node->setLogicalInputExpression(NULL);
    //   Py_RETURN_NONE;
    // } else {
      Expression* logic_expr = self->_network->parseSingleExpression(PyUnicode_AsUTF8(logic), &nodes_indexes);
      self->_node->setLogicalInputExpression(logic_expr);
    }
    
  } catch (BNException& e) {
    PyErr_SetString(PyBNException, e.getMessage().c_str());
    return NULL;
  }
  
  Py_RETURN_NONE;
}

static PyObject* cMaBoSSNode_getLogic(cMaBoSSNodeObject* self) 
{
  if (self->_node->getLogicalInputExpression() != NULL) {
    return PyUnicode_FromString(self->_node->getLogicalInputExpression()->toString().c_str());
  } else {
    Py_RETURN_NONE;
  }
  
}

static PyObject * cMaBoSSNode_setRawRateUp(cMaBoSSNodeObject* self, PyObject* args) 
{
  PyObject* rate_up = NULL;
  if (!PyArg_ParseTuple(args, "O", &rate_up))
    return NULL;
  
  try{
    std::map<std::string, NodeIndex> nodes_indexes;
    for (auto* node: self->_network->getNodes()) {
      nodes_indexes[node->getLabel()] = node->getIndex();
    }

    Expression* rate_up_expr = self->_network->parseSingleExpression(PyUnicode_AsUTF8(rate_up), &nodes_indexes);
    self->_node->setRateUpExpression(rate_up_expr);
    
  } catch (BNException& e) {
    PyErr_SetString(PyBNException, e.getMessage().c_str());
    return NULL;
  }
  
  Py_RETURN_NONE;
}

static PyObject * cMaBoSSNode_setRawRateDown(cMaBoSSNodeObject* self, PyObject* args) 
{

  PyObject* rate_down = NULL;
  if (!PyArg_ParseTuple(args, "O", &rate_down))
    return NULL;
  
  try{
    std::map<std::string, NodeIndex> nodes_indexes;
    for (auto* node: self->_network->getNodes()) {
      nodes_indexes[node->getLabel()] = node->getIndex();
    }

    Expression* rate_down_expr = self->_network->parseSingleExpression(PyUnicode_AsUTF8(rate_down), &nodes_indexes);
    self->_node->setRateUpExpression(rate_down_expr);
    
  } catch (BNException& e) {
    PyErr_SetString(PyBNException, e.getMessage().c_str());
    return NULL;
  }
  
  Py_RETURN_NONE;
}

static PyObject* cMaBoSSNode_setRate(cMaBoSSNodeObject* self, PyObject* args) 
{
  double rate_up = 0.0;
  double rate_down = 0.0;
  if (!PyArg_ParseTuple(args, "dd", &rate_up, &rate_down))
    return NULL;

  try{
    std::map<std::string, NodeIndex> nodes_indexes;
    for (auto* node: self->_network->getNodes()) {
      nodes_indexes[node->getLabel()] = node->getIndex();
    }

    self->_node->setRateUpExpression(
      new CondExpression(new AliasExpression("logic"), new ConstantExpression(rate_up), new ConstantExpression(0.0))
    );
    self->_node->setRateDownExpression(
      new CondExpression(new AliasExpression("logic"), new ConstantExpression(0.0), new ConstantExpression(rate_down))
    );
    
  } catch (BNException& e) {
    PyErr_SetString(PyBNException, e.getMessage().c_str());
    return NULL;
  }
  
  Py_RETURN_NONE;
}

static PyObject* cMaBoSSNode_getRateUp(cMaBoSSNodeObject* self) 
{
  PyObject* rate_up_str = PyUnicode_FromString(self->_node->getRateUpExpression()->toString().c_str());
  Py_INCREF(rate_up_str);
  return rate_up_str;
}

static PyObject* cMaBoSSNode_getRateDown(cMaBoSSNodeObject* self) 
{
  PyObject* rate_down_str = PyUnicode_FromString(self->_node->getRateDownExpression()->toString().c_str());
  Py_INCREF(rate_down_str);
  return rate_down_str;
}

static PyObject * cMaBoSSNode_new(PyTypeObject* type, PyObject *args, PyObject* kwargs) 
{
  char * name;
  cMaBoSSNetworkObject * network;
  static const char *kwargs_list[] = {"name", "network", NULL};
  if (!PyArg_ParseTupleAndKeywords(
    args, kwargs, "so", const_cast<char **>(kwargs_list), 
    &name, &network
  ))
    return NULL;
 
  cMaBoSSNodeObject * pynode = (cMaBoSSNodeObject *) type->tp_alloc(type, 0);
  
  try{
    pynode->_network = network->network;
    pynode->_node = network->network->getOrMakeNode(name);
    
  } catch (BNException& e) {
    PyErr_SetString(PyBNException, e.getMessage().c_str());
    return NULL;
  }
  
  return (PyObject*) pynode;
}


static PyMethodDef cMaBoSSNode_methods[] = {
    {"getLabel", (PyCFunction) cMaBoSSNode_getLabel, METH_NOARGS, "returns the node object"},
    {"set_logic", (PyCFunction) cMaBoSSNode_setLogic, METH_VARARGS, "sets the logic of the node"},
    {"get_logic", (PyCFunction) cMaBoSSNode_getLogic, METH_NOARGS, "returns the logic of the node"},
    {"set_rate", (PyCFunction) cMaBoSSNode_setRate, METH_VARARGS, "sets the rate of the node"},
    {"set_rate_up", (PyCFunction) cMaBoSSNode_setRawRateUp, METH_VARARGS, "sets the rate of the node"},
    {"set_rate_down", (PyCFunction) cMaBoSSNode_setRawRateDown, METH_VARARGS, "sets the rate of the node"},
    {"get_rate_up", (PyCFunction) cMaBoSSNode_getRateUp, METH_NOARGS, "returns the rate of the node"},
    {"get_rate_down", (PyCFunction) cMaBoSSNode_getRateDown, METH_NOARGS, "returns the rate of the node"},
    {NULL}  /* Sentinel */
};

static PyTypeObject cMaBoSSNode = []{
    PyTypeObject net{PyVarObject_HEAD_INIT(NULL, 0)};

    net.tp_name = "cmaboss.cMaBoSSNodeObject";
    net.tp_basicsize = sizeof(cMaBoSSNodeObject);
    net.tp_itemsize = 0;
    net.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    net.tp_doc = "cMaBoSS Node object";
    net.tp_new = cMaBoSSNode_new;
    net.tp_dealloc = (destructor) cMaBoSSNode_dealloc;
    net.tp_methods = cMaBoSSNode_methods;
    return net;
}();
#endif
