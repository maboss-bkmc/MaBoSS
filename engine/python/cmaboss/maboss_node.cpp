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

#include "maboss_node.h"
#include "maboss_commons.h"
#include "maboss_net.h"
#include "popmaboss_net.h"
#include "src/BooleanNetwork.h"

PyMethodDef cMaBoSSNode_methods[] = {
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

PyTypeObject cMaBoSSNode = []{
    PyTypeObject net{PyVarObject_HEAD_INIT(NULL, 0)};
    net.tp_name = build_type_name("cMaBoSSNodeObject");
    net.tp_basicsize = sizeof(cMaBoSSNodeObject);
    net.tp_itemsize = 0;
    net.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    net.tp_doc = "cMaBoSS Node object";
    net.tp_init = cMaBoSSNode_init;
    net.tp_new = cMaBoSSNode_new;
    net.tp_dealloc = (destructor) cMaBoSSNode_dealloc;
    net.tp_methods = cMaBoSSNode_methods;
    return net;
}();

void cMaBoSSNode_dealloc(cMaBoSSNodeObject *self)
{
    delete self->node;
    Py_TYPE(self)->tp_free((PyObject *) self);
}

PyObject* cMaBoSSNode_getLabel(cMaBoSSNodeObject* self) 
{
  return PyUnicode_FromString(self->node->getLabel().c_str());
}

PyObject* cMaBoSSNode_setLogic(cMaBoSSNodeObject* self, PyObject* args) 
{
  PyObject * logic = NULL;
  if (!PyArg_ParseTuple(args, "O", &logic))
    return NULL;
  
  try{
    if (logic != NULL) {
      Expression* logic_expr = self->network->parseSingleExpression(PyUnicode_AsUTF8(logic));
      self->node->setLogicalInputExpression(logic_expr);
    }
    
  } catch (BNException& e) {
    PyErr_SetString(PyBNException, e.getMessage().c_str());
    return NULL;
  }
  
  Py_RETURN_NONE;
}

PyObject* cMaBoSSNode_getLogic(cMaBoSSNodeObject* self) 
{
  if (self->node->getLogicalInputExpression() != NULL) {
    return PyUnicode_FromString(self->node->getLogicalInputExpression()->toString().c_str());
  } else {
    Py_RETURN_NONE;
  }
  
}

PyObject * cMaBoSSNode_setRawRateUp(cMaBoSSNodeObject* self, PyObject* args) 
{
  PyObject* rate_up = NULL;
  if (!PyArg_ParseTuple(args, "O", &rate_up))
    return NULL;
  
  try{
    Expression* rate_up_expr = self->network->parseSingleExpression(PyUnicode_AsUTF8(rate_up));
    self->node->setRateUpExpression(rate_up_expr);
    
  } catch (BNException& e) {
    PyErr_SetString(PyBNException, e.getMessage().c_str());
    return NULL;
  }
  
  Py_RETURN_NONE;
}

PyObject * cMaBoSSNode_setRawRateDown(cMaBoSSNodeObject* self, PyObject* args) 
{

  PyObject* rate_down = NULL;
  if (!PyArg_ParseTuple(args, "O", &rate_down))
    return NULL;
  
  try{
    Expression* rate_down_expr = self->network->parseSingleExpression(PyUnicode_AsUTF8(rate_down));
    self->node->setRateUpExpression(rate_down_expr);
    
  } catch (BNException& e) {
    PyErr_SetString(PyBNException, e.getMessage().c_str());
    return NULL;
  }
  
  Py_RETURN_NONE;
}

PyObject* cMaBoSSNode_setRate(cMaBoSSNodeObject* self, PyObject* args) 
{
  PyObject* rate_up = NULL;
  PyObject* rate_down = NULL;
  if (!PyArg_ParseTuple(args, "OO", &rate_up, &rate_down))
    return NULL;

  try{
    if (rate_up != NULL) 
    {
      Expression* rate_up_expr = NULL;
      
      if (PyObject_IsInstance(rate_up, (PyObject*) &PyFloat_Type))
      {
        rate_up_expr = new ConstantExpression(PyFloat_AsDouble(rate_up));
      } 
      else if (PyObject_IsInstance(rate_up, (PyObject*) &PyLong_Type)) 
      {
        rate_up_expr = new ConstantExpression(PyLong_AsDouble(rate_up));
      } 
      else if (PyObject_IsInstance(rate_up, (PyObject*) &PyUnicode_Type)) 
      {
        Expression* rate_up_expr = self->network->parseSingleExpression(PyUnicode_AsUTF8(rate_up));
        self->network->getSymbolTable()->defineUndefinedSymbols();
      }
      else {
        PyErr_SetString(PyBNException, "Unsupported type for rate up !");
        return NULL;
      }  
    
      if (rate_up_expr != NULL) 
      {
        if (self->node->getLogicalInputExpression() != NULL)
        {
          self->node->setRateUpExpression(
            new CondExpression(new AliasExpression("logic"), new ConstantExpression(PyFloat_AsDouble(rate_up)), new ConstantExpression(0.0))
          );
        } 
        else 
        {
          self->node->setRateUpExpression(rate_up_expr);
        }
      }
    }
    if (rate_down != NULL)
    {
      Expression* rate_down_expr  = NULL;
      
      if (PyObject_IsInstance(rate_down, (PyObject*) &PyFloat_Type))
      {
        rate_down_expr = new ConstantExpression(PyFloat_AsDouble(rate_down));
      }
      else if (PyObject_IsInstance(rate_down, (PyObject*) &PyLong_Type))
      { 
        rate_down_expr = new ConstantExpression(PyLong_AsDouble(rate_down));
      }
      else if (PyObject_IsInstance(rate_down, (PyObject*) &PyUnicode_Type))
      {
        Expression* rate_down_expr = self->network->parseSingleExpression(PyUnicode_AsUTF8(rate_down));
        self->network->getSymbolTable()->defineUndefinedSymbols();
      } 
      else 
      {
        PyErr_SetString(PyBNException, "Unsupported type for rate down !");
      }
      
      if (rate_down_expr != NULL)
      {
        if (self->node->getLogicalInputExpression() != NULL)
        {
          self->node->setRateDownExpression(
            new CondExpression(new AliasExpression("logic"), new ConstantExpression(0.0), rate_down_expr)
          );
        } 
        else 
        {
          self->node->setRateDownExpression(rate_down_expr);
        }

      }
    }
  } catch (BNException& e) {
    PyErr_SetString(PyBNException, e.getMessage().c_str());
    return NULL;
  }
  
  Py_RETURN_NONE;
}

PyObject* cMaBoSSNode_getRateUp(cMaBoSSNodeObject* self) 
{
  PyObject* rate_up_str = PyUnicode_FromString(self->node->getRateUpExpression()->toString().c_str());
  Py_INCREF(rate_up_str);
  return rate_up_str;
}

PyObject* cMaBoSSNode_getRateDown(cMaBoSSNodeObject* self) 
{
  PyObject* rate_down_str = PyUnicode_FromString(self->node->getRateDownExpression()->toString().c_str());
  Py_INCREF(rate_down_str);
  return rate_down_str;
}

PyObject * cMaBoSSNode_new(PyTypeObject* type, PyObject *args, PyObject* kwargs) 
{
  cMaBoSSNodeObject * py_node = (cMaBoSSNodeObject *) type->tp_alloc(type, 0);
  py_node->network = NULL;
  py_node->node = NULL;
  return (PyObject*) py_node;
}

int cMaBoSSNode_init(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyObject * name = Py_None;
  PyObject * py_network = Py_None;
  const char *kwargs_list[] = {"name", "network", NULL};
  if (!PyArg_ParseTupleAndKeywords(
    args, kwargs, "OO", const_cast<char **>(kwargs_list), 
    &name, &py_network
  ))
    return -1;

  cMaBoSSNodeObject * py_node = (cMaBoSSNodeObject *) self;

  try
  {

    if (PyObject_IsInstance(py_network, (PyObject*)&cMaBoSSNetwork))
    {
      py_node->network = ((cMaBoSSNetworkObject*) py_network)->network;
      
    } else if (PyObject_IsInstance(py_network, (PyObject*)&cPopMaBoSSNetwork))
    {
      py_node->network = ((cPopMaBoSSNetworkObject*) py_network)->network;
      
    } else {
      py_node = NULL;
      PyErr_SetString(PyBNException, "Invalid network object");
      return -1;
    }
    
    if (py_node->network != NULL){
      py_node->node = py_node->network->getOrMakeNode(PyUnicode_AsUTF8(name));
    }
    
  } catch (BNException& e) 
  {
    py_node = NULL;
    PyErr_SetString(PyBNException, e.getMessage().c_str());
    return -1;
  }
  
  return 0;
}
