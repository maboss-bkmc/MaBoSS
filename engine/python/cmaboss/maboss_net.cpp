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
     maboss_net.cpp

   Authors:
     Vincent Noël <vincent.noel@curie.fr>
 
   Date:
     January-March 2020
*/

#ifndef MABOSS_NETWORK
#define MABOSS_NETWORK
#include "maboss_net.h"
#include "maboss_node.cpp"



static void cMaBoSSNetwork_dealloc(cMaBoSSNetworkObject *self)
{
    delete self->network;
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject *cMaBoSSNetwork_str(PyObject *self) {
  PyObject* str = PyUnicode_FromString(((cMaBoSSNetworkObject* )self)->network->toString().c_str());
  Py_INCREF(str);
  return str;
}

static int cMaBoSSNetwork_NodesSetItem(cMaBoSSNetworkObject* self, PyObject *key, PyObject* value) 
{
  Py_INCREF(value);
  return PyDict_SetItem(self->nodes, key, value);
}

static PyObject * cMaBoSSNetwork_NodesGetItem(cMaBoSSNetworkObject* self, PyObject *key) 
{
  PyObject* item = PyDict_GetItem(self->nodes, key);
  Py_INCREF(item);
  return item;
}

static Py_ssize_t cMaBoSSNetwork_NodesLength(cMaBoSSNetworkObject* self)
{
  return PyObject_Length(self->nodes);
}

static PyObject* cMaBoSSNetwork_setOutput(cMaBoSSNetworkObject* self, PyObject *args) 
{
  PyObject* list;
  if (!PyArg_ParseTuple(args, "O", &list))
    return NULL;
  
  for (auto* node: self->network->getNodes()) 
  {
    if (PySequence_Contains(list, PyUnicode_FromString(node->getLabel().c_str()))) {
      node->isInternal(false);
    } else {
      node->isInternal(true);
    }
  }
  return Py_None;
}

static PyObject* cMaBoSSNetwork_getOutput(cMaBoSSNetworkObject* self) 
{
  PyObject* output = PyList_New(0);
  for (auto* node: self->network->getNodes()) 
  {
    if (!node->isInternal()) {
      PyList_Append(output, PyUnicode_FromString(node->getLabel().c_str()));
    }
  }
  Py_INCREF(output);
  return output;
}

static PyObject* cMaBoSSNetwork_setObservedGraphNode(cMaBoSSNetworkObject* self, PyObject *args) 
{
  PyObject* list;
  if (!PyArg_ParseTuple(args, "O", &list))
    return NULL;
  
  for (auto* node: self->network->getNodes()) 
  {
    if (PySequence_Contains(list, PyUnicode_FromString(node->getLabel().c_str()))) {
      node->inGraph(true);
    } else {
      node->inGraph(false);
    }
  }
  return Py_None;
}

static PyObject* cMaBoSSNetwork_getObservedGraphNode(cMaBoSSNetworkObject* self, PyObject *args) 
{
  PyObject* output = PyList_New(0);
  for (auto* node: self->network->getNodes()) 
  {
    if (node->inGraph()) {
      PyList_Append(output, PyUnicode_FromString(node->getLabel().c_str()));
    }
  }
  Py_INCREF(output);
  return output;
}

static PyObject* cMaBoSSNetwork_addNode(cMaBoSSNetworkObject* self, PyObject *args) 
{
  char * name;
  if (!PyArg_ParseTuple(args, "s", &name))
    return NULL;
  
  try{
    cMaBoSSNodeObject * pynode = (cMaBoSSNodeObject *) PyObject_New(cMaBoSSNodeObject, &cMaBoSSNode);
    pynode->_network = self->network;
    pynode->_node = self->network->getOrMakeNode(name);
    PyDict_SetItemString(self->nodes, name, (PyObject*) pynode);
    Py_INCREF(pynode);
    

  } catch (BNException& e) {
    PyErr_SetString(PyBNException, e.getMessage().c_str());
    return NULL;
  }
  
  return Py_None;
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
  
  try{
    pynetwork->network->parse(network_file);
    pynetwork->nodes = PyDict_New();

    for (auto* node: pynetwork->network->getNodes()) 
    { 
      
      cMaBoSSNodeObject * pynode = (cMaBoSSNodeObject *) PyObject_New(cMaBoSSNodeObject, &cMaBoSSNode);
      pynode->_node = node;
      PyDict_SetItemString(pynetwork->nodes, node->getLabel().c_str(), (PyObject*) pynode);
      Py_INCREF(pynode);
    }
 
  } catch (BNException& e) {
    PyErr_SetString(PyBNException, e.getMessage().c_str());
    return NULL;
  }
  
  return (PyObject*) pynetwork;
}


static PyMethodDef cMaBoSSNetwork_methods[] = {
    {"set_output", (PyCFunction) cMaBoSSNetwork_setOutput, METH_VARARGS, "sets the output nodes"},
    {"get_output", (PyCFunction) cMaBoSSNetwork_getOutput, METH_NOARGS, "gets the output nodes"},
    {"set_observed_graph_nodes", (PyCFunction) cMaBoSSNetwork_setObservedGraphNode, METH_VARARGS, "sets the observed graph nodes"},
    {"get_observed_graph_nodes", (PyCFunction) cMaBoSSNetwork_getObservedGraphNode, METH_VARARGS, "gets the observed graph nodes"},
    {"add_node", (PyCFunction) cMaBoSSNetwork_addNode, METH_VARARGS, "adds a node to the network"},
    {NULL}  /* Sentinel */
};

static PyMappingMethods cMaBoSSNetwork_mapping = {
	(lenfunc)cMaBoSSNetwork_NodesLength,		
	(binaryfunc)cMaBoSSNetwork_NodesGetItem,
	(objobjargproc)cMaBoSSNetwork_NodesSetItem,
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
    net.tp_as_mapping = &cMaBoSSNetwork_mapping;
    net.tp_str = cMaBoSSNetwork_str;
    return net;
}();
#endif