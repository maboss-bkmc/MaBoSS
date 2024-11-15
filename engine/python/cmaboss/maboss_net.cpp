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

#include "maboss_net.h"
#include "maboss_node.h"


PyMethodDef cMaBoSSNetwork_methods[] = {
    {"set_output", (PyCFunction) cMaBoSSNetwork_setOutput, METH_VARARGS, "sets the output nodes"},
    {"get_output", (PyCFunction) cMaBoSSNetwork_getOutput, METH_NOARGS, "gets the output nodes"},
    {"set_observed_graph_nodes", (PyCFunction) cMaBoSSNetwork_setObservedGraphNode, METH_VARARGS, "sets the observed graph nodes"},
    {"get_observed_graph_nodes", (PyCFunction) cMaBoSSNetwork_getObservedGraphNode, METH_VARARGS, "gets the observed graph nodes"},
    {"add_node", (PyCFunction) cMaBoSSNetwork_addNode, METH_VARARGS, "adds a node to the network"},
    {"set_istate", (PyCFunction) cMaBoSSNetwork_setIState, METH_VARARGS, "sets the initial state of the network"},
    {"get_istate", (PyCFunction) cMaBoSSNetwork_getIState, METH_NOARGS, "gets the initial state of the network"},
    {"keys", (PyCFunction) cMaBoSSNetwork_Keys, METH_NOARGS, "returns the keys of the nodes"},
    {"values", (PyCFunction) cMaBoSSNetwork_Values, METH_NOARGS, "returns the values of the nodes"},
    {"items", (PyCFunction) cMaBoSSNetwork_Items, METH_NOARGS, "returns the items of the nodes"},
    {NULL}  /* Sentinel */
};

PyMappingMethods cMaBoSSNetwork_mapping = {
	(lenfunc)cMaBoSSNetwork_NodesLength,		
	(binaryfunc)cMaBoSSNetwork_NodesGetItem,
	(objobjargproc)cMaBoSSNetwork_NodesSetItem,
};

PyTypeObject cMaBoSSNetwork = []{
    PyTypeObject net{PyVarObject_HEAD_INIT(NULL, 0)};

    net.tp_name = build_type_name("cMaBoSSNetworkObject");
    net.tp_basicsize = sizeof(cMaBoSSNetworkObject);
    net.tp_itemsize = 0;
    net.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    net.tp_doc = "cMaBoSS Network object";
    net.tp_init = cMaBoSSNetwork_init;
    net.tp_new = cMaBoSSNetwork_new;
    net.tp_dealloc = (destructor) cMaBoSSNetwork_dealloc;
    net.tp_methods = cMaBoSSNetwork_methods;
    net.tp_as_mapping = &cMaBoSSNetwork_mapping;
    net.tp_str = cMaBoSSNetwork_str;
    return net;
}();

void cMaBoSSNetwork_dealloc(cMaBoSSNetworkObject *self)
{
    delete self->network;
    Py_TYPE(self)->tp_free((PyObject *) self);
}

PyObject *cMaBoSSNetwork_str(PyObject *self) {
  PyObject* str = PyUnicode_FromString(((cMaBoSSNetworkObject* )self)->network->toString().c_str());
  Py_INCREF(str);
  return str;
}

int cMaBoSSNetwork_NodesSetItem(cMaBoSSNetworkObject* self, PyObject *key, PyObject* value) 
{
  Py_INCREF(value);
  return PyDict_SetItem(self->nodes, key, value);
}

PyObject * cMaBoSSNetwork_NodesGetItem(cMaBoSSNetworkObject* self, PyObject *key) 
{
  PyObject* item = PyDict_GetItem(self->nodes, key);
  Py_INCREF(item);
  return item;
}

Py_ssize_t cMaBoSSNetwork_NodesLength(cMaBoSSNetworkObject* self)
{
  return PyObject_Length(self->nodes);
}

PyObject * cMaBoSSNetwork_Keys(cMaBoSSNetworkObject* self) 
{
  PyObject* keys = PyDict_Keys(self->nodes);
  Py_INCREF(keys);
  return keys;
}

PyObject * cMaBoSSNetwork_Values(cMaBoSSNetworkObject* self) 
{
  PyObject* values = PyDict_Values(self->nodes);
  Py_INCREF(values);
  return values;
}

PyObject * cMaBoSSNetwork_Items(cMaBoSSNetworkObject* self) 
{
  PyObject* items = PyDict_Items(self->nodes);
  Py_INCREF(items);
  return items;
}

PyObject* cMaBoSSNetwork_setOutput(cMaBoSSNetworkObject* self, PyObject *args) 
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
  Py_RETURN_NONE;
}

PyObject* cMaBoSSNetwork_getOutput(cMaBoSSNetworkObject* self) 
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

PyObject* cMaBoSSNetwork_setObservedGraphNode(cMaBoSSNetworkObject* self, PyObject *args) 
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
  Py_RETURN_NONE;
}

PyObject* cMaBoSSNetwork_getObservedGraphNode(cMaBoSSNetworkObject* self, PyObject *args) 
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

PyObject* cMaBoSSNetwork_addNode(cMaBoSSNetworkObject* self, PyObject *args) 
{
  char * name;
  if (!PyArg_ParseTuple(args, "s", &name))
    return NULL;
  
  try{
    PyObject * py_node = PyObject_CallFunction((PyObject *) &cMaBoSSNode, "sO", name, self->network);
    PyDict_SetItemString(self->nodes, name, (PyObject*) py_node);
    Py_INCREF(py_node);
    
  } catch (BNException& e) {
    PyErr_SetString(PyBNException, e.getMessage().c_str());
    return NULL;
  }
  
  Py_RETURN_NONE;
}

PyObject* cMaBoSSNetwork_setIState(cMaBoSSNetworkObject* self, PyObject *args) 
{
  PyObject* node = NULL;
  PyObject* istate = NULL;
  if (!PyArg_ParseTuple(args, "OO", &node, &istate))
    return NULL;
  
  try {
    if (PyObject_IsInstance(node, (PyObject*)&PyUnicode_Type) && (
      PyObject_IsInstance(istate, (PyObject *)&PyFloat_Type) || PyObject_IsInstance(istate, (PyObject *)&PyLong_Type)
    ))
    {
    
      Node* maboss_node = self->network->getNode(PyUnicode_AsUTF8(node));
      if (PyObject_IsInstance(istate, (PyObject *)&PyFloat_Type)) {
        IStateGroup::setNodeProba(self->network, maboss_node, PyFloat_AsDouble(istate));
      } else {
        IStateGroup::setNodeProba(self->network, maboss_node, PyLong_AsDouble(istate));
      }
    
    } else if (PyObject_IsInstance(node, (PyObject*)&PyUnicode_Type) && PyObject_IsInstance(istate, (PyObject *)&PyList_Type))
    {
      Node* maboss_node = self->network->getNode(PyUnicode_AsUTF8(node));
      double proba = PyFloat_AsDouble(PyList_GetItem(istate, 1))/(PyFloat_AsDouble(PyList_GetItem(istate, 0)) + PyFloat_AsDouble(PyList_GetItem(istate, 1)));
      if (PyObject_IsInstance(istate, (PyObject *)&PyFloat_Type)) {
        IStateGroup::setNodeProba(self->network, maboss_node, proba);
      } else {
        IStateGroup::setNodeProba(self->network, maboss_node, proba);
      }
  
  } else if (PyObject_IsInstance(node, (PyObject*)&PyList_Type) && PyObject_IsInstance(istate, (PyObject *)&PyDict_Type))
  {
      std::vector<const Node*>* istate_nodes = new std::vector<const Node*>();
      std::map<std::vector<bool>, double> istate_map;
      
      for (Py_ssize_t i = 0; i < PyList_Size(node); i++) {
        Node* maboss_node = self->network->getNode(PyUnicode_AsUTF8(PyList_GetItem(node, i)));
        istate_nodes->push_back(maboss_node);
      }
      
      for (Py_ssize_t i = 0; i < PyList_Size(PyDict_Keys(istate)); i++) {
        
        std::vector<bool> istate_state;
        PyObject* boolean_state = PyList_GetItem(PyDict_Keys(istate), i);  
    
        if (PyTuple_Size(boolean_state) != PyList_Size(node)) {
          PyErr_SetString(PyBNException, "The number of nodes and the number of boolean values do not match");
          return NULL;
        }
    
        for (Py_ssize_t j=0; j < PyTuple_Size(boolean_state); j++) {
          istate_state.push_back(PyLong_AsLong(PyTuple_GetItem(boolean_state, j)) == 1);
        }
        istate_map.insert(std::pair<std::vector<bool>, double>(istate_state, PyFloat_AsDouble(PyDict_GetItem(istate, boolean_state))));
      }
      
      IStateGroup::setStatesProbas(self->network, istate_nodes, istate_map);
    } 
  } catch (BNException& e) {
    PyErr_SetString(PyBNException, e.getMessage().c_str());
    return NULL;
  }
  
  Py_RETURN_NONE;
}

PyObject* cMaBoSSNetwork_getIState(cMaBoSSNetworkObject* self) 
{
  Py_RETURN_NONE;
}

PyObject * cMaBoSSNetwork_new(PyTypeObject* type, PyObject *args, PyObject* kwargs) 
{
  cMaBoSSNetworkObject* py_network = (cMaBoSSNetworkObject *) type->tp_alloc(type, 0);
  py_network->network = new Network();
  py_network->nodes = PyDict_New();
  return (PyObject*) py_network;
}

int cMaBoSSNetwork_init(PyObject* self, PyObject *args, PyObject* kwargs) 
{
  PyObject * network_file = Py_None;
  PyObject * network_str = Py_None;
  PyObject * use_sbml_names = Py_False;
  const char *kwargs_list[] = {"network", "network_str", "use_sbml_names", NULL};
  if (!PyArg_ParseTupleAndKeywords(
    args, kwargs, "|OOO", const_cast<char **>(kwargs_list), 
    &network_file, &network_str, &use_sbml_names
  ))
    return -1;
  
  cMaBoSSNetworkObject * py_network = (cMaBoSSNetworkObject *) self;
  
  try
  {
    if (network_file != Py_None) 
    {
      std::string network_file_str = std::string(PyUnicode_AsUTF8(network_file));
#ifdef SBML_COMPAT
      if (network_file_str.substr(network_file_str.find_last_of(".") + 1) == "sbml" || network_file_str.substr(network_file_str.find_last_of(".") + 1) == "xml" ) {
        py_network->network->parseSBML(network_file_str.c_str(), NULL, (use_sbml_names == Py_True)); 
      } else {
#endif
        py_network->network->parse(network_file_str.c_str());
#ifdef SBML_COMPAT
      }
#endif
    } else if (network_str != Py_None) 
    {
      py_network->network->parseExpression(PyUnicode_AsUTF8(network_str));
      
    } else {
      PyErr_SetString(PyBNException, "No network file or string provided");
      return -1;
    }
      
    // Building dictionary of nodes
    for (auto* node: py_network->network->getNodes()) 
    {
      PyObject * py_node = PyObject_CallFunction((PyObject *) &cMaBoSSNode, "sO", node->getLabel().c_str(), py_network);
      PyDict_SetItemString(py_network->nodes, node->getLabel().c_str(), (PyObject*) py_node);
      Py_INCREF(py_node);
    }
 
  } catch (BNException& e) {
    py_network = NULL;
    PyErr_SetString(PyBNException, e.getMessage().c_str());
    return -1;
  }
  
  return 0;
  
}
