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
#include "popmaboss_net.h"
#include "maboss_node.h"


PyMethodDef cPopMaBoSSNetwork_methods[] = {
    // {"getNetwork", (PyCFunction) cPopMaBoSSNetwork_getNetwork, METH_NOARGS, "returns the network object"},
    // {"getNodes", (PyCFunction) cPopMaBoSSNetwork_getDictNodes, METH_NOARGS, "returns the dict of nodes"},
    {"set_output", (PyCFunction) cPopMaBoSSNetwork_setOutput, METH_VARARGS, "set the output nodes"},
    {"get_output", (PyCFunction) cPopMaBoSSNetwork_getOutput, METH_NOARGS, "returns the output nodes"},
    {"set_death_rate", (PyCFunction) cPopMaBoSSNetwork_setDeathRate, METH_VARARGS, "sets the death rate"},
    {"get_death_rate", (PyCFunction) cPopMaBoSSNetwork_getDeathRate, METH_NOARGS, "gets the death rate"},
    {"add_division_rule", (PyCFunction) cPopMaBoSSNetwork_addDivisionRule, METH_VARARGS, "adds a division rule"},
    {"get_division_rules", (PyCFunction) cPopMaBoSSNetwork_getDivisionRules, METH_NOARGS, "gets the division rules"},
    {"set_istate", (PyCFunction) cPopMaBoSSNetwork_setIstate, METH_VARARGS, "sets the initial state"},
    {"set_pop_istate", (PyCFunction) cPopMaBoSSNetwork_setPopIstate, METH_VARARGS, "sets the initial population state"},
    {"clear_pop_istate", (PyCFunction) cPopMaBoSSNetwork_clearPopIstate, METH_NOARGS, "clears the initial population state"},
    {"keys", (PyCFunction) cPopMaBoSSNetwork_Keys, METH_NOARGS, "returns the keys"},
    {"values", (PyCFunction) cPopMaBoSSNetwork_Values, METH_NOARGS, "returns the values"},
    {"items", (PyCFunction) cPopMaBoSSNetwork_Items, METH_NOARGS, "returns the items"},
    {NULL}  /* Sentinel */
};

PyMappingMethods cPopMaBoSSNetwork_mapping = {
	(lenfunc)cPopMaBoSSNetwork_NodesLength,		
	(binaryfunc)cPopMaBoSSNetwork_NodesGetItem,
	(objobjargproc)cPopMaBoSSNetwork_NodesSetItem,
};

PyTypeObject cPopMaBoSSNetwork = []{
    PyTypeObject net{PyVarObject_HEAD_INIT(NULL, 0)};

    net.tp_name = build_type_name("cPopMaBoSSNetworkObject");
    net.tp_basicsize = sizeof(cPopMaBoSSNetworkObject);
    net.tp_itemsize = 0;
    net.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    net.tp_doc = "cMaBoSS PopNetwork object";
    net.tp_init = cPopMaBoSSNetwork_init;
    net.tp_new = cPopMaBoSSNetwork_new;
    net.tp_dealloc = (destructor) cPopMaBoSSNetwork_dealloc;
    net.tp_methods = cPopMaBoSSNetwork_methods;
    net.tp_as_mapping = &cPopMaBoSSNetwork_mapping;
    net.tp_str = cPopMaBoSSNetwork_str;
    return net;
}();

void cPopMaBoSSNetwork_dealloc(cPopMaBoSSNetworkObject *self)
{
    delete self->network;
    Py_TYPE(self)->tp_free((PyObject *) self);
}

PyObject *cPopMaBoSSNetwork_str(PyObject *self) {
  PyObject* str = PyUnicode_FromString(((cPopMaBoSSNetworkObject* )self)->network->toString().c_str());
  Py_INCREF(str);
  return str;
}

int cPopMaBoSSNetwork_NodesSetItem(cPopMaBoSSNetworkObject* self, PyObject *key, PyObject* value) 
{
  Py_INCREF(value);
  return PyDict_SetItem(self->nodes, key, value);
}
PyObject * cPopMaBoSSNetwork_NodesGetItem(cPopMaBoSSNetworkObject* self, PyObject *key) 
{
  PyObject* item = PyDict_GetItem(self->nodes, key);
  Py_INCREF(item);
  return item;
}

Py_ssize_t cPopMaBoSSNetwork_NodesLength(cPopMaBoSSNetworkObject* self)
{
  return PyObject_Length(self->nodes);
}

PyObject* cPopMaBoSSNetwork_Keys(cPopMaBoSSNetworkObject* self) 
{
  PyObject* keys = PyDict_Keys(self->nodes);
  Py_INCREF(keys);
  return keys;
}

PyObject* cPopMaBoSSNetwork_Values(cPopMaBoSSNetworkObject* self) 
{
  PyObject* values = PyDict_Values(self->nodes);
  Py_INCREF(values);
  return values;
}

PyObject* cPopMaBoSSNetwork_Items(cPopMaBoSSNetworkObject* self) 
{
  PyObject* items = PyDict_Items(self->nodes);
  Py_INCREF(items);
  return items;
}

PyObject* cPopMaBoSSNetwork_getDeathRate(cPopMaBoSSNetworkObject* self) 
{
  const Expression* death_rate = self->network->getDeathRate();
  if (death_rate == NULL) {
    Py_INCREF(Py_None);
    return Py_None;
  }
  
  PyObject* death_rate_str = PyUnicode_FromString(
    death_rate->toString().c_str()
  );
  
  Py_INCREF(death_rate_str);
  return death_rate_str;
}

PyObject* cPopMaBoSSNetwork_setDeathRate(cPopMaBoSSNetworkObject* self, PyObject *args) 
{

  char* death_rate = NULL;
  if (!PyArg_ParseTuple(args, "s", &death_rate))
    return NULL;
  
  std::map<std::string, NodeIndex> nodes_indexes;
  for (auto* node: self->network->getNodes()) {
    nodes_indexes[node->getLabel()] = node->getIndex();
  }
  
  std::string death_rate_str = std::string("death {\nrate=") + std::string(death_rate) + std::string(";\n}");
  
  try{
    self->network->parseExpression(death_rate_str.c_str(), &nodes_indexes);
  
  } catch (BNException& e) {
    PyErr_SetString(PyBNException, (std::string(death_rate) + std::string(" is not a valid expression")).c_str());
    return NULL;
  }
  
  Py_INCREF(Py_None);
  return Py_None;
}

PyObject* cPopMaBoSSNetwork_setOutput(cPopMaBoSSNetworkObject* self, PyObject *args) 
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

PyObject* cPopMaBoSSNetwork_getOutput(cPopMaBoSSNetworkObject* self) 
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

PyObject* cPopMaBoSSNetwork_addDivisionRule(cPopMaBoSSNetworkObject* self, PyObject *args) 
{
  char* rule = NULL;
  PyObject* daugther_1 = NULL;
  PyObject* daugther_2 = NULL;
  if (!PyArg_ParseTuple(args, "s|OO", &rule,&daugther_1,&daugther_2))
    return NULL;
  
  std::map<std::string, NodeIndex> nodes_indexes;
  for (auto* node: self->network->getNodes()) {
    nodes_indexes[node->getLabel()] = node->getIndex();
  }
  
  try{
    std::string division_rule = std::string("division {\nrate=") + std::string(rule) + ";\n";
    if (daugther_1 != NULL){
      for (Py_ssize_t i=0; i < PyDict_Size(daugther_1); i++)
      {
        PyObject* key = PyList_GetItem(PyDict_Keys(daugther_1), i);
        std::string key_str = PyUnicode_AsUTF8(key);
        std::string value_str = std::to_string(PyLong_AsLong(PyDict_GetItem(daugther_1, key)));
        division_rule += key_str + std::string(".DAUGHTER1=") + value_str + ";\n";
      }
    }
    if (daugther_2 != NULL){
      for (Py_ssize_t i=0; i < PyDict_Size(daugther_2); i++)
      {
        PyObject* key = PyList_GetItem(PyDict_Keys(daugther_2), i);
        std::string key_str = PyUnicode_AsUTF8(key);
        std::string value_str = std::to_string(PyLong_AsLong(PyDict_GetItem(daugther_2, key)));
        division_rule += key_str + std::string(".DAUGHTER2=") + value_str + ";\n";
      }
    }
    
    division_rule += std::string("}");
    self->network->parseExpression(division_rule.c_str(), &nodes_indexes);
  } catch (BNException& e) {
    PyErr_SetString(PyBNException, e.getMessage().c_str());
    return NULL;
  }
  
  Py_INCREF(Py_None);
  return Py_None;
}

PyObject* cPopMaBoSSNetwork_getDivisionRules(cPopMaBoSSNetworkObject* self) 
{
  PyObject* rules = PyList_New(0);
  for (auto rule: self->network->getDivisionRules()) 
  {
    PyObject* rate = PyUnicode_FromString(rule->rate->toString().c_str());
    
    PyObject* daugther_1 = PyDict_New();
    for (auto map: rule->daughters[DivisionRule::DAUGHTER_1]) {
      
      PyDict_SetItemString(daugther_1, map.first->getLabel().c_str(), PyUnicode_FromString(map.second->toString().c_str()));
    }

    PyObject* daugther_2 = PyDict_New();
    for (auto map: rule->daughters[DivisionRule::DAUGHTER_2]) {
      PyDict_SetItemString(daugther_2, map.first->getLabel().c_str(), PyUnicode_FromString(map.second->toString().c_str()));
    }
    
    PyList_Append(rules, PyTuple_Pack(3, rate, daugther_1, daugther_2));
    
  }
  
  Py_INCREF(rules);
  return rules;
}

PyObject * cPopMaBoSSNetwork_new(PyTypeObject* type, PyObject *args, PyObject* kwargs) 
{
  cPopMaBoSSNetworkObject* py_network = (cPopMaBoSSNetworkObject *) type->tp_alloc(type, 0);
  py_network->network = new PopNetwork(); 
  py_network->nodes = PyDict_New();
  return (PyObject*) py_network;
}

int cPopMaBoSSNetwork_init(PyObject* self, PyObject *args, PyObject* kwargs) 
{
  PyObject * network_file = Py_None;
  PyObject * network_str = Py_None;
  const char *kwargs_list[] = {"network", "network_str", NULL};
  if (!PyArg_ParseTupleAndKeywords(
    args, kwargs, "|OO", const_cast<char **>(kwargs_list), 
    &network_file, &network_str
  ))
    return -1;
  
  cPopMaBoSSNetworkObject* py_network = (cPopMaBoSSNetworkObject *) self;
  
  try{
    if (network_file != Py_None) 
    {
      py_network->network->parse(PyUnicode_AsUTF8(network_file));  
      
    } else if (network_str != Py_None)
    {
      py_network->network->parseExpression(PyUnicode_AsUTF8(network_str));
      
    } else {
      py_network = NULL;
      PyErr_SetString(PyBNException, "No network file or string provided");
      return -1;
    }

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

PyObject* cPopMaBoSSNetwork_setIstate(cPopMaBoSSNetworkObject* self, PyObject *args)
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
    // [A,B,C] : { (0,1,1): 0.5, (1,0,1): 0.5 }
    // [A,B,C] = { ((0,1,1),5):10, ((1,0,0),2): 0.5}
    // [TumorCell,DC].pop_istate = 1.0 [{[1,0]:18} , {[0,1]:2}];

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

PyObject* cPopMaBoSSNetwork_setPopIstate(cPopMaBoSSNetworkObject* self, PyObject *args)
{
  PyObject* node = NULL;
  PyObject* istate = NULL;
  if (!PyArg_ParseTuple(args, "OO", &node, &istate))
    return NULL;
  
  try {
    if (PyObject_IsInstance(node, (PyObject*)&PyList_Type) && PyObject_IsInstance(istate, (PyObject *)&PyDict_Type))
    {
    
      // ['TumorCell','DC'] = { (((1,0),18), ((0,1),2)):1.0}
      // [TumorCell,DC].pop_istate = 1.0 [{[1,0]:18} , {[0,1]:2}];

      std::vector<const Node*>* istate_nodes = new std::vector<const Node*>();
      std::vector<PopIStateGroup::PopProbaIState*>* istate_map = new std::vector<PopIStateGroup::PopProbaIState*>();
      
      for (Py_ssize_t i = 0; i < PyList_Size(node); i++) {
        Node* maboss_node = self->network->getNode(PyUnicode_AsUTF8(PyList_GetItem(node, i)));
        istate_nodes->push_back(maboss_node);
      }
      
      //Parsing each key : each pop state
      for (Py_ssize_t i = 0; i < PyList_Size(PyDict_Keys(istate)); i++) 
      {  
        PyObject* py_pop_state = PyList_GetItem(PyDict_Keys(istate), i);
    
        // if (PyObject_IsInstance(PyTuple_GetItem(py_pop_state, 0), (PyObject *)&PyTuple_Type) && PyObject_IsInstance(PyTuple_GetItem(py_pop_state, 1), (PyObject *)&PyLong_Type)) {
        //   PyErr_SetString(PyBNException, "Keys should be tuples of tuples, integers");
        //   return NULL;
        // }
        
        std::vector<PopIStateGroup::PopProbaIState::PopIStateGroupIndividual*>* individual_pop_istates = new std::vector<PopIStateGroup::PopProbaIState::PopIStateGroupIndividual*>();
        
        // Parsing each cell state
        for (Py_ssize_t j=0; j < PyTuple_Size(py_pop_state); j++) 
        {  
          PyObject* py_boolean_state = PyTuple_GetItem(PyTuple_GetItem(py_pop_state, j), 0);
          std::vector<double> boolean_state;
          for (Py_ssize_t k=0; k < PyTuple_Size(py_boolean_state); k++) {
            boolean_state.push_back((double) (PyLong_AsLong(PyTuple_GetItem(py_boolean_state, k)) == 1));
          }

          unsigned int pop = PyLong_AsLong(PyTuple_GetItem(PyTuple_GetItem(py_pop_state, j), 1));
          
          individual_pop_istates->push_back(new PopIStateGroup::PopProbaIState::PopIStateGroupIndividual(boolean_state, pop));
        }
        
        double proba = PyFloat_AsDouble(PyDict_GetItem(istate, py_pop_state));
        istate_map->push_back(new PopIStateGroup::PopProbaIState(proba, individual_pop_istates));
      }
      
      std::string message = "Error loading complex pop istates";
      new PopIStateGroup(self->network, istate_nodes, istate_map, message);
    } 
  } catch (BNException& e) {
    PyErr_SetString(PyBNException, e.getMessage().c_str());
    return NULL;
  }
  
  Py_RETURN_NONE;
  
}

PyObject* cPopMaBoSSNetwork_clearPopIstate(cPopMaBoSSNetworkObject* self)
{
  self->network->clearPopIstates();
  Py_RETURN_NONE;
}
