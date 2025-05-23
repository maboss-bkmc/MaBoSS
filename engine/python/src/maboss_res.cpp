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
     maboss_res.cpp

   Authors:
     Vincent Noël <vincent.noel@curie.fr>
 
   Date:
     January-March 2020
*/

#include "maboss_res.h"

#include <structmember.h>
#include <fstream>
#include "displayers/FixedPointDisplayer.h"
#include "displayers/ProbTrajDisplayer.h"
#include "displayers/StatDistDisplayer.h"

#ifdef __GLIBC__
#include <malloc.h>
#endif

PyMemberDef cMaBoSSResult_members[] = {
    {(char*)"network", T_OBJECT_EX, offsetof(cMaBoSSResultObject, network), 0, (char*)"network"},
    {(char*)"runconfig", T_OBJECT_EX, offsetof(cMaBoSSResultObject, runconfig), 0, (char*)"runconfig"},
    {(char*)"engine", T_OBJECT_EX, offsetof(cMaBoSSResultObject, engine), 0, (char*)"engine"},
    {(char*)"start_time", T_LONG, offsetof(cMaBoSSResultObject, start_time), 0, (char*)"start_time"},
    {(char*)"end_time", T_LONG, offsetof(cMaBoSSResultObject, end_time), 0, (char*)"end_time"},
    {(char*)"probtraj", T_OBJECT_EX, offsetof(cMaBoSSResultObject, probtraj), 0, (char*)"probtraj"},
    {(char*)"last_probtraj", T_OBJECT_EX, offsetof(cMaBoSSResultObject, last_probtraj), 0, (char*)"last_probtraj"},
    {(char*)"observed_graph", T_OBJECT_EX, offsetof(cMaBoSSResultObject, observed_graph), 0, (char*)"observed_graph"},
    {NULL}  /* Sentinel */
};

PyMethodDef cMaBoSSResult_methods[] = {
    {"get_observed_graph", (PyCFunction) cMaBoSSResult_get_observed_graph, METH_NOARGS, "gets the observed graph table"},
    {"get_observed_durations", (PyCFunction) cMaBoSSResult_get_observed_durations, METH_NOARGS, "gets the observed durations table"},
    {"get_fp_table", (PyCFunction) cMaBoSSResult_get_fp_table, METH_NOARGS, "gets the fixpoints table"},
    {"get_probtraj", (PyCFunction) cMaBoSSResult_get_probtraj, METH_NOARGS, "gets the raw states probability trajectories of the simulation"},
    {"get_last_probtraj", (PyCFunction) cMaBoSSResult_get_last_probtraj, METH_NOARGS, "gets the raw states probability trajectories of the simulation"},
    {"get_nodes_probtraj", (PyCFunction) cMaBoSSResult_get_nodes_probtraj, METH_VARARGS, "gets the raw states probability trajectories of the simulation"},
    {"get_last_nodes_probtraj", (PyCFunction) cMaBoSSResult_get_last_nodes_probtraj, METH_VARARGS, "gets the raw states probability trajectories of the simulation"},
    {"display_fp", (PyCFunction) cMaBoSSResult_display_fp, METH_VARARGS, "prints the fixpoints to a file"},
    {"display_probtraj", (PyCFunction) cMaBoSSResult_display_probtraj, METH_VARARGS, "prints the probtraj to a file"},
    {"display_statdist", (PyCFunction) cMaBoSSResult_display_statdist, METH_VARARGS, "prints the statdist to a file"},
    {"display_run", (PyCFunction) cMaBoSSResult_display_run, METH_VARARGS, "prints the run of the simulation to a file"},
    {NULL}  /* Sentinel */
};


PyTypeObject cMaBoSSResult = {
  PyVarObject_HEAD_INIT(NULL, 0)
  build_type_name("cMaBoSSResultObject"),               /* tp_name */
  sizeof(cMaBoSSResultObject),               /* tp_basicsize */
    0,                              /* tp_itemsize */
  (destructor) cMaBoSSResult_dealloc,      /* tp_dealloc */
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
  "cMaBoSS Result object",                   /* tp_doc */
    0,                              /* tp_traverse */
    0,                              /* tp_clear */
    0,                              /* tp_richcompare */
    0,                              /* tp_weaklistoffset */
    0,                              /* tp_iter */
    0,                              /* tp_iternext */
  cMaBoSSResult_methods,                              /* tp_methods */
  cMaBoSSResult_members,                              /* tp_members */
    0,                              /* tp_getset */
    0,                              /* tp_base */
    0,                              /* tp_dict */
    0,                              /* tp_descr_get */
    0,                              /* tp_descr_set */
    0,                              /* tp_dictoffset */
    0,                              /* tp_init */
    0,                              /* tp_alloc */
  cMaBoSSResult_new,                      /* tp_new */    
};

void cMaBoSSResult_dealloc(cMaBoSSResultObject *self)
{
  delete self->engine;
  
#ifdef __GLIBC__
  malloc_trim(0);
#endif 

  Py_TYPE(self)->tp_free((PyObject *) self);
}

PyObject * cMaBoSSResult_new(PyTypeObject* type, PyObject *args, PyObject* kwargs) 
{
  cMaBoSSResultObject* res;
  res = (cMaBoSSResultObject *) type->tp_alloc(type, 0);
  res->probtraj = Py_None;
  res->last_probtraj = Py_None;
  res->observed_graph = Py_None;
  res->observed_durations = Py_None;
  return (PyObject*) res;
}

PyObject* cMaBoSSResult_get_fp_table(cMaBoSSResultObject* self) {

  PyObject *dict = PyDict_New();

  for (auto& result: self->engine->getFixPointsDists()) {
    PyObject *tuple = PyTuple_Pack(2, 
      PyFloat_FromDouble(result.second.second),
      PyUnicode_FromString(result.second.first.getName(self->network).c_str())
    );

    PyDict_SetItem(dict, PyLong_FromUnsignedLong(result.first), tuple);
  }

  return dict;
}

PyObject* cMaBoSSResult_get_observed_graph(cMaBoSSResultObject* self) {

  if (self->observed_graph == Py_None)
  {
    self->observed_graph = self->engine->getNumpyObservedGraph();
  }
  
  Py_INCREF(self->observed_graph);

  return self->observed_graph;
}

PyObject* cMaBoSSResult_get_observed_durations(cMaBoSSResultObject* self) {

  if (self->observed_durations == Py_None)
  {
    self->observed_durations = self->engine->getNumpyObservedDurations();
  }
  
  Py_INCREF(self->observed_durations);

  return self->observed_durations;
}

PyObject* cMaBoSSResult_get_probtraj(cMaBoSSResultObject* self) {
  if (self->probtraj == Py_None) {
    self->probtraj = self->engine->getMergedCumulator()->getNumpyStatesDists(self->network);
  }
  
  Py_INCREF(self->probtraj);

  return self->probtraj;
}

PyObject* cMaBoSSResult_get_last_probtraj(cMaBoSSResultObject* self) {
  if (self->last_probtraj == Py_None) {
    self->last_probtraj = self->engine->getMergedCumulator()->getNumpyLastStatesDists(self->network);
  }
  
  Py_INCREF(self->last_probtraj);
  return self->last_probtraj;
}

PyObject* cMaBoSSResult_get_nodes_probtraj(cMaBoSSResultObject* self, PyObject* args) {

  std::vector<Node*> list_nodes;
  PyObject* pList = Py_None;
  
  if (!PyArg_ParseTuple(args, "|O", &pList)) {
    PyErr_SetString(PyExc_TypeError, "Error parsing arguments");
    return NULL;
  }
  
  if (pList != Py_None) {
    PyObject* pItem;
    int n = PyList_Size(pList);
    for (int i=0; i<n; i++) {
        pItem = PyList_GetItem(pList, i);
        list_nodes.push_back(self->network->getNode(std::string(PyUnicode_AsUTF8(pItem))));
    }
  }
  
  return self->engine->getMergedCumulator()->getNumpyNodesDists(self->network, list_nodes);
}

PyObject* cMaBoSSResult_get_last_nodes_probtraj(cMaBoSSResultObject* self, PyObject* args) {

  std::vector<Node*> list_nodes;
  PyObject* pList = Py_None;
  
  if (!PyArg_ParseTuple(args, "|O", &pList)) {
    PyErr_SetString(PyExc_TypeError, "Error parsing arguments");
    return NULL;
  }
  
  if (pList != Py_None) {
    PyObject* pItem;
    int n = PyList_Size(pList);
    for (int i=0; i<n; i++) {
        pItem = PyList_GetItem(pList, i);
        list_nodes.push_back(self->network->getNode(std::string(PyUnicode_AsUTF8(pItem))));
    }
  }
  
  return self->engine->getMergedCumulator()->getNumpyLastNodesDists(self->network, list_nodes);
}

PyObject* cMaBoSSResult_display_fp(cMaBoSSResultObject* self, PyObject *args) 
{
  char * filename = NULL;
  int hexfloat = 0;
  if (!PyArg_ParseTuple(args, "s|i", &filename, &hexfloat))
    return NULL;
    
  std::ostream* output_fp = new std::ofstream(filename);
  CSVFixedPointDisplayer* fp_displayer = new CSVFixedPointDisplayer(self->network, *output_fp, (bool)hexfloat);
  self->engine->displayFixpoints(fp_displayer);
  delete(fp_displayer);
  ((std::ofstream*) output_fp)->close();
  delete output_fp;

  Py_RETURN_NONE;
}

PyObject* cMaBoSSResult_display_probtraj(cMaBoSSResultObject* self, PyObject *args) 
{
  char * filename = NULL;
  int hexfloat = 0;
  if (!PyArg_ParseTuple(args, "s|i", &filename, &hexfloat))
    return NULL;
    
  std::ostream* output_probtraj = new std::ofstream(filename);
  CSVProbTrajDisplayer<NetworkState>* probtraj_displayer = new CSVProbTrajDisplayer<NetworkState>(self->network, *output_probtraj, (bool)hexfloat);
  
  self->engine->displayProbTraj(probtraj_displayer);
  
  delete probtraj_displayer;
  ((std::ofstream*) output_probtraj)->close();
  delete output_probtraj;

  Py_RETURN_NONE;
}

PyObject* cMaBoSSResult_display_statdist(cMaBoSSResultObject* self, PyObject *args) 
{
  char * filename = NULL;
  int hexfloat = 0;
  if (!PyArg_ParseTuple(args, "s|i", &filename, &hexfloat))
    return NULL;
    
  std::ostream* output_statdist = new std::ofstream(filename);
  CSVStatDistDisplayer*  statdist_displayer = new CSVStatDistDisplayer(self->network, *output_statdist, (bool)hexfloat);
  self->engine->displayStatDist(statdist_displayer);
  delete statdist_displayer;
  ((std::ofstream*) output_statdist)->close();
  delete output_statdist;

  Py_RETURN_NONE;
}

PyObject* cMaBoSSResult_display_run(cMaBoSSResultObject* self, PyObject* args) 
{
  char * filename = NULL;
  int hexfloat = 0;
  if (!PyArg_ParseTuple(args, "s|i", &filename, &hexfloat))
    return NULL;
    
  std::ostream* output_run = new std::ofstream(filename);
  self->engine->displayRunStats(*output_run, self->start_time, self->end_time);
  ((std::ofstream*) output_run)->close();
  delete output_run;

  Py_RETURN_NONE;
}
