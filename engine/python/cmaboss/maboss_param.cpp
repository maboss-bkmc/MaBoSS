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
     maboss_param.cpp

   Authors:
     Vincent Noël <vincent.noel@curie.fr>
 
   Date:
     January-March 2020
*/

#ifndef MABOSS_PARAM
#define MABOSS_PARAM

#define PY_SSIZE_T_CLEAN

#include <Python.h>

typedef struct {
  PyObject_HEAD
  Network* network;
  RunConfig* config;
} cMaBoSSParamObject;

static void cMaBoSSParam_dealloc(cMaBoSSParamObject *self)
{
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject* cMaBoSSParam_update_parameters(cMaBoSSParamObject* self, PyObject *args, PyObject* kwargs) 
{
  PyObject* key, *value;
  Py_ssize_t pos = 0;
  while (PyDict_Next(kwargs, &pos, &key, &value)) 
  {  
    if (PyUnicode_CompareWithASCIIString(key, "time_tick") == 0) {
      self->config->setParameter("time_tick", PyFloat_AsDouble(value));
    } else if (PyUnicode_CompareWithASCIIString(key, "max_time") == 0) {
      self->config->setParameter("max_time", PyFloat_AsDouble(value));
    } else if (PyUnicode_CompareWithASCIIString(key, "sample_count") == 0) {
      self->config->setParameter("sample_count", PyLong_AsLong(value));
    } else if (PyUnicode_CompareWithASCIIString(key, "init_pop") == 0) {
      self->config->setParameter("init_pop", PyLong_AsLong(value));
    } else if (PyUnicode_CompareWithASCIIString(key, "discrete_time") == 0) {
      self->config->setParameter("discrete_time", PyLong_AsLong(value));
    } else if (PyUnicode_CompareWithASCIIString(key, "use_physrandgen") == 0) {
      self->config->setParameter("use_physrandgen", PyLong_AsLong(value));
    } else if (PyUnicode_CompareWithASCIIString(key, "use_mtrandgen") == 0) {
      self->config->setParameter("use_mtrandgen", PyLong_AsLong(value));
    } else if (PyUnicode_CompareWithASCIIString(key, "use_glibcrandgen") == 0) {
      self->config->setParameter("use_glibcrandgen", PyLong_AsLong(value));
    } else if (PyUnicode_CompareWithASCIIString(key, "seed_pseudorandom") == 0) {
      self->config->setParameter("seed_pseudorandom", PyFloat_AsDouble(value));
    } else if (PyUnicode_CompareWithASCIIString(key, "thread_count") == 0) {
      self->config->setParameter("thread_count", PyLong_AsLong(value));
    } else if (PyUnicode_CompareWithASCIIString(key, "display_traj") == 0) {
      self->config->setParameter("display_traj", PyLong_AsLong(value));
    } else if (PyUnicode_CompareWithASCIIString(key, "statdist_traj_count") == 0) {
      self->config->setParameter("statdist_traj_count", PyLong_AsLong(value));
    } else if (PyUnicode_CompareWithASCIIString(key, "statdist_cluster_threshold") == 0) {
      self->config->setParameter("statdist_cluster_threshold", PyFloat_AsDouble(value));
    } else if (PyUnicode_CompareWithASCIIString(key, "statdist_similarity_cache_max_size") == 0) {
      self->config->setParameter("statdist_similarity_cache_max_size", PyLong_AsLong(value));
    } else {
      const char * key_str = PyUnicode_AsUTF8(key);
      if (key_str[0] == '$') {
        SymbolTable* st = self->network->getSymbolTable();
        st->setSymbolValue(st->getSymbol(key_str), PyFloat_AsDouble(value));
      } else {
        PyErr_SetString(PyExc_KeyError, "Unknown parameter");
        return NULL;
      }
    }
  }
  
  return Py_None;
}

static int cMaBoSSParam_SetItem(cMaBoSSParamObject* self, PyObject *key, PyObject* value) 
{
  PyObject* empty_tumple = PyTuple_New(0);
  Py_INCREF(empty_tumple);
  cMaBoSSParam_update_parameters(self, empty_tumple, Py_BuildValue("{s:O}", PyUnicode_AsUTF8(key), value));
  return 0;
}

static PyObject * cMaBoSSParam_GetItem(cMaBoSSParamObject* self, PyObject *key) 
{
  if (PyUnicode_CompareWithASCIIString(key, "time_tick") == 0) {
    PyObject* time_tick = PyFloat_FromDouble(self->config->getTimeTick());
    Py_INCREF(time_tick);
    return time_tick;
    
  } else if (PyUnicode_CompareWithASCIIString(key, "max_time") == 0) {
    PyObject* max_time = PyFloat_FromDouble(self->config->getMaxTime());
    Py_INCREF(max_time);
    return max_time;
    
  } else if (PyUnicode_CompareWithASCIIString(key, "sample_count") == 0) {
    PyObject* sample_count = PyLong_FromUnsignedLong(self->config->getSampleCount());
    Py_INCREF(sample_count);
    return sample_count;
    
  } else if (PyUnicode_CompareWithASCIIString(key, "discrete_time") == 0) {
    PyObject* discrete_time = self->config->isDiscreteTime() ? Py_True : Py_False;
    Py_INCREF(discrete_time);
    return discrete_time;
  
  } else if (PyUnicode_CompareWithASCIIString(key, "use_physrandgen") == 0) {
    PyObject* use_physrandgen = self->config->usePhysRandGen() ? Py_True : Py_False;
    Py_INCREF(use_physrandgen);
    return use_physrandgen;
  
  } else if (PyUnicode_CompareWithASCIIString(key, "use_mtrandgen") == 0) {
    PyObject* use_mtrandgen = self->config->useMTRandGen() ? Py_True : Py_False;
    Py_INCREF(use_mtrandgen);
    return use_mtrandgen;
  
  } else if (PyUnicode_CompareWithASCIIString(key, "use_glibcrandgen") == 0) {
    PyObject* use_glibcrandgen = self->config->useGlibcRandGen() ? Py_True : Py_False;
    Py_INCREF(use_glibcrandgen);
    return use_glibcrandgen;
  
  } else if (PyUnicode_CompareWithASCIIString(key, "seed_pseudorandom") == 0) {
    PyObject* seed_pseudorandom = PyLong_FromLong(self->config->getSeedPseudoRandom());
    Py_INCREF(seed_pseudorandom);
    return seed_pseudorandom;
  
  } else if (PyUnicode_CompareWithASCIIString(key, "thread_count") == 0) {
    PyObject* thread_count = PyLong_FromUnsignedLong(self->config->getThreadCount());
    Py_INCREF(thread_count);
    return thread_count;
  
  } else if (PyUnicode_CompareWithASCIIString(key, "display_traj") == 0) {
    PyObject* display_traj = PyLong_FromUnsignedLong(self->config->getDisplayTrajectories());
    Py_INCREF(display_traj);
    return display_traj;
    
  } else if (PyUnicode_CompareWithASCIIString(key, "statdist_traj_count") == 0) {
    PyObject* statdist_traj_count = PyLong_FromUnsignedLong(self->config->getStatDistTrajCount());
    Py_INCREF(statdist_traj_count);
    return statdist_traj_count;
  
  } else if (PyUnicode_CompareWithASCIIString(key, "statdist_cluster_threshold") == 0) {
    PyObject* statdist_cluster_threshold = PyFloat_FromDouble(self->config->getStatdistClusterThreshold());
    Py_INCREF(statdist_cluster_threshold);
    return statdist_cluster_threshold;
    
  } else if (PyUnicode_CompareWithASCIIString(key, "statdist_similarity_cache_max_size") == 0) {
    PyObject* statdist_similarity_cache_max_size = PyLong_FromUnsignedLong(self->config->getStatDistSimilarityCacheMaxSize());
    Py_INCREF(statdist_similarity_cache_max_size);
    return statdist_similarity_cache_max_size;
    
  } else if (PyUnicode_CompareWithASCIIString(key, "init_pop") == 0) {
    PyObject* init_pop = PyLong_FromUnsignedLong(self->config->getInitPop());
    Py_INCREF(init_pop);
    return init_pop;
    
  } else {
    
    const char * key_str = PyUnicode_AsUTF8(key);
    if (key_str[0] == '$') {
      SymbolTable* st = self->network->getSymbolTable();
      PyObject* symbol_value = PyFloat_FromDouble(st->getSymbolValue(st->getSymbol(key_str)));
      Py_INCREF(symbol_value);
      return symbol_value;
    } else {
      PyErr_SetString(PyExc_KeyError, "Unknown parameter");
      return NULL;
    }
  }
  
  return NULL;
}

static Py_ssize_t cMaBoSSParam_Length(cMaBoSSParamObject* self)
{
  return 15 + self->network->getSymbolTable()->getSymbolsNames().size();
}

static PyObject* cMaBoSSParam_getKeys(cMaBoSSParamObject* self)
{
  SymbolTable* st = self->network->getSymbolTable();
  PyObject* keys = PyList_New(15 + st->getSymbolsNames().size());
  PyList_SetItem(keys, 0, PyUnicode_FromString("time_tick"));
  PyList_SetItem(keys, 1, PyUnicode_FromString("max_time"));
  PyList_SetItem(keys, 2, PyUnicode_FromString("sample_count"));
  PyList_SetItem(keys, 3, PyUnicode_FromString("init_pop"));
  PyList_SetItem(keys, 4, PyUnicode_FromString("discrete_time"));
  PyList_SetItem(keys, 5, PyUnicode_FromString("use_physrandgen"));
  PyList_SetItem(keys, 6, PyUnicode_FromString("use_glibcrandgen"));
  PyList_SetItem(keys, 7, PyUnicode_FromString("use_mtrandgen"));
  PyList_SetItem(keys, 8, PyUnicode_FromString("seed_pseudorandom"));
  PyList_SetItem(keys, 9, PyUnicode_FromString("display_traj"));
  PyList_SetItem(keys, 10, PyUnicode_FromString("statdist_traj_count"));
  PyList_SetItem(keys, 11, PyUnicode_FromString("statdist_cluster_threshold"));
  PyList_SetItem(keys, 12, PyUnicode_FromString("thread_count"));
  PyList_SetItem(keys, 13, PyUnicode_FromString("statdist_similarity_cache_max_size"));
  PyList_SetItem(keys, 14, PyUnicode_FromString("init_pop"));
  int i = 0;
  for (auto const& item_name : st->getSymbolsNames()) {
    PyList_SetItem(keys, 15 + i, PyUnicode_FromString(item_name.c_str()));
    i++;
  }
  return keys;
}

static PyObject* cMaBoSSParam_getValues(cMaBoSSParamObject* self)
{
  SymbolTable* st = self->network->getSymbolTable();
  PyObject* values = PyList_New(15 + st->getSymbolsNames().size());
  PyList_SetItem(values, 0, PyFloat_FromDouble(self->config->getTimeTick()));
  PyList_SetItem(values, 1, PyFloat_FromDouble(self->config->getMaxTime()));
  PyList_SetItem(values, 2, PyLong_FromUnsignedLong(self->config->getSampleCount()));
  PyList_SetItem(values, 3, PyLong_FromUnsignedLong(self->config->getInitPop()));
  PyList_SetItem(values, 4, self->config->isDiscreteTime() ? Py_True : Py_False);
  PyList_SetItem(values, 5, self->config->usePhysRandGen() ? Py_True : Py_False);
  PyList_SetItem(values, 6, self->config->useGlibcRandGen() ? Py_True : Py_False);
  PyList_SetItem(values, 7, self->config->useMTRandGen() ? Py_True : Py_False);
  PyList_SetItem(values, 8, PyLong_FromLong(self->config->getSeedPseudoRandom()));
  PyList_SetItem(values, 9, PyLong_FromUnsignedLong(self->config->getDisplayTrajectories()));
  PyList_SetItem(values, 10, PyLong_FromUnsignedLong(self->config->getStatDistTrajCount()));
  PyList_SetItem(values, 11, PyFloat_FromDouble(self->config->getStatdistClusterThreshold()));
  PyList_SetItem(values, 12, PyLong_FromUnsignedLong(self->config->getThreadCount()));
  PyList_SetItem(values, 13, PyLong_FromUnsignedLong(self->config->getStatDistSimilarityCacheMaxSize()));
  PyList_SetItem(values, 14, PyLong_FromUnsignedLong(self->config->getInitPop()));
  int i = 0;
  for (auto const& item_name : st->getSymbolsNames()) {
    PyList_SetItem(values, 15 + i, PyFloat_FromDouble(st->getSymbolValue(st->getSymbol(item_name))));
    i++;
  }
  return values;
}

static PyObject* cMaBoSSParam_getItems(cMaBoSSParamObject* self)
{
  SymbolTable* st = self->network->getSymbolTable();
  PyObject* items = PyList_New(15 + st->getSymbolsNames().size());
  PyList_SetItem(items, 0, PyTuple_Pack(2, PyUnicode_FromString("time_tick"), PyFloat_FromDouble(self->config->getTimeTick())));
  PyList_SetItem(items, 1, PyTuple_Pack(2, PyUnicode_FromString("max_time"), PyFloat_FromDouble(self->config->getMaxTime())));
  PyList_SetItem(items, 2, PyTuple_Pack(2, PyUnicode_FromString("sample_count"), PyLong_FromUnsignedLong(self->config->getSampleCount())));
  PyList_SetItem(items, 3, PyTuple_Pack(2, PyUnicode_FromString("init_pop"), PyLong_FromUnsignedLong(self->config->getInitPop())));
  PyList_SetItem(items, 4, PyTuple_Pack(2, PyUnicode_FromString("discrete_time"), self->config->isDiscreteTime() ? Py_True : Py_False));
  PyList_SetItem(items, 5, PyTuple_Pack(2, PyUnicode_FromString("use_physrandgen"), self->config->usePhysRandGen() ? Py_True : Py_False));
  PyList_SetItem(items, 6, PyTuple_Pack(2, PyUnicode_FromString("use_glibcrandgen"), self->config->useGlibcRandGen() ? Py_True : Py_False));
  PyList_SetItem(items, 7, PyTuple_Pack(2, PyUnicode_FromString("use_mtrandgen"), self->config->useMTRandGen() ? Py_True : Py_False));
  PyList_SetItem(items, 8, PyTuple_Pack(2, PyUnicode_FromString("seed_pseudorandom"), PyLong_FromLong(self->config->getSeedPseudoRandom())));
  PyList_SetItem(items, 9, PyTuple_Pack(2, PyUnicode_FromString("display_traj"), PyLong_FromUnsignedLong(self->config->getDisplayTrajectories())));
  PyList_SetItem(items, 10, PyTuple_Pack(2, PyUnicode_FromString("statdist_traj_count"), PyLong_FromUnsignedLong(self->config->getStatDistTrajCount())));
  PyList_SetItem(items, 11, PyTuple_Pack(2, PyUnicode_FromString("statdist_cluster_threshold"), PyFloat_FromDouble(self-> config->getStatdistClusterThreshold())));
  PyList_SetItem(items, 12, PyTuple_Pack(2, PyUnicode_FromString("thread_count"), PyLong_FromUnsignedLong(self->config->getThreadCount())));
  PyList_SetItem(items, 13, PyTuple_Pack(2, PyUnicode_FromString("statdist_similarity_cache_max_size"), PyLong_FromUnsignedLong(self->config->getStatDistSimilarityCacheMaxSize())));
  PyList_SetItem(items, 14, PyTuple_Pack(2, PyUnicode_FromString("init_pop"), PyLong_FromUnsignedLong(self->config->getInitPop())));
  int i = 0;
  for (auto const& item_name : st->getSymbolsNames()) {
    PyList_SetItem(items, 15 + i, PyTuple_Pack(2, PyUnicode_FromString(item_name.c_str()), PyFloat_FromDouble(st->getSymbolValue(st->getSymbol(item_name)))));
    i++;
  }
  return items;
}
  
static PyMethodDef cMaBoSSParam_methods[] = {
  {"keys", (PyCFunction) cMaBoSSParam_getKeys, METH_NOARGS, "returns the keys"},
  {"values", (PyCFunction) cMaBoSSParam_getValues, METH_NOARGS, "returns the values"},
  {"items", (PyCFunction) cMaBoSSParam_getItems, METH_NOARGS, "returns the items"},
  {NULL}  /* Sentinel */
};

static PyMappingMethods cMaBoSSParam_mapping = {
	(lenfunc)cMaBoSSParam_Length,		// lenfunc PyMappingMethods.mp_length
	(binaryfunc)cMaBoSSParam_GetItem,		// binaryfunc PyMappingMethods.mp_subscript
	(objobjargproc)cMaBoSSParam_SetItem,		// objobjargproc PyMappingMethods.mp_ass_subscript
};

static PyTypeObject cMaBoSSParam = []{
    PyTypeObject net{PyVarObject_HEAD_INIT(NULL, 0)};

    net.tp_name = "cmaboss.cMaBoSSParamObject";
    net.tp_basicsize = sizeof(cMaBoSSParamObject);
    net.tp_itemsize = 0;
    net.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    net.tp_doc = "cMaBoSS Params object";
    // net.tp_new = cMaBoSSParam_new;
    net.tp_dealloc = (destructor) cMaBoSSParam_dealloc;
    net.tp_methods = cMaBoSSParam_methods;
    net.tp_as_mapping = &cMaBoSSParam_mapping;
    return net;
}();
#endif