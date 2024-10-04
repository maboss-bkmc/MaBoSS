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
  // SymbolTable* st = self->network->getSymbolTable();
  // for (auto const& item_name : st->getSymbolsNames()) {
  //   std::cout << item_name << " = " << st->getSymbolValue(st->getSymbol(item_name));
  // }
  
  
  PyObject * time_tick = NULL;
  PyObject * max_time = NULL;
  PyObject * sample_count = NULL;
  PyObject * init_pop = NULL;
  PyObject * discrete_time = NULL;
  PyObject * use_physrandgen = NULL;
  PyObject * use_glibcrandgen = NULL;
  PyObject * use_mtrandgen = NULL;
  PyObject * seed_pseudorandom = NULL;
  PyObject * display_traj = NULL;
  PyObject * statdist_traj_count = NULL;
  PyObject * statdist_cluster_threshold = NULL;
  PyObject * thread_count = NULL;
  PyObject * statdist_similarity_cache_max_size = NULL; 
 
  static const char *kwargs_list[] = {
    "time_tick", "max_time", "sample_count", "init_pop",
    "discrete_time", "use_physrandgen", "use_glibcrandgen",
    "use_mtrandgen", "seed_pseudorandom", "display_traj", 
    "statdist_traj_count", "statdist_cluster_threshold", 
    "thread_count", "statdist_similarity_cache_max_size",
    NULL
  };
  
  if (!PyArg_ParseTupleAndKeywords(
    args, kwargs, "|OOOOOOOOOOOOOO", const_cast<char **>(kwargs_list), 
    &time_tick, &max_time, &sample_count, &init_pop, 
    &discrete_time, &use_physrandgen, &use_glibcrandgen, 
    &use_mtrandgen, &seed_pseudorandom, &display_traj, 
    &statdist_traj_count, &statdist_cluster_threshold, 
    &thread_count, &statdist_similarity_cache_max_size
  ))
    return NULL;
    
  if (time_tick != NULL) {
    self->config->setParameter("time_tick", PyFloat_AsDouble(time_tick));
  }
  if (max_time != NULL) {
    self->config->setParameter("max_time", PyFloat_AsDouble(max_time));
  }
  if (sample_count != NULL) {
    self->config->setParameter("sample_count", PyLong_AsLong(sample_count));
  }
  if (init_pop != NULL) {
    self->config->setParameter("init_pop", PyLong_AsLong(init_pop));
  }
  if (discrete_time != NULL) {
    self->config->setParameter("discrete_time", PyLong_AsLong(discrete_time));
  }
  if (use_physrandgen != NULL) {
    self->config->setParameter("use_physrandgen", PyLong_AsLong(use_physrandgen));
  }
  if (use_glibcrandgen != NULL) {
    self->config->setParameter("use_glibcrandgen", PyLong_AsLong(use_glibcrandgen));
  }
  if (use_mtrandgen != NULL) {
    self->config->setParameter("use_mtrandgen", PyLong_AsLong(use_mtrandgen));
  }
  if (seed_pseudorandom != NULL) {
    self->config->setParameter("seed_pseudorandom", PyFloat_AsDouble(seed_pseudorandom));
  }
  if (display_traj != NULL) {
    self->config->setParameter("display_traj", PyLong_AsLong(display_traj));
  }
  if (statdist_traj_count != NULL) {
    self->config->setParameter("statdist_traj_count", PyLong_AsLong(statdist_traj_count));
  }
  if (statdist_cluster_threshold != NULL) {
    self->config->setParameter("statdist_cluster_threshold", PyFloat_AsDouble(statdist_cluster_threshold));
  }
  if (thread_count != NULL) {
    self->config->setParameter("thread_count", PyLong_AsLong(thread_count));
  }
  if (statdist_similarity_cache_max_size != NULL) {
    self->config->setParameter("statdist_similarity_cache_max_size", PyLong_AsLong(statdist_similarity_cache_max_size));
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
  
  } 
  return NULL;
}

static Py_ssize_t cMaBoSSParam_Length(cMaBoSSParamObject* self)
{
  // return PyObject_Length(self->nodes);
  return 13;
}

static PyMethodDef cMaBoSSParam_methods[] = {
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