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
     sedml_sim.cpp

   Authors:
     Vincent Noël <vincent.noel@curie.fr>
 
   Date:
     April 2025
*/

#include "sedml_sim.h"
#include "maboss_commons.h"

#include "sedml/SedEngine.h"

#ifdef __GLIBC__
#include <malloc.h>
#endif

PyMethodDef sedmlSim_methods[] = {
  {"get_plots", (PyCFunction) sedmlSim_get_plots, METH_NOARGS, "returns the list of plots"},
  {"get_reports", (PyCFunction) sedmlSim_get_reports, METH_NOARGS, "returns the list of reports"},
  {NULL}  /* Sentinel */
};

PyMemberDef sedmlSim_members[] = {
    {NULL}  /* Sentinel */
};

PyTypeObject sedmlSim = {
  PyVarObject_HEAD_INIT(NULL, 0)
  build_type_name("sedmlSimObject"),               /* tp_name */
  sizeof(sedmlSimObject),               /* tp_basicsize */
    0,                              /* tp_itemsize */
  (destructor) sedmlSim_dealloc,      /* tp_dealloc */
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
  "cMaBoSS SEDML Simulation object",                   /* tp_doc */
    0,                              /* tp_traverse */
    0,                              /* tp_clear */
    0,                              /* tp_richcompare */
    0,                              /* tp_weaklistoffset */
    0,                              /* tp_iter */
    0,                              /* tp_iternext */
  sedmlSim_methods,                              /* tp_methods */
  sedmlSim_members,                              /* tp_members */
    0,                              /* tp_getset */
    0,                              /* tp_base */
    0,                              /* tp_dict */
    0,                              /* tp_descr_get */
    0,                              /* tp_descr_set */
    0,                              /* tp_dictoffset */
  sedmlSim_init,                              /* tp_init */
    0,                              /* tp_alloc */
  sedmlSim_new,                      /* tp_new */ 
};

void sedmlSim_dealloc(sedmlSimObject *self)
{
    Py_TYPE(self)->tp_free((PyObject *) self);
}

int sedmlSim_init(PyObject* self, PyObject *args, PyObject* kwargs)  
{
  PyObject * sedml_file = Py_None;
  const char *kwargs_list[] = {"sedml_file", NULL};
  if (!PyArg_ParseTupleAndKeywords(
    args, kwargs, "|O", const_cast<char **>(kwargs_list), 
    &sedml_file
  )) {
    return -1;
  }
  sedmlSimObject* py_simulation = (sedmlSimObject *) self;

  try {
    SedEngine* engine = new SedEngine();
    py_simulation->engine = engine;  
    engine->parse(PyUnicode_AsUTF8(sedml_file));
    engine->run();
  }
  catch (BNException& e) {
    PyErr_SetString(PyBNException, e.getMessage().c_str());
    return -1;
  }
  
  return 0;
}

PyObject * sedmlSim_new(PyTypeObject* type, PyObject *args, PyObject* kwargs) 
{
  sedmlSimObject* py_simulation = (sedmlSimObject *) type->tp_alloc(type, 0);
  py_simulation->engine = NULL;
  return (PyObject *) py_simulation;
}

PyObject* sedmlSim_get_plots(sedmlSimObject* self)
{
  std::vector<Plot2D> sedplots = self->engine->getPlots();
  PyObject* list_plots = PyList_New(sedplots.size());
  size_t i=0;
  
  for (const auto& plot: sedplots)
  {
    PyObject* plot_data = plot.getPlotData();
    PyList_SetItem(list_plots, i, plot_data);
    i++;
  }
  
  Py_INCREF(list_plots);
  return list_plots;
}

PyObject* sedmlSim_get_reports(sedmlSimObject* self)
{
  std::vector<Report> sedreports = self->engine->getReports();
  PyObject* list_reports = PyList_New(sedreports.size());
  size_t i=0;
  
  for (const auto& report: sedreports)
  {
    PyObject* report_data = report.getReportData();
    PyList_SetItem(list_reports, i, report_data);
    i++;
  }
  
  Py_INCREF(list_reports);
  return list_reports;
}
