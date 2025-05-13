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
     maboss_cfg.cpp

   Authors:
     Vincent Noël <vincent.noel@curie.fr>
 
   Date:
     January-March 2020
*/

#include "maboss_cfg.h"
#include "maboss_net.h"
#include "popmaboss_net.h"


PyMethodDef cMaBoSSConfig_methods[] = {
    {NULL}  /* Sentinel */
};

PyTypeObject cMaBoSSConfig = {
  PyVarObject_HEAD_INIT(NULL, 0)
  build_type_name("cMaBoSSConfigObject"),               /* tp_name */
  sizeof(cMaBoSSConfigObject),               /* tp_basicsize */
    0,                              /* tp_itemsize */
  (destructor) cMaBoSSConfig_dealloc,      /* tp_dealloc */
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
  "cMaBoSS Config object",                   /* tp_doc */
    0,                              /* tp_traverse */
    0,                              /* tp_clear */
    0,                              /* tp_richcompare */
    0,                              /* tp_weaklistoffset */
    0,                              /* tp_iter */
    0,                              /* tp_iternext */
  cMaBoSSConfig_methods,                              /* tp_methods */
    0,                              /* tp_members */
    0,                              /* tp_getset */
    0,                              /* tp_base */
    0,                              /* tp_dict */
    0,                              /* tp_descr_get */
    0,                              /* tp_descr_set */
    0,                              /* tp_dictoffset */
  cMaBoSSConfig_init,                              /* tp_init */
    0,                              /* tp_alloc */
  cMaBoSSConfig_new,                      /* tp_new */
};

void cMaBoSSConfig_dealloc(cMaBoSSConfigObject *self)
{
    delete self->config;
    Py_TYPE(self)->tp_free((PyObject *) self);
}

PyObject * cMaBoSSConfig_new(PyTypeObject* type, PyObject *args, PyObject* kwargs) 
{
  cMaBoSSConfigObject* py_config = (cMaBoSSConfigObject *) type->tp_alloc(type, 0);
  py_config->config = new RunConfig();
  return (PyObject*) py_config;
}
int cMaBoSSConfig_init(PyObject* self, PyObject *args, PyObject* kwargs) 
{
  PyObject * py_network = Py_None;
  PyObject * config_file = Py_None;
  PyObject * config_files = Py_None;
  PyObject * config_str = Py_None;
  
  const char *kwargs_list[] = {"network", "config_file", "config_files", "config_str", NULL};
  if (!PyArg_ParseTupleAndKeywords(
    args, kwargs, "|OOOO", const_cast<char **>(kwargs_list), 
    &py_network, &config_file, &config_files, &config_str
  ))
    return -1;
  
  Network* network = NULL;
  
  if (py_network != Py_None && PyObject_IsInstance(py_network, (PyObject*)&cMaBoSSNetwork))
  {
    network = ((cMaBoSSNetworkObject*) py_network)->network;
    
  } else if (py_network != Py_None && PyObject_IsInstance(py_network, (PyObject*)&cPopMaBoSSNetwork))
  {
    network = ((cPopMaBoSSNetworkObject*) py_network)->network;
    
  } else {
    PyErr_SetString(PyBNException, "Invalid network object");
    return -1;
  }
  
  cMaBoSSConfigObject* py_config = (cMaBoSSConfigObject *) self;

  try 
  {
    if (config_file != Py_None) 
    {
       IStateGroup::reset(network);
       py_config->config->parse(network, PyUnicode_AsUTF8(config_file));
      
    } else if (config_files != Py_None)
    {
      IStateGroup::reset(network);
      for (int i = 0; i < PyList_Size(config_files); i++) {
        PyObject* item = PyList_GetItem(config_files, i);
        py_config->config->parse(network, PyUnicode_AsUTF8(item));
      }
      
    } else if (config_str != Py_None)
    {
      IStateGroup::reset(network);
      py_config->config->parseExpression(network, PyUnicode_AsUTF8(config_str));
      
    } 
    
  } catch (BNException& e) {
    py_config = NULL;
    PyErr_SetString(PyBNException, e.getMessage().c_str());
    return -1;
  }
  return 0;
}
