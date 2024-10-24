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

#include "maboss_commons.h"

#include "src/BooleanNetwork.h"
#include "src/RunConfig.h"

typedef struct {
  PyObject_HEAD
  Network* network;
  RunConfig* config;
} cMaBoSSParamObject;

void cMaBoSSParam_dealloc(PyObject *self);
PyObject* cMaBoSSParam_new(PyTypeObject* type, PyObject *args, PyObject* kwargs);
int cMaBoSSParam_init(PyObject* self, PyObject *args, PyObject* kwargs);
PyObject* cMaBoSSParam_update_parameters(cMaBoSSParamObject* self, PyObject *args, PyObject* kwargs);
int cMaBoSSParam_SetItem(cMaBoSSParamObject* self, PyObject *key, PyObject* value);
PyObject * cMaBoSSParam_GetItem(cMaBoSSParamObject* self, PyObject *key);
Py_ssize_t cMaBoSSParam_Length(cMaBoSSParamObject* self);
PyObject* cMaBoSSParam_getKeys(cMaBoSSParamObject* self);
PyObject* cMaBoSSParam_getValues(cMaBoSSParamObject* self);
PyObject* cMaBoSSParam_getItems(cMaBoSSParamObject* self);

//     // .tp_repr = (reprfunc)myobj_repr,
    
    
//     PyTypeObject net{PyVarObject_HEAD_INIT(NULL, 0)};

//     net.tp_name = 
//     net.tp_basicsize = ;
//     net.tp_itemsize = 0;
//     net.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
//     net.tp_doc = ;
//     net.tp_call = PyObject_Call;
//     net.tp_init = ;
//     net.tp_new = ;
//     net.tp_dealloc = ;
//     net;
//     net;
//     return net;
// }();
#endif