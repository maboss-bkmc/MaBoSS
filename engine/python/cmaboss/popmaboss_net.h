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
     popmaboss_net.h

   Authors:
     Vincent Noël <vincent.noel@curie.fr>
 
   Date:
     January-March 2020
*/

#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <set>
#include "src/BooleanNetwork.h"
#include "src/MaBEstEngine.h"

typedef struct {
  PyObject_HEAD
  PopNetwork* network;
} cMaBoSSPopNetworkObject;


static void cMaBoSSPopNetwork_dealloc(cMaBoSSPopNetworkObject *self);

static PyObject * cMaBoSSPopNetwork_new(PyTypeObject* type, PyObject *args, PyObject* kwargs);

static PyMethodDef cMaBoSSPopNetwork_methods[] = {
    {NULL}  /* Sentinel */
};

static PyTypeObject cMaBoSSPopNetwork = []{
    PyTypeObject net{PyVarObject_HEAD_INIT(NULL, 0)};

    net.tp_name = "cmaboss.cMaBoSSPopNetworkObject";
    net.tp_basicsize = sizeof(cMaBoSSPopNetworkObject);
    net.tp_itemsize = 0;
    net.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    net.tp_doc = "cMaBoSS PopNetwork object";
    net.tp_new = cMaBoSSPopNetwork_new;
    net.tp_dealloc = (destructor) cMaBoSSPopNetwork_dealloc;
    net.tp_methods = cMaBoSSPopNetwork_methods;
    return net;
}();