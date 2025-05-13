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
     maboss_net.h

   Authors:
     Vincent Noël <vincent.noel@curie.fr>
 
   Date:
     January-March 2020
*/

#ifndef MABOSS_NETWORK_H
#define MABOSS_NETWORK_H

#include "maboss_commons.h"

#include "Network.h"
typedef struct {
  PyObject_HEAD
  Network* network;
  PyObject* nodes;
} cMaBoSSNetworkObject;


void cMaBoSSNetwork_dealloc(cMaBoSSNetworkObject *self);
PyObject *cMaBoSSNetwork_str(PyObject *self);
int cMaBoSSNetwork_NodesSetItem(cMaBoSSNetworkObject* self, PyObject *key, PyObject* value);
PyObject * cMaBoSSNetwork_NodesGetItem(cMaBoSSNetworkObject* self, PyObject *key);
Py_ssize_t cMaBoSSNetwork_NodesLength(cMaBoSSNetworkObject* self);
PyObject * cMaBoSSNetwork_Keys(cMaBoSSNetworkObject* self);
PyObject * cMaBoSSNetwork_Values(cMaBoSSNetworkObject* self);
PyObject * cMaBoSSNetwork_Items(cMaBoSSNetworkObject* self);
PyObject* cMaBoSSNetwork_setOutput(cMaBoSSNetworkObject* self, PyObject *args);
PyObject* cMaBoSSNetwork_getOutput(cMaBoSSNetworkObject* self);
PyObject* cMaBoSSNetwork_setObservedGraphNode(cMaBoSSNetworkObject* self, PyObject *args);
PyObject* cMaBoSSNetwork_getObservedGraphNode(cMaBoSSNetworkObject* self, PyObject *args);
PyObject* cMaBoSSNetwork_addNode(cMaBoSSNetworkObject* self, PyObject *args);
PyObject* cMaBoSSNetwork_setIState(cMaBoSSNetworkObject* self, PyObject *args); 
PyObject* cMaBoSSNetwork_getIState(cMaBoSSNetworkObject* self);
PyObject * cMaBoSSNetwork_new(PyTypeObject* type, PyObject *args, PyObject* kwargs);
int cMaBoSSNetwork_init(PyObject* self, PyObject *args, PyObject* kwargs);

#endif

