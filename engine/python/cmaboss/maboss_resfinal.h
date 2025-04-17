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
     maboss_resfinal.h

   Authors:
     Vincent Noël <vincent.noel@curie.fr>
 
   Date:
     January-March 2020
*/

#ifndef MABOSS_RESFINAL
#define MABOSS_RESFINAL

#include "maboss_commons.h"

#include "src/BooleanNetwork.h"
#include "src/RunConfig.h"
#include "src/engines/FinalStateSimulationEngine.h"

typedef struct {
  PyObject_HEAD
  Network* network;
  RunConfig* runconfig;
  FinalStateSimulationEngine* engine;
  time_t start_time;
  time_t end_time;
  PyObject* last_probtraj;
} cMaBoSSResultFinalObject;

void cMaBoSSResultFinal_dealloc(cMaBoSSResultFinalObject *self);
PyObject * cMaBoSSResultFinal_new(PyTypeObject* type, PyObject *args, PyObject* kwargs);
PyObject* cMaBoSSResultFinal_get_last_probtraj(cMaBoSSResultFinalObject* self);
PyObject* cMaBoSSResultFinal_get_last_nodes_probtraj(cMaBoSSResultFinalObject* self, PyObject* args);
PyObject* cMaBoSSResultFinal_display_final_states(cMaBoSSResultFinalObject* self, PyObject* args);
PyObject* cMaBoSSResultFinal_get_final_time(cMaBoSSResultFinalObject* self);
PyObject* cMaBoSSResultFinal_display_run(cMaBoSSResultFinalObject* self, PyObject* args);

#endif