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
     FixedPointEngine.h

   Authors:
     Vincent Noel <contact@vincent-noel.fr>
 
   Date:
     March 2021
*/

#ifndef _FIXEDPOINTENGINE_H_
#define _FIXEDPOINTENGINE_H_

#include <map>
#include <vector>
#include <assert.h>

#ifdef MPI_COMPAT
#include <mpi.h>
#endif

#include "MetaEngine.h"
#include "../Network.h"
#include "../RunConfig.h"
#include "../displayers/FixedPointDisplayer.h"
#include "../maps_header.h"

struct EnsembleArgWrapper;
typedef STATE_MAP<NetworkState_Impl, unsigned int> FixedPoints;
class FixedPointEngine : public MetaEngine {

protected:

  FixedPoints* fixpoints;
  std::vector<FixedPoints*> fixpoint_map_v;
  static void mergePairOfFixpoints(FixedPoints* fixpoints_1, FixedPoints* fixpoints_2);

#ifdef MPI_COMPAT
  static void mergePairOfMPIFixpoints(FixedPoints* fixpoints, int world_rank, int dest, int origin, bool pack=true);

  static void MPI_Unpack_Fixpoints(FixedPoints* fp_map, char* buff, unsigned int buff_size);
  static char* MPI_Pack_Fixpoints(const FixedPoints* fp_map, int dest, unsigned int * buff_size);
  static void MPI_Send_Fixpoints(const FixedPoints* fp_map, int dest);
  static void MPI_Recv_Fixpoints(FixedPoints* fp_map, int origin);
  
#endif

public:

#ifdef MPI_COMPAT
  FixedPointEngine(Network * network, RunConfig* runconfig, int world_size, int world_rank) : MetaEngine(network, runconfig, world_size, world_rank) {}
#else
  FixedPointEngine(Network * network, RunConfig* runconfig) : MetaEngine(network, runconfig) {}
#endif

  bool converges() const {return fixpoints->size() > 0;}
  const FixedPoints* getFixpoints() const {return fixpoints;}
  const std::map<unsigned int, std::pair<NetworkState, double> > getFixPointsDists() const;

  void displayFixpoints(FixedPointDisplayer* displayer) const;

};

#endif
