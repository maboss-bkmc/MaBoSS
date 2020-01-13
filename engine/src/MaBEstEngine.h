/* 
   MaBoSS (Markov Boolean Stochastic Simulator)
   Copyright (C) 2011-2018 Institut Curie, 26 rue d'Ulm, Paris, France
   
   MaBoSS is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.
   
   MaBoSS is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.
   
   You should have received a copy of the GNU Lesser General Public
   License along with this library; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA 
*/

/*
   Module:
     MaBEstEngine.h

   Authors:
     Eric Viara <viara@sysra.com>
     Gautier Stoll <gautier.stoll@curie.fr>
 
   Date:
     January-March 2011
*/

#ifndef _MABESTENGINE_H_
#define _MABESTENGINE_H_

#include <string>
#include <map>
#include <vector>
#include <assert.h>

#include "MetaEngine.h"
#include "BooleanNetwork.h"
#include "Cumulator.h"
#include "RandomGenerator.h"
#include "RunConfig.h"

struct ArgWrapper;

class MaBEstEngine : public MetaEngine {

  std::vector<ArgWrapper*> arg_wrapper_v;
  NodeIndex getTargetNode(RandomGenerator* random_generator, const MAP<NodeIndex, double>& nodeTransitionRates, double total_rate) const;
  double computeTH(const MAP<NodeIndex, double>& nodeTransitionRates, double total_rate) const;
  void epilogue();
  static void* threadWrapper(void *arg);
  void runThread(Cumulator* cumulator, unsigned int start_count_thread, unsigned int sample_count_thread, RandomGeneratorFactory* randgen_factory, int seed, STATE_MAP<NetworkState_Impl, unsigned int>* fixpoint_map, std::ostream* output_traj);
  STATE_MAP<NetworkState_Impl, unsigned int>* mergeFixpointMaps();

public:
  static const std::string VERSION;
  
  MaBEstEngine(Network* network, RunConfig* runconfig);

  void run(std::ostream* output_traj);

  ~MaBEstEngine();
};

#endif
