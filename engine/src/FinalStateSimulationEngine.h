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

#ifndef _FINAL_STATE_SIMULATION_ENGINE_H_
#define _FINAL_STATE_SIMULATION_ENGINE_H_

#include <string>
#include <map>
#include <vector>
#include <assert.h>

#include "MetaEngine.h"
#include "BooleanNetwork.h"
#include "Cumulator.h"
#include "RandomGenerator.h"
#include "RunConfig.h"

struct FinalStateArgWrapper;

class FinalStateSimulationEngine {
  
  Network* network;
  RunConfig* runconfig;

  double time_tick;
  double max_time;
  unsigned int sample_count;
  bool discrete_time;
  unsigned int thread_count;
  bool has_internal = false;
  NetworkState internal_state;

  NetworkState reference_state;
  unsigned int refnode_count;

  std::vector<unsigned int> sample_count_per_thread;

  std::vector<FinalStateArgWrapper*> arg_wrapper_v;
  NodeIndex getTargetNode(RandomGenerator* random_generator, const MAP<NodeIndex, double>& nodeTransitionRates, double total_rate) const;
  void epilogue();
  static void* threadWrapper(void *arg);
  void runThread(unsigned int start_count_thread, unsigned int sample_count_thread, RandomGeneratorFactory* randgen_factory, int seed, STATE_MAP<NetworkState_Impl, unsigned int>* final_state_map, std::ostream* output_traj);
  
  STATE_MAP<NetworkState_Impl, unsigned int>* mergeFinalStateMaps();
  STATE_MAP<NetworkState_Impl, double> final_states;
  std::vector<STATE_MAP<NetworkState_Impl, unsigned int>*> final_states_map_v;

public:
  static const std::string VERSION;
  
  FinalStateSimulationEngine(Network* network, RunConfig* runconfig);

  void run(std::ostream* output_traj);
  ~FinalStateSimulationEngine();

  const STATE_MAP<NetworkState_Impl, double> getFinalStates() const {return final_states;}
  const STATE_MAP<Node*, double> getFinalNodes() const;
  const double getFinalTime() const { return max_time; }

#ifdef PYTHON_API
  PyObject* getNumpyLastStatesDists() const;
  std::vector<Node*> getNodes() const;
  PyObject* getNumpyLastNodesDists() const;
#endif

  void displayFinal(std::ostream& output_final, bool hexfloat=false) const;

};

#endif
