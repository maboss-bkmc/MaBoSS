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
     StochasticSimulationEngine.h

   Authors:
     Eric Viara <viara@sysra.com>
     Gautier Stoll <gautier.stoll@curie.fr>
 
   Date:
     January-March 2011
*/

#ifndef _STOCHASTICSIMULATIONENGINE_H_
#define _STOCHASTICSIMULATIONENGINE_H_

#include <string>
#include <map>
#include <vector>
#include <assert.h>

#include "BooleanNetwork.h"
#include "RandomGenerator.h"
#include "RunConfig.h"


class StochasticSimulationEngine {

  Network* network;
  RunConfig* runconfig;

  double time_tick;
  double max_time;
  bool discrete_time;
  int seed; 
  NodeIndex getTargetNode(RandomGenerator* random_generator, const MAP<NodeIndex, double>& nodeTransitionRates, double total_rate) const;

public:
  static const std::string VERSION;
  
  StochasticSimulationEngine(Network* network, RunConfig* runconfig): network(network), runconfig(runconfig), time_tick(runconfig->getTimeTick()), max_time(runconfig->getMaxTime()), discrete_time(runconfig->isDiscreteTime()), seed(runconfig->getSeedPseudoRandom()) {}
  void setSeed(int _seed) { seed = _seed; }
  NetworkState_Impl run(NetworkState_Impl* initial_state = NULL, std::ostream* output_traj = NULL);
};

#endif
