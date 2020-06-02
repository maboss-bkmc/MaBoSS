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

  // Duration of the simulation
  double max_time;
  
  // Using discrete time
  bool discrete_time;
  
  // Time tick for discrete time
  double time_tick;
  

  NodeIndex getTargetNode(RandomGenerator* random_generator, const MAP<NodeIndex, double>& nodeTransitionRates, double total_rate) const;

public:
  static const std::string VERSION;
  RandomGenerator *random_generator;

  
  StochasticSimulationEngine(Network* network, RunConfig* runconfig, int seed): network(network), runconfig(runconfig), max_time(runconfig->getMaxTime()), discrete_time(runconfig->isDiscreteTime()), time_tick(runconfig->getTimeTick()) {
    random_generator = runconfig->getRandomGeneratorFactory()->generateRandomGenerator(seed);
  }
  ~StochasticSimulationEngine() { delete random_generator; }
  
  
  void setSeed(int _seed) { 
    random_generator->setSeed(_seed); 
  }
  void setMaxTime(double _max_time) { this->max_time = _max_time; }
  void setDiscreteTime(bool _discrete_time) { this->discrete_time = _discrete_time; }
  void setTimeTick(double _time_tick) { this->time_tick = _time_tick; }
  
  NetworkState run(NetworkState& initial_state, std::ostream* output_traj = NULL);
};

#endif
