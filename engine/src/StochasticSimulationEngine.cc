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
     StochasticSimulationEngine.cc

   Authors:
     Eric Viara <viara@sysra.com>
     Gautier Stoll <gautier.stoll@curie.fr>
 
   Date:
     January-March 2011
*/

#include "StochasticSimulationEngine.h"
#include "Probe.h"
#include <stdlib.h>
#include <math.h>
#include <iomanip>
#include <iostream>

const std::string StochasticSimulationEngine::VERSION = "1.0.0";

NodeIndex StochasticSimulationEngine::getTargetNode(RandomGenerator *random_generator, const MAP<NodeIndex, double> &nodeTransitionRates, double total_rate) const
{
  double U_rand2 = random_generator->generate();
  double random_rate = U_rand2 * total_rate;
  MAP<NodeIndex, double>::const_iterator begin = nodeTransitionRates.begin();
  MAP<NodeIndex, double>::const_iterator end = nodeTransitionRates.end();
  NodeIndex node_idx = INVALID_NODE_INDEX;
  while (begin != end && random_rate > 0.)
  {
    node_idx = (*begin).first;
    double rate = (*begin).second;
    random_rate -= rate;
    ++begin;
  }

  assert(node_idx != INVALID_NODE_INDEX);
  assert(network->getNode(node_idx)->getIndex() == node_idx);
  return node_idx;
}

NetworkState_Impl StochasticSimulationEngine::run(NetworkState_Impl* initial_state, std::ostream *output_traj)
{
  const std::vector<Node *> &nodes = network->getNodes();
  std::vector<Node *>::const_iterator begin = nodes.begin();
  std::vector<Node *>::const_iterator end = nodes.end();
  NetworkState network_state;

  RandomGeneratorFactory *randgen_factory = runconfig->getRandomGeneratorFactory();
  RandomGenerator *random_generator = randgen_factory->generateRandomGenerator(seed);
    
  if (initial_state != NULL) {
    network_state = *initial_state;
  } else {
    network->initStates(network_state, random_generator);
  }
  
  double tm = 0.;
  unsigned int step = 0;
  if (NULL != output_traj)
  {
    // (*output_traj) << "\nTrajectory #" << (nn+1) << '\n';
    (*output_traj) << " istate\t";
    network_state.displayOneLine(*output_traj, network);
    (*output_traj) << '\n';
  }
  while (tm < max_time)
  {
    double total_rate = 0.;
    MAP<NodeIndex, double> nodeTransitionRates;
    begin = nodes.begin();

    while (begin != end)
    {
      Node *node = *begin;
      NodeIndex node_idx = node->getIndex();
      if (node->getNodeState(network_state))
      {
        double r_down = node->getRateDown(network_state);
        if (r_down != 0.0)
        {
          total_rate += r_down;
          nodeTransitionRates[node_idx] = r_down;
        }
      }
      else
      {
        double r_up = node->getRateUp(network_state);
        if (r_up != 0.0)
        {
          total_rate += r_up;
          nodeTransitionRates[node_idx] = r_up;
        }
      }
      ++begin;
    }

    // double TH;
    if (total_rate == 0) {
      tm = max_time;
    } else {
      
      double transition_time ;
      if (discrete_time) {
        transition_time = time_tick;
      } else {
        double U_rand1 = random_generator->generate();
        transition_time = -log(U_rand1) / total_rate;
      }
      
      tm += transition_time;
    }

    if (NULL != output_traj)
    {
      (*output_traj) << std::setprecision(10) << tm << '\t';
      network_state.displayOneLine(*output_traj, network);
      (*output_traj) << std::endl;
    }

    if (tm >= max_time)
      break;

    NodeIndex node_idx = getTargetNode(random_generator, nodeTransitionRates, total_rate);
    network_state.flipState(network->getNode(node_idx));
    step++;
  }
  
  delete random_generator;
  return network_state.getState();
}
