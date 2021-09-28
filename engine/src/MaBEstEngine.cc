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
     MaBEstEngine.cc

   Authors:
     Eric Viara <viara@sysra.com>
     Gautier Stoll <gautier.stoll@curie.fr>
     Vincent Noël <vincent.noel@curie.fr>
 
   Date:
     January-March 2011
*/

#include "MaBEstEngine.h"
#include "Probe.h"
#include <stdlib.h>
#include <math.h>
#include <iomanip>
#include <iostream>

const std::string MaBEstEngine::VERSION = "2.4.2b";
size_t RandomGenerator::generated_number_count = 0;

MaBEstEngine::MaBEstEngine(Network* network, RunConfig* runconfig) :
  MetaEngine(network, runconfig)
  {

  if (thread_count > sample_count) {
    thread_count = sample_count;
  }

  if (thread_count > 1 && !runconfig->getRandomGeneratorFactory()->isThreadSafe()) {
    std::cerr << "Warning: non reentrant random, may not work properly in multi-threaded mode\n";
  }

  const std::vector<Node*>& nodes = network->getNodes();
  std::vector<Node*>::const_iterator begin = nodes.begin();
  std::vector<Node*>::const_iterator end = nodes.end();

  NetworkState internal_state;
  bool has_internal = false;
  refnode_count = 0;
  while (begin != end) {
    Node* node = *begin;
    if (node->isInternal()) {
      has_internal = true;
      internal_state.setNodeState(node, true);
    }
    if (node->isReference()) {
      reference_state.setNodeState(node, node->getReferenceState());
      refnode_mask.setNodeState(node, true);
      refnode_count++;
    }
    ++begin;
  }

  merged_cumulator = NULL;
  cumulator_v.resize(thread_count);
  unsigned int count = sample_count / thread_count;
  unsigned int firstcount = count + sample_count - count * thread_count;

  unsigned int scount = statdist_trajcount / thread_count;
  unsigned int first_scount = scount + statdist_trajcount - scount * thread_count;

  for (unsigned int nn = 0; nn < thread_count; ++nn) {
    Cumulator* cumulator = new Cumulator(runconfig, runconfig->getTimeTick(), runconfig->getMaxTime(), (nn == 0 ? firstcount : count), (nn == 0 ? first_scount : scount ));
    
    if (has_internal) {
#ifdef USE_STATIC_BITSET
      NetworkState_Impl state2 = ~internal_state.getState();
      cumulator->setOutputMask(state2);
#else
      cumulator->setOutputMask(~internal_state.getState());
#endif
    }
    cumulator->setRefnodeMask(refnode_mask.getState());
    cumulator_v[nn] = cumulator;
  }
}

struct ArgWrapper {
  MaBEstEngine* mabest;
  unsigned int start_count_thread;
  unsigned int sample_count_thread;
  Cumulator* cumulator;
  RandomGeneratorFactory* randgen_factory;
  int seed;
  STATE_MAP<NetworkState_Impl, unsigned int>* fixpoint_map;
  std::ostream* output_traj;

  ArgWrapper(MaBEstEngine* mabest, unsigned int start_count_thread, unsigned int sample_count_thread, Cumulator* cumulator, RandomGeneratorFactory* randgen_factory, int seed, STATE_MAP<NetworkState_Impl, unsigned int>* fixpoint_map, std::ostream* output_traj) :
    mabest(mabest), start_count_thread(start_count_thread), sample_count_thread(sample_count_thread), cumulator(cumulator), randgen_factory(randgen_factory), seed(seed), fixpoint_map(fixpoint_map), output_traj(output_traj) { }
};

void* MaBEstEngine::threadWrapper(void *arg)
{
#ifdef USE_DYNAMIC_BITSET
  MBDynBitset::init_pthread();
#endif
  ArgWrapper* warg = (ArgWrapper*)arg;
  try {
    warg->mabest->runThread(warg->cumulator, warg->start_count_thread, warg->sample_count_thread, warg->randgen_factory, warg->seed, warg->fixpoint_map, warg->output_traj);
  } catch(const BNException& e) {
    std::cerr << e;
  }
#ifdef USE_DYNAMIC_BITSET
  MBDynBitset::end_pthread();
#endif
  return NULL;
}

void MaBEstEngine::runThread(Cumulator* cumulator, unsigned int start_count_thread, unsigned int sample_count_thread, RandomGeneratorFactory* randgen_factory, int seed, STATE_MAP<NetworkState_Impl, unsigned int>* fixpoint_map, std::ostream* output_traj)
{
  const std::vector<Node*>& nodes = network->getNodes();
  std::vector<Node*>::const_iterator begin = nodes.begin();
  std::vector<Node*>::const_iterator end = nodes.end();
  unsigned int stable_cnt = 0;
  NetworkState network_state; 

#ifdef MPI_COMPAT
  // std::cout << "Running samples " << start_count_thread << " to " << (start_count_thread + sample_count_thread-1) << " on node " << world_rank << std::endl;
#else
  // std::cout << "Running samples " << start_count_thread << " to " << (start_count_thread + sample_count_thread-1) << std::endl;
#endif
  
  RandomGenerator* random_generator = randgen_factory->generateRandomGenerator(seed);
  for (unsigned int nn = 0; nn < sample_count_thread; ++nn) {
#ifdef MPI_COMPAT
    // std::cout << "Running thread with seed " << seed+start_count_thread+nn  << " on node " << world_rank << std::endl;
#else 
    // std::cout << "Running thread with seed " << seed+start_count_thread+nn << std::endl;
#endif
    random_generator->setSeed(seed+start_count_thread+nn);
    cumulator->rewind();
    network->initStates(network_state, random_generator);
    double tm = 0.;
    unsigned int step = 0;
    if (NULL != output_traj) {
      (*output_traj) << "\nTrajectory #" << (nn+1) << '\n';
      (*output_traj) << " istate\t";
      network_state.displayOneLine(*output_traj, network);
      (*output_traj) << '\n';
    }
    while (tm < max_time) {
      double total_rate = 0.;
      MAP<NodeIndex, double> nodeTransitionRates;
      begin = nodes.begin();

      while (begin != end) {
	Node* node = *begin;
	NodeIndex node_idx = node->getIndex();
	if (node->getNodeState(network_state)) {
	  double r_down = node->getRateDown(network_state);
	  if (r_down != 0.0) {
	    total_rate += r_down;
	    nodeTransitionRates[node_idx] = r_down;
	  }
	} else {
	  double r_up = node->getRateUp(network_state);
	  if (r_up != 0.0) {
	    total_rate += r_up;
	    nodeTransitionRates[node_idx] = r_up;
	  }
	}
	++begin;
      }

      // EV: 2018-12-19 suppressed this block and integrated fixed point management below
      /*
      if (total_rate == 0.0) {
	std::cerr << "FP\n";
	// may have several fixpoint maps
	if (fixpoint_map->find(network_state.getState()) == fixpoint_map->end()) {
	  (*fixpoint_map)[network_state.getState()] = 1;
	} else {
	  (*fixpoint_map)[network_state.getState()]++;
	}
	cumulator->cumul(network_state, max_time, 0.);
	tm = max_time;
	stable_cnt++;
	break;
      }
      */

      double TH;
      if (total_rate == 0) {
	tm = max_time;
	TH = 0.;
	if (fixpoint_map->find(network_state.getState()) == fixpoint_map->end()) {
	  (*fixpoint_map)[network_state.getState()] = 1;
	} else {
	  (*fixpoint_map)[network_state.getState()]++;
	}
	stable_cnt++;
      } else {
	double transition_time ;
	if (discrete_time) {
	  transition_time = time_tick;
	} else {
	  double U_rand1 = random_generator->generate();
	  transition_time = -log(U_rand1) / total_rate;
	}
	
	tm += transition_time;
	TH = computeTH(network, nodeTransitionRates, total_rate);
      }

      if (NULL != output_traj) {
	(*output_traj) << std::setprecision(10) << tm << '\t';
	network_state.displayOneLine(*output_traj, network);
	(*output_traj) << '\t' << TH << '\n';
      }

      cumulator->cumul(network_state, tm, TH);

      if (tm >= max_time) {
	break;
      }

      NodeIndex node_idx = getTargetNode(network, random_generator, nodeTransitionRates, total_rate);
      network_state.flipState(network->getNode(node_idx));
      step++;
    }
    cumulator->trajectoryEpilogue();
  }
  delete random_generator;
}

void MaBEstEngine::run(std::ostream* output_traj)
{
  pthread_t* tid = new pthread_t[thread_count];
  RandomGeneratorFactory* randgen_factory = runconfig->getRandomGeneratorFactory();
  int seed = runconfig->getSeedPseudoRandom();
#ifdef MPI_COMPAT
  
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  seed += sample_count*world_rank;
  
#endif
  unsigned int start_sample_count = 0;
  Probe probe;
  for (unsigned int nn = 0; nn < thread_count; ++nn) {
    STATE_MAP<NetworkState_Impl, unsigned int>* fixpoint_map = new STATE_MAP<NetworkState_Impl, unsigned int>();
    fixpoint_map_v.push_back(fixpoint_map);
    ArgWrapper* warg = new ArgWrapper(this, start_sample_count, cumulator_v[nn]->getSampleCount(), cumulator_v[nn], randgen_factory, seed, fixpoint_map, output_traj);
    pthread_create(&tid[nn], NULL, MaBEstEngine::threadWrapper, warg);
    arg_wrapper_v.push_back(warg);

    start_sample_count += cumulator_v[nn]->getSampleCount();
  }
  for (unsigned int nn = 0; nn < thread_count; ++nn) {
    // std::cout << "Lauching thread " << nn << ", seed = " << seed << std::endl;
    pthread_join(tid[nn], NULL);
  }
  probe.stop();
  elapsed_core_runtime = probe.elapsed_msecs();
  user_core_runtime = probe.user_msecs();
  // std::cout << "Trajectories computed, running epilogue" << std::endl;
  probe.start();
  epilogue();
  probe.stop();
  elapsed_epilogue_runtime = probe.elapsed_msecs();
  user_epilogue_runtime = probe.user_msecs();
  // std::cout << "Epilogue done, quitting run()" << std::endl;
  delete [] tid;
}  

STATE_MAP<NetworkState_Impl, unsigned int>* MaBEstEngine::mergeFixpointMaps()
{
  STATE_MAP<NetworkState_Impl, unsigned int>* fixpoint_map;
  // First, each node do its thing
  if (1 == fixpoint_map_v.size()) {
    fixpoint_map = new STATE_MAP<NetworkState_Impl, unsigned int>(*fixpoint_map_v[0]);
  } else {

    fixpoint_map = new STATE_MAP<NetworkState_Impl, unsigned int>();
    std::vector<STATE_MAP<NetworkState_Impl, unsigned int>*>::iterator begin = fixpoint_map_v.begin();
    std::vector<STATE_MAP<NetworkState_Impl, unsigned int>*>::iterator end = fixpoint_map_v.end();
    while (begin != end) {
      STATE_MAP<NetworkState_Impl, unsigned int>* fp_map = *begin;
      STATE_MAP<NetworkState_Impl, unsigned int>::const_iterator b = fp_map->begin();
      STATE_MAP<NetworkState_Impl, unsigned int>::const_iterator e = fp_map->end();
      while (b != e) {
        //NetworkState_Impl state = (*b).first;
        const NetworkState_Impl& state = b->first;
        if (fixpoint_map->find(state) == fixpoint_map->end()) {
    (*fixpoint_map)[state] = (*b).second;
        } else {
    (*fixpoint_map)[state] += (*b).second;
        }
        ++b;
      }
      ++begin;
    }
  }
  
  return fixpoint_map;
}


#ifdef MPI_COMPAT
STATE_MAP<NetworkState_Impl, unsigned int>* MaBEstEngine::mergeMPIFixpointMaps(STATE_MAP<NetworkState_Impl, unsigned int>* t_fixpoint_map)
{
  // If we are, but only on one node, we don't need to do anything
  if (world_size == 1) {
    // std::cout << "Single node simulation. Returning fixpoints" << std::endl;
    return t_fixpoint_map;
  } else {
    
    for (int i = 1; i < world_size; i++) {
      
      if (world_rank == 0) {
        
        // std::cout << "Here in rank zero we wait for result of rank " << i << std::endl;
        
        // First we want to know the number of fixpoints we're going to receive   
        int nb_fixpoints = -1;
        MPI_Recv(&nb_fixpoints, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // std::cout << "We received the number of fixpoints from node " << i << " : " << nb_fixpoints << std::endl;
        
        for (int j = 0; j < nb_fixpoints; j++) {
          NetworkState_Impl state;
          MPI_Recv(&state, 1, MPI_UNSIGNED_LONG_LONG, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          unsigned int count = -1;
          MPI_Recv(&count, 1, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          
          if (t_fixpoint_map->find(state) == t_fixpoint_map->end()) {
            (*t_fixpoint_map)[state] = count;
          } else {
            (*t_fixpoint_map)[state] += count;
          }
        }
         
      } else {
        
        // std::cout << "Here in rank " << i << " we send the result to rank 0" << std::endl;
        
        int nb_fixpoints = t_fixpoint_map->size();
        // First we send the number of fixpoints we have
        MPI_Send(&nb_fixpoints, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        
        // std::cout << "We send the number of fixpoints to node 0 : " << t_fixpoint_map->size() << std::endl;
        
        for (auto& fixpoint: *t_fixpoint_map) {
          NetworkState_Impl state = fixpoint.first;
          unsigned int count = fixpoint.second;
          
          MPI_Send(&state, 1, MPI_UNSIGNED_LONG_LONG, 0, 0, MPI_COMM_WORLD);
          MPI_Send(&count, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD);
          
        } 
      }      
    }

    return t_fixpoint_map; 
  }
}
#endif

void MaBEstEngine::epilogue()
{
  merged_cumulator = Cumulator::mergeCumulatorsParallel(runconfig, cumulator_v);
  
#ifdef MPI_COMPAT
  merged_cumulator = Cumulator::mergeMPICumulators(runconfig, merged_cumulator, world_size, world_rank);

  if (world_rank == 0)
  {
#endif
  merged_cumulator->epilogue(network, reference_state);
#ifdef MPI_COMPAT
  }
#endif 

  STATE_MAP<NetworkState_Impl, unsigned int>* merged_fixpoint_map = mergeFixpointMaps();

#ifdef MPI_COMPAT
  merged_fixpoint_map = mergeMPIFixpointMaps(merged_fixpoint_map);
#endif

  STATE_MAP<NetworkState_Impl, unsigned int>::const_iterator b = merged_fixpoint_map->begin();
  STATE_MAP<NetworkState_Impl, unsigned int>::const_iterator e = merged_fixpoint_map->end();

  while (b != e) {
    fixpoints[NetworkState((*b).first).getState()] = (*b).second;
    ++b;
  }
  delete merged_fixpoint_map;
}

MaBEstEngine::~MaBEstEngine()
{
#ifdef MPI_COMPAT
  MPI_Finalize();
#endif
  
  for (auto t_fixpoint_map: fixpoint_map_v)
    delete t_fixpoint_map;
  
  for (auto t_arg_wrapper: arg_wrapper_v)
    delete t_arg_wrapper;

  delete merged_cumulator;
}

