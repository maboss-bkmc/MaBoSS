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
     PopMaBEstEngine.cc

   Authors:
     Vincent Noël <vincent.noel@curie.fr>
 
   Date:
     March 2021
*/

#include "PopMaBEstEngine.h"
#include "../BooleanNetwork.h"
#include "../Cumulator.h"
#include "MetaEngine.h"
#include "../displayers/ProbTrajDisplayer.h"
#include "../Probe.h"
#include <stdlib.h>
#include <math.h>
#include <iomanip>
#include <iostream>

const std::string PopMaBEstEngine::VERSION = "0.0.1";
// extern size_t RandomGenerator::generated_number_count;


#ifdef MPI_COMPAT
PopMaBEstEngine::PopMaBEstEngine(PopNetwork *pop_network, RunConfig *runconfig, int world_size, int world_rank) : MetaEngine(pop_network, runconfig, world_size, world_rank), pop_network(pop_network)
#else
PopMaBEstEngine::PopMaBEstEngine(PopNetwork *pop_network, RunConfig *runconfig) : MetaEngine(pop_network, runconfig), pop_network(pop_network)
#endif
{
  
  elapsed_core_runtime = user_core_runtime = elapsed_statdist_runtime = user_statdist_runtime = elapsed_epilogue_runtime = user_epilogue_runtime = 0;

  if (thread_count > sample_count)
  {
    thread_count = sample_count;
  }

  if (thread_count > 1 && !runconfig->getRandomGeneratorFactory()->isThreadSafe())
  {
    std::cerr << "Warning: non reentrant random, may not work properly in multi-threaded mode\n";
  }

  const std::vector<Node *> &nodes = pop_network->getNodes();
  NetworkState internal_state;
  bool has_internal = false;
  refnode_count = 0;

  if (runconfig->hasCustomPopOutput())
  {
    const Expression* custom_pop_output = runconfig->getCustomPopOutputExpression();
    std::vector<Node*> custom_nodes = custom_pop_output->getNodes();
    for (auto* node: nodes) {
      node->isInternal(std::find(custom_nodes.begin(), custom_nodes.end(), node) == custom_nodes.end());
    }
  }

  for (const auto * node : nodes)
  {
    if (node->isInternal())
    {
      has_internal = true;
      internal_state.setNodeState(node, true);
    }
    if (node->isReference()) {
      reference_state.setNodeState(node, node->getReferenceState());
      refnode_mask.setNodeState(node, true);
      refnode_count++;
    }
  }

  merged_cumulator = NULL;
  custom_pop_cumulator = NULL;
  custom_pop_cumulator_v.resize(thread_count, NULL);
  cumulator_v.resize(thread_count, NULL);
  unsigned int count = sample_count / thread_count;
  unsigned int firstcount = count + sample_count - count * thread_count;
  
  unsigned int scount = statdist_trajcount / thread_count;
  unsigned int first_scount = scount + statdist_trajcount - scount * thread_count;

  for (unsigned int nn = 0; nn < thread_count; ++nn)
  {
    Cumulator<PopSize> *custom_cumulator = new Cumulator<PopSize>(runconfig, runconfig->getTimeTick(), runconfig->getMaxTime(), (nn == 0 ? firstcount : count), (nn == 0 ? first_scount : scount));
    custom_pop_cumulator_v[nn] = custom_cumulator;
    Cumulator<PopNetworkState> *cumulator = new Cumulator<PopNetworkState>(runconfig, runconfig->getTimeTick(), runconfig->getMaxTime(), (nn == 0 ? firstcount : count), (nn == 0 ? first_scount : scount ));
    if (has_internal)
    {
  #ifdef USE_STATIC_BITSET
      NetworkState_Impl state2 = ~internal_state.getState();
      cumulator->setOutputMask(PopNetworkState(state2, 1));
  #else
      cumulator->setOutputMask(PopNetworkState(~internal_state.getState(), 1));
  #endif
    }
    cumulator_v[nn] = cumulator;
  }
}

PopNetworkState PopMaBEstEngine::getTargetNode(RandomGenerator *random_generator, const PopNetworkStateMap& popNodeTransitionRates, double total_rate) const
{
  double U_rand2 = random_generator->generate();
  double random_rate = U_rand2 * total_rate;
  PopNetworkState result;
  for (const auto & node_tr_rate : popNodeTransitionRates) 
  {
    double rate = node_tr_rate.second;
    random_rate -= rate;
    result = node_tr_rate.first;
    
    if (random_rate <= 0)
      break;
  }

  return result;
}

double PopMaBEstEngine::computeTH(const MAP<NodeIndex, double> &nodeTransitionRates, double total_rate) const
{
  if (nodeTransitionRates.size() == 1)
  {
    return 0.;
  }

  MAP<NodeIndex, double>::const_iterator begin = nodeTransitionRates.begin();

  double TH = 0.;
  double rate_internal = 0.;

  while (begin != nodeTransitionRates.end())
  {
    NodeIndex index = (*begin).first;
    double rate = (*begin).second;
    if (pop_network->getNode(index)->isInternal())
    {
      rate_internal += rate;
    }
    ++begin;
  }

  double total_rate_non_internal = total_rate - rate_internal;

  begin = nodeTransitionRates.begin();

  while (begin != nodeTransitionRates.end())
  {
    NodeIndex index = (*begin).first;
    double rate = (*begin).second;
    if (!pop_network->getNode(index)->isInternal())
    {
      double proba = rate / total_rate_non_internal;
      TH -= log2(proba) * proba;
    }
    ++begin;
  }

  return TH;
}

struct ArgWrapper
{
  PopMaBEstEngine *mabest;
  unsigned int start_count_thread;
  unsigned int sample_count_thread;
  Cumulator<PopNetworkState> *cumulator;
  Cumulator<PopSize>* custom_cumulator;
  RandomGeneratorFactory *randgen_factory;
  long long int* elapsed_time;
  int seed;
  FixedPoints *fixpoint_map;
  std::ostream *output_traj;

  ArgWrapper(PopMaBEstEngine *mabest, unsigned int start_count_thread, unsigned int sample_count_thread, Cumulator<PopNetworkState> *cumulator, Cumulator<PopSize>* custom_cumulator, RandomGeneratorFactory *randgen_factory, long long int * elapsed_time, int seed, FixedPoints *fixpoint_map, std::ostream *output_traj) : mabest(mabest), start_count_thread(start_count_thread), sample_count_thread(sample_count_thread), cumulator(cumulator), custom_cumulator(custom_cumulator), randgen_factory(randgen_factory), elapsed_time(elapsed_time), seed(seed), fixpoint_map(fixpoint_map), output_traj(output_traj) {}
};

void *PopMaBEstEngine::threadWrapper(void *arg)
{
#ifdef USE_DYNAMIC_BITSET
  MBDynBitset::init_pthread();
#endif
  ArgWrapper *warg = (ArgWrapper *)arg;
  try
  {
    warg->mabest->runThread(warg->cumulator, warg->custom_cumulator, warg->start_count_thread, warg->sample_count_thread, warg->randgen_factory, warg->seed, warg->fixpoint_map, warg->output_traj);
  }
  catch (const BNException &e)
  {
    std::cerr << e;
  }
#ifdef USE_DYNAMIC_BITSET
  MBDynBitset::end_pthread();
#endif
  return NULL;
}

void PopMaBEstEngine::runThread(Cumulator<PopNetworkState> *cumulator, Cumulator<PopSize>* custom_cumulator, unsigned int start_count_thread, unsigned int sample_count_thread, RandomGeneratorFactory *randgen_factory, int seed, FixedPoints *fixpoint_map, std::ostream *output_traj)
{
  const std::vector<Node *> &nodes = pop_network->getNodes();
  PopNetworkState pop_network_state;

  RandomGenerator *random_generator = randgen_factory->generateRandomGenerator(seed);
  for (unsigned int nn = 0; nn < sample_count_thread; ++nn)
  {
    random_generator->setSeed(seed + start_count_thread + nn);
    cumulator->rewind();
    custom_cumulator->rewind();
    
    if (RunConfig::getVerbose() > 1) 
      std::cout << "> New simulation" << std::endl;
  
    pop_network->initPopStates(pop_network_state, random_generator, runconfig->getInitPop());
    
    if (RunConfig::getVerbose() > 1) {
      std::cout << ">> Initial state : ";
      pop_network_state.displayOneLine(std::cout, pop_network);
      std::cout << std::endl;
    }
      
    double tm = 0.;
    if (NULL != output_traj) {
      (*output_traj) << "\nTrajectory #" << (nn+1) << '\n';
      (*output_traj) << " istate\t";
      pop_network_state.displayOneLine(*output_traj, pop_network);
      (*output_traj) << '\n';
    }
    while (tm < max_time)
    {
      double total_rate = 0.;
      
      PopNetworkStateMap popNodeTransitionRates;
      // forall S ∈ Σ such that ψ(S) > 0 do
      for (auto pop : pop_network_state.getMap())
      {
        if (pop.second > 0)
        {

#ifdef USE_DYNAMIC_BITSET
          NetworkState t_network_state(pop.first, 1);
#else
          NetworkState t_network_state(pop.first);
#endif
          double total_pop_rate = 0.;
          // forall S' such that there is only one different node state compare to S do
          for (auto node : nodes)
          {

            double nodeTransitionRate = 0;
            if (node->getNodeState(t_network_state))
            {
              double r_down = node->getRateDown(t_network_state, pop_network_state);
              if (r_down != 0.0)
              {
                total_pop_rate += r_down;
                nodeTransitionRate = r_down;
              }
            }
            else
            {
              double r_up = node->getRateUp(t_network_state, pop_network_state);
              if (r_up != 0.0)
              {
                total_pop_rate += r_up;
                nodeTransitionRate = r_up;
              }
            }

            // if ρS→S' > 0 then
            if (nodeTransitionRate > 0.0)
            {
              // Construct ψ' from (ψ, S and S')
              // ψ'(S'') ≡ ψ(S'' ), ∀S'' != (S, S')
              PopNetworkState new_pop_network_state = PopNetworkState(pop_network_state);
              
              // ψ'(S) ≡ ψ(S) − 1
              new_pop_network_state.decr(t_network_state);
              
              // ψ'(S') ≡ ψ(S') + 1
#ifdef USE_DYNAMIC_BITSET
              NetworkState new_network_state(t_network_state, 1);
#else
              NetworkState new_network_state = t_network_state;
#endif
              new_network_state.flipState(node);
              new_pop_network_state.incr(new_network_state); 
              
              // Compute the transition rate ρψ→ψ' using
              // ρψ→ψ' ≡ ψ(S)ρS→S'
              nodeTransitionRate *= pop.second;

              total_rate += nodeTransitionRate;
              // Put (ψ' , ρψ→ψ' ) in Ω
              popNodeTransitionRates.insert(std::pair<PopNetworkState, double>(new_pop_network_state, nodeTransitionRate));
            }
          }
          
          // forall Possible division do
          for (auto division_rule: pop_network->getDivisionRules()) {
            
            // Compute the transition rate of cell division from MaBoSS language, ρS→division
            double division_rate = division_rule->getRate(pop.first, pop_network_state);
            
            // if ρS→division > 0 then
            if (division_rate > 0){
              
              
              // Construct the two daughter cell states S' and S'' using MaBoSS language
              // Construct the new state ψ 0 
              PopNetworkState new_pop_network_state = PopNetworkState(pop_network_state);
              new_pop_network_state.decr(t_network_state);
              
              NetworkState state_daughter1 = division_rule->applyRules(DivisionRule::DAUGHTER_1, pop.first, pop_network_state);
              new_pop_network_state.incr(state_daughter1);
              
              NetworkState state_daughter2 = division_rule->applyRules(DivisionRule::DAUGHTER_2, pop.first, pop_network_state);
              new_pop_network_state.incr(state_daughter2);
              
              // Compute the transition rate ρψ→ψ' = ψ(S)ρS→division
              division_rate *= pop.second;
              
              // Put (ψ' , ρψ→ψ' ) in Ω
              popNodeTransitionRates.insert(std::pair<PopNetworkState, double>(new_pop_network_state, division_rate));
              total_rate += division_rate;
              
            }
          }
          double rate_death = pop_network->getDeathRate(pop.first, pop_network_state);
          if (rate_death > 0){
            rate_death *= pop.second;
            total_rate += rate_death;
            
            PopNetworkState new_pop_network_state = PopNetworkState(pop_network_state);
            new_pop_network_state.decr(t_network_state);
            popNodeTransitionRates.insert(std::pair<PopNetworkState, double>(new_pop_network_state, rate_death));
          }
          
          if (total_pop_rate == 0)
          {
            FixedPoints::iterator iter = fixpoint_map->find(t_network_state.getState());
            if (iter == fixpoint_map->end())
            {
              (*fixpoint_map)[t_network_state.getState()] = 1;
            }
            else
            {
              iter->second++;
            }
          }
        }
      }
      
      if (RunConfig::getVerbose() > 2) {
        for (const auto &transition : popNodeTransitionRates)
        {
          std::cout << ">>> Transition : ";
          PopNetworkState t_state(transition.first);
          t_state.displayOneLine(std::cout, pop_network);
          std::cout << ", proba=" << (int)(100*transition.second/total_rate) << std::endl;
        }
      }
      double TH;
      if (total_rate == 0)
      {
        tm = max_time;
        TH = 0.;
      }
      else
      {
        double transition_time;
        if (discrete_time)
        {
          transition_time = time_tick;
        }
        else
        {
          double U_rand1 = random_generator->generate();
          transition_time = -log(U_rand1) / total_rate;
        }

        tm += transition_time;
        // Commenting for now
        TH = 0.;
        // TH = computeTH(nodeTransitionRates, total_rate);
      }

      if (NULL != output_traj) {
        (*output_traj) << std::setprecision(10) << tm << '\t';
        pop_network_state.displayOneLine(*output_traj, pop_network);
        (*output_traj) << '\t' << TH << '\n';
      }

      if (runconfig->hasCustomPopOutput())
      {
        NetworkState s;
#ifdef USE_DYNAMIC_BITSET
        PopNetworkState t_popstate(pop_network_state & cumulator->getOutputMask().getMap().begin()->first, 1);
#else
        PopNetworkState t_popstate(pop_network_state & cumulator->getOutputMask().getMap().begin()->first);
#endif
        double eval = runconfig->getCustomPopOutputExpression()->eval(NULL, s, t_popstate);
        if (eval >= 0){
          PopSize pop_size((unsigned int) eval);
          custom_cumulator->cumul(pop_size, tm, TH); 
        } else {
          cumulator->cumulEmpty(tm);
        }
      } else {
        cumulator->cumul(pop_network_state, tm, TH);
      }
      
      if (tm >= max_time)
      {
        break;
      }

      pop_network_state = getTargetNode(random_generator, popNodeTransitionRates, total_rate);

      if (RunConfig::getVerbose() > 0) 
        std::cout << "> time = " << tm << std::endl;
        
      if (RunConfig::getVerbose() > 1) {
        std::cout << ">> Present state : ";
        pop_network_state.displayOneLine(std::cout, pop_network);
        std::cout << std::endl;
      }
      
    }
    if (RunConfig::getVerbose() > 0)
      std::cout << std::endl;
      
    if (runconfig->hasCustomPopOutput())
      custom_cumulator->trajectoryEpilogue();
    else
      cumulator->trajectoryEpilogue();

  }
  delete random_generator;
}

void PopMaBEstEngine::run(std::ostream *output_traj)
{
#ifdef STD_THREAD
  std::vector<std::thread*> tid(thread_count);
#else
  pthread_t *tid = new pthread_t[thread_count];
#endif
  RandomGeneratorFactory *randgen_factory = runconfig->getRandomGeneratorFactory();
  int seed = runconfig->getSeedPseudoRandom();
  
#ifdef MPI_COMPAT
  unsigned int start_sample_count = sample_count * world_rank;
#else
  unsigned int start_sample_count = 0;
#endif
  
#ifdef MPI_COMPAT
  thread_elapsed_runtimes[world_rank].resize(thread_count);
#else
  thread_elapsed_runtimes.resize(thread_count);
#endif
  
  Probe probe;
  for (unsigned int nn = 0; nn < thread_count; ++nn)
  {
    FixedPoints *fixpoint_map = new FixedPoints();
    fixpoint_map_v.push_back(fixpoint_map);
    
#ifdef MPI_COMPAT
    ArgWrapper* warg = new ArgWrapper(this, start_sample_count, cumulator_v[nn]->getSampleCount(), cumulator_v[nn], custom_pop_cumulator_v[nn], randgen_factory, &(thread_elapsed_runtimes[world_rank][nn]), seed, fixpoint_map, output_traj);
#else
    ArgWrapper* warg = new ArgWrapper(this, start_sample_count, cumulator_v[nn]->getSampleCount(), cumulator_v[nn], custom_pop_cumulator_v[nn], randgen_factory, &(thread_elapsed_runtimes[nn]), seed, fixpoint_map, output_traj);
#endif
#ifdef STD_THREAD
    tid[nn] = new std::thread(PopMaBEstEngine::threadWrapper, warg);
#else
    pthread_create(&tid[nn], NULL, PopMaBEstEngine::threadWrapper, warg);
#endif
    arg_wrapper_v.push_back(warg);

    start_sample_count += cumulator_v[nn]->getSampleCount();
  }
  for (unsigned int nn = 0; nn < thread_count; ++nn)
  {
#ifdef STD_THREAD
    tid[nn]->join();
#else
    pthread_join(tid[nn], NULL);
#endif
  }
  probe.stop();
  elapsed_core_runtime = probe.elapsed_msecs();
  user_core_runtime = probe.user_msecs();
  probe.start();
  epilogue();
  probe.stop();
  elapsed_epilogue_runtime = probe.elapsed_msecs();
  user_epilogue_runtime = probe.user_msecs();

#ifdef STD_THREAD
  for (std::thread* t:tid)
    delete t;
  tid.clear();
#else
  delete[] tid;
#endif
  
#ifdef MPI_COMPAT
  
  if (world_rank == 0)
  {
    elapsed_core_runtimes.resize(world_size);
    user_core_runtimes.resize(world_size);
    elapsed_epilogue_runtimes.resize(world_size);
    user_epilogue_runtimes.resize(world_size);
  
  }
  
  MPI_Gather(&elapsed_core_runtime, 1, MPI_LONG_LONG_INT, elapsed_core_runtimes.data(), 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
  MPI_Gather(&user_core_runtime, 1, MPI_LONG_LONG_INT, user_core_runtimes.data(), 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);  
  MPI_Gather(&elapsed_epilogue_runtime, 1, MPI_LONG_LONG_INT, elapsed_epilogue_runtimes.data(), 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
  MPI_Gather(&user_epilogue_runtime, 1, MPI_LONG_LONG_INT, user_epilogue_runtimes.data(), 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
  
#endif
}

void PopMaBEstEngine::mergePairOfFixpoints(FixedPoints* fixpoints_1, FixedPoints* fixpoints_2)
{
  for (const auto& fixpoint: *fixpoints_2) {
    
    FixedPoints::iterator iter = fixpoints_1->find(fixpoint.first);
    if (iter == fixpoints_1->end()) {
      (*fixpoints_1)[fixpoint.first] = fixpoint.second;
    
    } else {
      iter->second += fixpoint.second;
    
    }
  }
  delete fixpoints_2; 
}

struct PopMergeWrapper {
  Cumulator<PopNetworkState>* cumulator_1;
  Cumulator<PopNetworkState>* cumulator_2;
  Cumulator<PopSize>* custom_cumulator_1;
  Cumulator<PopSize>* custom_cumulator_2;
  FixedPoints* fixpoints_1;
  FixedPoints* fixpoints_2;
  
  PopMergeWrapper(Cumulator<PopNetworkState>* cumulator_1, Cumulator<PopNetworkState>* cumulator_2, Cumulator<PopSize>* custom_cumulator_1, Cumulator<PopSize>* custom_cumulator_2, FixedPoints* fixpoints_1, FixedPoints* fixpoints_2) :
    cumulator_1(cumulator_1), cumulator_2(cumulator_2), custom_cumulator_1(custom_cumulator_1), custom_cumulator_2(custom_cumulator_2), fixpoints_1(fixpoints_1), fixpoints_2(fixpoints_2) { }
};

void* PopMaBEstEngine::threadMergeWrapper(void *arg)
{
#ifdef USE_DYNAMIC_BITSET
  MBDynBitset::init_pthread();
#endif
  PopMergeWrapper* warg = (PopMergeWrapper*)arg;
  try {
    Cumulator<PopSize>::mergePairOfCumulators(warg->custom_cumulator_1, warg->custom_cumulator_2);
    Cumulator<PopNetworkState>::mergePairOfCumulators(warg->cumulator_1, warg->cumulator_2);
    PopMaBEstEngine::mergePairOfFixpoints(warg->fixpoints_1, warg->fixpoints_2);
  } catch(const BNException& e) {
    std::cerr << e;
  }
#ifdef USE_DYNAMIC_BITSET
  MBDynBitset::end_pthread();
#endif
  return NULL;
}


void PopMaBEstEngine::mergeResults()
{
  size_t size = cumulator_v.size();
  
  if (size > 1) {
    
    
    unsigned int lvl=1;
    unsigned int max_lvl = (unsigned int) ceil(log2(size));

    while(lvl <= max_lvl) {      
    
      unsigned int step_lvl = (unsigned int) pow(2, lvl-1);
      unsigned int width_lvl = (unsigned int) floor(size/(step_lvl*2)) + 1;
#ifdef STD_THREAD
      std::vector<std::thread*> tid(width_lvl);
#else
      pthread_t* tid = new pthread_t[width_lvl];
#endif
      unsigned int nb_threads = 0;
      std::vector<PopMergeWrapper*> wargs;
      for(unsigned int i=0; i < size; i+=(step_lvl*2)) {
        
        if (i+step_lvl < size) {
          PopMergeWrapper* warg = new PopMergeWrapper(cumulator_v[i], cumulator_v[i+step_lvl], custom_pop_cumulator_v[i], custom_pop_cumulator_v[i+step_lvl], fixpoint_map_v[i], fixpoint_map_v[i+step_lvl]);
#ifdef STD_THREAD
          tid[nb_threads] = new std::thread(PopMaBEstEngine::threadMergeWrapper, warg);
#else
          pthread_create(&tid[nb_threads], NULL, PopMaBEstEngine::threadMergeWrapper, warg);
#endif
          nb_threads++;
          wargs.push_back(warg);
        } 
      }
      
      for(unsigned int i=0; i < nb_threads; i++) {   
#ifdef STD_THREAD
          tid[i]->join();
#else
          pthread_join(tid[i], NULL);
#endif
      }
      
      for (auto warg: wargs) {
        delete warg;
      }
#ifdef STD_THREAD
      for (std::thread * t: tid)
        delete t;
      tid.clear();
#else
      delete [] tid;
#endif
      lvl++;
    }   
  }
}


#ifdef MPI_COMPAT


void PopMaBEstEngine::mergeMPIResults(RunConfig* runconfig, Cumulator<PopNetworkState>* ret_cumul, FixedPoints* fixpoints, int world_size, int world_rank, bool pack)
{  
  if (world_size> 1) {
    
    int lvl=1;
    int max_lvl = ceil(log2(world_size));

    while(lvl <= max_lvl) {
    
      int step_lvl = pow(2, lvl-1);
      MPI_Barrier(MPI_COMM_WORLD);
      for(int i=0; i < world_size; i+=(step_lvl*2)) {
        
        if (i+step_lvl < world_size) {
          if (world_rank == i || world_rank == (i+step_lvl)){
            Cumulator<PopNetworkState>::mergePairOfMPICumulators(ret_cumul, world_rank, i, i+step_lvl, runconfig, pack);
            mergePairOfMPIFixpoints(fixpoints, world_rank, i, i+step_lvl, pack);
          }
        } 
      }
      
      lvl++;
    }
  }
}

void PopMaBEstEngine::MPI_Unpack_Fixpoints(FixedPoints* fp_map, char* buff, unsigned int buff_size)
{
        
  int position = 0;
  unsigned int nb_fixpoints;
  MPI_Unpack(buff, buff_size, &position, &nb_fixpoints, 1, MPI_UNSIGNED, MPI_COMM_WORLD);
  
  if (nb_fixpoints > 0) {
    if (fp_map == NULL) {
      fp_map = new FixedPoints();
    }
    for (unsigned int j=0; j < nb_fixpoints; j++) {
      NetworkState state;
      state.my_MPI_Unpack(buff, buff_size, &position);
      unsigned int count = 0;
      MPI_Unpack(buff, buff_size, &position, &count, 1, MPI_UNSIGNED, MPI_COMM_WORLD);
      
      if (fp_map->find(state.getState()) == fp_map->end()) {
        (*fp_map)[state.getState()] = count;
      } else {
        (*fp_map)[state.getState()] += count;
      }
    }
  }
}

char* PopMaBEstEngine::MPI_Pack_Fixpoints(const FixedPoints* fp_map, int dest, unsigned int * buff_size)
{
  unsigned int nb_fixpoints = fp_map == NULL ? 0 : fp_map->size();
  *buff_size = sizeof(unsigned int);
  for (auto& fixpoint: *fp_map) {
    NetworkState state(fixpoint.first);
    *buff_size += state.my_MPI_Pack_Size() + sizeof(unsigned int);
  }
  char* buff = new char[*buff_size];
  int position = 0;
  
  MPI_Pack(&nb_fixpoints, 1, MPI_UNSIGNED, buff, *buff_size, &position, MPI_COMM_WORLD);

  if (nb_fixpoints > 0) {
    for (auto& fixpoint: *fp_map) {  
      NetworkState state(fixpoint.first);
      unsigned int count = fixpoint.second;
      state.my_MPI_Pack(buff, *buff_size, &position);
      MPI_Pack(&count, 1, MPI_UNSIGNED, buff, *buff_size, &position, MPI_COMM_WORLD);
    }
  }
  return buff;
}

void PopMaBEstEngine::MPI_Send_Fixpoints(const FixedPoints* fp_map, int dest) 
{
  int nb_fixpoints = fp_map->size();
  MPI_Send(&nb_fixpoints, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
  
  for (auto& fixpoint: *fp_map) {
    NetworkState state(fixpoint.first);
    unsigned int count = fixpoint.second;
    
    state.my_MPI_Send(dest);
    MPI_Send(&count, 1, MPI_UNSIGNED, dest, 0, MPI_COMM_WORLD);
    
  } 
}

void PopMaBEstEngine::MPI_Recv_Fixpoints(FixedPoints* fp_map, int origin) 
{
  int nb_fixpoints = -1;
  MPI_Recv(&nb_fixpoints, 1, MPI_INT, origin, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  
  for (int j = 0; j < nb_fixpoints; j++) {
    NetworkState state;
    state.my_MPI_Recv(origin);
    
    unsigned int count = -1;
    MPI_Recv(&count, 1, MPI_UNSIGNED, origin, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    if (fp_map->find(state.getState()) == fp_map->end()) {
      (*fp_map)[state.getState()] = count;
    } else {
      (*fp_map)[state.getState()] += count;
    }
  }
}

void PopMaBEstEngine::mergePairOfMPIFixpoints(FixedPoints* fixpoints, int world_rank, int dest, int origin, bool pack) 
{
   if (world_rank == dest) 
   {
   
    if (pack) {
      unsigned int buff_size;
      MPI_Recv( &buff_size, 1, MPI_UNSIGNED, origin, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
      char* buff = new char[buff_size];
      MPI_Recv( buff, buff_size, MPI_PACKED, origin, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
          
      MPI_Unpack_Fixpoints(fixpoints, buff, buff_size);
      delete [] buff;
      
    } else {
      MPI_Recv_Fixpoints(fixpoints, origin);
    }
    
  } else if (world_rank == origin) {

    if (pack) {

      unsigned int buff_size;
      char* buff = MPI_Pack_Fixpoints(fixpoints, dest, &buff_size);

      MPI_Send(&buff_size, 1, MPI_UNSIGNED, dest, 0, MPI_COMM_WORLD);
      MPI_Send( buff, buff_size, MPI_PACKED, dest, 0, MPI_COMM_WORLD); 
      delete [] buff;            
      
    } else {
     
      MPI_Send_Fixpoints(fixpoints, dest);
    }
  }
}

#endif



void PopMaBEstEngine::epilogue()
{
  mergeResults();

  merged_cumulator = cumulator_v[0];
  custom_pop_cumulator = custom_pop_cumulator_v[0];
  fixpoints = fixpoint_map_v[0];
  
#ifdef MPI_COMPAT
  
  mergeMPIResults(runconfig, merged_cumulator, fixpoints, world_size, world_rank);
  
  if (world_rank == 0)
  {
#endif
  if (runconfig->hasCustomPopOutput())
    custom_pop_cumulator->epilogue(pop_network, reference_state);
  else
    merged_cumulator->epilogue(pop_network, reference_state);
  
#ifdef MPI_COMPAT
  }
#endif 
  
}

PopMaBEstEngine::~PopMaBEstEngine()
{
  delete fixpoint_map_v[0];
  
  for (auto t_arg_wrapper : arg_wrapper_v)
    delete t_arg_wrapper;

  delete merged_cumulator;
  delete custom_pop_cumulator;
}

void PopMaBEstEngine::displayFixpoints(FixedPointDisplayer *displayer) const
{
#ifdef MPI_COMPAT
if (getWorldRank() == 0) {
#endif
  displayer->begin(fixpoints->size());

  size_t nn = 0;
  for (const auto& fp : *fixpoints)
  {
    const NetworkState &network_state = fp.first;
    displayer->displayFixedPoint(nn + 1, network_state, fp.second, sample_count);
    nn++;
  }
  displayer->end();
#ifdef MPI_COMPAT
}
#endif
}

void PopMaBEstEngine::displayPopProbTraj(ProbTrajDisplayer<PopNetworkState> *displayer) const
{
#ifdef MPI_COMPAT
if (getWorldRank() == 0) {
#endif
  if (!runconfig->hasCustomPopOutput())
    merged_cumulator->displayProbTraj(refnode_count, displayer);
  
#ifdef MPI_COMPAT
}
#endif
}

void PopMaBEstEngine::display(ProbTrajDisplayer<PopNetworkState> *pop_probtraj_displayer, FixedPointDisplayer *fp_displayer) const
{
#ifdef MPI_COMPAT
if (getWorldRank() == 0) {
#endif
  displayPopProbTraj(pop_probtraj_displayer);
  displayFixpoints(fp_displayer);
#ifdef MPI_COMPAT
}
#endif
}


void PopMaBEstEngine::displayRunStats(std::ostream& os, time_t start_time, time_t end_time) const {
#ifdef MPI_COMPAT
if (getWorldRank() == 0) {
#endif
  const char sepfmt[] = "-----------------------------------------------%s-----------------------------------------------\n";
  char bufstr[1024];

  os << '\n';
  snprintf(bufstr, 1024, sepfmt, "--- Run ---");
  os << bufstr;

  os << "MaBoSS version: " << VERSION << " [networks up to " << MAXNODES << " nodes]\n";
  os << "\nRun start time: " << ctime(&start_time);
  os << "Run end time: " << ctime(&end_time);

  os << "\nCore user runtime: " << (getUserCoreRunTime()/1000.) << " secs using " << thread_count << " thread" << (thread_count > 1 ? "s" : "") << "\n";
  os << "Core elapsed runtime: " << (getElapsedCoreRunTime()/1000.) << " secs using " << thread_count << " thread" << (thread_count > 1 ? "s" : "") << "\n\n";

  os << "Epilogue user runtime: " << (getUserEpilogueRunTime()/1000.) << " secs using 1 thread\n";
  os << "Epilogue elapsed runtime: " << (getElapsedEpilogueRunTime()/1000.) << " secs using 1 thread\n\n";

  os << "StatDist user runtime: " << (getUserStatDistRunTime()/1000.) << " secs using 1 thread\n";
  os << "StatDist elapsed runtime: " << (getElapsedStatDistRunTime()/1000.) << " secs using 1 thread\n\n";
  
  // os << "Number of PopNetworkState_Impl created : " << PopNetworkState_Impl::generated_number_count << std::endl << std::endl;
  
  runconfig->display(pop_network, start_time, end_time, os);
#ifdef MPI_COMPAT
}
#endif
}

void PopMaBEstEngine::displayCustomPopProbTraj(ProbTrajDisplayer<PopSize>* displayer) const
{
  custom_pop_cumulator->displayProbTraj(refnode_count, displayer);
}
const std::map<unsigned int, std::pair<NetworkState, double> > PopMaBEstEngine::getFixPointsDists() const {
  
  std::map<unsigned int, std::pair<NetworkState, double> > res;
  if (0 == fixpoints->size()) {
    return res;
  }

  unsigned int nn = 0;
  for (const auto & fp : *fixpoints) {
    const NetworkState& network_state = fp.first;
    res[nn++] = std::make_pair(network_state,(double) fp.second / sample_count);
  }
  return res;
}
