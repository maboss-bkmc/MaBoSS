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
#include "Probe.h"
#include <stdlib.h>
#include <math.h>
#include <iomanip>
#include <iostream>
#include "Utils.h"
#ifndef WINDOWS
#include <dlfcn.h>
#else
#include <windows.h>
#endif

static const char *MABOSS_USER_FUNC_INIT = "maboss_user_func_init";

const std::string PopMaBEstEngine::VERSION = "0.0.1";
// extern size_t RandomGenerator::generated_number_count;

PopMaBEstEngine::PopMaBEstEngine(PopNetwork *pop_network, RunConfig *runconfig) : pop_network(pop_network), runconfig(runconfig),
                                                                           time_tick(runconfig->getTimeTick()),
                                                                           max_time(runconfig->getMaxTime()),
                                                                           sample_count(runconfig->getSampleCount()),
                                                                           discrete_time(runconfig->isDiscreteTime()),
                                                                           thread_count(runconfig->getThreadCount())
{

  if (thread_count > sample_count)
  {
    thread_count = sample_count;
  }

  if (thread_count > 1 && !runconfig->getRandomGeneratorFactory()->isThreadSafe())
  {
    std::cerr << "Warning: non reentrant random, may not work properly in multi-threaded mode\n";
  }

  const std::vector<Node *> &nodes = pop_network->getNodes();
  std::vector<Node *>::const_iterator begin = nodes.begin();
  std::vector<Node *>::const_iterator end = nodes.end();

  NetworkState internal_state;
  bool has_internal = false;
  refnode_count = 0;
  while (begin != end)
  {
    Node *node = *begin;
    if (node->isInternal())
    {
      has_internal = true;
      internal_state.setNodeState(node, true);
    }
    // if (node->isReference()) {
    //   reference_state.setNodeState(node, node->getReferenceState());
    //   refnode_count++;
    // }
    ++begin;
  }

  merged_cumulator = NULL;
  cumulator_v.resize(thread_count);
  unsigned int count = sample_count / thread_count;
  unsigned int firstcount = count + sample_count - count * thread_count;
  for (unsigned int nn = 0; nn < thread_count; ++nn)
  {
    PopCumulator *cumulator = new PopCumulator(runconfig, runconfig->getTimeTick(), runconfig->getMaxTime(), (nn == 0 ? firstcount : count));
    if (has_internal)
    {
#ifdef USE_STATIC_BITSET
      NetworkState_Impl state2 = ~internal_state.getState();
      cumulator->setOutputMask(state2);
#else
      cumulator->setOutputMask(~internal_state.getState());
#endif
    }
    cumulator_v[nn] = cumulator;
  }
}

PopNetworkState_Impl PopMaBEstEngine::getTargetNode(RandomGenerator *random_generator, const STATE_MAP<PopNetworkState_Impl, double, PopNetworkState_ImplHash, PopNetworkState_ImplEquality> popNodeTransitionRates, double total_rate) const
{
  double U_rand2 = random_generator->generate();
  double random_rate = U_rand2 * total_rate;
  STATE_MAP<PopNetworkState_Impl, double, PopNetworkState_ImplHash, PopNetworkState_ImplEquality>::const_iterator begin = popNodeTransitionRates.begin();
  STATE_MAP<PopNetworkState_Impl, double, PopNetworkState_ImplHash, PopNetworkState_ImplEquality>::const_iterator end = popNodeTransitionRates.end();

  PopNetworkState_Impl result = PopNetworkState_Impl();
  while (begin != end && random_rate > 0.)
  {
    double rate = begin->second;
    random_rate -= rate;
    result = begin->first;

    ++begin;
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
  MAP<NodeIndex, double>::const_iterator end = nodeTransitionRates.end();
  double TH = 0.;
  double rate_internal = 0.;

  while (begin != end)
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

  while (begin != end)
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
  PopCumulator *cumulator;
  RandomGeneratorFactory *randgen_factory;
  int seed;
  STATE_MAP<NetworkState_Impl, unsigned int> *fixpoint_map;
  std::ostream *output_traj;

  ArgWrapper(PopMaBEstEngine *mabest, unsigned int start_count_thread, unsigned int sample_count_thread, PopCumulator *cumulator, RandomGeneratorFactory *randgen_factory, int seed, STATE_MAP<NetworkState_Impl, unsigned int> *fixpoint_map, std::ostream *output_traj) : mabest(mabest), start_count_thread(start_count_thread), sample_count_thread(sample_count_thread), cumulator(cumulator), randgen_factory(randgen_factory), seed(seed), fixpoint_map(fixpoint_map), output_traj(output_traj) {}
};

void *PopMaBEstEngine::threadWrapper(void *arg)
{
#ifdef USE_DYNAMIC_BITSET
  MBDynBitset::init_pthread();
#endif
  ArgWrapper *warg = (ArgWrapper *)arg;
  try
  {
    warg->mabest->runThread(warg->cumulator, warg->start_count_thread, warg->sample_count_thread, warg->randgen_factory, warg->seed, warg->fixpoint_map, warg->output_traj);
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

void PopMaBEstEngine::runThread(PopCumulator *cumulator, unsigned int start_count_thread, unsigned int sample_count_thread, RandomGeneratorFactory *randgen_factory, int seed, STATE_MAP<NetworkState_Impl, unsigned int> *fixpoint_map, std::ostream *output_traj)
{
  const std::vector<Node *> &nodes = pop_network->getNodes();
  // std::vector<Node*>::const_iterator begin = nodes.begin();
  // std::vector<Node*>::const_iterator end = nodes.end();
  unsigned int stable_cnt = 0;
  // NetworkState network_state;
  PopNetworkState pop_network_state;

  RandomGenerator *random_generator = randgen_factory->generateRandomGenerator(seed);
  for (unsigned int nn = 0; nn < sample_count_thread; ++nn)
  {
    random_generator->setSeed(seed + start_count_thread + nn);
    cumulator->rewind();
    std::cout << "> New simulation" << std::endl;
    // network->initStates(network_state, random_generator);
    // std::cout << "Initializing pop state" << std::endl;
    pop_network->initPopStates(pop_network_state, random_generator, runconfig->getInitPop());
    // std::cout << "Done" << std::endl;
    double tm = 0.;
    unsigned int step = 0;
    // if (NULL != output_traj) {
    //   (*output_traj) << "\nTrajectory #" << (nn+1) << '\n';
    //   (*output_traj) << " istate\t";
    //   network_state.displayOneLine(*output_traj, network);
    //   (*output_traj) << '\n';
    // }
    while (tm < max_time)
    {
      double total_rate = 0.;

      std::cout << ">> Present state : ";
      pop_network_state.getState().display(pop_network, std::cout);
      std::cout << std::endl;

      STATE_MAP<PopNetworkState_Impl, double, PopNetworkState_ImplHash, PopNetworkState_ImplEquality> popNodeTransitionRates;
      // forall S ∈ Σ such that ψ(S) > 0 do
      for (auto pop : pop_network_state.getState())
      {
        if (pop.second > 0)
        {

          NetworkState t_network_state(pop.first);

          double total_pop_rate = 0.;
          // forall S' such that there is only one different node state compare to S do
          for (auto node : nodes)
          {

            double nodeTransitionRate = 0;
            NodeIndex node_idx = node->getIndex();
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

              PopNetworkState new_pop_network_state = PopNetworkState(pop_network_state);
              // Construct ψ' from (ψ, S and S')
              // std::cout << "Just copied from Phi";
              // new_pop_network_state.getState().display(network, std::cout);
              // std::cout << std::endl;
              
              // std::cout << "S state : " << t_network_state.getName(network) << std::endl;
              // ψ'(S) ≡ ψ(S) − 1
              // new_pop_network_state.getState()[t_network_state.getState()]--;
              new_pop_network_state.decr(t_network_state.getState());
              // std::cout << "After decr";
              // new_pop_network_state.getState().display(network, std::cout);
              // std::cout << std::endl;
              
              // std::cout << "New pop state : ";
              // new_pop_network_state.getState().display(network, std::cout);

              // ψ'(S') ≡ ψ(S') + 1
              NetworkState new_network_state = t_network_state;
              new_network_state.flipState(node);
              // std::cout << "S' state : " << new_network_state.getName(network) << std::endl;

              new_pop_network_state.incr(new_network_state.getState());
              
              // ψ'(S'') ≡ ψ(S'' ), ∀S'' != (S, S')
              // Done by initial copy

              // Compute the transition rate ρψ→ψ' using
              // ρψ→ψ' ≡ ψ(S)ρS→S'
              nodeTransitionRate *= pop_network_state.getState()[t_network_state.getState()];
              // nodeTransitionRate *= pop.second;

              total_rate += nodeTransitionRate;
              // Put (ψ' , ρψ→ψ' ) in Ω
              popNodeTransitionRates.insert(std::pair<PopNetworkState_Impl, double>(new_pop_network_state.getState(), nodeTransitionRate));
            }
          }
          
          
          // Here only looking at the first defined division... for now
          std::vector<double> rate_divisions = pop_network->getDivisionRates(pop.first, pop_network_state);
          for (auto rate_division: rate_divisions) {
            if (rate_division > 0){
              rate_division *= pop.second;
              total_rate += rate_division;

              PopNetworkState new_pop_network_state = PopNetworkState(pop_network_state);
              new_pop_network_state.incr(t_network_state.getState());
              popNodeTransitionRates.insert(std::pair<PopNetworkState_Impl, double>(new_pop_network_state.getState(), rate_division));
            }
          }
          double rate_death = pop_network->getDeathRate(pop.first, pop_network_state);
          if (rate_death > 0){
            rate_death *= pop.second;
            total_rate += rate_death;
            
            PopNetworkState new_pop_network_state = PopNetworkState(pop_network_state);
            new_pop_network_state.decr(t_network_state.getState());
            popNodeTransitionRates.insert(std::pair<PopNetworkState_Impl, double>(new_pop_network_state.getState(), rate_death));
          }
          
          if (total_pop_rate == 0)
          {
            if (fixpoint_map->find(t_network_state.getState()) == fixpoint_map->end())
            {
              (*fixpoint_map)[t_network_state.getState()] = 1;
            }
            else
            {
              (*fixpoint_map)[t_network_state.getState()]++;
            }
          }
        }
      }

      // std::cout << "Total rate : " << total_rate << std::endl;
      for (const auto &transition : popNodeTransitionRates)
      {
        std::cout << " >>> Transition : ";
        // PopNetworkState_Impl t_state = transition.first;
        transition.first.display(pop_network, std::cout);
        std::cout << ", proba=" << (int)(100*transition.second/total_rate) << std::endl;
      }
      double TH;
      if (total_rate == 0)
      {
        tm = max_time;
        TH = 0.;
        // if (fixpoint_map->find(network_state.getState()) == fixpoint_map->end()) {
        //   (*fixpoint_map)[network_state.getState()] = 1;
        // } else {
        //   (*fixpoint_map)[network_state.getState()]++;
        // }
        stable_cnt++;
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

      // if (NULL != output_traj) {
      //   (*output_traj) << std::setprecision(10) << tm << '\t';
      //   network_state.displayOneLine(*output_traj, network);
      //   (*output_traj) << '\t' << TH << '\n';
      // }

      cumulator->cumul(pop_network_state, tm, TH);

      if (tm >= max_time)
      {
        break;
      }

      pop_network_state = getTargetNode(random_generator, popNodeTransitionRates, total_rate);
      // network_state.flipState(network->getNode(node_idx));

      step++;
    }
    cumulator->trajectoryEpilogue();
  }
  delete random_generator;
}

void PopMaBEstEngine::run(std::ostream *output_traj)
{
  pthread_t *tid = new pthread_t[thread_count];
  RandomGeneratorFactory *randgen_factory = runconfig->getRandomGeneratorFactory();
  int seed = runconfig->getSeedPseudoRandom();
  unsigned int start_sample_count = 0;
  Probe probe;
  for (unsigned int nn = 0; nn < thread_count; ++nn)
  {
    STATE_MAP<NetworkState_Impl, unsigned int> *fixpoint_map = new STATE_MAP<NetworkState_Impl, unsigned int>();
    fixpoint_map_v.push_back(fixpoint_map);
    ArgWrapper *warg = new ArgWrapper(this, start_sample_count, cumulator_v[nn]->getSampleCount(), cumulator_v[nn], randgen_factory, seed, fixpoint_map, output_traj);
    pthread_create(&tid[nn], NULL, PopMaBEstEngine::threadWrapper, warg);
    arg_wrapper_v.push_back(warg);

    start_sample_count += cumulator_v[nn]->getSampleCount();
  }
  for (unsigned int nn = 0; nn < thread_count; ++nn)
  {
    pthread_join(tid[nn], NULL);
  }
  probe.stop();
  elapsed_core_runtime = probe.elapsed_msecs();
  user_core_runtime = probe.user_msecs();
  probe.start();
  epilogue();
  probe.stop();
  elapsed_epilogue_runtime = probe.elapsed_msecs();
  user_epilogue_runtime = probe.user_msecs();
  delete[] tid;
}

STATE_MAP<NetworkState_Impl, unsigned int> *PopMaBEstEngine::mergeFixpointMaps()
{
  if (1 == fixpoint_map_v.size())
  {
    return new STATE_MAP<NetworkState_Impl, unsigned int>(*fixpoint_map_v[0]);
  }

  STATE_MAP<NetworkState_Impl, unsigned int> *fixpoint_map = new STATE_MAP<NetworkState_Impl, unsigned int>();
  std::vector<STATE_MAP<NetworkState_Impl, unsigned int> *>::iterator begin = fixpoint_map_v.begin();
  std::vector<STATE_MAP<NetworkState_Impl, unsigned int> *>::iterator end = fixpoint_map_v.end();
  while (begin != end)
  {
    STATE_MAP<NetworkState_Impl, unsigned int> *fp_map = *begin;
    STATE_MAP<NetworkState_Impl, unsigned int>::const_iterator b = fp_map->begin();
    STATE_MAP<NetworkState_Impl, unsigned int>::const_iterator e = fp_map->end();
    while (b != e)
    {
      //NetworkState_Impl state = (*b).first;
      const NetworkState_Impl &state = b->first;
      if (fixpoint_map->find(state) == fixpoint_map->end())
      {
        (*fixpoint_map)[state] = (*b).second;
      }
      else
      {
        (*fixpoint_map)[state] += (*b).second;
      }
      ++b;
    }
    ++begin;
  }
  return fixpoint_map;
}

void PopMaBEstEngine::epilogue()
{
  merged_cumulator = PopCumulator::mergePopCumulators(runconfig, cumulator_v);
  merged_cumulator->epilogue(pop_network, reference_state);

  for (auto t_cumulator : cumulator_v)
    delete t_cumulator;

  STATE_MAP<NetworkState_Impl, unsigned int> *merged_fixpoint_map = mergeFixpointMaps();

  STATE_MAP<NetworkState_Impl, unsigned int>::const_iterator b = merged_fixpoint_map->begin();
  STATE_MAP<NetworkState_Impl, unsigned int>::const_iterator e = merged_fixpoint_map->end();

  while (b != e)
  {
    fixpoints[NetworkState((*b).first).getState()] = (*b).second;
    ++b;
  }
  delete merged_fixpoint_map;
}

PopMaBEstEngine::~PopMaBEstEngine()
{
  for (auto t_fixpoint_map : fixpoint_map_v)
    delete t_fixpoint_map;

  for (auto t_arg_wrapper : arg_wrapper_v)
    delete t_arg_wrapper;

  delete merged_cumulator;
}

void PopMaBEstEngine::init()
{
  extern void builtin_functions_init();
  builtin_functions_init();
}

void PopMaBEstEngine::loadUserFuncs(const char *module)
{
  init();

#ifndef WINDOWS
  void *dl = dlopen(module, RTLD_LAZY);
#else
  void *dl = LoadLibrary(module);
#endif

  if (NULL == dl)
  {
#ifndef WINDOWS
    std::cerr << dlerror() << std::endl;
#else
    std::cerr << GetLastError() << std::endl;
#endif
    exit(1);
  }

#ifndef WINDOWS
  void *sym = dlsym(dl, MABOSS_USER_FUNC_INIT);
#else
  typedef void(__cdecl * MYPROC)(std::map<std::string, Function *> *);
  MYPROC sym = (MYPROC)GetProcAddress((HINSTANCE)dl, MABOSS_USER_FUNC_INIT);
#endif

  if (sym == NULL)
  {
    std::cerr << "symbol " << MABOSS_USER_FUNC_INIT << "() not found in user func module: " << module << "\n";
    exit(1);
  }
  typedef void (*init_t)(std::map<std::string, Function *> *);
  init_t init_fun = (init_t)sym;
  init_fun(Function::getFuncMap());
}

void PopMaBEstEngine::displayFixpoints(FixedPointDisplayer *displayer) const
{
  displayer->begin(fixpoints.size());
  /*
  output_fp << "Fixed Points (" << fixpoints.size() << ")\n";
  if (0 == fixpoints.size()) {
    return;
  }
  */

  STATE_MAP<NetworkState_Impl, unsigned int>::const_iterator begin = fixpoints.begin();
  STATE_MAP<NetworkState_Impl, unsigned int>::const_iterator end = fixpoints.end();

  //output_fp << "FP\tProba\tState\t";
  //network->displayHeader(output_fp);
  for (unsigned int nn = 0; begin != end; ++nn)
  {
    const NetworkState &network_state = begin->first;
    displayer->displayFixedPoint(nn + 1, network_state, begin->second, sample_count);
    /*
    output_fp << "#" << (nn+1) << "\t";
    if (hexfloat) {
      output_fp << fmthexdouble((double)begin->second / sample_count) <<  "\t";
    } else {
      output_fp << ((double)begin->second / sample_count) <<  "\t";
    }
    network_state.displayOneLine(output_fp, network);
    output_fp << '\t';
    network_state.display(output_fp, network);
    */
    ++begin;
  }
  displayer->end();
}

void PopMaBEstEngine::displayPopProbTraj(PopProbTrajDisplayer *displayer) const
{
  merged_cumulator->displayPopProbTraj(pop_network, refnode_count, displayer);
}

void PopMaBEstEngine::display(PopProbTrajDisplayer *pop_probtraj_displayer, FixedPointDisplayer *fp_displayer) const
{
  displayPopProbTraj(pop_probtraj_displayer);
  displayFixpoints(fp_displayer);
}