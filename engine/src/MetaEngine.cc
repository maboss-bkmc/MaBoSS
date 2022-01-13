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
     MetaEngine.cc

   Authors:
     Eric Viara <viara@sysra.com>
     Gautier Stoll <gautier.stoll@curie.fr>
     Vincent Noël <vincent.noel@curie.fr>
 
   Date:
     January-March 2011
*/

#include "MetaEngine.h"
#include "Probe.h"
#include "Utils.h"
#ifndef WINDOWS
  #include <dlfcn.h>
#else
  #include <windows.h>
#endif

static const char* MABOSS_USER_FUNC_INIT = "maboss_user_func_init";

void MetaEngine::init()
{
  extern void builtin_functions_init();
  builtin_functions_init();
}

void MetaEngine::loadUserFuncs(const char* module)
{
  init();

#ifndef WINDOWS
  void* dl = dlopen(module, RTLD_LAZY);
#else
  void* dl = LoadLibrary(module);
#endif

  if (NULL == dl) {
#ifndef WINDOWS    
    std::cerr << dlerror() << std::endl;
#else
    std::cerr << GetLastError() << std::endl;
#endif
    exit(1);
  }

#ifndef WINDOWS
  void* sym = dlsym(dl, MABOSS_USER_FUNC_INIT);
#else
  typedef void (__cdecl *MYPROC)(std::map<std::string, Function*>*);
  MYPROC sym = (MYPROC) GetProcAddress((HINSTANCE) dl, MABOSS_USER_FUNC_INIT);
#endif

  if (sym == NULL) {
    std::cerr << "symbol " << MABOSS_USER_FUNC_INIT << "() not found in user func module: " << module << "\n";
    exit(1);
  }
  typedef void (*init_t)(std::map<std::string, Function*>*);
  init_t init_fun = (init_t)sym;
  init_fun(Function::getFuncMap());
}

const std::map<unsigned int, std::pair<NetworkState, double> > MetaEngine::getFixPointsDists() const {
  
  std::map<unsigned int, std::pair<NetworkState, double> > res;
  if (0 == fixpoints.size()) {
    return res;
  }

  STATE_MAP<NetworkState_Impl, unsigned int>::const_iterator begin = fixpoints.begin();
  STATE_MAP<NetworkState_Impl, unsigned int>::const_iterator end = fixpoints.end();
  
  for (unsigned int nn = 0; begin != end; ++nn) {
    const NetworkState& network_state = (*begin).first;
    res[nn] = std::make_pair(network_state,(double) (*begin).second / sample_count);
    ++begin;
  }
  return res;
}


NodeIndex MetaEngine::getTargetNode(Network* _network, RandomGenerator* random_generator, const MAP<NodeIndex, double>& nodeTransitionRates, double total_rate) const
{
  double U_rand2 = random_generator->generate();
  double random_rate = U_rand2 * total_rate;
  MAP<NodeIndex, double>::const_iterator begin = nodeTransitionRates.begin();
  MAP<NodeIndex, double>::const_iterator end = nodeTransitionRates.end();
  NodeIndex node_idx = INVALID_NODE_INDEX;
  while (begin != end && random_rate >= 0.) {
    node_idx = (*begin).first;
    double rate = (*begin).second;
    random_rate -= rate;
    ++begin;
  }

  assert(node_idx != INVALID_NODE_INDEX);
  assert(_network->getNode(node_idx)->getIndex() == node_idx);
  return node_idx;
}

double MetaEngine::computeTH(Network* _network, const MAP<NodeIndex, double>& nodeTransitionRates, double total_rate) const
{
  if (nodeTransitionRates.size() == 1) {
    return 0.;
  }

  MAP<NodeIndex, double>::const_iterator begin = nodeTransitionRates.begin();
  MAP<NodeIndex, double>::const_iterator end = nodeTransitionRates.end();
  double TH = 0.;
  double rate_internal = 0.;

  while (begin != end) {
    NodeIndex index = (*begin).first;
    double rate = (*begin).second;
    if (_network->getNode(index)->isInternal()) {
      rate_internal += rate;
    }
    ++begin;
  }

  double total_rate_non_internal = total_rate - rate_internal;

  begin = nodeTransitionRates.begin();

  while (begin != end) {
    NodeIndex index = (*begin).first;
    double rate = (*begin).second;
    if (!_network->getNode(index)->isInternal()) {
      double proba = rate / total_rate_non_internal;
      TH -= log2(proba) * proba;
    }
    ++begin;
  }

  return TH;
}

const std::map<double, STATE_MAP<NetworkState_Impl, double> > MetaEngine::getStateDists() const {
  return merged_cumulator->getStateDists();
}

const STATE_MAP<NetworkState_Impl, double> MetaEngine::getNthStateDist(int nn) const {
  return merged_cumulator->getNthStateDist(nn);
}

const STATE_MAP<NetworkState_Impl, double> MetaEngine::getAsymptoticStateDist() const {
  return merged_cumulator->getAsymptoticStateDist();
}

const std::map<double, std::map<Node *, double> > MetaEngine::getNodesDists() const {

  std::map<double, std::map<Node *, double> > result;

  const std::map<double, STATE_MAP<NetworkState_Impl, double> > state_dist = merged_cumulator->getStateDists();

  std::map<double, STATE_MAP<NetworkState_Impl, double> >::const_iterator begin = state_dist.begin();
  std::map<double, STATE_MAP<NetworkState_Impl, double> >::const_iterator end = state_dist.end();
  
  const std::vector<Node*>& nodes = network->getNodes();

  while(begin != end) {

    std::map<Node *, double> node_dist;
    STATE_MAP<NetworkState_Impl, double> present_state_dist = begin->second;

    std::vector<Node *>::const_iterator nodes_begin = nodes.begin();
    std::vector<Node *>::const_iterator nodes_end = nodes.end();

    while(nodes_begin != nodes_end) {

      if (!(*nodes_begin)->isInternal())
      {
        double dist = 0;

        STATE_MAP<NetworkState_Impl, double>::const_iterator states_begin = present_state_dist.begin();
        STATE_MAP<NetworkState_Impl, double>::const_iterator states_end = present_state_dist.end();
      
        while(states_begin != states_end) {

          NetworkState state = states_begin->first;
          dist += states_begin->second * ((double) state.getNodeState(*nodes_begin));

          states_begin++;
        }

        node_dist[*nodes_begin] = dist;
      }
      nodes_begin++;
    }

    result[begin->first] = node_dist;

    begin++;
  }

  return result;
}

const std::map<Node*, double> MetaEngine::getNthNodesDist(int nn) const {
  std::map<Node *, double> result;

  const STATE_MAP<NetworkState_Impl, double> state_dist = merged_cumulator->getNthStateDist(nn);
  
  const std::vector<Node*>& nodes = network->getNodes();
  std::vector<Node *>::const_iterator nodes_begin = nodes.begin();
  std::vector<Node *>::const_iterator nodes_end = nodes.end();

  while(nodes_begin != nodes_end) {

    if (!(*nodes_begin)->isInternal())
    {
      double dist = 0;

      STATE_MAP<NetworkState_Impl, double>::const_iterator states_begin = state_dist.begin();
      STATE_MAP<NetworkState_Impl, double>::const_iterator states_end = state_dist.end();
    
      while(states_begin != states_end) {

        NetworkState state = states_begin->first;
        dist += states_begin->second * ((double) state.getNodeState(*nodes_begin));

        states_begin++;
      }

      result[*nodes_begin] = dist;
    }
    nodes_begin++;
  }

  return result;  
}

const std::map<Node*, double> MetaEngine::getAsymptoticNodesDist() const {
  std::map<Node *, double> result;

  const STATE_MAP<NetworkState_Impl, double> state_dist = merged_cumulator->getAsymptoticStateDist();
  
  const std::vector<Node*>& nodes = network->getNodes();
  std::vector<Node *>::const_iterator nodes_begin = nodes.begin();
  std::vector<Node *>::const_iterator nodes_end = nodes.end();

  while(nodes_begin != nodes_end) {

    if (!(*nodes_begin)->isInternal())
    {
      double dist = 0;

      STATE_MAP<NetworkState_Impl, double>::const_iterator states_begin = state_dist.begin();
      STATE_MAP<NetworkState_Impl, double>::const_iterator states_end = state_dist.end();
    
      while(states_begin != states_end) {

        NetworkState state = states_begin->first;
        dist += states_begin->second * ((double) state.getNodeState(*nodes_begin));

        states_begin++;
      }

      result[*nodes_begin] = dist;
    }
    nodes_begin++;
  }

  return result;  
}

const std::map<double, double> MetaEngine::getNodeDists(Node * node) const {
 
  std::map<double, double> result;

  const std::map<double, STATE_MAP<NetworkState_Impl, double> > state_dist = merged_cumulator->getStateDists();

  std::map<double, STATE_MAP<NetworkState_Impl, double> >::const_iterator begin = state_dist.begin();
  std::map<double, STATE_MAP<NetworkState_Impl, double> >::const_iterator end = state_dist.end();

  while(begin != end) {

    STATE_MAP<NetworkState_Impl, double> present_state_dist = begin->second;
    double dist = 0;

    STATE_MAP<NetworkState_Impl, double>::const_iterator states_begin = present_state_dist.begin();
    STATE_MAP<NetworkState_Impl, double>::const_iterator states_end = present_state_dist.end();
  
    while(states_begin != states_end) {

      NetworkState state = states_begin->first;
      dist += states_begin->second * ((double) state.getNodeState(node));

      states_begin++;
    }
    result[begin->first] = dist;

    begin++;
  }

  return result; 
}

double MetaEngine::getNthNodeDist(Node * node, int nn) const {

  double result = 0;

  const STATE_MAP<NetworkState_Impl, double> state_dist = merged_cumulator->getNthStateDist(nn);
  
  STATE_MAP<NetworkState_Impl, double>::const_iterator states_begin = state_dist.begin();
  STATE_MAP<NetworkState_Impl, double>::const_iterator states_end = state_dist.end();

  while(states_begin != states_end) {

    NetworkState state = states_begin->first;
    result += states_begin->second * ((double) state.getNodeState(node));

    states_begin++;
  }

  return result;  
}

double MetaEngine::getAsymptoticNodeDist(Node * node) const {

  double result = 0;

  const STATE_MAP<NetworkState_Impl, double> state_dist = merged_cumulator->getAsymptoticStateDist();
  
  STATE_MAP<NetworkState_Impl, double>::const_iterator states_begin = state_dist.begin();
  STATE_MAP<NetworkState_Impl, double>::const_iterator states_end = state_dist.end();

  while(states_begin != states_end) {

    NetworkState state = states_begin->first;
    result += states_begin->second * ((double) state.getNodeState(node));

    states_begin++;
  }

  return result;  
}

const double MetaEngine::getFinalTime() const {
  return merged_cumulator->getFinalTime();
}

void MetaEngine::displayFixpoints(FixedPointDisplayer* displayer) const 
{  
#ifdef MPI_COMPAT
  if (world_rank == 0) {
#endif

  displayer->begin(fixpoints.size());
  STATE_MAP<NetworkState_Impl, unsigned int>::const_iterator begin = fixpoints.begin();
  STATE_MAP<NetworkState_Impl, unsigned int>::const_iterator end = fixpoints.end();
  
  for (unsigned int nn = 0; begin != end; ++nn) {
    const NetworkState& network_state = begin->first;
#ifdef MPI_COMPAT
    displayer->displayFixedPoint(nn+1, network_state, begin->second, global_sample_count);
#else
    displayer->displayFixedPoint(nn+1, network_state, begin->second, sample_count);
#endif
    ++begin;
  }
  displayer->end();
  
#ifdef MPI_COMPAT
  }
#endif
}

void MetaEngine::displayProbTraj(ProbTrajDisplayer* displayer) const {
#ifdef MPI_COMPAT
  if (world_rank == 0) {
#endif

  merged_cumulator->displayProbTraj(network, refnode_count, displayer);

#ifdef MPI_COMPAT
  }
#endif
}

void MetaEngine::displayStatDist(StatDistDisplayer* statdist_displayer) const {
#ifdef MPI_COMPAT
  if (world_rank == 0) {
#endif

  Probe probe;
  merged_cumulator->displayStatDist(network, refnode_count, statdist_displayer);
  probe.stop();
  elapsed_statdist_runtime = probe.elapsed_msecs();
  user_statdist_runtime = probe.user_msecs();

#ifdef MPI_COMPAT
  }
#endif

}

void MetaEngine::display(ProbTrajDisplayer* probtraj_displayer, StatDistDisplayer* statdist_displayer, FixedPointDisplayer* fp_displayer) const
{
  displayProbTraj(probtraj_displayer);
  displayStatDist(statdist_displayer);
  displayFixpoints(fp_displayer);
}

void MetaEngine::displayAsymptotic(std::ostream& output_asymptprob, bool hexfloat, bool proba) const
{
#ifdef MPI_COMPAT
  if (world_rank == 0) {
#endif
  merged_cumulator->displayAsymptoticCSV(network, refnode_count, output_asymptprob, hexfloat, proba);
#ifdef MPI_COMPAT
  }
#endif
}

void MetaEngine::mergePairOfFixpoints(STATE_MAP<NetworkState_Impl, unsigned int>* fixpoints_1, STATE_MAP<NetworkState_Impl, unsigned int>* fixpoints_2)
{
  for (auto& fixpoint: *fixpoints_2) {
    
    STATE_MAP<NetworkState_Impl, unsigned int>::iterator t_fixpoint = fixpoints_1->find(fixpoint.first);
    if (fixpoints_1->find(fixpoint.first) == fixpoints_1->end()) {
      t_fixpoint->second = fixpoint.second;
    
    } else {
      t_fixpoint->second += fixpoint.second;
    
    }
  }
  delete fixpoints_2; 
}


STATE_MAP<NetworkState_Impl, unsigned int>* MetaEngine::mergeFixpointMaps()
{
  if (1 == fixpoint_map_v.size()) {
    return new STATE_MAP<NetworkState_Impl, unsigned int>(*fixpoint_map_v[0]);
  }

  STATE_MAP<NetworkState_Impl, unsigned int>* fixpoint_map = new STATE_MAP<NetworkState_Impl, unsigned int>();
  std::vector<STATE_MAP<NetworkState_Impl, unsigned int>*>::iterator begin = fixpoint_map_v.begin();
  std::vector<STATE_MAP<NetworkState_Impl, unsigned int>*>::iterator end = fixpoint_map_v.end();
  while (begin != end) {
    STATE_MAP<NetworkState_Impl, unsigned int>* fp_map = *begin;
    STATE_MAP<NetworkState_Impl, unsigned int>::const_iterator b = fp_map->begin();
    STATE_MAP<NetworkState_Impl, unsigned int>::const_iterator e = fp_map->end();
    while (b != e) {
      //NetworkState_Impl state = (*b).first;
      const NetworkState_Impl& state = (*b).first;
      if (fixpoint_map->find(state) == fixpoint_map->end()) {
	(*fixpoint_map)[state] = (*b).second;
      } else {
	(*fixpoint_map)[state] += (*b).second;
      }
      ++b;
    }
    ++begin;
  }
  return fixpoint_map;
}



struct MergeWrapper {
  Cumulator* cumulator_1;
  Cumulator* cumulator_2;
  STATE_MAP<NetworkState_Impl, unsigned int>* fixpoints_1;
  STATE_MAP<NetworkState_Impl, unsigned int>* fixpoints_2;
  
  MergeWrapper(Cumulator* cumulator_1, Cumulator* cumulator_2, STATE_MAP<NetworkState_Impl, unsigned int>* fixpoints_1, STATE_MAP<NetworkState_Impl, unsigned int>* fixpoints_2) :
    cumulator_1(cumulator_1), cumulator_2(cumulator_2), fixpoints_1(fixpoints_1), fixpoints_2(fixpoints_2) { }
};

void* MetaEngine::threadMergeWrapper(void *arg)
{
#ifdef USE_DYNAMIC_BITSET
  MBDynBitset::init_pthread();
#endif
  MergeWrapper* warg = (MergeWrapper*)arg;
  try {
    Cumulator::mergePairOfCumulators(warg->cumulator_1, warg->cumulator_2);
    MetaEngine::mergePairOfFixpoints(warg->fixpoints_1, warg->fixpoints_2);
  } catch(const BNException& e) {
    std::cerr << e;
  }
#ifdef USE_DYNAMIC_BITSET
  MBDynBitset::end_pthread();
#endif
  return NULL;
}


std::pair<Cumulator*, STATE_MAP<NetworkState_Impl, unsigned int>*> MetaEngine::mergeResults(std::vector<Cumulator*>& cumulator_v, std::vector<STATE_MAP<NetworkState_Impl, unsigned int> *>& fixpoint_map_v) {
  
  size_t size = cumulator_v.size();
  
  if (size == 0) {
    return std::make_pair((Cumulator*) NULL, new STATE_MAP<NetworkState_Impl, unsigned int>());
  }
  
  if (size > 1) {
    
    
    unsigned int lvl=1;
    unsigned int max_lvl = ceil(log2(size));

    while(lvl <= max_lvl) {      
    
      unsigned int step_lvl = pow(2, lvl-1);
      unsigned int width_lvl = floor(size/(step_lvl*2)) + 1;
      pthread_t* tid = new pthread_t[width_lvl];
      unsigned int nb_threads = 0;
      std::vector<MergeWrapper*> wargs;
      for(unsigned int i=0; i < size; i+=(step_lvl*2)) {
        
        if (i+step_lvl < size) {
          MergeWrapper* warg = new MergeWrapper(cumulator_v[i], cumulator_v[i+step_lvl], fixpoint_map_v[i], fixpoint_map_v[i+step_lvl]);
          pthread_create(&tid[nb_threads], NULL, MetaEngine::threadMergeWrapper, warg);
          nb_threads++;
          wargs.push_back(warg);
        } 
      }
      
      for(unsigned int i=0; i < nb_threads; i++) {   
          pthread_join(tid[i], NULL);
          
      }
      
      for (auto warg: wargs) {
        delete warg;
      }
      delete [] tid;
      lvl++;
    }
  
   
  }
  
  return std::make_pair(cumulator_v[0], fixpoint_map_v[0]);
}

#ifdef MPI_COMPAT
STATE_MAP<NetworkState_Impl, unsigned int>* MetaEngine::MPI_Unpack_Fixpoints(STATE_MAP<NetworkState_Impl, unsigned int>* fp_map, char* buff, unsigned int buff_size)
{
        
  int position = 0;
  unsigned int nb_fixpoints;
  MPI_Unpack(buff, buff_size, &position, &nb_fixpoints, 1, MPI_UNSIGNED, MPI_COMM_WORLD);
  
  if (nb_fixpoints > 0) {
    if (fp_map == NULL) {
      fp_map = new STATE_MAP<NetworkState_Impl, unsigned int>();
      // std::cout << "Creating new fp map for " << nb_fixpoints << " fixpoints" << std::endl;
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
  // std::cout << "Added " << fp_map->size() <<  " fixpoints to the map" << std::endl;
  return fp_map;
}

char* MetaEngine::MPI_Pack_Fixpoints(const STATE_MAP<NetworkState_Impl, unsigned int>* fp_map, int dest, unsigned int * buff_size)
{
  unsigned int nb_fixpoints = fp_map == NULL ? 0 : fp_map->size();
  *buff_size = sizeof(unsigned int) + (sizeof(unsigned int) + NetworkState::my_MPI_Pack_Size()) * nb_fixpoints;
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

void MetaEngine::MPI_Send_Fixpoints(const STATE_MAP<NetworkState_Impl, unsigned int>* fp_map, int dest) 
{
  // MPI_Send version
  // First we send the number of fixpoints we have
  int nb_fixpoints = fp_map->size();
  MPI_Send(&nb_fixpoints, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
  
  for (auto& fixpoint: *fp_map) {
    NetworkState state(fixpoint.first);
    unsigned int count = fixpoint.second;
    
    state.my_MPI_Send(dest);
    MPI_Send(&count, 1, MPI_UNSIGNED, dest, 0, MPI_COMM_WORLD);
    
  } 
}

void MetaEngine::MPI_Recv_Fixpoints(STATE_MAP<NetworkState_Impl, unsigned int>* fp_map, int origin) 
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

STATE_MAP<NetworkState_Impl, unsigned int>* MetaEngine::mergeMPIFixpointMaps(STATE_MAP<NetworkState_Impl, unsigned int>* t_fixpoint_map, bool pack)
{
  // If we are, but only on one node, we don't need to do anything
  if (world_size == 1) {
    return t_fixpoint_map;
  } else {
    
    for (int i = 1; i < world_size; i++) {
      
      if (world_rank == 0) {
        
        int rank = i;
        MPI_Bcast(&rank, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (pack) {
          // MPI_Unpack version
          unsigned int buff_size;
          MPI_Recv( &buff_size, 1, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          
          char* buff = new char[buff_size];
          MPI_Recv( buff, buff_size, MPI_PACKED, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
          
          MPI_Unpack_Fixpoints(t_fixpoint_map, buff, buff_size);
          delete buff;
          
        } else {
          MPI_Recv_Fixpoints(t_fixpoint_map, i);
        }
         
      } else {
        
        int rank;
        MPI_Bcast(&rank, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank == world_rank) {
          
          if (pack) {
            unsigned int buff_size;
            char* buff = MPI_Pack_Fixpoints(t_fixpoint_map, 0, &buff_size);

            MPI_Send(&buff_size, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD);
            MPI_Send( buff, buff_size, MPI_PACKED, 0, 0, MPI_COMM_WORLD); 
            delete buff;
            
          } else {
            MPI_Send_Fixpoints(t_fixpoint_map, 0);
          }
        }
      }      
    }

    return t_fixpoint_map; 
  }
}
#endif
