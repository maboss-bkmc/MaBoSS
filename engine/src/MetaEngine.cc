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
     MaBEstEngine.cc

   Authors:
     Eric Viara <viara@sysra.com>
     Gautier Stoll <gautier.stoll@curie.fr>
 
   Date:
     January-March 2011
*/

#include "MetaEngine.h"
#include "Probe.h"
#include "Utils.h"
#include <dlfcn.h>

static const char* MABOSS_USER_FUNC_INIT = "maboss_user_func_init";

void MetaEngine::init()
{
  extern void builtin_functions_init();
  builtin_functions_init();
}

void MetaEngine::loadUserFuncs(const char* module)
{
  init();

  void* dl = dlopen(module, RTLD_LAZY);
  if (NULL == dl) {
    std::cerr << dlerror() << "\n";
    exit(1);
  }

  void* sym = dlsym(dl, MABOSS_USER_FUNC_INIT);
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


void MetaEngine::displayFixpoints(std::ostream& output_fp, bool hexfloat) const 
{
  output_fp << "Fixed Points (" << fixpoints.size() << ")\n";
  if (0 == fixpoints.size()) {
    return;
  }

#ifdef HAS_STD_HEXFLOAT
  if (hexfloat) {
    output_fp << std::hexfloat;
  }
#endif

  STATE_MAP<NetworkState_Impl, unsigned int>::const_iterator begin = fixpoints.begin();
  STATE_MAP<NetworkState_Impl, unsigned int>::const_iterator end = fixpoints.end();
  
  output_fp << "FP\tProba\tState\t";
  network->displayHeader(output_fp);
  for (unsigned int nn = 0; begin != end; ++nn) {
    const NetworkState& network_state = (*begin).first;
    output_fp << "#" << (nn+1) << "\t";
    if (hexfloat) {
      output_fp << fmthexdouble((double)(*begin).second / sample_count) <<  "\t";
    } else {
      output_fp << ((double)(*begin).second / sample_count) <<  "\t";
    }
    network_state.displayOneLine(output_fp, network);
    output_fp << '\t';
    network_state.display(output_fp, network);
    ++begin;
  }
}

void MetaEngine::displayProbTraj(std::ostream& output_probtraj, bool hexfloat) const {
  merged_cumulator->displayProbTrajCSV(network, refnode_count, output_probtraj, hexfloat);
}

void MetaEngine::displayStatDist(std::ostream& output_statdist, bool hexfloat) const {
  Probe probe;
  merged_cumulator->displayStatDistCSV(network, refnode_count, output_statdist, hexfloat);
  probe.stop();
  elapsed_statdist_runtime = probe.elapsed_msecs();
  user_statdist_runtime = probe.user_msecs();

  unsigned int statdist_traj_count = runconfig->getStatDistTrajCount();
  if (statdist_traj_count == 0) {
    output_statdist << "Trajectory\tState\tProba\n";
  }
}
void MetaEngine::display(std::ostream& output_probtraj, std::ostream& output_statdist, std::ostream& output_fp, bool hexfloat) const
{
  displayProbTraj(output_probtraj, hexfloat);
  displayStatDist(output_statdist, hexfloat);
  displayFixpoints(output_fp, hexfloat);
}

void MetaEngine::displayAsymptotic(std::ostream& output_asymptprob, bool hexfloat, bool proba) const
{
  merged_cumulator->displayAsymptoticCSV(network, refnode_count, output_asymptprob, hexfloat, proba);
}