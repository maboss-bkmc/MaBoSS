/* 
   MaBoSS (Markov Boolean Stochastic Simulator)
   Copyright (C) 2011-2019 Institut Curie, 26 rue d'Ulm, Paris, France
   
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
     EnsembleEngine.h

   Authors:
     Vincent Noel <contact@vincent-noel.fr>
 
   Date:
     March 2019
*/

#ifndef _METAENGINE_H_
#define _METAENGINE_H_

#include <string>
#include <map>
#include <vector>
#include <assert.h>

#include "BooleanNetwork.h"
#include "Cumulator.h"
#include "RandomGenerator.h"
#include "RunConfig.h"

struct EnsembleArgWrapper;

class MetaEngine {

protected:

  Network* network;
  RunConfig* runconfig;

  double time_tick;
  double max_time;
  unsigned int sample_count;
  bool discrete_time;
  unsigned int thread_count;
  
  NetworkState reference_state;
  unsigned int refnode_count;

  mutable long long elapsed_core_runtime, user_core_runtime, elapsed_statdist_runtime, user_statdist_runtime, elapsed_epilogue_runtime, user_epilogue_runtime;
  STATE_MAP<NetworkState_Impl, unsigned int> fixpoints;
  std::vector<STATE_MAP<NetworkState_Impl, unsigned int>*> fixpoint_map_v;
  
  Cumulator* merged_cumulator;
  std::vector<Cumulator*> cumulator_v;

  STATE_MAP<NetworkState_Impl, unsigned int>* mergeFixpointMaps();

public:

  MetaEngine(Network* network, RunConfig* runconfig) : 
    network(network), runconfig(runconfig),
    time_tick(runconfig->getTimeTick()), 
    max_time(runconfig->getMaxTime()), 
    sample_count(runconfig->getSampleCount()), 
    discrete_time(runconfig->isDiscreteTime()), 
    thread_count(runconfig->getThreadCount()) {}

  static void init();
  static void loadUserFuncs(const char* module);

  long long getElapsedCoreRunTime() const {return elapsed_core_runtime;}
  long long getUserCoreRunTime() const {return user_core_runtime;}

  long long getElapsedEpilogueRunTime() const {return elapsed_epilogue_runtime;}
  long long getUserEpilogueRunTime() const {return user_epilogue_runtime;}

  long long getElapsedStatDistRunTime() const {return elapsed_statdist_runtime;}
  long long getUserStatDistRunTime() const {return user_statdist_runtime;}

  bool converges() const {return fixpoints.size() > 0;}
  const STATE_MAP<NetworkState_Impl, unsigned int>& getFixpoints() const {return fixpoints;}

  const std::map<double, STATE_MAP<NetworkState_Impl, double> > getStateDists() const;
  const STATE_MAP<NetworkState_Impl, double> getNthStateDist(int nn) const;
  const STATE_MAP<NetworkState_Impl, double> getAsymptoticStateDist() const;

  const std::map<double, std::map<Node *, double> > getNodesDists() const;
  const std::map<Node*, double> getNthNodesDist(int nn) const;
  const std::map<Node*, double> getAsymptoticNodesDist() const;

  const std::map<double, double> getNodeDists(Node * node) const;
  double getNthNodeDist(Node * node, int nn) const;
  double getAsymptoticNodeDist(Node * node) const;
  
  const std::map<unsigned int, std::pair<NetworkState, double> > getFixPointsDists() const;
  int getMaxTickIndex() const {return merged_cumulator->getMaxTickIndex();} 
  const double getFinalTime() const;

  void display(std::ostream& output_probtraj, std::ostream& output_statdist, std::ostream& output_fp, bool hexfloat = false) const;
  void displayStatDist(std::ostream& output_statdist, bool hexfloat = false) const;
  void displayProbTraj(std::ostream& output_probtraj, bool hexfloat = false) const;
  void displayFixpoints(std::ostream& output_fp, bool hexfloat = false) const;
  void displayAsymptotic(std::ostream& output_asymptprob, bool hexfloat = false, bool proba = true) const;


};

#endif
