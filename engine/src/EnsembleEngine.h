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

#ifndef _ENSEMBLEENGINE_H_
#define _ENSEMBLEENGINE_H_

#include <string>
#include <map>
#include <vector>
#include <assert.h>

#include "MetaEngine.h"
#include "BooleanNetwork.h"
#include "Cumulator.h"
#include "RandomGenerator.h"
#include "RunConfig.h"

struct EnsembleArgWrapper;

class EnsembleEngine : public MetaEngine {

  std::vector<Network*> networks;
  std::vector<Cumulator*> cumulators_per_model; // The final Cumulators for each model
  std::vector<STATE_MAP<NetworkState_Impl, unsigned int>* > fixpoints_per_model; // The final fixpoints for each model
  
  bool save_individual_result; // Do we want to save individual model simulation result
  bool random_sampling; // Randomly select the number of simulation per model

  std::vector<std::vector<unsigned int> > simulation_indices_v; // The list of indices of models to simulate for each thread
  std::vector<std::vector<Cumulator*> > cumulator_models_v; // The results for each model, by thread
  std::vector<std::vector<Cumulator*> > cumulators_thread_v; // The results for each model, by model
  std::vector<std::vector<STATE_MAP<NetworkState_Impl, unsigned int>*> > fixpoints_models_v; // The fixpoints for each model, by thread
  std::vector<std::vector<STATE_MAP<NetworkState_Impl, unsigned int>*> > fixpoints_threads_v; // The fixpoints for each model, by thread

  std::vector<EnsembleArgWrapper*> arg_wrapper_v;
  NodeIndex getTargetNode(Network* network, RandomGenerator* random_generator, const MAP<NodeIndex, double>& nodeTransitionRates, double total_rate) const;
  double computeTH(const MAP<NodeIndex, double>& nodeTransitionRates, double total_rate) const;
  void epilogue();
  static void* threadWrapper(void *arg);
  void runThread(Cumulator* cumulator, unsigned int start_count_thread, unsigned int sample_count_thread, RandomGeneratorFactory* randgen_factory, int seed, STATE_MAP<NetworkState_Impl, unsigned int>* fixpoint_map, std::ostream* output_traj, std::vector<unsigned int> simulation_ind, std::vector<Cumulator*> t_models_cumulators, std::vector<STATE_MAP<NetworkState_Impl, unsigned int>* > t_models_fixpoints);
  void mergeEnsembleFixpointMaps();

public:
  static const std::string VERSION;

  EnsembleEngine(std::vector<Network*> network, RunConfig* runconfig, bool save_individual_result=false, bool random_sampling=false);

  void run(std::ostream* output_traj);

  void displayIndividual(unsigned int model_id, std::ostream& output_probtraj, std::ostream& output_statdist, std::ostream& output_fp, bool hexfloat = false) const;

  ~EnsembleEngine();
};

#endif
