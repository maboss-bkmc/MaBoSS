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
     ProbTrajEngine.cc

   Authors:
     Eric Viara <viara@sysra.com>
     Gautier Stoll <gautier.stoll@curie.fr>
     Vincent Noël <vincent.noel@curie.fr>
 
   Date:
     March 2021
*/

#include "ProbTrajEngine.h"
#include "../Probe.h"
#include "../Utils.h"

struct MergeWrapper {
  Cumulator<NetworkState>* cumulator_1;
  Cumulator<NetworkState>* cumulator_2;
  FixedPoints* fixpoints_1;
  FixedPoints* fixpoints_2;
  ObservedGraph* observed_graph_1;
  ObservedGraph* observed_graph_2;
  
  MergeWrapper(Cumulator<NetworkState>* cumulator_1, Cumulator<NetworkState>* cumulator_2, FixedPoints* fixpoints_1, FixedPoints* fixpoints_2, ObservedGraph* observed_graph_1, ObservedGraph* observed_graph_2) :
    cumulator_1(cumulator_1), cumulator_2(cumulator_2), fixpoints_1(fixpoints_1), fixpoints_2(fixpoints_2), observed_graph_1(observed_graph_1), observed_graph_2(observed_graph_2) { }
};

void* ProbTrajEngine::threadMergeWrapper(void *arg)
{
#ifdef USE_DYNAMIC_BITSET
  MBDynBitset::init_pthread();
#endif
  MergeWrapper* warg = (MergeWrapper*)arg;
  try {
    Cumulator<NetworkState>::mergePairOfCumulators(warg->cumulator_1, warg->cumulator_2);
    ProbTrajEngine::mergePairOfFixpoints(warg->fixpoints_1, warg->fixpoints_2);
    warg->observed_graph_1->mergePairOfObservedGraph(warg->observed_graph_2);
  } catch(const BNException& e) {
    std::cerr << e;
  }
#ifdef USE_DYNAMIC_BITSET
  MBDynBitset::end_pthread();
#endif
  return NULL;
}


void ProbTrajEngine::mergeResults(std::vector<Cumulator<NetworkState>*>& cumulator_v, std::vector<FixedPoints *>& fixpoint_map_v, std::vector<ObservedGraph* >& observed_graph_v) {
  
  size_t size = cumulator_v.size();

  if (size > 1) {
    
    
    unsigned int lvl=1;
    unsigned int max_lvl = (unsigned int) ceil(log2(size));

    while(lvl <= max_lvl) {      
    
      unsigned int step_lvl = (unsigned int) pow(2, lvl-1);
      unsigned int width_lvl = (unsigned int) floor(size/(step_lvl*2)) + 1;
#ifdef STD_THREAD
      std::vector<std::thread *> tid(width_lvl);
#else
      pthread_t* tid = new pthread_t[width_lvl];
#endif
      unsigned int nb_threads = 0;
      std::vector<MergeWrapper*> wargs;
      for(unsigned int i=0; i < size; i+=(step_lvl*2)) {
        
        if (i+step_lvl < size) {
          MergeWrapper* warg = new MergeWrapper(cumulator_v[i], cumulator_v[i+step_lvl], fixpoint_map_v[i], fixpoint_map_v[i+step_lvl], observed_graph_v[i], observed_graph_v[i+step_lvl]);
#ifdef STD_THREAD
          tid[nb_threads] = new std::thread(ProbTrajEngine::threadMergeWrapper, warg);
#else
          pthread_create(&tid[nb_threads], NULL, ProbTrajEngine::threadMergeWrapper, warg);
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
      for (std::thread* t: tid) {
        delete t;
      }
      tid.clear();
#else
      delete [] tid;
#endif
      lvl++;
    }
  
   
  }
}

#ifdef MPI_COMPAT

void ProbTrajEngine::mergeMPIResults(RunConfig* runconfig, Cumulator<NetworkState>* ret_cumul, FixedPoints* fixpoints, ObservedGraph* graph, int world_size, int world_rank, bool pack)
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
            Cumulator<NetworkState>::mergePairOfMPICumulators(ret_cumul, world_rank, i, i+step_lvl, runconfig, pack);
            mergePairOfMPIFixpoints(fixpoints, world_rank, i, i+step_lvl, pack);
            ObservedGraph::mergePairOfMPIObservedGraph(graph, world_rank, i, i+step_lvl, pack);
          }
        } 
      }
      
      lvl++;
    }
  }
}
#endif


void ProbTrajEngine::displayProbTraj(ProbTrajDisplayer<NetworkState>* displayer) const {

#ifdef MPI_COMPAT
if (getWorldRank() == 0) {
#endif

  merged_cumulator->displayProbTraj(refnode_count, displayer);

#ifdef MPI_COMPAT
}
#endif
}

void ProbTrajEngine::displayStatDist(StatDistDisplayer* statdist_displayer) const {

#ifdef MPI_COMPAT
if (getWorldRank() == 0) {
#endif

  Probe probe;
  merged_cumulator->displayStatDist(statdist_displayer);
  probe.stop();
  elapsed_statdist_runtime = probe.elapsed_msecs();
  user_statdist_runtime = probe.user_msecs();

  /*
  unsigned int statdist_traj_count = runconfig->getStatDistTrajCount();
  if (statdist_traj_count == 0) {
    output_statdist << "Trajectory\tState\tProba\n";
  }
  */
#ifdef MPI_COMPAT
}
#endif

}

void ProbTrajEngine::display(ProbTrajDisplayer<NetworkState>* probtraj_displayer, StatDistDisplayer* statdist_displayer, FixedPointDisplayer* fp_displayer) const
{
  displayProbTraj(probtraj_displayer);
  displayStatDist(statdist_displayer);
  displayFixpoints(fp_displayer);
}

void ProbTrajEngine::displayObservedGraph(std::ostream* output_observed_graph, std::ostream * output_observed_durations){

#ifdef MPI_COMPAT
if (getWorldRank() == 0) {
#endif

  observed_graph->display(output_observed_graph, output_observed_durations, network);
  
#ifdef MPI_COMPAT
}
#endif
}

#ifdef PYTHON_API

PyObject* ProbTrajEngine::getNumpyObservedGraph()
{
  if (observed_graph != NULL)
  {
    return observed_graph->getNumpyObservedGraph(network);
  
  } else {
    Py_RETURN_NONE;
  }
}
PyObject* ProbTrajEngine::getNumpyObservedDurations()
{
  if (observed_graph != NULL)
  {
    return observed_graph->getNumpyObservedDurations(network);
  
  } else {
    Py_RETURN_NONE;
  }
}

#endif

void ProbTrajEngine::buildSchedule() {
  // std::cout << "Building schedule..." << std::endl;
  // Clear previous schedule if any
  for (auto schedule_entry : schedule) {
    delete schedule_entry.second;
  }
  schedule.clear();
  schedule_times.clear();
  
  // Build new schedule
  for (const auto& node : network->getNodes()) {
    std::map<double, Expression*>* node_schedule = node->getScheduledFlips();
    if (node_schedule != NULL) {
      for (const auto& entry : *node_schedule) {
        double time = entry.first;
        Expression* expr = entry.second;
        if (schedule.find(time) == schedule.end()) {
          schedule[time] = new std::map<Node*, Expression*>();
        }
        (*schedule[time])[node] = expr;
        schedule_times.push_back(time);
        
      }
    }
  }
  
  // Making sure schedule times are unique and sorted
  std::sort(schedule_times.begin(), schedule_times.end());  
  auto last = std::unique(schedule_times.begin(), schedule_times.end());
  schedule_times.erase(last, schedule_times.end());
}

void ProbTrajEngine::applySchedule(RandomGenerator* random_generator, NetworkState& network_state, std::vector<double>& times, double time) {
  
  for (double t : times) {
    if (t <= time) {
      auto it = schedule.find(t);
      if (it != schedule.end()) {
        for (const auto& entry : *(it->second)) {
          Node* node = entry.first;
          Expression* expr = entry.second;
          double val = expr->eval(node, network_state);
          double rand = random_generator->generate();
          network_state.setNodeState(node, (val >= rand)); 
        }
      }
      times.erase(times.begin()); // Remove the time from the list
      break;
    }
    else { break; } // No need to check further if time is already past the schedule time
  }
}