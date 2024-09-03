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
#include "Probe.h"
#include "Utils.h"

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

void ProbTrajEngine::mergePairOfObservedGraph(ObservedGraph* observed_graph_1, ObservedGraph* observed_graph_2)
{
  for (auto origin_state: *observed_graph_2){
    for (auto destination_state: origin_state.second) {
      (*observed_graph_1)[origin_state.first][destination_state.first] += destination_state.second;
    }
  }
  delete observed_graph_2;
  observed_graph_2 = NULL;
}

void* ProbTrajEngine::threadMergeWrapper(void *arg)
{
#ifdef USE_DYNAMIC_BITSET
  MBDynBitset::init_pthread();
#endif
  MergeWrapper* warg = (MergeWrapper*)arg;
  try {
    Cumulator<NetworkState>::mergePairOfCumulators(warg->cumulator_1, warg->cumulator_2);
    ProbTrajEngine::mergePairOfFixpoints(warg->fixpoints_1, warg->fixpoints_2);
    ProbTrajEngine::mergePairOfObservedGraph(warg->observed_graph_1, warg->observed_graph_2);
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
    unsigned int max_lvl = ceil(log2(size));

    while(lvl <= max_lvl) {      
    
      unsigned int step_lvl = pow(2, lvl-1);
      unsigned int width_lvl = floor(size/(step_lvl*2)) + 1;
      pthread_t* tid = new pthread_t[width_lvl];
      unsigned int nb_threads = 0;
      std::vector<MergeWrapper*> wargs;
      for(unsigned int i=0; i < size; i+=(step_lvl*2)) {
        
        if (i+step_lvl < size) {
          MergeWrapper* warg = new MergeWrapper(cumulator_v[i], cumulator_v[i+step_lvl], fixpoint_map_v[i], fixpoint_map_v[i+step_lvl], observed_graph_v[i], observed_graph_v[i+step_lvl]);
          pthread_create(&tid[nb_threads], NULL, ProbTrajEngine::threadMergeWrapper, warg);
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
            mergePairOfMPIObservedGraph(graph, world_rank, i, i+step_lvl, pack); 
          }
        } 
      }
      
      lvl++;
    }
  }
}
#endif


const double ProbTrajEngine::getFinalTime() const {
  return merged_cumulator->getFinalTime();
}

void ProbTrajEngine::displayProbTraj(ProbTrajDisplayer<NetworkState>* displayer) const {

#ifdef MPI_COMPAT
if (getWorldRank() == 0) {
#endif

  merged_cumulator->displayProbTraj(network, refnode_count, displayer);

#ifdef MPI_COMPAT
}
#endif
}

void ProbTrajEngine::displayStatDist(StatDistDisplayer* statdist_displayer) const {

#ifdef MPI_COMPAT
if (getWorldRank() == 0) {
#endif

  Probe probe;
  merged_cumulator->displayStatDist(network, refnode_count, statdist_displayer);
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

void ProbTrajEngine::displayObservedGraph(std::ostream* output_observed_graph){

#ifdef MPI_COMPAT
if (getWorldRank() == 0) {
#endif

  if (graph_states.size() > 0)
  {
    (*output_observed_graph) << "State";
    for (auto state: graph_states) {
      (*output_observed_graph) << "\t" << NetworkState(state).getName(network);
    }
    (*output_observed_graph) << std::endl;
    
    for (auto origin_state: graph_states) {
      (*output_observed_graph) << NetworkState(origin_state).getName(network);
      
      for (auto destination_state: graph_states) {
        (*output_observed_graph) << "\t" << (*(observed_graph))[origin_state][destination_state];
      }
      
      (*output_observed_graph) << std::endl;
    }
    
  }
#ifdef MPI_COMPAT
}
#endif
  
}
  
#ifdef MPI_COMPAT

void ProbTrajEngine::mergePairOfMPIObservedGraph(ObservedGraph* graph, int world_rank, int dest, int origin, bool pack)
{
  if (world_rank == dest) 
  {
  
    if (pack) {
      unsigned int buff_size = -1;
      MPI_Recv( &buff_size, 1, MPI_UNSIGNED, origin, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);    
      char* buff = new char[buff_size];
      MPI_Recv( buff, buff_size, MPI_PACKED, origin, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
      MPI_Unpack_ObservedGraph(graph, buff, buff_size);
      delete [] buff;
      
    } else {
      MPI_Recv_ObservedGraph(graph, origin);
    }
    
  } else if (world_rank == origin) {

    if (pack) {
      unsigned int buff_size = -1;
      char* buff = MPI_Pack_ObservedGraph(graph, dest, &buff_size);      
      MPI_Send(&buff_size, 1, MPI_UNSIGNED, dest, 0, MPI_COMM_WORLD);
      MPI_Send( buff, buff_size, MPI_PACKED, dest, 0, MPI_COMM_WORLD); 
      delete [] buff;            
      
    } else {
    
      MPI_Send_ObservedGraph(graph, dest);
    }
  }
}

unsigned int ProbTrajEngine::MPI_Pack_Size_ObservedGraph(const ObservedGraph* graph) {
  unsigned int pack_size = sizeof(unsigned int);
  if (graph != NULL) {
    for (auto& row: *graph) {
      NetworkState s(row.first);
      pack_size += s.my_MPI_Pack_Size();
      pack_size += row.second.size() * sizeof(unsigned int);
    }
  }
  return pack_size;
}

void ProbTrajEngine::MPI_Unpack_ObservedGraph(ObservedGraph* graph, char* buff, unsigned int buff_size)
{
  int position = 0;
  
  unsigned int size = -1;
  MPI_Unpack(buff, buff_size, &position, &size, 1, MPI_UNSIGNED, MPI_COMM_WORLD);
  
  if (size > 0) {
    
    std::vector<NetworkState_Impl> states;
    for (unsigned int i=0; i < size; i++) {
      NetworkState s;
      s.my_MPI_Unpack(buff, buff_size, &position);
      states.push_back(s.getState());
    }
    
    if (graph == NULL){
      graph = new ObservedGraph();  
      for (auto& state: states) {
        (*graph)[state] = std::map<NetworkState_Impl, unsigned int>();
        for (auto& state2: states) {
          (*graph)[state][state2] = 0;
        }
      }
    }
    
    for (auto& row: *graph) {
      for (auto& cell: row.second) {
        unsigned int count = -1;
        MPI_Unpack(buff, buff_size, &position, &count, 1, MPI_UNSIGNED, MPI_COMM_WORLD);
        cell.second += count;
      }
    }
  }
}

char* ProbTrajEngine::MPI_Pack_ObservedGraph(const ObservedGraph* graph, int dest, unsigned int * buff_size)
{
  
  *buff_size = MPI_Pack_Size_ObservedGraph(graph);
  
  char* buff = new char[*buff_size];
  int position = 0;
  
  unsigned int size = graph == NULL ? 0 : graph->size();
  MPI_Pack(&size, 1, MPI_UNSIGNED, buff, *buff_size, &position, MPI_COMM_WORLD);
  
  if (graph != NULL){
    
    for (auto& row: *graph) {
      NetworkState s(row.first);
      s.my_MPI_Pack(buff, *buff_size, &position);
    }
    for (auto& row: *graph) {
      for (auto& cell: row.second) {
        unsigned int count = cell.second;
        MPI_Pack(&count, 1, MPI_UNSIGNED, buff, *buff_size, &position, MPI_COMM_WORLD);
      }
    }
  }
  return buff;
}

void ProbTrajEngine::MPI_Send_ObservedGraph(const ObservedGraph* graph, int dest)
{
  unsigned int nb_states = graph == NULL ? 0 : graph->size();
  MPI_Send(&nb_states, 1, MPI_UNSIGNED, dest, 0, MPI_COMM_WORLD);
  
  if (nb_states > 0) {
    for (auto& row: *graph) {
      NetworkState s(row.first);
      s.my_MPI_Send(dest);
    }
  
    for (auto& row: *graph) {
      for (auto& cell: row.second) {
        unsigned int count = cell.second;
        MPI_Send(&count, 1, MPI_UNSIGNED, dest, 0, MPI_COMM_WORLD);
      }
    } 
  }
}

void ProbTrajEngine::MPI_Recv_ObservedGraph(ObservedGraph* graph, int origin)
{
  unsigned int nb_states = -1;
  MPI_Recv(&nb_states, 1, MPI_UNSIGNED, origin, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  if (nb_states > 0) {
    std::vector<NetworkState_Impl> states;
    for (unsigned int i=0; i < nb_states; i++) {
      NetworkState s;
      s.my_MPI_Recv(origin);
      states.push_back(s.getState());
    }
    
    if (graph == NULL) {
      graph = new ObservedGraph();
      for (auto& state: states) {
        (*graph)[state] = std::map<NetworkState_Impl, unsigned int>();
        for (auto& state2: states) {
          (*graph)[state][state2] = 0;
        }
      }
    }
    
    for (auto& row: *graph) {
      for (auto& cell: row.second) {
        unsigned int count = -1;
        MPI_Recv(&count, 1, MPI_UNSIGNED, origin, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cell.second += count;
      }
    } 
  }
}

#endif

#ifdef PYTHON_API

PyObject* ProbTrajEngine::getNumpyObservedGraph()
{
  if (observed_graph != NULL)
  {
    npy_intp dims[2] = {(npy_intp) observed_graph->size(), (npy_intp) observed_graph->size()};
    PyArrayObject* graph = (PyArrayObject *) PyArray_ZEROS(2,dims,NPY_DOUBLE, 0); 
    PyObject* states = PyList_New(observed_graph->size());

    int i=0;
    for (auto& row: *observed_graph) 
    {
      PyList_SetItem(states, i, PyUnicode_FromString(NetworkState(row.first).getName(network).c_str()));
      int j=0;
      for (auto& cell: row.second) {
        void* ptr_val = PyArray_GETPTR2(graph, i, j);

        PyArray_SETITEM(graph, (char*) ptr_val, PyFloat_FromDouble(cell.second));
        j++;
      }
      i++;
    }

    return PyTuple_Pack(2, PyArray_Return(graph), states);
  
  } else {
    return Py_None;
  }
}

#endif