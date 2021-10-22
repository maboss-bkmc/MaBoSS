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
     Cumulator.h

   Authors:
     Eric Viara <viara@sysra.com>
     Gautier Stoll <gautier.stoll@curie.fr>
     Vincent Noël <vincent.noel@curie.fr>
 
   Date:
     January-March 2011
*/

#ifndef _CUMULATOR_H_
#define _CUMULATOR_H_

#include <string>
#include <map>
#include <set>
#include <vector>
#include <iostream>
#include <assert.h>

#ifdef PYTHON_API
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL MABOSS_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#endif

#ifdef MPI_COMPAT
#include <mpi.h>

#include <stdint.h>
#include <limits.h>

#if SIZE_MAX == UCHAR_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED_CHAR
#elif SIZE_MAX == USHRT_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED_SHORT
#elif SIZE_MAX == UINT_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED
#elif SIZE_MAX == ULONG_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#else
   #error "what is happening here?"
#endif

#endif

static bool COMPUTE_ERRORS = true;

#include "ProbaDist.h"
#include "RunConfig.h"

class Network;
class ProbTrajDisplayer;
class StatDistDisplayer;

class Cumulator {

  struct TickValue {
    double tm_slice;
    double TH;
    double tm_slice_square;
    TickValue() {tm_slice = 0.; TH = 0.; tm_slice_square = 0.;}
    TickValue(double tm_slice, double TH) : tm_slice(tm_slice), TH(TH), tm_slice_square(0.0) { }
  };

  struct LastTickValue {
    double tm_slice;
    double TH;
    LastTickValue() : tm_slice(0.0), TH(0.0) {}
    LastTickValue(double tm_slice, double TH) : tm_slice(tm_slice), TH(TH) {}
  };

  class CumulMap {
    STATE_MAP<NetworkState_Impl, TickValue> mp;

  public:
    size_t size() const {
      return mp.size();
    }

    void incr(const NetworkState_Impl& state, double tm_slice, double TH) {
      STATE_MAP<NetworkState_Impl, TickValue>::iterator iter = mp.find(state);
      if (iter == mp.end()) {
	mp[state] = TickValue(tm_slice, tm_slice * TH);
      } else {
	(*iter).second.tm_slice += tm_slice;
	(*iter).second.TH += tm_slice * TH;
      }
    }

    void cumulTMSliceSquare(const NetworkState_Impl& state, double tm_slice) {
      STATE_MAP<NetworkState_Impl, TickValue>::iterator iter = mp.find(state);
      assert(iter != mp.end());
      (*iter).second.tm_slice_square += tm_slice * tm_slice;
    }
    
    void add(const NetworkState_Impl& state, const TickValue& tick_value) {
      STATE_MAP<NetworkState_Impl, TickValue>::iterator iter = mp.find(state);
      if (iter == mp.end()) {
	mp[state] = tick_value;
      } else {
	TickValue& to_tick_value = (*iter).second;
	to_tick_value.tm_slice += tick_value.tm_slice;
	if (COMPUTE_ERRORS) {
	  to_tick_value.tm_slice_square += tick_value.tm_slice_square;
	}
	to_tick_value.TH += tick_value.TH;
      }
    }
    
#ifdef MPI_COMPAT
    size_t my_MPI_Size() {
      return sizeof(size_t) + size() * (sizeof(double)*2 + NetworkState::my_MPI_Pack_Size());
    }

    void my_MPI_Pack(void* buff, unsigned int size_pack, int* position) {
       // First, the cumulMap
      // and first, it's size
      size_t s_cumulMap = size();
      MPI_Pack(&s_cumulMap, 1, my_MPI_SIZE_T, buff, size_pack, position, MPI_COMM_WORLD);

      CumulMap::Iterator t_iterator = iterator();
      
      TickValue t_tick_value;
      while ( t_iterator.hasNext()) {

        const NetworkState_Impl& state = t_iterator.next2(t_tick_value);        
        NetworkState t_state(state);
        
        t_state.my_MPI_Pack(buff, size_pack, position);

        MPI_Pack(&(t_tick_value.tm_slice), 1, MPI_DOUBLE, buff, size_pack, position, MPI_COMM_WORLD);
        MPI_Pack(&(t_tick_value.TH), 1, MPI_DOUBLE, buff, size_pack, position, MPI_COMM_WORLD);

      }   
    }

    void my_MPI_Unpack(void* buff, unsigned int buff_size, int* position) 
    {
      size_t s_cumulMap;
      MPI_Unpack(buff, buff_size, position, &s_cumulMap, 1, my_MPI_SIZE_T, MPI_COMM_WORLD);

      for (size_t j=0; j < s_cumulMap; j++) {
        
        NetworkState t_state;
        t_state.my_MPI_Unpack(buff, buff_size, position);
     
        double t_tm_slice;
        MPI_Unpack(buff, buff_size, position, &t_tm_slice, 1, MPI_DOUBLE, MPI_COMM_WORLD);
     
        double t_TH;
        MPI_Unpack(buff, buff_size, position, &t_TH, 1, MPI_DOUBLE, MPI_COMM_WORLD);

        TickValue t_tick_value(t_tm_slice, t_TH);
        add(t_state.getState(), t_tick_value); 
      }

    }
    
    void my_MPI_Recv(int source) {
      
      size_t s_cumulMap;
      MPI_Recv(&s_cumulMap, 1, my_MPI_SIZE_T, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      for (size_t j=0; j < s_cumulMap; j++) {
        
        NetworkState t_state;
        t_state.my_MPI_Recv(source);
     
        double t_tm_slice;
        double t_TH;
        MPI_Recv(&t_tm_slice, 1, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&t_TH, 1, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        TickValue t_tick_value(t_tm_slice, t_TH);
        add(t_state.getState(), t_tick_value); 
      }
      // std::cout << "Finished receiving cumulMap" << std::endl;
    }
    
    void my_MPI_Send(int dest) {
      // First, the cumulMap
      // and first, it's size
      size_t s_cumulMap = size();
      // std::cout << "Will send a cumul map of size " << s_cumulMap << std::endl;
      MPI_Send(&s_cumulMap, 1, my_MPI_SIZE_T, dest, 0, MPI_COMM_WORLD);

      CumulMap::Iterator t_iterator = iterator();
      
      // NetworkState t_state;
      TickValue t_tick_value;
      while ( t_iterator.hasNext()) {

        const NetworkState_Impl& state = t_iterator.next2(t_tick_value);
        
        NetworkState t_state(state);
        t_state.my_MPI_Send(dest);
        
        // MPI_Send(&t_state, 1, MPI_UNSIGNED_LONG_LONG, dest, 0, MPI_COMM_WORLD);
      
        // TickValue t_tick_value = entry.second;
        MPI_Send(&(t_tick_value.tm_slice), 1, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
        MPI_Send(&(t_tick_value.TH), 1, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);

      }   
      // std::cout << "Finished sending cumulMap" << std::endl;
    }
#endif

    class Iterator {
    
      const CumulMap& cumul_map;
      STATE_MAP<NetworkState_Impl, TickValue>::const_iterator iter, end;

    public:
      Iterator(const CumulMap& cumul_map) : cumul_map(cumul_map) {
	iter = cumul_map.mp.begin();
	end = cumul_map.mp.end();
      }
	
      void rewind() {
	iter = cumul_map.mp.begin();
      }

      bool hasNext() const {
	return iter != end;
      }

      void next(NetworkState_Impl& state, TickValue& tick_value) {
	state = (*iter).first;
	tick_value = (*iter).second;
	++iter;
      }
	
      const NetworkState_Impl& next2(TickValue& tick_value) {
	tick_value = (*iter).second;
	return (*iter++).first;
      }
	
      void next(TickValue& tick_value) {
	tick_value = (*iter).second;
	++iter;
      }
    };

    Iterator iterator() {return Iterator(*this);}
    Iterator iterator() const {return Iterator(*this);}
  };
  class HDCumulMap {
    STATE_MAP<NetworkState_Impl, double> mp;

  public:
    size_t size() const {
      return mp.size();
    }

    void incr(const NetworkState_Impl& fullstate, double tm_slice) {
      STATE_MAP<NetworkState_Impl, double>::iterator iter = mp.find(fullstate);
      if (iter == mp.end()) {
	mp[fullstate] = tm_slice;
      } else {
	(*iter).second += tm_slice;
      }
    }

    void add(const NetworkState_Impl& fullstate, double tm_slice) {
      STATE_MAP<NetworkState_Impl, double>::iterator iter = mp.find(fullstate);
      if (iter == mp.end()) {
	mp[fullstate] = tm_slice;
      } else {
	(*iter).second += tm_slice;
      }
    }

#ifdef MPI_COMPAT
    size_t my_MPI_Size() {
      return sizeof(size_t) + size() * (sizeof(double) + NetworkState::my_MPI_Pack_Size());
    }
    
    void my_MPI_Pack(void* buff, unsigned int size_pack, int* position) 
    {
      size_t s_HDCumulMap = size();
      MPI_Pack(&s_HDCumulMap, 1, my_MPI_SIZE_T, buff, size_pack, position, MPI_COMM_WORLD);

      HDCumulMap::Iterator t_hd_iterator = iterator();
      
      double tm_slice;
      while ( t_hd_iterator.hasNext()) {

        const NetworkState_Impl& state = t_hd_iterator.next2(tm_slice);
        NetworkState t_state(state);
      
        t_state.my_MPI_Pack(buff, size_pack, position);
        MPI_Pack(&tm_slice, 1, MPI_DOUBLE, buff, size_pack, position, MPI_COMM_WORLD);
      }
    }
    
    void my_MPI_Unpack(void* buff, unsigned int buff_size, int* position) 
    {
      // First we need the size
      size_t s_HDCumulMap;
      MPI_Unpack(buff, buff_size, position, &s_HDCumulMap, 1, my_MPI_SIZE_T, MPI_COMM_WORLD);
      
      for (size_t j=0; j < s_HDCumulMap; j++) {
        
        NetworkState t_state;
        t_state.my_MPI_Unpack(buff, buff_size, position);
      
        double t_tm_slice;
        MPI_Unpack(buff, buff_size, position, &t_tm_slice, 1, MPI_DOUBLE, MPI_COMM_WORLD);
      
        add(t_state.getState(), t_tm_slice);  
      }
    }
    
    void my_MPI_Recv(int source) {
      // First we need the size
      size_t s_HDCumulMap;
      MPI_Recv(&s_HDCumulMap, 1, my_MPI_SIZE_T, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      for (size_t j=0; j < s_HDCumulMap; j++) {
        
        NetworkState t_state;
        t_state.my_MPI_Recv(source);

        double t_tm_slice;
        MPI_Recv(&t_tm_slice, 1, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        add(t_state.getState(), t_tm_slice);  
      }
    }
    
    void my_MPI_Send(int dest) {
      size_t s_HDCumulMap = size();
      MPI_Send(&s_HDCumulMap, 1, my_MPI_SIZE_T, dest, 0, MPI_COMM_WORLD);

      HDCumulMap::Iterator t_hd_iterator = iterator();
      
      // NetworkState t_state;
      double tm_slice;
      while ( t_hd_iterator.hasNext()) {

        const NetworkState_Impl& state = t_hd_iterator.next2(tm_slice);
        
        NetworkState t_state(state);
        t_state.my_MPI_Send(dest);
        MPI_Send(&tm_slice, 1, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);

      }
    }
    
#endif

    class Iterator {
    
      const HDCumulMap& hd_cumul_map;
      STATE_MAP<NetworkState_Impl, double>::const_iterator iter, end;

    public:
      Iterator(const HDCumulMap& hd_cumul_map) : hd_cumul_map(hd_cumul_map) {
	iter = hd_cumul_map.mp.begin();
	end = hd_cumul_map.mp.end();
      }
	
      void rewind() {
	iter = hd_cumul_map.mp.begin();
      }

      bool hasNext() const {
	return iter != end;
      }

      void next(NetworkState_Impl& state, double& tm_slice) {
	state = (*iter).first;
	tm_slice = (*iter).second;
	++iter;
      }
	
      const NetworkState_Impl& next2(double& tm_slice) {
	tm_slice = (*iter).second;
	return (*iter++).first;
      }

      void next(double& tm_slice) {
	tm_slice = (*iter).second;
	++iter;
      }
    };

    Iterator iterator() {return Iterator(*this);}
    Iterator iterator() const {return Iterator(*this);}
  };

  RunConfig* runconfig;
  double time_tick;
  unsigned int sample_count;
  unsigned int sample_num;
  double last_tm;
  int tick_index;
  std::vector<double> H_v;
  std::vector<double> TH_v;
  std::vector<MAP<unsigned int, double> > HD_v;
  std::vector<double> TH_square_v;
  unsigned int maxcols;
  int max_size;
  int max_tick_index;
  NetworkState_Impl output_mask;
  std::vector<CumulMap> cumul_map_v;
  std::vector<HDCumulMap> hd_cumul_map_v;
  unsigned int statdist_trajcount;
  NetworkState_Impl refnode_mask;
  std::vector<ProbaDist> proba_dist_v;
  ProbaDist curtraj_proba_dist;
  STATE_MAP<NetworkState_Impl, LastTickValue> last_tick_map;
  bool tick_completed;

  CumulMap& get_map() {
    assert((size_t)tick_index < cumul_map_v.size());
    return cumul_map_v[tick_index];
  }

  CumulMap& get_map(unsigned int nn) {
    assert(nn < cumul_map_v.size());
    return cumul_map_v[nn];
  }

  const CumulMap& get_map(unsigned int nn) const {
    assert(nn < cumul_map_v.size());
    return cumul_map_v[nn];
  }

  HDCumulMap& get_hd_map() {
    assert((size_t)tick_index < hd_cumul_map_v.size());
    return hd_cumul_map_v[tick_index];
  }

  HDCumulMap& get_hd_map(unsigned int nn) {
    assert(nn < hd_cumul_map_v.size());
    return hd_cumul_map_v[nn];
  }

  const HDCumulMap& get_hd_map(unsigned int nn) const {
    assert(nn < hd_cumul_map_v.size());
    return hd_cumul_map_v[nn];
  }

#ifdef MPI_COMPAT

static void MPI_Send_Cumulator(Cumulator* ret_cumul, int dest);
static void MPI_Recv_Cumulator(Cumulator* mpi_ret_cumul, int origin);
static Cumulator* initializeMPICumulator(Cumulator* ret_cumul, RunConfig* runconfig, int world_rank);

#endif
  double cumultime(int at_tick_index = -1) {
    if (at_tick_index < 0) {
      at_tick_index = tick_index;
    }
    return at_tick_index * time_tick;
  }

  bool incr(const NetworkState_Impl& state, double tm_slice, double TH, const NetworkState_Impl& fullstate) {
    if (tm_slice == 0.0) {
      return true;
    }

    if (sample_num < statdist_trajcount) {
      curtraj_proba_dist.incr(fullstate, tm_slice);
    }
    if (tick_index >= max_size) {
      return false;
    }
    tick_completed = false;
    CumulMap& mp = get_map();
    mp.incr(state, tm_slice, TH);

    HDCumulMap& hd_mp = get_hd_map();
    hd_mp.incr(fullstate, tm_slice);

    STATE_MAP<NetworkState_Impl, LastTickValue>::iterator last_tick_iter = last_tick_map.find(state);
    if (last_tick_iter == last_tick_map.end()) {
      last_tick_map[state] = LastTickValue(tm_slice, tm_slice * TH);
    } else {
      (*last_tick_iter).second.tm_slice += tm_slice;
      (*last_tick_iter).second.TH += tm_slice * TH;
    }

    return true;
  }

  void check() const;

  void add(unsigned int where, const CumulMap& add_cumul_map);
  void add(unsigned int where, const HDCumulMap& add_hd_cumul_map);
  
public:

  Cumulator(RunConfig* runconfig, double time_tick, double max_time, unsigned int sample_count, unsigned int statdist_trajcount) :
    runconfig(runconfig), time_tick(time_tick), sample_count(sample_count), sample_num(0), last_tm(0.), tick_index(0), statdist_trajcount(statdist_trajcount) {
#ifdef USE_STATIC_BITSET
    output_mask.set();
    refnode_mask.reset();
#elif defined(USE_BOOST_BITSET) || defined(USE_DYNAMIC_BITSET)
    // EV: 2020-10-23
    //output_mask.resize(MAXNODES);
    output_mask.resize(Network::getMaxNodeSize());
    output_mask.set();
    refnode_mask.resize(Network::getMaxNodeSize());
    refnode_mask.reset();
#else
    output_mask = ~0ULL;
    refnode_mask = 0ULL;
#endif
    max_size = (int)(max_time/time_tick)+2;
    max_tick_index = max_size;
    cumul_map_v.resize(max_size);
    hd_cumul_map_v.resize(max_size);

    if (COMPUTE_ERRORS) {
      TH_square_v.resize(max_size);
      for (int nn = 0; nn < max_size; ++nn) {
	TH_square_v[nn] = 0.;
      }
    }
    proba_dist_v.resize(statdist_trajcount);
    tick_completed = false;
  }

  void rewind() {
    if (last_tm) {
      computeMaxTickIndex();
    }

    tick_index = 0;
    last_tm = 0.;
    last_tick_map.clear();
    curtraj_proba_dist.clear();
    tick_completed = false;
  }

  void next() {
    if (tick_index < max_size) {
      STATE_MAP<NetworkState_Impl, LastTickValue>::iterator begin = last_tick_map.begin();
      STATE_MAP<NetworkState_Impl, LastTickValue>::iterator end = last_tick_map.end();
      CumulMap& mp = get_map();
      double TH = 0.0;
      while (begin != end) {
	//NetworkState_Impl state = (*begin).first;
	const NetworkState_Impl& state = (*begin).first;
	double tm_slice = (*begin).second.tm_slice;
	TH += (*begin).second.TH;
	if (COMPUTE_ERRORS) {
	  mp.cumulTMSliceSquare(state, tm_slice);
	}
	++begin;
      }
      if (COMPUTE_ERRORS) {
	TH_square_v[tick_index] += TH * TH;
      }
    }
    ++tick_index;
    tick_completed = true;
    last_tick_map.clear();
  }

  void cumul(const NetworkState& network_state, double tm, double TH) {
#ifdef USE_DYNAMIC_BITSET
    NetworkState_Impl fullstate(network_state.getState() & refnode_mask, 1);
#else
    NetworkState_Impl fullstate(network_state.getState() & refnode_mask);
#endif    
    NetworkState_Impl state(network_state.getState() & output_mask);
    double time_1 = cumultime(tick_index+1);
    if (tm < time_1) {
      incr(state, tm - last_tm, TH, fullstate);
      last_tm = tm;
      return;
    }

    if (!incr(state, time_1 - last_tm, TH, fullstate)) {
      last_tm = tm;
      return;
    }
    next();

    for (; cumultime(tick_index+1) < tm; next()) {
      if (!incr(state, time_tick, TH, fullstate)) {
	last_tm = tm;
	return;
      }
    }
      
    incr(state, tm - cumultime(), TH, fullstate);
    last_tm = tm;
  }

  void setOutputMask(const NetworkState_Impl& output_mask) {
    this->output_mask = output_mask;
  }
  
  void setRefnodeMask(const NetworkState_Impl& refnode_mask) {
    this->refnode_mask = refnode_mask;
  }

  void displayProbTraj(Network* network, unsigned int refnode_count, ProbTrajDisplayer* displayer) const;
  void displayStatDist(Network* network, unsigned int refnode_count, StatDistDisplayer* displayer) const;
  void displayAsymptoticCSV(Network* network, unsigned int refnode_count, std::ostream& os_asymptprob = std::cout, bool hexfloat = false, bool proba = true) const;


#ifdef PYTHON_API

  PyObject* getNumpyStatesDists(Network* network) const;
  PyObject* getNumpyLastStatesDists(Network* network) const;
  std::set<NetworkState_Impl> getStates() const;
  std::vector<NetworkState_Impl> getLastStates() const;
  PyObject* getNumpyNodesDists(Network* network, std::vector<Node*> output_nodes) const;
  PyObject* getNumpyLastNodesDists(Network* network, std::vector<Node*> output_nodes) const;
  std::vector<Node*> getNodes(Network* network) const;
  
#endif
  const std::map<double, STATE_MAP<NetworkState_Impl, double> > getStateDists() const;
  const STATE_MAP<NetworkState_Impl, double> getNthStateDist(int nn) const;
  const STATE_MAP<NetworkState_Impl, double> getAsymptoticStateDist() const;
  const double getFinalTime() const;

  void computeMaxTickIndex();
  int getMaxTickIndex() const { return max_tick_index;} 

  void epilogue(Network* network, const NetworkState& reference_state);
  void trajectoryEpilogue();

  unsigned int getSampleCount() const {return sample_count;}

  static Cumulator* mergeCumulatorsParallel(RunConfig* runconfig, std::vector<Cumulator*>& cumulator_v);
  static void mergePairOfCumulators(Cumulator* cumulator_1, Cumulator* cumulator_2);
  static void* threadMergeCumulatorWrapper(void *arg);

#ifdef MPI_COMPAT
  static Cumulator* mergeMPICumulators(RunConfig* runconfig, Cumulator* ret_cumul, int world_size, int world_rank, bool pack=true);
  static size_t MPI_Size_Cumulator(Cumulator* ret_cumul);
  static char* MPI_Pack_Cumulator(Cumulator* ret_cumul, int dest, unsigned int * buff_size);
  static void MPI_Unpack_Cumulator(Cumulator* mpi_ret_cumul, char* buff, unsigned int buff_size);

#endif

};

#endif
