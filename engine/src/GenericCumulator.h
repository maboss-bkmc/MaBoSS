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
     GenericCumulator.h

   Authors:
     Eric Viara <viara@sysra.com>
     Gautier Stoll <gautier.stoll@curie.fr>
     Vincent Noël <vincent.noel@curie.fr>
 
   Date:
     January-March 2011
*/

#ifndef _GENERIC_CUMULATOR_H_
#define _GENERIC_CUMULATOR_H_

#define USE_NEXT_OPT

#include <string>
#include <map>
#include <set>
#include <vector>
#include <iostream>
#include <assert.h>
#include <cfloat>

#ifdef PYTHON_API
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL MABOSS_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#endif

static bool COMPUTE_ERRORS = true;

#define HD_BUG

#include "RunConfig.h"
#include "ProbaDist.h"
#include "StatDistDisplayer.h"
#include "ProbTrajDisplayer.h"

class Network;
template <class S> class ProbTrajDisplayer;
// class StatDistDisplayer;

template <class S>
class GenericCumulator {

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
    STATE_MAP<S, TickValue> mp;

  public:
    size_t size() const {
      return mp.size();
    }

    void incr(const S& state, double tm_slice, double TH) {
      typename STATE_MAP<S, TickValue>::iterator iter = mp.find(state);
      if (iter == mp.end()) {
	mp[state] = TickValue(tm_slice, tm_slice * TH);
      } else {
	(*iter).second.tm_slice += tm_slice;
	(*iter).second.TH += tm_slice * TH;
      }
    }

    void cumulTMSliceSquare(const S& state, double tm_slice) {
      typename STATE_MAP<S, TickValue>::iterator iter = mp.find(state);
      assert(iter != mp.end());
      (*iter).second.tm_slice_square += tm_slice * tm_slice;
    }
    
    void add(const S& state, const TickValue& tick_value) {
      typename STATE_MAP<S, TickValue>::iterator iter = mp.find(state);
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

    class Iterator {
    
      const CumulMap& cumul_map;
      typename STATE_MAP<S, TickValue>::const_iterator iter, end;

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

      void next(S& state, TickValue& tick_value) {
	state = (*iter).first;
	tick_value = (*iter).second;
	++iter;
      }
	
#ifdef USE_NEXT_OPT
      const S& next2(TickValue& tick_value) {
	tick_value = (*iter).second;
	return (*iter++).first;
      }
#endif
	
      void next(TickValue& tick_value) {
	tick_value = (*iter).second;
	++iter;
      }
    };

    Iterator iterator() {return Iterator(*this);}
    Iterator iterator() const {return Iterator(*this);}
  };

// #ifdef HD_BUG
  class HDCumulMap {
    typename STATE_MAP<S, double> mp;

  public:
    void incr(const S& fullstate, double tm_slice) {
      typename STATE_MAP<S, double>::iterator iter = mp.find(fullstate);
      if (iter == mp.end()) {
	mp[fullstate] = tm_slice;
      } else {
	(*iter).second += tm_slice;
      }
    }

    void add(const S& fullstate, double tm_slice) {
      typename STATE_MAP<S, double>::iterator iter = mp.find(fullstate);
      if (iter == mp.end()) {
	mp[fullstate] = tm_slice;
      } else {
	(*iter).second += tm_slice;
      }
    }

    class Iterator {
    
      const HDCumulMap& hd_cumul_map;
      typename STATE_MAP<S, double>::const_iterator iter, end;

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

      void next(S& state, double& tm_slice) {
	state = (*iter).first;
	tm_slice = (*iter).second;
	++iter;
      }
	
#ifdef USE_NEXT_OPT
      const S& next2(double& tm_slice) {
	tm_slice = (*iter).second;
	return (*iter++).first;
      }
#endif

      void next(double& tm_slice) {
	tm_slice = (*iter).second;
	++iter;
      }
    };

    Iterator iterator() {return Iterator(*this);}
    Iterator iterator() const {return Iterator(*this);}
  };
// #endif
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
  NetworkState output_mask;
  std::vector<CumulMap> cumul_map_v;
// #ifdef HD_BUG
  std::vector<HDCumulMap> hd_cumul_map_v;
// #endif
  std::vector<ProbaDist<S> > proba_dist_v;
  ProbaDist<S> curtraj_proba_dist;
  STATE_MAP<S, LastTickValue> last_tick_map;
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

// #ifdef HD_BUG
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
// #endif

  double cumultime(int at_tick_index = -1) {
    if (at_tick_index < 0) {
      at_tick_index = tick_index;
    }
    return at_tick_index * time_tick;
  }

  bool incr(const S& state, double tm_slice, double TH, const S& fullstate) {
    if (tm_slice == 0.0) {
      return true;
    }

    curtraj_proba_dist.incr(S(fullstate), tm_slice);

    if (tick_index >= max_size) {
      return false;
    }
    tick_completed = false;
    CumulMap& mp = get_map();
    mp.incr(state, tm_slice, TH);
// #ifdef HD_BUG
    HDCumulMap& hd_mp = get_hd_map();
    hd_mp.incr(fullstate, tm_slice);
// #endif

    typename STATE_MAP<S, LastTickValue>::iterator last_tick_iter = last_tick_map.find(state);
    if (last_tick_iter == last_tick_map.end()) {
      last_tick_map[state] = LastTickValue(tm_slice, tm_slice * TH);
    } else {
      (*last_tick_iter).second.tm_slice += tm_slice;
      (*last_tick_iter).second.TH += tm_slice * TH;
    }

    return true;
  }

  void check() const {
    // check that for each tick (except the last one), the sigma of each map == 1.
    for (int nn = 0; nn < max_tick_index; ++nn) {
    const CumulMap& mp = get_map(nn);
    typename CumulMap::Iterator iter = mp.iterator();
    double sum = 0.;
    while (iter.hasNext()) {
      TickValue tick_value;
      iter.next(tick_value);
      sum += tick_value.tm_slice;
    }
    sum /= time_tick*sample_count;
    assert(sum >= 1. - 0.0001 && sum <= 1. + 0.0001);
  }

  }

  void add(unsigned int where, const CumulMap& add_cumul_map) 
  {
    CumulMap& to_cumul_map = get_map(where);

    typename CumulMap::Iterator iter = add_cumul_map.iterator();
    while (iter.hasNext()) {
      TickValue tick_value;
  #ifdef USE_NEXT_OPT
      const S& state = iter.next2(tick_value);
  #else
      S state;
      iter.next(state, tick_value);
  #endif
      to_cumul_map.add(state, tick_value);
    }

  }
// #ifdef HD_BUG
  void add(unsigned int where, const HDCumulMap& add_hd_cumul_map) 
  {
    HDCumulMap& to_hd_cumul_map = get_hd_map(where);

    typename HDCumulMap::Iterator iter = add_hd_cumul_map.iterator();
    while (iter.hasNext()) {
      double tm_slice;
  #ifdef USE_NEXT_OPT
      const S& state = iter.next2(tm_slice);
  #else
      S state;
      iter.next(state, tm_slice);
  #endif
      to_hd_cumul_map.add(state, tm_slice);
    }

  }
// #endif
  
public:

  GenericCumulator(RunConfig* runconfig, double time_tick, double max_time, unsigned int sample_count) :
    runconfig(runconfig), time_tick(time_tick), sample_count(sample_count), sample_num(0), last_tm(0.), tick_index(0) {
#ifdef USE_STATIC_BITSET
    output_mask.set();
#elif defined(USE_BOOST_BITSET) || defined(USE_DYNAMIC_BITSET)
    // EV: 2020-10-23
    //output_mask.resize(MAXNODES);
    output_mask.resize(Network::getMaxNodeSize());
    output_mask.set();
#else
    output_mask = ~0ULL;
#endif
    max_size = (int)(max_time/time_tick)+2;
    max_tick_index = max_size;
    cumul_map_v.resize(max_size);
// #ifdef HD_BUG
    hd_cumul_map_v.resize(max_size);
// #endif
    if (COMPUTE_ERRORS) {
      TH_square_v.resize(max_size);
      for (int nn = 0; nn < max_size; ++nn) {
	TH_square_v[nn] = 0.;
      }
    }
    proba_dist_v.resize(sample_count);
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
      typename STATE_MAP<S, LastTickValue>::iterator begin = last_tick_map.begin();
      typename STATE_MAP<S, LastTickValue>::iterator end = last_tick_map.end();
      CumulMap& mp = get_map();
      double TH = 0.0;
      while (begin != end) {
	//S state = (*begin).first;
	const S& state = (*begin).first;
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

  void cumul(const S& network_state, double tm, double TH) {
#ifdef USE_DYNAMIC_BITSET
    S fullstate(network_state, 1);
#else
    S fullstate(network_state);
#endif
    S state(fullstate & output_mask);
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

  void setOutputMask(const NetworkState& output_mask) {
    this->output_mask = output_mask;
  }

  void displayProbTraj(Network* network, unsigned int refnode_count, ProbTrajDisplayer<S>* displayer) const 
  {
    std::vector<Node*>::const_iterator begin_network;

    displayer->begin(COMPUTE_ERRORS, maxcols, refnode_count);

    double time_tick2 = time_tick * time_tick;
    double ratio = time_tick*sample_count;
    for (int nn = 0; nn < max_tick_index; ++nn) {
      displayer->beginTimeTick(nn*time_tick);
      // TH
      const CumulMap& mp = get_map(nn);
      typename CumulMap::Iterator iter = mp.iterator();
      displayer->setTH(TH_v[nn]);

      // ErrorTH
      //assert((size_t)nn < TH_square_v.size());
      if (COMPUTE_ERRORS) {
        double TH_square = TH_square_v[nn];
        double TH = TH_v[nn]; // == TH
        double variance_TH = (TH_square / ((sample_count-1) * time_tick2)) - (TH*TH*sample_count/(sample_count-1));
        double err_TH;
        double variance_TH_sample_count = variance_TH/sample_count;
        //assert(variance_TH > 0.0);
        if (variance_TH_sample_count >= 0.0) {
    err_TH = sqrt(variance_TH_sample_count);
        } else {
    err_TH = 0.;
        }
        displayer->setErrorTH(err_TH);
      }

      // H
      displayer->setH(H_v[nn]);

      std::string zero_hexfloat = fmthexdouble(0.0);
      // HD
      const MAP<unsigned int, double>& hd_m = HD_v[nn];
      for (unsigned int hd = 0; hd <= refnode_count; ++hd) { 
        MAP<unsigned int, double>::const_iterator hd_m_iter = hd_m.find(hd);
        if (hd_m_iter != hd_m.end()) {
    displayer->setHD(hd, hd_m_iter->second);
        } else {
    displayer->setHD(hd, 0.);
        }
      }

      // Proba, ErrorProba
      while (iter.hasNext()) {
        TickValue tick_value;
  #ifdef USE_NEXT_OPT
        const S& state = S(iter.next2(tick_value), 1);
        // Here I'm copying because it was a copy in the displayer
        // But do we really need it ?
        // NetworkState state(t_state, 1);
  #else
        S state;
        iter.next(state, tick_value);
        // NetworkState state(t_state);
  #endif
        double proba = tick_value.tm_slice / ratio;      
        if (COMPUTE_ERRORS) {
    double tm_slice_square = tick_value.tm_slice_square;
    double variance_proba = (tm_slice_square / ((sample_count-1) * time_tick2)) - (proba*proba*sample_count/(sample_count-1));
    double err_proba;
    double variance_proba_sample_count = variance_proba/sample_count;
    if (variance_proba_sample_count >= DBL_MIN) {
      err_proba = sqrt(variance_proba_sample_count);
    } else {
      err_proba = 0.;
    }
    displayer->addProba(state, proba, err_proba);
        } else {
    displayer->addProba(state, proba, 0.);
        }
      }
      displayer->endTimeTick();
    }
    displayer->end();
  }
  void displayStatDist(Network* network, unsigned int refnode_count, StatDistDisplayer* displayer) const {
    // should not be in cumulator, but somehwere in ProbaDist*

    // Probability distribution
    unsigned int statdist_traj_count = runconfig->getStatDistTrajCount();
    if (statdist_traj_count == 0) {
      return;
    }

    unsigned int max_size = 0;
    unsigned int cnt = 0;
    unsigned int proba_dist_size = proba_dist_v.size();
    for (unsigned int nn = 0; nn < proba_dist_size; ++nn) {
      const ProbaDist<S>& proba_dist = proba_dist_v[nn];
      if (proba_dist.size() > max_size) {
        max_size = proba_dist.size();
      }
      cnt++;
      if (cnt > statdist_traj_count) {
        break;
      }
    }

    displayer->begin(max_size, statdist_traj_count);
    cnt = 0;
    displayer->beginStatDistDisplay();
    for (unsigned int nn = 0; nn < proba_dist_size; ++nn) {
      const ProbaDist<S>& proba_dist = proba_dist_v[nn];
      displayer->beginStateProba(cnt+1);
      cnt++;

      proba_dist.display(displayer);
      displayer->endStateProba();
      if (cnt >= statdist_traj_count) {
        break;
      }
    }
    displayer->endStatDistDisplay();

    // should not be called from here, but from MaBestEngine
    // ProbaDistClusterFactory* clusterFactory = new ProbaDistClusterFactory(proba_dist_v, statdist_traj_count);
    // clusterFactory->makeClusters(runconfig);
    // clusterFactory->display(displayer);
    // clusterFactory->computeStationaryDistribution();
    // clusterFactory->displayStationaryDistribution(displayer);
    displayer->end();
    // delete clusterFactory;
  }
  void displayAsymptoticCSV(Network* network, unsigned int refnode_count, std::ostream& os_asymptprob = std::cout, bool hexfloat = false, bool proba = true) const
  {
    double ratio;
  if (proba)
  {
    ratio = time_tick * sample_count;
  }
  else
  {
    ratio = time_tick;
  }

  // Choosing the last tick
  int nn = max_tick_index - 1;

#ifdef HAS_STD_HEXFLOAT
  if (hexfloat)
  {
    os_asymptprob << std::hexfloat;
  }
#endif
  // TH
  const CumulMap &mp = get_map(nn);
  typename CumulMap::Iterator iter = mp.iterator();


  while (iter.hasNext())
  {
    TickValue tick_value;
#ifdef USE_NEXT_OPT
    const S& state = iter.next2(tick_value);
#else
    S state;
    iter.next(state, tick_value);
#endif
    double proba = tick_value.tm_slice / ratio;
    if (proba)
    {
      if (hexfloat)
      {
        os_asymptprob << std::setprecision(6) << fmthexdouble(proba);
      }
      else
      {
        os_asymptprob << std::setprecision(6) << proba;
      }
    }
    else
    {
      int t_proba = static_cast<int>(round(proba));
      os_asymptprob << std::fixed << t_proba;
    }

    os_asymptprob << '\t';
    // NetworkState network_state(state);
    state.displayOneLine(os_asymptprob, network);

    os_asymptprob << '\n';

  }
  }


// #ifdef PYTHON_API

//   PyObject* getNumpyStatesDists(Network* network) const;
//   PyObject* getNumpyLastStatesDists(Network* network) const;
//   std::set<S> getStates() const;
//   std::vector<S> getLastStates() const;
//   PyObject* getNumpyNodesDists(Network* network) const;
//   PyObject* getNumpyLastNodesDists(Network* network) const;
//   std::vector<Node*> getNodes(Network* network) const;
  
// #endif
//   const std::map<double, STATE_MAP<S, double> > getStateDists() const;
//   const STATE_MAP<S, double> getNthStateDist(int nn) const;
//   const STATE_MAP<S, double> getAsymptoticStateDist() const;
  const double getFinalTime() const {
      return time_tick*(getMaxTickIndex()-1);

  }

  void computeMaxTickIndex() {
    if (max_tick_index > tick_index) {
      max_tick_index = tick_index;
    }
  }
  
  int getMaxTickIndex() const { return max_tick_index;} 

  void epilogue(Network* network, const S& reference_state) 
  {
    computeMaxTickIndex();

    // compute H (Entropy), TH (Transition entropy) and HD (Hamming distance)
    H_v.resize(max_tick_index);
    TH_v.resize(max_tick_index);

    maxcols = 0;
    double ratio = time_tick * sample_count;
    for (int nn = 0; nn < max_tick_index; ++nn) { // time tick
      const CumulMap& mp = get_map(nn);
      typename CumulMap::Iterator iter = mp.iterator();
      H_v[nn] = 0.;
      TH_v[nn] = 0.;
      while (iter.hasNext()) {
        TickValue tick_value;
  #ifdef USE_NEXT_OPT
        const S &state = iter.next2(tick_value);
  #else
        S state;
        iter.next(state, tick_value);
  #endif
        double tm_slice = tick_value.tm_slice;
        double proba = tm_slice / ratio;      
        double TH = tick_value.TH / sample_count;
        H_v[nn] += -log2(proba) * proba;
        TH_v[nn] += TH;
      }
      TH_v[nn] /= time_tick;
      if (mp.size() > maxcols) {
        maxcols = mp.size();
      }
    }

    HD_v.resize(max_tick_index);

    for (int nn = 0; nn < max_tick_index; ++nn) { // time tick
      const HDCumulMap& hd_mp = get_hd_map(nn);
      typename HDCumulMap::Iterator iter = hd_mp.iterator();
      MAP<unsigned int, double>& hd_m = HD_v[nn];
      while (iter.hasNext()) {
        double tm_slice;
  #ifdef USE_NEXT_OPT
        const S &state = iter.next2(tm_slice);
  #else
        S state;
        iter.next(state, tm_slice);
  #endif
        double proba = tm_slice / ratio;      
        int hd = reference_state.hamming(network, state);
        if (hd_m.find(hd) == hd_m.end()) {
    hd_m[hd] = proba;
        } else {
    hd_m[hd] += proba;
        }
      }
    }
  }
  
  void trajectoryEpilogue() 
  {
    assert(sample_num < sample_count);

    typename ProbaDist<S>::Iterator curtraj_proba_dist_iter = curtraj_proba_dist.iterator();

    double proba_max_time = 0.;

    while (curtraj_proba_dist_iter.hasNext()) {
      double tm_slice;
      curtraj_proba_dist_iter.next(tm_slice);
      proba_max_time += tm_slice;
    }

    //std::cout << "Trajepilogue #" << (sample_num+1) << " " << proba_max_time << '\n';
    double proba = 0;
    curtraj_proba_dist_iter.rewind();

    ProbaDist<S>& proba_dist = proba_dist_v[sample_num++];
    while (curtraj_proba_dist_iter.hasNext()) {
      S state;
      double tm_slice;
      curtraj_proba_dist_iter.next(state, tm_slice);
      //assert(proba_dist.find(state) == proba_dist.end());
      double new_tm_slice = tm_slice / proba_max_time;
      proba_dist.set(state, new_tm_slice);
      proba += new_tm_slice;
    }

    assert(proba >= 0.9999 && proba <= 1.0001);
  }

  unsigned int getSampleCount() const {return sample_count;}

  static GenericCumulator* mergeCumulators(RunConfig* runconfig, std::vector<GenericCumulator*>& cumulator_v) {
    size_t size = cumulator_v.size();
    if (1 == size) {
      GenericCumulator* cumulator = cumulator_v[0];
      return new GenericCumulator(*cumulator);
    }

    unsigned int t_cumulator_size = 0;
    for (auto& cum: cumulator_v) {
      t_cumulator_size += cum->sample_count;
    }

    GenericCumulator* ret_cumul = new GenericCumulator(runconfig, runconfig->getTimeTick(), runconfig->getMaxTime(), t_cumulator_size);
    size_t min_cumul_size = ~0ULL;
    size_t min_tick_index_size = ~0ULL;
    typename std::vector<GenericCumulator*>::iterator begin = cumulator_v.begin();
    typename std::vector<GenericCumulator*>::iterator end = cumulator_v.end();
    while (begin != end) {
      GenericCumulator* cumulator = *begin;
      cumulator->computeMaxTickIndex();
      if (cumulator->cumul_map_v.size() < min_cumul_size) {
        min_cumul_size = cumulator->cumul_map_v.size();
      }
      if ((size_t)cumulator->max_tick_index < min_tick_index_size) {
        min_tick_index_size = cumulator->max_tick_index;
      }
      ++begin;
    }

    ret_cumul->cumul_map_v.resize(min_cumul_size);
  // #ifdef HD_BUG
    ret_cumul->hd_cumul_map_v.resize(min_cumul_size);
  // #endif
    ret_cumul->max_tick_index = ret_cumul->tick_index = min_tick_index_size;

    begin = cumulator_v.begin();
    unsigned int rr = 0;
    for (unsigned int jj = 0; begin != end; ++jj) {
      GenericCumulator* cumulator = *begin;
      for (unsigned int nn = 0; nn < min_cumul_size; ++nn) {
        ret_cumul->add(nn, cumulator->cumul_map_v[nn]);
  // #ifdef HD_BUG
        ret_cumul->add(nn, cumulator->hd_cumul_map_v[nn]);
  // #endif
        ret_cumul->TH_square_v[nn] += cumulator->TH_square_v[nn];
      }
      unsigned int proba_dist_size = cumulator->proba_dist_v.size();
      for (unsigned int ii = 0; ii < proba_dist_size; ++ii) {
        assert(ret_cumul->proba_dist_v.size() > rr);
        ret_cumul->proba_dist_v[rr++] = cumulator->proba_dist_v[ii];
      }
      ++begin;
    }
    return ret_cumul;
  }
};

#endif
