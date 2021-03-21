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
     PopCumulator.cc

   Authors:
     Vincent Noël <vincent.noel@curie.fr>
 
   Date:
     March 2021
*/

#include "BooleanNetwork.h"
#include "PopCumulator.h"
#include "RunConfig.h"
#include "PopProbTrajDisplayer.h"
#include "Utils.h"
#include <sstream>
#include <iomanip>
#include <math.h>
#include <float.h>

double distance(const STATE_MAP<PopNetworkState_Impl, double, PopNetworkState_ImplHash, PopNetworkState_ImplEquality>& proba_dist1, const STATE_MAP<PopNetworkState_Impl, double, PopNetworkState_ImplHash, PopNetworkState_ImplEquality>& proba_dist2)
{
  return 0.;
}

void PopCumulator::check() const
{
  // check that for each tick (except the last one), the sigma of each map == 1.
  for (int nn = 0; nn < max_tick_index; ++nn) {
    const PopCumulMap& mp = get_map(nn);
    PopCumulMap::Iterator iter = mp.iterator();
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

void PopCumulator::trajectoryEpilogue()
{
  assert(sample_num < sample_count);

  PopProbaDist::Iterator curtraj_proba_dist_iter = curtraj_proba_dist.iterator();

  double proba_max_time = 0.;
  while (curtraj_proba_dist_iter.hasNext()) {
    double tm_slice;
    curtraj_proba_dist_iter.next(tm_slice);
    proba_max_time += tm_slice;
  }

  //std::cout << "Trajepilogue #" << (sample_num+1) << " " << proba_max_time << '\n';
  double proba = 0;
  curtraj_proba_dist_iter.rewind();

  PopProbaDist& proba_dist = proba_dist_v[sample_num++];
  while (curtraj_proba_dist_iter.hasNext()) {
    PopNetworkState_Impl state;
    double tm_slice;
    curtraj_proba_dist_iter.next(state, tm_slice);
    //assert(proba_dist.find(state) == proba_dist.end());
    double new_tm_slice = tm_slice / proba_max_time;
    proba_dist.set(state, new_tm_slice);
    proba += new_tm_slice;
  }

  assert(proba >= 0.9999 && proba <= 1.0001);
}

void PopCumulator::computeMaxTickIndex()
{
  /*
  unsigned int tmp_tick_index = tick_index + !tick_completed;
  if (max_tick_index > tmp_tick_index) {
    max_tick_index = tmp_tick_index;
  }
  */
  if (max_tick_index > tick_index) {
    max_tick_index = tick_index;
  }
}

void PopCumulator::epilogue(Network* network, const PopNetworkState& reference_state)
{
  computeMaxTickIndex();

  //check();

  // compute H (Entropy), TH (Transition entropy) and HD (Hamming distance)
  H_v.resize(max_tick_index);
  TH_v.resize(max_tick_index);
#ifndef HD_BUG
  HD_v.resize(max_tick_index);
#endif

  maxcols = 0;
  double ratio = time_tick * sample_count;
  for (int nn = 0; nn < max_tick_index; ++nn) { // time tick
    const PopCumulMap& mp = get_map(nn);
    PopCumulMap::Iterator iter = mp.iterator();
    H_v[nn] = 0.;
    TH_v[nn] = 0.;
#ifndef HD_BUG
    MAP<unsigned int, double>& hd_m = HD_v[nn];
#endif
    while (iter.hasNext()) {
      TickValue tick_value;
#ifdef USE_NEXT_OPT
      const PopNetworkState_Impl &state = iter.next2(tick_value);
#else
      PopNetworkState_Impl state;
      iter.next(state, tick_value);
#endif
      double tm_slice = tick_value.tm_slice;
      double proba = tm_slice / ratio;      
      double TH = tick_value.TH / sample_count;
      H_v[nn] += -log2(proba) * proba;
#ifndef HD_BUG
      int hd = reference_state.hamming(network, state);
      if (hd_m.find(hd) == hd_m.end()) {
	hd_m[hd] = proba;
      } else {
	hd_m[hd] += proba;
      }
#endif
      TH_v[nn] += TH;
    }
    TH_v[nn] /= time_tick;
    if (mp.size() > maxcols) {
      maxcols = mp.size();
    }
  }

#ifdef HD_BUG
  HD_v.resize(max_tick_index);

  for (int nn = 0; nn < max_tick_index; ++nn) { // time tick
    const HDPopCumulMap& hd_mp = get_hd_map(nn);
    HDPopCumulMap::Iterator iter = hd_mp.iterator();
    MAP<unsigned int, double>& hd_m = HD_v[nn];
    while (iter.hasNext()) {
      double tm_slice;
#ifdef USE_NEXT_OPT
      const PopNetworkState_Impl &state = iter.next2(tm_slice);
#else
      PopNetworkState_Impl state;
      iter.next(state, tm_slice);
#endif
      double proba = tm_slice / ratio;      
  //     int hd = reference_state.hamming(network, state);
  //     if (hd_m.find(hd) == hd_m.end()) {
	// hd_m[hd] = proba;
  //     } else {
	// hd_m[hd] += proba;
  //     }
    }
  }
#endif
}

void PopCumulator::displayPopProbTraj(Network* network, unsigned int refnode_count, PopProbTrajDisplayer* displayer) const
{
  std::vector<Node*>::const_iterator begin_network;

  displayer->begin(POP_COMPUTE_ERRORS, maxcols, refnode_count);

  double time_tick2 = time_tick * time_tick;
  double ratio = time_tick*sample_count;
  for (int nn = 0; nn < max_tick_index; ++nn) {
    displayer->beginTimeTick(nn*time_tick);
    // TH
    const PopCumulMap& mp = get_map(nn);
    PopCumulMap::Iterator iter = mp.iterator();
    displayer->setTH(TH_v[nn]);

    // ErrorTH
    //assert((size_t)nn < TH_square_v.size());
    if (POP_COMPUTE_ERRORS) {
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
      const PopNetworkState_Impl& state = iter.next2(tick_value);
#else
      PopNetworkState_Impl state;
      iter.next(state, tick_value);
#endif
      double proba = tick_value.tm_slice / ratio;      
      if (POP_COMPUTE_ERRORS) {
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

void PopCumulator::add(unsigned int where, const PopCumulMap& add_cumul_map)
{
  PopCumulMap& to_cumul_map = get_map(where);

  PopCumulMap::Iterator iter = add_cumul_map.iterator();
  while (iter.hasNext()) {
    TickValue tick_value;
#ifdef USE_NEXT_OPT
    const PopNetworkState_Impl& state = iter.next2(tick_value);
#else
    PopNetworkState_Impl state;
    iter.next(state, tick_value);
#endif
    to_cumul_map.add(state, tick_value);
  }
}

#ifdef HD_BUG
void PopCumulator::add(unsigned int where, const HDPopCumulMap& add_hd_cumul_map)
{
  HDPopCumulMap& to_hd_cumul_map = get_hd_map(where);

  HDPopCumulMap::Iterator iter = add_hd_cumul_map.iterator();
  while (iter.hasNext()) {
    double tm_slice;
#ifdef USE_NEXT_OPT
    const PopNetworkState_Impl& state = iter.next2(tm_slice);
#else
    PopNetworkState_Impl state;
    iter.next(state, tm_slice);
#endif
    to_hd_cumul_map.add(state, tm_slice);
  }
}
#endif

PopCumulator* PopCumulator::mergePopCumulators(RunConfig* runconfig, std::vector<PopCumulator*>& cumulator_v)
{
  std::cout << "Merging cumulators" << std::endl;
  size_t size = cumulator_v.size();
  if (1 == size) {
    PopCumulator* cumulator = cumulator_v[0];
    return new PopCumulator(*cumulator);
  }

  unsigned int t_cumulator_size = 0;
  for (auto& cum: cumulator_v) {
    t_cumulator_size += cum->sample_count;
  }

  PopCumulator* ret_cumul = new PopCumulator(runconfig, runconfig->getTimeTick(), runconfig->getMaxTime(), t_cumulator_size);
  size_t min_cumul_size = ~0ULL;
  size_t min_tick_index_size = ~0ULL;
  std::vector<PopCumulator*>::iterator begin = cumulator_v.begin();
  std::vector<PopCumulator*>::iterator end = cumulator_v.end();
  while (begin != end) {
    PopCumulator* cumulator = *begin;
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
#ifdef HD_BUG
  ret_cumul->hd_cumul_map_v.resize(min_cumul_size);
#endif
  ret_cumul->max_tick_index = ret_cumul->tick_index = min_tick_index_size;

  begin = cumulator_v.begin();
  unsigned int rr = 0;
  for (unsigned int jj = 0; begin != end; ++jj) {
    PopCumulator* cumulator = *begin;
    for (unsigned int nn = 0; nn < min_cumul_size; ++nn) {
      ret_cumul->add(nn, cumulator->cumul_map_v[nn]);
#ifdef HD_BUG
      ret_cumul->add(nn, cumulator->hd_cumul_map_v[nn]);
#endif
      ret_cumul->TH_square_v[nn] += cumulator->TH_square_v[nn];
    }
    
    // Commenting for now
    unsigned int proba_dist_size = cumulator->proba_dist_v.size();
    for (unsigned int ii = 0; ii < proba_dist_size; ++ii) {
      assert(ret_cumul->proba_dist_v.size() > rr);
      ret_cumul->proba_dist_v[rr++] = cumulator->proba_dist_v[ii];
    }
    ++begin;
  }
  return ret_cumul;
}

