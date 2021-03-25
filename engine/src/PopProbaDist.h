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
    PopProbaDist.h

   Authors:
     Vincent Noël <vincent.noel@curie.fr>
 
   Date:
     March 2021
*/

#ifndef _POPPROBADIST_H_
#define _POPPROBADIST_H_

#include <assert.h>
#include <string>
#include <map>
#include <vector>
#include <iostream>

#include "BooleanNetwork.h"
// class StatDistDisplayer;

#define CLUSTER_OPTIM

class PopProbaDist {
 
  STATE_MAP<PopNetworkState_Impl, double> mp;

 public:
 
 PopProbaDist() {
   mp = STATE_MAP<PopNetworkState_Impl, double>();
 }

  size_t size() const {
    return mp.size();
  }

  void incr(const PopNetworkState_Impl& state, double tm_slice) {
    STATE_MAP<PopNetworkState_Impl, double>::iterator proba_iter = mp.find(state);
    if (proba_iter == mp.end()) {
      mp[state] = tm_slice;
    } else {
      (*proba_iter).second += tm_slice;
    }
  }

  void clear() {
    mp.clear();
  }

  void set(const PopNetworkState_Impl& state, double tm_slice) {
    mp[state] = tm_slice;
  }

  bool hasState(const PopNetworkState_Impl& state, double& tm_slice) const {
    STATE_MAP<PopNetworkState_Impl, double>::const_iterator iter = mp.find(state);
    if (iter != mp.end()) {
      tm_slice = (*iter).second;
      return true;
    }
    return false;
  }

  class Iterator {
    
    const PopProbaDist& proba_dist_map;
    STATE_MAP<PopNetworkState_Impl, double>::const_iterator iter, end;

  public:
  Iterator(const PopProbaDist& proba_dist_map) : proba_dist_map(proba_dist_map) {
      iter = proba_dist_map.mp.begin();
      end = proba_dist_map.mp.end();
    }
	
    void rewind() {
      iter = proba_dist_map.mp.begin();
    }

    bool hasNext() const {
      return iter != end;
    }

    void next(PopNetworkState_Impl& state, double& tm_slice) {
      state = (*iter).first;
      tm_slice = (*iter).second;
      ++iter;
    }

    void next(PopNetworkState_Impl& state) {
      state = (*iter).first;
      ++iter;
    }

    const PopNetworkState_Impl& next2(double& tm_slice) {
      tm_slice = (*iter).second;
      return (*iter++).first;
    }

    const PopNetworkState_Impl& next2() {
      return (*iter++).first;
    }

    void next(double& tm_slice) {
      tm_slice = (*iter).second;
      ++iter;
    }
  };	

  void display(std::ostream& os, Network* network, bool hexfloat) const;

  Iterator iterator() {return Iterator(*this);}
  Iterator iterator() const {return Iterator(*this);}
};

#endif
