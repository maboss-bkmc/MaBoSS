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
     BooleanNetwork.h

   Authors:
     Eric Viara <viara@sysra.com>
     Gautier Stoll <gautier.stoll@curie.fr>
     Vincent Noël <vincent.noel@curie.fr>
 
   Date:
     January-March 2011
*/

#ifndef _NETWORKSTATE_IMPL_H
#define _NETWORKSTATE_IMPL_H

#ifdef USE_DYNAMIC_BITSET

  #undef MAXNODES
  #define MAXNODES 0xFFFFFFF

#elif MAXNODES>64

  #define USE_STATIC_BITSET

#endif

#ifdef USE_STATIC_BITSET
#include <bitset>
typedef std::bitset<MAXNODES> NetworkState_Impl;

// #ifdef USE_UNORDERED_MAP

namespace std {
//   template <> struct HASH_STRUCT<bitset<MAXNODES> >
//   {
//     size_t operator()(const bitset<MAXNODES>& val) const {
// #ifdef COMPARE_BITSET_AND_ULONG
//       return val.to_ulong();
// #else
//       static const bitset<MAXNODES> MASK(0xFFFFFFFFUL);
//       return (val & MASK).to_ulong();
// #endif
//     }
//   };

//   template <> struct equal_to<bitset<MAXNODES> >
//   {
//     size_t operator()(const bitset<MAXNODES>& val1, const bitset<MAXNODES>& val2) const {
//       return val1 == val2;
//     }
//   };

  // Added less operator, necessary for maps, sets. Code from https://stackoverflow.com/a/21245301/11713763
  template <> struct less<bitset<MAXNODES> >
  {
    size_t operator()(const bitset<MAXNODES>& val1, const bitset<MAXNODES>& val2) const {
    for (int i = MAXNODES-1; i >= 0; i--) {
        if (val1[i] ^ val2[i]) return val2[i];
    }
    return false;

    }
  };
}

// #else

// template <int N> class sbitset : public std::bitset<N> {

//  public:
//   sbitset() : std::bitset<N>() { }
//   sbitset(const sbitset<N>& sbitset) : std::bitset<N>(sbitset) { }
//   sbitset(const std::bitset<N>& bitset) : std::bitset<N>(bitset) { }

//   int operator<(const sbitset<N>& bitset1) const {
// #ifdef COMPARE_BITSET_AND_ULONG
//     return this->to_ulong() < bitset1.to_ulong();
// #else
//     for (int nn = N-1; nn >= 0; --nn) {
//       int delta = this->test(nn) - bitset1.test(nn);
//       if (delta < 0) {
// 	return 1;
//       }
//       if (delta > 0) {
// 	return 0;
//       }
//     }
//     return 0;
// #endif
//   }
// };

// typedef sbitset<MAXNODES> NetworkState_Impl;
// #endif


// 
#elif defined(USE_DYNAMIC_BITSET)
#include "MBDynBitset.h"
typedef MBDynBitset NetworkState_Impl;

#else
typedef unsigned long long NetworkState_Impl;
#endif

#endif
