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

#ifndef _NETWORKSTATE_H_
#define _NETWORKSTATE_H_

#include <set>

#include "NetworkState_Impl.h"
#include "Node.h"
#include "Network.h"

#ifdef MPI_COMPAT
#include "MPI_headers.h"
#endif

// global state of the boolean network
class NetworkState {
  NetworkState_Impl state;

#if !defined(USE_STATIC_BITSET) && !defined(USE_DYNAMIC_BITSET)
  static NetworkState_Impl nodeBit(const Node* node) {
    return node->getNodeBit();
  }

public:
  static NetworkState_Impl nodeBit(NodeIndex node_index) {
    return (1ULL << node_index);
  }
#endif

public:
  NetworkState(const NetworkState_Impl& state) : state(state) { }
  NetworkState(const NetworkState& state) : state(state.getState()) {}
#ifdef USE_DYNAMIC_BITSET
  NetworkState(const NetworkState_Impl& state, int copy) : state(state, 1) { }
  NetworkState(const NetworkState& state, int copy) : state(state.getState(1), 1) {}
#else
  NetworkState(const NetworkState_Impl& state, int copy) : state(state) { }
  NetworkState(const NetworkState& state, int copy) : state(state.getState()) {}
#endif

  NetworkState operator&(const NetworkState& mask) const { 
#ifdef USE_DYNAMIC_BITSET
    return NetworkState(state & mask.getState(), 1);
#else
    return NetworkState(state & mask.getState());
#endif
  }
  
  NetworkState applyMask(const NetworkState& mask, std::map<unsigned int, unsigned int>& scale) const {
#ifdef USE_DYNAMIC_BITSET
    return NetworkState(state & mask.getState(), 1);
#else
    return NetworkState(state & mask.getState());
#endif
  }

#ifdef USE_STATIC_BITSET
  NetworkState() { }
#elif defined(USE_DYNAMIC_BITSET)
  // EV: 2020-10-23
  //NetworkState() : state(MAXNODES) { }
  NetworkState() : state(Network::getMaxNodeSize()) { }
  // EV: 2020-12-01 would be better to create a 0-size state and then call resize dynamically
  //NetworkState() : state(0) { }
#else
  NetworkState() : state(0ULL) { }
#endif

  void set() {
#ifdef USE_STATIC_BITSET
    state.set();
#elif defined(USE_DYNAMIC_BITSET)
    // EV: 2020-10-23
    //output_mask.resize(MAXNODES);
    state.resize(Network::getMaxNodeSize());
    state.set();
#else
    state = ~0ULL;
#endif
  }

  void reset() {
#ifdef USE_STATIC_BITSET
    state.reset();
#elif defined(USE_DYNAMIC_BITSET)
    // EV: 2020-10-23
    //output_mask.resize(MAXNODES);
    state.resize(Network::getMaxNodeSize());
    state.reset();
#else
    state = 0ULL;
#endif
  }

  NodeState getNodeState(const Node* node) const {
#if defined(USE_STATIC_BITSET) || defined(USE_DYNAMIC_BITSET)
    return state.test(node->getIndex());
#else
    return state & nodeBit(node);
#endif
  }

  void setNodeState(const Node* node, NodeState node_state) {
#if defined(USE_STATIC_BITSET) || defined(USE_DYNAMIC_BITSET)
    state.set(node->getIndex(), node_state);
#else
    if (node_state) {
      state |= nodeBit(node);
    } else {
      state &= ~nodeBit(node);
    }
#endif
  }

  void flipState(const Node* node) {
#if defined(USE_STATIC_BITSET) || defined(USE_DYNAMIC_BITSET)
    //state.set(node->getIndex(), !state.test(node->getIndex()));
    state.flip(node->getIndex());
#else
    state ^= nodeBit(node);
#endif
  }

  // returns true if and only if there is a logical input expression that allows to compute state from input nodes
  bool computeNodeState(const Node* node, NodeState& node_state);

  static bool isPopState() { return false; }
  std::set<NetworkState_Impl>* getNetworkStates() const {
    return new std::set<NetworkState_Impl>({state});
  }
  
#ifdef USE_DYNAMIC_BITSET
  NetworkState_Impl getState(int copy) const {return NetworkState_Impl(state, copy);}
#endif
  NetworkState_Impl getState() const {return state;}


  void display(std::ostream& os, const Network* network) const;

  std::string getName(const Network * network, const std::string& sep=" -- ") const;
 
  void displayOneLine(std::ostream& os, const Network* network, const std::string& sep = " -- ") const;
  void displayJSON(std::ostream& os, const Network* network, const std::string& sep = " -- ") const;

#ifndef USE_UNORDERED_MAP
  bool operator<(const NetworkState& network_state) const {
    return state < network_state.state;
  }
#endif
  unsigned int hamming(Network* network, const NetworkState_Impl& state) const;
  unsigned int hamming(Network* network, const NetworkState& state) const;


#ifdef MPI_COMPAT
  size_t my_MPI_Pack_Size() const {
#ifdef USE_STATIC_BITSET
    return (MAXNODES/64 + (MAXNODES%64 > 0 ? 1 : 0)) * sizeof(unsigned long long) + sizeof(size_t);

#elif defined(USE_DYNAMIC_BITSET)
    return 0;
    
#else
    return sizeof(unsigned long long);
    
#endif
    
  }

  void my_MPI_Pack(void* buff, unsigned int size_pack, int* position) const {
#ifdef USE_STATIC_BITSET
    
    std::vector<unsigned long long> arr = to_ullongs(state);
    size_t nb_ullongs = arr.size();
    MPI_Pack(&nb_ullongs, 1, my_MPI_SIZE_T, buff, size_pack, position, MPI_COMM_WORLD);
    
    for (size_t i = 0; i < arr.size(); i++) {
      MPI_Pack(&(arr[i]), 1, MPI_UNSIGNED_LONG_LONG, buff, size_pack, position, MPI_COMM_WORLD);
    }
    
#elif defined(USE_DYNAMIC_BITSET)
    
    
#else
    MPI_Pack(&state, 1, MPI_UNSIGNED_LONG_LONG, buff, size_pack, position, MPI_COMM_WORLD);
    
#endif
  }
  
  void my_MPI_Unpack(void* buff, unsigned int buff_size, int* position) {
#ifdef USE_STATIC_BITSET
    std::vector<unsigned long long> v;
    size_t nb_ullongs;
    MPI_Unpack(buff, buff_size, position, &nb_ullongs, 1, my_MPI_SIZE_T, MPI_COMM_WORLD);
    
    for (size_t i = 0; i < nb_ullongs; i++) {
      unsigned long long t_ullong;
      MPI_Unpack(buff, buff_size, position, &t_ullong, 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
      v.push_back(t_ullong);
    }
    
    state = to_bitset(v);

#elif defined(USE_DYNAMIC_BITSET)
    
    
#else
    MPI_Unpack(buff, buff_size, position, &state, 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
    
#endif
    
  }
  void my_MPI_Recv(int source) 
  {
#ifdef USE_STATIC_BITSET
    std::vector<unsigned long long> v;
    size_t nb_ullongs;
    MPI_Recv(&nb_ullongs, 1, my_MPI_SIZE_T, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    for (size_t i = 0; i < nb_ullongs; i++) {
      unsigned long long t_ullong;
      MPI_Recv(&t_ullong, 1, MPI_UNSIGNED_LONG_LONG, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      v.push_back(t_ullong);
    }
    
    state = to_bitset(v);
    
#elif defined(USE_DYNAMIC_BITSET)
    
#else
    MPI_Recv(&state, 1, MPI_UNSIGNED_LONG_LONG, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#endif
  }
  
  void my_MPI_Send(int dest) const
  {
#ifdef USE_STATIC_BITSET
    std::vector<unsigned long long> arr = to_ullongs(state);
    size_t nb_ullongs = arr.size();
    MPI_Send(&nb_ullongs, 1, my_MPI_SIZE_T, dest, 0, MPI_COMM_WORLD);
    
    for (size_t i = 0; i < arr.size(); i++) {
      MPI_Send(&(arr[i]), 1, MPI_UNSIGNED_LONG_LONG, dest, 0, MPI_COMM_WORLD);
    }
    
#elif defined(USE_DYNAMIC_BITSET)

#else
    MPI_Send(&state, 1, MPI_UNSIGNED_LONG_LONG, dest, 0, MPI_COMM_WORLD);

#endif
  }
#endif

  static NodeState getState(Node* node, const NetworkState_Impl &state) {
#if defined(USE_STATIC_BITSET) || defined(USE_DYNAMIC_BITSET)
    return state.test(node->getIndex());
#else
    return state & nodeBit(node);
#endif
  }

#ifdef USE_STATIC_BITSET
  
  static std::vector<unsigned long long> to_ullongs(std::bitset<MAXNODES> bs) {
    std::vector<unsigned long long> ret;
    ret.clear();

    unsigned int nb_arrays = MAXNODES/64 + (MAXNODES%64 > 0 ? 1 : 0);
    for (unsigned int i=0; i < nb_arrays; i++) {
        ret.push_back(((bs<<(i*64))>>(MAXNODES-64)).to_ullong());   
    }
  
    return ret;
  }

  static std::bitset<MAXNODES> to_bitset(std::vector<unsigned long long> arr) {
    std::bitset<MAXNODES> ret;
    
    for (unsigned int i=0; i < (arr.size()-1); i++) 
    {
        ret |= arr[i];
    
        if (i < (arr.size()-2)) {
            ret = ret << 64;
        } else {
            ret = ret << (MAXNODES-((i+1)*64));        
            ret |= (arr[i+1] >> (MAXNODES-((i+1)*64)));
        } 
    } 
    return ret;
  }
#endif

};
namespace std {
  template <> struct hash<NetworkState>
  {
    size_t operator()(const NetworkState& val) const {
#ifdef USE_DYNAMIC_BITSET
      return std::hash<NetworkState_Impl>{}(val.getState(1));
#else
      return std::hash<NetworkState_Impl>{}(val.getState());
#endif
    }
  };
  
  template <> struct equal_to<NetworkState>
  {
    size_t operator()(const NetworkState& val1, const NetworkState& val2) const {
#ifdef USE_DYNAMIC_BITSET
      NetworkState_Impl state_1(val1.getState(1));
      NetworkState_Impl state_2(val2.getState(1));
      return std::equal_to<NetworkState_Impl>{}(state_1, state_2);
#else
      return std::equal_to<NetworkState_Impl>{}(val1.getState(), val2.getState());
#endif
    }
  };

  // Added less operator, necessary for maps, sets. Code from https://stackoverflow.com/a/21245301/11713763
  template <> struct less<NetworkState>
  {
    size_t operator()(const NetworkState& val1, const NetworkState& val2) const {
#ifdef USE_DYNAMIC_BITSET
      const NetworkState_Impl& state_1 = val1.getState(1);
      const NetworkState_Impl& state_2 = val2.getState(1);
      return std::less<NetworkState_Impl>{}(state_1, state_2);
#else
      return std::less<NetworkState_Impl>{}(val1.getState(), val2.getState());
#endif
    }
  };
}

// global state of the population boolean network
class PopNetworkState {
  
  std::map<NetworkState_Impl, unsigned int> mp;
  mutable size_t hash;
  mutable bool hash_init;

public:

  const std::map<NetworkState_Impl, unsigned int>& getMap() const {
    return mp;
  }

  size_t getHash() const { 
    // EV 2021-08-23: invalid comparison as hash == 0 can be a valid computed hash; replaced by hash_init
    //if (hash == 0) {
    if (!hash_init) {
      hash = compute_hash();
      hash_init = true;
    }
    return hash; 
  }

 PopNetworkState() : mp(std::map<NetworkState_Impl, unsigned int>()), hash(0) , hash_init(false) { }
 PopNetworkState(const PopNetworkState &p ) : hash(0), hash_init(false) { *this = p; }
#ifdef USE_DYNAMIC_BITSET
 PopNetworkState(const PopNetworkState &p , int copy) : hash(0), hash_init(false) 
 { 
    mp.clear();
    for (const auto& item : p.getMap())
    {
      NetworkState_Impl state(item.first, copy);
      mp[state] = item.second;
    } 
  }
#endif
 PopNetworkState(std::map<NetworkState_Impl, unsigned int> mp ) : mp(mp), hash(0), hash_init(false) { }

 PopNetworkState(NetworkState_Impl state, unsigned int value) : mp(std::map<NetworkState_Impl, unsigned int>()), hash(0) , hash_init(false) {
#ifdef USE_DYNAMIC_BITSET
  mp[NetworkState_Impl(state, 1)] = value;
#else
  mp[state] = value;
#endif
  }
  
  void set() {
    mp.clear();
    hash_init = false;
    hash = 0;
    NetworkState new_state;
    new_state.set();
#ifdef USE_DYNAMIC_BITSET
    mp[new_state.getState(1)] = 1;
#else
    mp[new_state.getState()] = 1;
#endif
  }
  
  PopNetworkState& operator=(const PopNetworkState &p ) 
  {     
#ifdef USE_DYNAMIC_BITSET
    mp.clear();
    for (const auto& item : p.getMap())
    {
      NetworkState_Impl state(item.first, 1);
      mp[state] = item.second;
    }
#else
    mp = std::map<NetworkState_Impl, unsigned int>(p.getMap());

    
#endif  
    // EV 2021-10-28
    hash = 0;
    hash_init = false;
    return *this;
  }

  PopNetworkState applyMask(const PopNetworkState& mask, std::map<unsigned int, unsigned int>& scale) const {
    std::map<NetworkState_Impl, unsigned int> new_map;
    NetworkState networkstate_mask = mask.getMap().begin()->first;
        
    for (const auto & elem: mp) {
      NetworkState_Impl new_state = elem.first & networkstate_mask.getState();
#ifdef USE_DYNAMIC_BITSET
      new_map[NetworkState_Impl(new_state, 1)] = scale[elem.second];
#else
      new_map[new_state] = scale[elem.second];
#endif
    }
    return PopNetworkState(new_map);
  }

  void addStatePop(const NetworkState_Impl& state, unsigned int pop) {
    auto iter = mp.find(state);
    if (iter == mp.end()) {
#ifdef USE_DYNAMIC_BITSET
      mp[NetworkState_Impl(state,1)] = pop;
#else
      mp[state] = pop;
#endif
    } else {
      iter->second += pop;
    }
    // EV 2021-10: the following code was missing
    hash = 0;
    hash_init = false;
  }
  
  // & operator for applying the mask
  PopNetworkState operator&(const NetworkState_Impl& mask) const { 
    
    PopNetworkState masked_pop_state;
    for (const auto &network_state_pop : mp) {
#ifdef USE_DYNAMIC_BITSET
      NetworkState_Impl masked_network_state(network_state_pop.first & mask, 1);
#else
      NetworkState_Impl masked_network_state(network_state_pop.first & mask);
#endif
      masked_pop_state.addStatePop(masked_network_state, network_state_pop.second);
    }
    
#ifdef USE_DYNAMIC_BITSET
    return PopNetworkState(masked_pop_state, 1); 
#else
    return PopNetworkState(masked_pop_state); 
#endif
  }
  
  // & operator for applying the mask
  PopNetworkState operator&(const NetworkState& mask) const { 
    
    PopNetworkState masked_pop_state;
    for (const auto &network_state_pop : mp) {
#ifdef USE_DYNAMIC_BITSET
      NetworkState_Impl masked_network_state(network_state_pop.first & mask.getState(), 1);
#else
NetworkState_Impl masked_network_state(network_state_pop.first & mask.getState());
#endif
      masked_pop_state.addStatePop(masked_network_state, network_state_pop.second);
    }
    
#ifdef USE_DYNAMIC_BITSET
    return PopNetworkState(masked_pop_state, 1); 
#else
    return PopNetworkState(masked_pop_state); 
#endif  
  }
  
  bool operator==(const PopNetworkState& pop_state) const {

    // So when are two PopNetworkState inequals ?
    // First, if they don't have the same length of states in the population
    const std::map<NetworkState_Impl, unsigned int>& other_mp = pop_state.getMap();

    if (mp.size() != other_mp.size()) {
      return false;
    }
    
   // EV 2021-10-28: std::map are ordered, so it is just necessary to compare
    // return std::equal(mp.begin(), mp.end(), other_mp.begin());
    
    std::map<NetworkState_Impl, unsigned int>::const_iterator iter = mp.begin();
    std::map<NetworkState_Impl, unsigned int>::const_iterator other_iter = other_mp.begin();
    for ( ; iter != mp.end(); ++iter, ++other_iter) {
      if ((iter->first != other_iter->first) || (iter->second != other_iter->second)) {
	return false;
      }
    }
    return true;
  }
  
  // Increases the population of the state
  void incr(const NetworkState& net_state) {
    
    auto iter = mp.find(net_state.getState());
    if (iter == mp.end()) {
#ifdef USE_DYNAMIC_BITSET
        mp[net_state.getState(1)] = 1;
#else
        mp[net_state.getState()] = 1;
#endif
    } else {
      iter->second++;
    }
    hash = 0;
    hash_init = false;
  }

  // Decreases the population of the state
  void decr(const NetworkState& net_state) {
    NetworkState_Impl t_state = net_state.getState();
    auto iter = mp.find(t_state);
    assert(iter != mp.end());
    if (iter->second > 1) {
      iter->second--;
    } else {
      mp.erase(t_state);
    }
    hash = 0;
    hash_init = false;
  }
  
  // Returns if the state exists
  bool exists(const NetworkState& net_state) const {
    return mp.find(net_state.getState()) != mp.end();
  }
  
  // EV 2021-10-28: useful in case of using std::map<PopNetworkState, ...>
  bool operator<(const PopNetworkState& pop_state) const {

    const std::map<NetworkState_Impl, unsigned int>& other_mp = pop_state.getMap();

    if (mp.size() != other_mp.size()) {
      return mp.size() < other_mp.size();
    }
    
    // EV 2021-10-28: std::map are ordered => this code is ok
    std::map<NetworkState_Impl, unsigned int>::const_iterator iter = mp.begin();
    std::map<NetworkState_Impl, unsigned int>::const_iterator other_iter = other_mp.begin();
    for ( ; iter != mp.end(); ++iter, ++other_iter) {
      if (iter->first != other_iter->first) {
	return std::less<NetworkState_Impl>{}(iter->first, other_iter->first);
      } else if (iter->second != other_iter->second) {
	return iter->second < other_iter->second;
      }
    }
    return false;
  }

  size_t compute_hash() const {

#ifdef USE_DYNAMIC_BITSET
    
    size_t result = 1;
    for (const auto &network_state_pop: mp) {
      result += std::hash<NetworkState_Impl>{}(network_state_pop.first);
      result += std::hash<unsigned int>{}(network_state_pop.second);
    }
    return result;
#else    
    // New one : for all state:pop, compute sum_i = state_i * pop_i;
    // Expensive, but should be a good one ?
    
    size_t result = 1;
    for (auto &network_state_pop: mp) {
      NetworkState_Impl t_state = network_state_pop.first;
      const unsigned char* p = (const unsigned char*)&t_state;
      for (size_t nn = 0; nn < sizeof(t_state); nn++) {
        unsigned char val = *p++;
        if (val) {
          result *= val;
          result ^= result >> 8;
        }
      }
      p = (const unsigned char*)&network_state_pop.second;
      if (*p) {
	result *= *p;
	result ^= result >> 8;
      }
    }
    return result;
#endif
    // EV: 2021-11-24 note: returning another hash code changes the results, for instance:
    // return (size_t)(result*1.1);
  }
  
  // Count the population satisfying an expression
  unsigned int count(Expression * expr) const;
  void clear() { mp.clear(); hash_init = false; }
  std::string getName(const Network * network, const std::string& sep=" -- ") const;
  void displayOneLine(std::ostream& os, const Network* network, const std::string& sep = " -- ") const;
  void displayJSON(std::ostream& os, const Network* network, const std::string& sep = " -- ") const;

  unsigned int hamming(Network* network, const NetworkState& state) const;
  
  static bool isPopState() { return true; }
  std::set<NetworkState_Impl>* getNetworkStates() const {
    std::set<NetworkState_Impl>* result = new std::set<NetworkState_Impl>();
    for (auto network_state_pop : mp) {
      result->insert(network_state_pop.first);
    }
    return result;
  }
  
#ifdef MPI_COMPAT
  size_t my_MPI_Pack_Size() const {
    return sizeof(size_t) + mp.size() * (sizeof(NetworkState_Impl) + sizeof(unsigned int));
  }

  void my_MPI_Pack(void* buff, unsigned int size_pack, int* position) const {

    size_t nb_populations = mp.size();
    MPI_Pack(&nb_populations, 1, my_MPI_SIZE_T, buff, size_pack, position, MPI_COMM_WORLD);
    
    for (auto &network_state_pop : mp) {
      NetworkState s(network_state_pop.first); 
      s.my_MPI_Pack(buff, size_pack, position);
      MPI_Pack(&(network_state_pop.second), 1, MPI_UNSIGNED, buff, size_pack, position, MPI_COMM_WORLD);
    }
  }
  
  void my_MPI_Unpack(void* buff, unsigned int buff_size, int* position) {
    mp.clear();
    size_t nb_populations;
    MPI_Unpack(buff, buff_size, position, &nb_populations, 1, my_MPI_SIZE_T, MPI_COMM_WORLD);
    
    for (size_t i = 0; i < nb_populations; i++) {
      NetworkState t_state;
      t_state.my_MPI_Unpack(buff, buff_size, position);
      unsigned int pop;
      MPI_Unpack(buff, buff_size, position, &pop, 1, MPI_UNSIGNED, MPI_COMM_WORLD);
      mp[t_state.getState()] = pop;
    }
  }
  
  void my_MPI_Recv(int source) 
  {
    mp.clear();
    size_t nb_populations;
    MPI_Recv(&nb_populations, 1, my_MPI_SIZE_T, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    for (size_t i = 0; i < nb_populations; i++) {
      NetworkState t_state;
      t_state.my_MPI_Recv(source);
      unsigned int pop;
      MPI_Recv(&pop, 1, MPI_UNSIGNED, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      mp[t_state.getState()] = pop;
    }
  }
  
  void my_MPI_Send(int dest) const
  {
    size_t nb_populations = mp.size();
    MPI_Send(&nb_populations, 1, my_MPI_SIZE_T, dest, 0, MPI_COMM_WORLD);
    
    for (auto &network_state_pop : mp) {
      NetworkState s(network_state_pop.first);
      s.my_MPI_Send(dest);
      MPI_Send(&(network_state_pop.second), 1, MPI_UNSIGNED, dest, 0, MPI_COMM_WORLD);
    }
  }
#endif  
};

namespace std {
  template <> struct hash<PopNetworkState>
  {
    size_t operator()(const PopNetworkState & x) const
    {
      return x.getHash();
    }
  };
  
  template <> struct equal_to<PopNetworkState >
  {
    size_t operator()(const PopNetworkState& val1, const PopNetworkState& val2) const {
      return val1 == val2;
    }
  };

  template <> struct not_equal_to<PopNetworkState >
  {
    size_t operator()(const PopNetworkState& val1, const PopNetworkState& val2) const {
      return !(val1 == val2);
    }
  };
  
  // Added less operator, necessary for maps, sets. Code from https://stackoverflow.com/a/21245301/11713763
  template <> struct less<PopNetworkState>
  {
    size_t operator()(const PopNetworkState& val1, const PopNetworkState& val2) const {
      return val1 < val2;
    }
  };
}

class PopSize {
  unsigned int size;
public:
  PopSize(unsigned int _size) : size(_size) { }
  PopSize() : size(0) { }
  PopSize(const PopSize& p, int copy ) {
    this->size = p.getSize();
  }
  void set() {size = 0;}
  unsigned int getSize() const {return size;}
  static bool isPopState() {return false;}
  
  // & operator for applying the mask
  PopSize operator&(const NetworkState_Impl& mask) const { 
    return PopSize(size); 
  }
  
  // & operator for applying the mask
  PopSize operator&(const NetworkState& mask) const { 
    return PopSize(size);
  }
  
  PopSize applyMask(const PopSize& mask, std::map<unsigned int, unsigned int>& scale) const {
    return PopSize(size);
  }
  
  bool operator==(const PopSize& pop_size) const {
    return pop_size.getSize() == size;
  }
  
  bool operator<(const PopSize& pop_size) const {
    return size < pop_size.getSize();
  }
  
  int hamming(Network* network, const NetworkState& state) const {
    return 0;
  }
  std::set<NetworkState_Impl>* getNetworkStates() const {
    return new std::set<NetworkState_Impl>();
  }
  
  std::string getName(Network * network, const std::string& sep=" -- ") const {
    return std::to_string(size);
  }
  
  void displayOneLine(std::ostream& os, Network* network, const std::string& sep = " -- ") const {
    os << getName(network, sep);
  }
  
  
};

namespace std {
  template <> struct hash<PopSize>
  {
    size_t operator()(const PopSize & x) const
    {
      return x.getSize();
    }
  };
}
#endif
