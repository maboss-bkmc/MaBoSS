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
     BooleanNetwork.cc

   Authors:
     Eric Viara <viara@sysra.com>
     Gautier Stoll <gautier.stoll@curie.fr>
     Vincent Noël <vincent.noel@curie.fr>
 
   Date:
     January-March 2011
     updated October 2014
*/
#include "NetworkState.h"
#include "Expressions.h"

bool NetworkState::computeNodeState(const Node* node, NodeState& node_state)
{
  const Expression* expr = node->getLogicalInputExpression();
  if (NULL != expr) {
    double d = expr->eval(node, *this);
    node_state = d != 0.;
    setNodeState(node, node_state);
    return true;
  }
  return false;
}

#define HAMMING_METHOD2

unsigned int NetworkState::hamming(Network* network, const NetworkState_Impl& state2) const
{
  unsigned int hd = 0;
#ifdef HAMMING_METHOD1
  // faster way
  unsigned long s = (state ^ (state2 & state));
  unsigned int node_count = network->getNodes().size();
  for (unsigned int nn = 0; nn < node_count; ++nn) {
    if ((1ULL << nn) & s) {
      hd++;
    }
  }
  return hd;
#endif

#ifdef HAMMING_METHOD2
  NetworkState network_state2(state2, 1);
  const std::vector<Node*>& nodes = network->getNodes();
  
  for (const auto * node : nodes) {
    if (node->isReference()) {
      NodeState node_state1 = getNodeState(node);
      NodeState node_state2 = network_state2.getNodeState(node);
      hd += 1 - (node_state1 == node_state2);
    }
  }

  return hd;
#endif
}

unsigned int NetworkState::hamming(Network* network, const NetworkState& state2) const
{
  return hamming(network, state2.getState());
}

void NetworkState::display(std::ostream& os, const Network* network) const
{
  const std::vector<Node*>& nodes = network->getNodes();
  int nn = 0;
  for (const auto * node : nodes) {
    os << (nn > 0 ? "\t" : "") << getNodeState(node);
    nn++;
  }
  os << '\n';
}

std::string NetworkState::getName(const Network* network, const std::string& sep) const {
#if defined(USE_STATIC_BITSET) || defined(USE_DYNAMIC_BITSET)
  if (state.none()) {
    return "<nil>";
  }
#else
  if (!state) {
    return "<nil>";
  }
#endif

  std::string result = "";
  const std::vector<Node*>& nodes = network->getNodes();
  
  bool displayed = false;
  for (const auto * node : nodes) {
    if (getNodeState(node)) {
      if (displayed) {
	      result += sep;
      } else {
	      displayed = true;
      }
      result += node->getLabel();
    }
  }
  return result;
  }


void NetworkState::displayOneLine(std::ostream& os, const Network* network, const std::string& sep) const
{
  os << getName(network, sep);
}

void NetworkState::displayJSON(std::ostream& os, const Network* network, const std::string& sep) const
{
  os << getName(network, sep);
}

std::string PopNetworkState::getName(const Network * network, const std::string& sep) const {
  
  std::string res = "[";
  
  size_t i = mp.size();
  for (auto pop : mp) {
    NetworkState t_state(pop.first);
    res += "{" + t_state.getName(network) + ":" + std::to_string(pop.second) + "}";
    if (--i > 0) {
      res += ",";
    }
  }
  res += "]";
  return res;
}

void PopNetworkState::displayOneLine(std::ostream &strm, const Network* network, const std::string& sep) const 
{    
  strm << getName(network, sep);
}

void PopNetworkState::displayJSON(std::ostream &strm, const Network* network, const std::string& sep) const 
{    
  strm << "[";
  size_t i = mp.size();
  for (auto pop : mp) {
    NetworkState t_state(pop.first);
    strm << "{'" << t_state.getName(network) << "':" << pop.second << "}";
    if (--i > 0) {
      strm << ",";
    }
  }
  strm << "]";
}

unsigned int PopNetworkState::count(Expression * expr) const
{
  unsigned int res = 0;
  
  for (auto network_state_proba : mp) {
    NetworkState network_state = NetworkState(network_state_proba.first);
    if (expr == NULL || (bool)expr->eval(NULL, network_state)) {
      res += network_state_proba.second;
    }
  }
  
  return res;
}


unsigned int PopNetworkState::hamming(Network* network, const NetworkState& state2) const
{
  unsigned int hd = 0;
// #ifdef HAMMING_METHOD1
//   // faster way
//   unsigned long s = (state ^ (state2 & state));
//   unsigned int node_count = network->getNodes().size();
//   for (unsigned int nn = 0; nn < node_count; ++nn) {
//     if ((1ULL << nn) & s) {
//       hd++;
//     }
//   }
//   return hd;
// #endif

// #ifdef HAMMING_METHOD2
//   NetworkState network_state2(state2, 1);
//   const std::vector<Node*>& nodes = network->getNodes();
//   std::vector<Node*>::const_iterator begin = nodes.begin();
//   std::vector<Node*>::const_iterator end = nodes.end();

//   while (begin != end) {
//     Node* node = *begin;
//     if (node->isReference()) {
//       NodeState node_state1 = getNodeState(node);
//       NodeState node_state2 = network_state2.getNodeState(node);
//       hd += 1 - (node_state1 == node_state2);
//     }
//     ++begin;
//   }
// #endif

  return hd;
}
