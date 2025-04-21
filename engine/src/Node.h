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
     Node.h

   Authors:
     Eric Viara <viara@sysra.com>
     Gautier Stoll <gautier.stoll@curie.fr>
     Vincent Noël <vincent.noel@curie.fr>
 
   Date:
     January-March 2011
*/

#ifndef _NODE_H_
#define _NODE_H_

#include <stdlib.h>
#include <string>
#include <sstream>
#include <map>

#ifdef SBML_COMPAT
#include <sbml/SBMLTypes.h>
#include "sbml/packages/qual/common/QualExtensionTypes.h"
 
LIBSBML_CPP_NAMESPACE_USE
#endif

#include "NetworkState_Impl.h"
#include "RandomGenerator.h"
#include "maps_header.h"

typedef unsigned int NodeIndex;
typedef bool NodeState; // for now... could be a class


static const std::string ATTR_RATE_UP = "rate_up";
static const std::string ATTR_RATE_DOWN = "rate_down";
static const std::string ATTR_LOGIC = "logic";
static const std::string ATTR_DESCRIPTION = "description";
static const NodeIndex INVALID_NODE_INDEX = (NodeIndex)~0U;

class Expression;
class Network;
class NetworkState;
class PopNetworkState;
class LogicalExprGenContext;
// extern std::ostream& operator<<(std::ostream& os, const BNException& e);

class Node {
  static bool override;
  static bool augment;
  std::string label;
  std::string description;
  bool istate_set;
  bool is_internal;
  bool is_reference;
  bool in_graph;
  bool is_mutable;
  NodeState referenceState;
  const Expression* logicalInputExpr;
  const Expression* rateUpExpr;
  const Expression* rateDownExpr;
  mutable NodeState istate;
  NodeIndex index;
  
  MAP<std::string, const Expression*> attr_expr_map;
  MAP<std::string, std::string> attr_str_map;
  Expression* rewriteLogicalExpression(Expression* rateUpExpr, Expression* rateDownExpr) const;
  NetworkState_Impl node_bit;

 public:
  Node(const std::string& label, const std::string& description, NodeIndex index);

  void setIndex(NodeIndex new_index) {
    index = new_index;
#if !defined(USE_STATIC_BITSET) && !defined(USE_DYNAMIC_BITSET)
    node_bit = 1ULL << new_index;
#endif
  }

  const std::string& getLabel() const {
    return label;
  }

  void setDescription(const std::string& _description) {
    this->description = _description;
  }

  const std::string& getDescription() const {
    return description;
  }

  void setLogicalInputExpression(const Expression* logicalInputExpr);

  void setRateUpExpression(const Expression* expr);

  void setRateDownExpression(const Expression* expr);

  const Expression* getLogicalInputExpression() const {
    return logicalInputExpr;
  }

  const Expression* getRateUpExpression() const {
    return rateUpExpr;
  }

  const Expression* getRateDownExpression() const {
    return rateDownExpr;
  }

  void setAttributeExpression(const std::string& attr_name, const Expression* expr) {
    if (attr_name == ATTR_RATE_UP) {
      setRateUpExpression(expr);
      return;
    }
    if (attr_name == ATTR_RATE_DOWN) {
      setRateDownExpression(expr);
      return;
    }
    if (attr_name == ATTR_LOGIC) {
      setLogicalInputExpression(expr);
      return;
    }
    attr_expr_map[attr_name] = expr;
  }

  void mutate(double value);
  void makeMutable(Network* network);
  NodeState getIState(const Network* network, RandomGenerator* randgen) const;

  void setIState(NodeState _istate) {
    istate_set = true;
    this->istate = _istate;
  }

  bool istateSetRandomly() const {
    return !istate_set;
  }

  bool isInternal() const {
    return is_internal;
  }
  
  bool inGraph() const {
    return in_graph;
  }

  void isInternal(bool _is_internal) {
    this->is_internal = _is_internal;
  }
  
  void inGraph(bool _in_graph) {
    this->in_graph = _in_graph;
  }

  bool isReference() const {
    return is_reference;
  }

  void setReference(bool _is_reference) {
    this->is_reference = _is_reference;
  }

  NodeState getReferenceState() const {
    return referenceState;
  }

  void setReferenceState(NodeState _referenceState) {
    this->is_reference = true;
    this->referenceState = _referenceState;
  }

  const Expression* getAttributeExpression(const std::string& attr_name) const {
    MAP<std::string, const Expression*>::const_iterator iter = attr_expr_map.find(attr_name);
    if (iter == attr_expr_map.end()) {
      if (attr_name == ATTR_RATE_UP) {
	return getRateUpExpression();
      }
      if (attr_name == ATTR_RATE_DOWN) {
	return getRateDownExpression();
      }
      if (attr_name == ATTR_LOGIC) {
	return getLogicalInputExpression();
      }
      return NULL;
    }
    return (*iter).second;
  }

  void setAttributeString(const std::string& attr_name, const std::string& str) {
    if (attr_name == ATTR_DESCRIPTION) {
      setDescription(str);
      return;
    }

    attr_str_map[attr_name] = str;
  }

  std::string getAttributeString(const std::string& attr_name) const {
    MAP<std::string, std::string>::const_iterator iter = attr_str_map.find(attr_name);
    if (iter == attr_str_map.end()) {
      if (attr_name == ATTR_DESCRIPTION) {
	return getDescription();
      }
      return "";
    }
    return (*iter).second;
  }

  NodeIndex getIndex() const {return index;}

#if !defined(USE_STATIC_BITSET) && !defined(USE_DYNAMIC_BITSET)
  NetworkState_Impl getNodeBit() const {return node_bit;}
#endif

  const MAP<std::string, const Expression*>& getAttributeExpressionMap() const {
    return attr_expr_map;
  }

  const MAP<std::string, std::string>& getAttributeStringMap() const {
    return attr_str_map;
  }

  bool isInputNode() const; // true if node state does not depend on other node states

  double getRateUp(const NetworkState& network_state) const;
  double getRateDown(const NetworkState& network_state) const;
  double getRateUp(const NetworkState& network_state, const PopNetworkState& pop) const;
  double getRateDown(const NetworkState& network_state, const PopNetworkState& pop) const;
  NodeState getNodeState(const NetworkState& network_state) const;
  void setNodeState(NetworkState& network_state, NodeState state);

  // returns true if and only if there is a logical input expression that allows to compute state from input nodes
  bool computeNodeState(NetworkState& network_state, NodeState& node_state) const;

  std::string toString() const {
    std::ostringstream ostr;
    display(ostr);
    return ostr.str();
  }

  void display(std::ostream& os) const;
  Expression* generateRawLogicalExpression() const;
  void generateLogicalExpression(LogicalExprGenContext& gen) const;

  static void setOverride(bool _override) {
    Node::override = _override;
  }

  static bool isOverride() {return override;}

  static void setAugment(bool _augment) {
    Node::augment = _augment;
  }

  static bool isAugment() {return augment;}

#ifdef SBML_COMPAT
  void writeSBML(QualitativeSpecies* qs) 
  {
    qs->setId(this->getLabel());
    qs->setCompartment("c");
    qs->setConstant(false);
    qs->setInitialLevel(1);
    qs->setMaxLevel(1);
    qs->setName(this->getLabel());
  }
#endif

  void reset();

  ~Node();
};

#endif
