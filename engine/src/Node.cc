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

#include "Node.h"
#include "NetworkState.h"
#include "Expressions.h"

bool Node::override = false;
bool Node::augment = false;

Node::Node(const std::string& label, const std::string& description, NodeIndex index) : label(label), description(description), istate_set(false), is_internal(false), is_reference(false), in_graph(false), referenceState(false), logicalInputExpr(NULL), rateUpExpr(NULL), rateDownExpr(NULL), index(index)
{
#if !defined(USE_STATIC_BITSET) && !defined(USE_DYNAMIC_BITSET)
  node_bit = NetworkState::nodeBit(index);
#endif
}


void Node::reset()
{
  description = "";
  istate_set = false;
  is_internal = false;
  is_reference = false;
  in_graph = false;
  is_mutable = false;
  referenceState = false;
  delete logicalInputExpr;
  logicalInputExpr = NULL;
  delete rateUpExpr;
  rateUpExpr = NULL;
  delete rateDownExpr;
  rateDownExpr = NULL;
}

void Node::setLogicalInputExpression(const Expression* _logicalInputExpr) {
  delete this->logicalInputExpr;
  this->logicalInputExpr = _logicalInputExpr;
}

void Node::setRateUpExpression(const Expression* expr) {
  delete this->rateUpExpr;
  this->rateUpExpr = expr;
}

void Node::setRateDownExpression(const Expression* expr) {
  delete this->rateDownExpr;
  this->rateDownExpr = expr;
}

bool Node::isInputNode() const
{
  return getLogicalInputExpression() == NULL && getRateUpExpression() == NULL && getRateDownExpression() == NULL;
}

double Node::getRateUp(const NetworkState& network_state) const
{
  if (getRateUpExpression() == NULL) {
    if (NULL != getLogicalInputExpression()) {
      double d = getLogicalInputExpression()->eval(this, network_state);
      return (0.0 != d) ? 1.0 : 0.0;
    }
    return 0.0;
  }
  return getRateUpExpression()->eval(this, network_state);
}

double Node::getRateUp(const NetworkState& network_state, const PopNetworkState& pop) const
{
  if (getRateUpExpression() == NULL) {
    if (NULL != getLogicalInputExpression()) {
      double d = getLogicalInputExpression()->eval(this, network_state, pop);
      return (0.0 != d) ? 1.0 : 0.0;
    }
    return 0.0;
  }
  return getRateUpExpression()->eval(this, network_state, pop);
}


double Node::getRateDown(const NetworkState& network_state) const
{
  if (getRateDownExpression() == NULL) {
    if (NULL != getLogicalInputExpression()) {
      double d = getLogicalInputExpression()->eval(this, network_state);
      return (0.0 != d) ? 0.0 : 1.0;
    }
    return 0.0;
  }
  return getRateDownExpression()->eval(this, network_state);
}

double Node::getRateDown(const NetworkState& network_state, const PopNetworkState& pop) const
{
  if (getRateDownExpression() == NULL) {
    if (NULL != getLogicalInputExpression()) {
      double d = getLogicalInputExpression()->eval(this, network_state, pop);
      return (0.0 != d) ? 0.0 : 1.0;
    }
    return 0.0;
  }
  return getRateDownExpression()->eval(this, network_state, pop);
}


void Node::mutate(double value) 
{
    delete logicalInputExpr;
    logicalInputExpr = new ConstantExpression(value);
    delete rateUpExpr;
    rateUpExpr = NULL;
    delete rateDownExpr;
    rateDownExpr = NULL;
}

void Node::makeMutable(Network* network)
{
  if (!this->is_mutable) 
  {
    
    
    const Symbol* lowvar = network->getSymbolTable()->getOrMakeSymbol("$Low_" + label);
    const Symbol* highvar = network->getSymbolTable()->getOrMakeSymbol("$High_" + label);
    const Symbol* nb_mutable = network->getSymbolTable()->getOrMakeSymbol("$nb_mutable");
    
    Expression* new_rate_up = NULL;
    Expression* new_rate_down = NULL;
    
    if (rateUpExpr == NULL) {
      
      new_rate_up = new CondExpression(
        logicalInputExpr->clone(),
        new ConstantExpression(1.0),
        new ConstantExpression(0.0)
      );
       
    } else {
      new_rate_up = rateUpExpr->clone();  
      delete rateUpExpr;
    }
    
    new_rate_up = new CondExpression(
      new EqualExpression(new SymbolExpression(network->getSymbolTable(), highvar), new ConstantExpression(1.0)),
      new DivExpression(new ConstantExpression(std::numeric_limits<double>::max()), new SymbolExpression(network->getSymbolTable(), nb_mutable)),
      new_rate_up
    );
    rateUpExpr = new CondExpression(
      new EqualExpression(new SymbolExpression(network->getSymbolTable(), lowvar), new ConstantExpression(1.0)),
      new ConstantExpression(0.0),
      new_rate_up
    );
    
    if (rateDownExpr == NULL) {
      new_rate_down = new CondExpression(
        logicalInputExpr->clone(),
        new ConstantExpression(0.0),
        new ConstantExpression(1.0)
      );
    } else {
      new_rate_down = rateDownExpr->clone();
      delete rateDownExpr;
    }
    
    new_rate_down = new CondExpression(
      new EqualExpression(new SymbolExpression(network->getSymbolTable(), lowvar), new ConstantExpression(1.0)),
      new DivExpression(new ConstantExpression(std::numeric_limits<double>::max()), new SymbolExpression(network->getSymbolTable(), nb_mutable)),
      new_rate_down
    );
    rateDownExpr = new CondExpression(
      new EqualExpression(new SymbolExpression(network->getSymbolTable(), highvar), new ConstantExpression(1.0)),
      new ConstantExpression(0.0),
      new_rate_down
    );
  
    network->getSymbolTable()->setSymbolValue(nb_mutable, network->getSymbolTable()->getSymbolValue(nb_mutable) + 1);
    this->is_mutable = true;
  }
}

NodeState Node::getNodeState(const NetworkState& network_state) const
{
  return network_state.getNodeState(this);
}

void Node::setNodeState(NetworkState& network_state, NodeState state)
{
  network_state.setNodeState(this, state);
}

bool Node::computeNodeState(NetworkState& network_state, NodeState& node_state) const
{
  return network_state.computeNodeState(this, node_state);
}

NodeState Node::getIState(const Network* network, RandomGenerator* rangen) const
{
  if (!istate_set) {
    double rand = rangen->generate();
    istate = rand > 0.5; // >= 0.5 ?
  }
  return istate;
}

void Node::display(std::ostream& os) const
{
  os << "node " << label << " {\n";
  if (description.length() > 0) {
    os << "  description = \"" << description << "\";\n";
  }
  if (NULL != logicalInputExpr) {
    os << "  logic = ";
    logicalInputExpr->display(os);
    os << ";\n";
  }
  if (NULL != rateUpExpr) {
    os << "  rate_up = ";
    rateUpExpr->display(os);
    os << ";\n";
  }
  if (NULL != rateDownExpr) {
    os << "  rate_down = ";
    rateDownExpr->display(os);
    os << ";\n";
  }

  if (attr_expr_map.size() || attr_str_map.size()) {
    os << "\n  // extra attributes\n";
    for (const auto & attr_expr : attr_expr_map) {
      os << "  " << attr_expr.first << " = ";
      attr_expr.second->display(os);
      os << ";\n";
    }

    for (const auto & attr_str : attr_str_map) {
      os << "  " << attr_str.first << " = \"" << attr_str.second << "\";\n";
    }
  }
  os << "}\n";
}

Node::~Node()
{
  delete logicalInputExpr;
  delete rateUpExpr;
  delete rateDownExpr;

  for (const auto & attr_expr : attr_expr_map) {
    delete attr_expr.second;
  }
}
