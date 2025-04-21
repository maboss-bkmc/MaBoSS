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
     IStates.h

   Authors:
     Eric Viara <viara@sysra.com>
     Gautier Stoll <gautier.stoll@curie.fr>
     Vincent Noël <vincent.noel@curie.fr>
 
   Date:
     January-March 2011
*/

#ifndef _ISTATES_H_
#define _ISTATES_H_

#include <vector>

#include "NetworkState.h"
#include "Expressions.h"

class PopNetwork;

class PopIStateGroup {
  
public:
  class PopProbaIState {
    public:

    class PopIStateGroupIndividual {
      public:
      std::vector<double> state_value_list;
      unsigned int pop_size;
      
      PopIStateGroupIndividual(std::vector<Expression*>* state_expr_list, Expression* t_pop_size) 
      {
        NetworkState network_state;

        for (auto state_expr: *state_expr_list) {
          state_value_list.push_back(state_expr->eval(NULL, network_state));
        }
        
        pop_size = (unsigned int) t_pop_size->eval(NULL, network_state);  
      }
      
      PopIStateGroupIndividual(std::vector<double> state_value_list, unsigned int pop_size) : state_value_list(state_value_list), pop_size(pop_size) {}
      
      std::vector<double> getStateValueList() { return state_value_list; }
      unsigned int getPopSize() { return pop_size; }  
    };



    double proba_value;
    Expression* proba_expr;
    std::vector<PopIStateGroupIndividual*>* individual_list;
    
    PopProbaIState(Expression* proba_expr, std::vector<PopIStateGroupIndividual*>* individual_list) 
    {
      this->proba_expr = proba_expr;
      NetworkState network_state;
      proba_value = proba_expr->eval(NULL, network_state);
      
      this->individual_list = individual_list;
    }
    
    PopProbaIState(double proba_value, std::vector<PopIStateGroupIndividual*>* individual_list) : proba_value(proba_value), proba_expr(NULL), individual_list(individual_list) {}
    
    std::vector<PopIStateGroupIndividual*>* getIndividualList() { return individual_list; }
    double getProbaValue() { return proba_value; }
    ~PopProbaIState() {
      delete proba_expr;
      for (auto individual: *individual_list) {
        delete individual;
      }
      delete individual_list;
    }
  };
  
  std::vector<const Node*>* nodes;
  std::vector<PopProbaIState*>* proba_istates;
  
  PopIStateGroup(PopNetwork* network, std::vector<const Node*>* nodes, std::vector<PopProbaIState*>* proba_istates, std::string& error_msg) : nodes(nodes), proba_istates(proba_istates) 
  {
    epilogue(network); 
  }
  ~PopIStateGroup() {
    delete nodes;
    for (auto * proba_istate : *proba_istates)
      delete proba_istate;
    delete proba_istates;
  }
  void epilogue(PopNetwork* network);
  
  std::vector<const Node*>* getNodes() { return nodes; }
  std::vector<PopProbaIState*>* getPopProbaIStates() { return proba_istates; }
  
  static void initPopStates(PopNetwork* network, PopNetworkState& initial_state, RandomGenerator* randgen, unsigned int pop);
  static void display(PopNetwork* network, std::ostream& os);

};



class IStateGroup {

public:
  struct ProbaIState {
    double proba_value;
    std::vector<double>* state_value_list;

    ProbaIState(Expression* proba_expr, std::vector<Expression*>* state_expr_list) {
      NetworkState network_state;
      proba_value = proba_expr->eval(NULL, network_state);
      
      state_value_list = new std::vector<double>();
      for (auto * state_expr : *state_expr_list)
      {
        state_value_list->push_back(state_expr->eval(NULL, network_state));
	    }
    }
  
    ProbaIState(double proba_value, std::vector<double>* state_value_list) {
      this->proba_value = proba_value;
      this->state_value_list = state_value_list;
    }
  
    // only one node
    ProbaIState(double proba_value, Expression* state_expr) {
      this->proba_value = proba_value;
      state_value_list = new std::vector<double>();
      NetworkState network_state;
      state_value_list->push_back(state_expr->eval(NULL, network_state));
    }

    ProbaIState(double proba_value, double istate_value) {
      this->proba_value = proba_value;
      state_value_list = new std::vector<double>();
      state_value_list->push_back(istate_value);
    }
    
    ProbaIState(ProbaIState* obj) {
      this->proba_value = obj->getProbaValue();
      this->state_value_list = new std::vector<double>(*(obj->getStateValueList()));
    }

    ~ProbaIState() {
      delete state_value_list;
    }
    double getProbaValue() {return proba_value;}
    std::vector<double>* getStateValueList() {return state_value_list;}
    void normalizeProbaValue(double proba_sum) {proba_value /= proba_sum;}
  };
  
  IStateGroup(Network* network, std::vector<const Node*>* nodes, std::vector<ProbaIState*>* proba_istates, std::string& error_msg) : nodes(nodes), proba_istates(proba_istates) {
    is_random = false;
    size_t node_size = nodes->size();
    for (auto * proba_istate : *proba_istates)
    {
      if (proba_istate->getStateValueList()->size() != node_size) {
        std::ostringstream ostr;
        ostr << "size inconsistency in istate expression: got " <<  proba_istate->getStateValueList()->size() << " states, has " << node_size << " nodes";
        error_msg = ostr.str();
        return;
      }
    }
    epilogue(network);
 }

  IStateGroup(Network * network, const Node* node) {
    is_random = true;
    nodes = new std::vector<const Node*>();
    nodes->push_back(node);
    proba_istates = new std::vector<ProbaIState*>();
    proba_istates->push_back(new ProbaIState(0.5, 0.));
    proba_istates->push_back(new ProbaIState(0.5, 1.));
    epilogue(network);
  }

  IStateGroup(Network * network, const Node* node, Expression* expr) {
    is_random = false;
    nodes = new std::vector<const Node*>();
    nodes->push_back(node);
    proba_istates = new std::vector<ProbaIState*>();
    proba_istates->push_back(new ProbaIState(1., expr));
    epilogue(network);
  }
  
  IStateGroup(IStateGroup* obj, Network* network) {
    this->is_random = obj->isRandom();
    this->nodes = new std::vector<const Node*>();
    for (const auto node: *(obj->getNodes())) {
      this->nodes->push_back(node);
    }
    this->proba_istates = new std::vector<ProbaIState*>();
    for(auto proba_istate: *(obj->getProbaIStates())) {
      this->proba_istates->push_back(new ProbaIState(proba_istate));
    }
    epilogue(network);
  }
  
  ~IStateGroup() {
    delete nodes;
    for (auto * proba_istate : *proba_istates)
      delete proba_istate;

    delete proba_istates;
  }

  std::vector<const Node*>* getNodes() {return nodes;}
  std::vector<ProbaIState*>* getProbaIStates() {return proba_istates;}
  double getProbaSum() const {return proba_sum;}

  bool isRandom() const {return is_random;}

  bool hasNode(const Node * node) {
    for (auto * t_node : *nodes)
      if (node == t_node)
        return true;
      
    return false;
  }

  static void checkAndComplete(Network* network);
  static void initStates(Network* network, NetworkState& initial_state, RandomGenerator * randgen);
  static void display(Network* network, std::ostream& os);
  static void reset(Network* network);
  

  static void removeNode(Network * network, const Node * node) {
      
    // Initialized at size(), so we can use < size() to check if something changed
    size_t to_delete = network->getIStateGroup()->size();
    for (size_t i=0; i < network->getIStateGroup()->size(); i++) 
    {
      auto * istate_group = network->getIStateGroup()->at(i);  
      if (istate_group->hasNode(node)) {
        if (istate_group->getNodes()->size() == 1) {
          if (to_delete < network->getIStateGroup()->size()) {
            throw BNException("Two IStateGroup with the same node");
          }
          to_delete = i;
          // the node should be in a unique place, so it should work ?
          break;
          
        } else {
          size_t ii;
          std::vector<const Node*>* group_nodes = istate_group->getNodes();
          for(ii = 0; ii < group_nodes->size(); ii++) {
              if (group_nodes->at(ii) == node) {
                group_nodes->erase(group_nodes->begin() + (std::ptrdiff_t) ii);
                break;
              }
          }
          
          std::vector<IStateGroup::ProbaIState*>* proba_istates = istate_group->getProbaIStates();
          for (auto * proba_istate : *proba_istates)
          {
            proba_istate->state_value_list->erase(proba_istate->state_value_list->begin() + (std::ptrdiff_t) i);
          }
        } 
      }
    }
    if (to_delete < network->getIStateGroup()->size())
      network->getIStateGroup()->erase(network->getIStateGroup()->begin() + (std::ptrdiff_t) to_delete);
  }
  
  static void setStatesProbas(Network * network, std::vector<const Node*>* nodes, std::map<std::vector<bool>, double>& probas) {
    for (auto* node: *nodes) {
      IStateGroup::removeNode(network, node);
    }
    
    std::vector<IStateGroup::ProbaIState*>* new_proba_istates = new std::vector<IStateGroup::ProbaIState*>();
    for (auto& state_proba: probas) {
      
      std::vector<double>* state_values = new std::vector<double>();
      for (auto node_state: state_proba.first) {
        state_values->push_back(node_state ? 1.0 : 0.0);
          
      }
      new_proba_istates->push_back(new ProbaIState(state_proba.second, state_values));
        
    }
    
    
    std::string message = "";
    new IStateGroup(network, nodes, new_proba_istates, message);
  }
  
  static void setNodeProba(Network * network, Node * node, double value) {

    IStateGroup::removeNode(network, node);

    std::vector<const Node*>* new_nodes = new std::vector<const Node*>();

    new_nodes->push_back(node);

    std::vector<IStateGroup::ProbaIState*>* new_proba_istates = new std::vector<IStateGroup::ProbaIState*>();

    if (value == 0.0 || value == 1.0) {            
      new_proba_istates->push_back(new ProbaIState(1.0, value));

    } else {
      new_proba_istates->push_back(new ProbaIState(1.0-value, 0.0));
      new_proba_istates->push_back(new ProbaIState(value, 1.0));
    }

    std::string message = "";

    new IStateGroup(network, new_nodes, new_proba_istates, message);
  }

static void setInitialState(Network * network, NetworkState * state) {

  for (auto * node : network->getNodes())
  {
    setNodeProba(network, node, node->getNodeState(*state));
  }
  
}
private:
  std::vector<const Node*>* nodes;
  std::vector<ProbaIState*>* proba_istates;
  double proba_sum;
  bool is_random;

  void computeProbaSum() 
  {  
    proba_sum = 0;
    for (auto * proba_istate : *proba_istates)
      proba_sum += proba_istate->getProbaValue();
    
    for (auto * proba_istate : *proba_istates)
      proba_istate->normalizeProbaValue(proba_sum);
    
  }

  void epilogue(Network* network) {
    computeProbaSum();
    network->getIStateGroup()->push_back(this);
  }
};

extern const bool backward_istate;

#endif
