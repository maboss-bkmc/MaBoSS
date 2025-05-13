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
     PopNetwork.h

   Authors:
     Eric Viara <viara@sysra.com>
     Gautier Stoll <gautier.stoll@curie.fr>
     Vincent Noël <vincent.noel@curie.fr>
 
   Date:
     January-March 2011
*/

#ifndef _POPNETWORK_H_
#define _POPNETWORK_H_

#include "IStates.h"
#include "Network.h"
#include "Expressions.h"

class DivisionRule {
  
  public:

  //During a division, you remove the cell and create two new cells
  static const int DAUGHTER_1;
  // and 
  static const int DAUGHTER_2;
  
  // Each one has it's own map which will change a node value according to an expression
  std::map<Node*, Expression*> daughter1;
  std::map<Node*, Expression*> daughter2;
  
  // And we have an map to select the map of each daughter
  std::map<int, std::map<Node*, Expression*> > daughters = {{DAUGHTER_1, daughter1}, {DAUGHTER_2, daughter2}};
  
  // Division also have a rate
  Expression* rate;
  
  
  DivisionRule() {
    daughter1.clear();
    daughter2.clear();
    rate = NULL;
  }
  
  ~DivisionRule() {
    for (auto& daughter : daughters) {        
      for (auto node_expr : daughter.second) {
        delete node_expr.second;
      }      
    }
    delete rate;
  }
  
  void setRate(Expression* rate);
  double getRate(const NetworkState& state, const PopNetworkState& pop);
  
  void addDaughterNode(int daughter, Node* node, Expression* expression) {
    daughters[daughter][node] = expression;
  }
  
  // This will return a new state based on the mother cell, properly modified according to the maps
  NetworkState applyRules(int daughter, const NetworkState& state, const PopNetworkState& pop);
  
  void display(std::ostream& os) const
  {
    os << "division {" << std::endl;
    os << "  rate=" << this->rate->toString() << ";" << std::endl;
    for (const auto node_expr : daughters.at(DAUGHTER_1)) {
      os << "  " << node_expr.first->getLabel() << ".DAUGHTER1 = " << node_expr.second->toString() << ";" << std::endl;
    }
    for (const auto node_expr : daughters.at(DAUGHTER_2)) {
      os << "  " << node_expr.first->getLabel() << ".DAUGHTER2 = " << node_expr.second->toString() << ";" << std::endl;
    }

    os << "}" << std::endl;
  }
};

class PopIStateGroup;

class PopNetwork : public Network {
  public:
  
  // Population networks have two extra fields :
  
  // Rules for division (speed, state modifications)
  std::vector<DivisionRule*> divisionRules;
  
  // Death rate
  Expression* deathRate;
  
  std::vector<PopIStateGroup*>* pop_istate_group_list;

  PopNetwork();
  ~PopNetwork();
  PopNetwork(const PopNetwork& network);
  PopNetwork& operator=(const PopNetwork& network);

  int parse(const char* file = NULL, std::map<std::string, NodeIndex>* nodes_indexes = NULL, bool is_temp_file = false);
  int parseExpression(const char* content, std::map<std::string, NodeIndex>* nodes_indexes = NULL);
  Expression* parseSingleExpression(const char* content, std::map<std::string, NodeIndex>* nodes_indexes = NULL);
  void initPopStates(PopNetworkState& initial_pop_state, RandomGenerator* randgen, unsigned int pop);

  void addDivisionRule(DivisionRule* rule) { divisionRules.push_back(rule); }
  void removeDivisionRule(size_t index) { 
    DivisionRule* dr = divisionRules[index]; 
    divisionRules.erase(divisionRules.begin() + index); 
    delete dr;
  }
  void setDeathRate(Expression* expr) { deathRate = expr; }
  
  const std::vector<DivisionRule*> getDivisionRules() const { return divisionRules; }
  const Expression* getDeathRate() const { return deathRate; }
  
  // Evaluation of the death rate according to the state
  double getDeathRate(const NetworkState& state, const PopNetworkState& pop) const;
  
  std::vector<PopIStateGroup*>* getPopIStateGroup() const { return pop_istate_group_list; }
  
  std::string toString() const {
    std::ostringstream ostr;
    display(ostr);
    return ostr.str();
  }

  void display(std::ostream& os) const;
  void clearPopIstates() {
    pop_istate_group_list->clear();
  }
  bool isPopNetwork() { return true; }

};

#endif
