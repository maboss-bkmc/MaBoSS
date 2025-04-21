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
     PopNetwork.cc

   Authors:
     Eric Viara <viara@sysra.com>
     Gautier Stoll <gautier.stoll@curie.fr>
     Vincent Noël <vincent.noel@curie.fr>
 
   Date:
     January-March 2011
     updated October 2014
*/

#include "PopNetwork.h"
#include "parsers/BooleanGrammar.h"

extern FILE* ctbndlin;
extern void ctbndl_scan_expression(const char *);
extern int ctbndlparse();
extern int ctbndllex_destroy();
const bool backward_istate = getenv("MABOSS_BACKWARD_ISTATE") != NULL;

const int DivisionRule::DAUGHTER_1 = 1;
const int DivisionRule::DAUGHTER_2 = 2;

PopNetwork::PopNetwork() : Network() 
{ 
  deathRate = NULL; 
  divisionRules.clear();
  pop_istate_group_list = new std::vector<PopIStateGroup*>();
}

int PopNetwork::parse(const char* file, std::map<std::string, NodeIndex>* nodes_indexes, bool is_temp_file)
{
  set_pop_network(this);
  int res = Network::parse(file, nodes_indexes, is_temp_file);
  
  set_pop_network(NULL);
  return res;
}


int PopNetwork::parseExpression(const char* content, std::map<std::string, NodeIndex>* nodes_indexes){
  
  set_pop_network(this);
  int res = Network::parseExpression(content, nodes_indexes);
  
  set_pop_network(NULL);
  return res;
}

Expression* PopNetwork::parseSingleExpression(const char* content, std::map<std::string, NodeIndex>* nodes_indexes)
{
  set_pop_network(this);
  set_expression(NULL);
  const char * se = "SINGLE_EXPRESSION ";
  const char * ee = ";";
  std::string new_content = se;
  new_content += content;
  new_content += ee;
  Network::parseExpression(new_content.c_str(), nodes_indexes);
  
  set_pop_network(NULL);
  return get_expression();
}


void DivisionRule::setRate(Expression* _rate) {  
  this->rate = _rate;
}

double DivisionRule::getRate(const NetworkState& state, const PopNetworkState& pop) {
  return rate->eval(NULL, state, pop);
}

NetworkState DivisionRule::applyRules(int daughter, const NetworkState& state, const PopNetworkState& pop) {
#ifdef USE_DYNAMIC_BITSET
  NetworkState res(state, 1);
#else
  NetworkState res(state);
#endif
  for (auto daughter_rule : daughters[daughter]) {
    res.setNodeState(daughter_rule.first, (bool)daughter_rule.second->eval(NULL, state, pop));
  }
  
  return res;
}

double PopNetwork::getDeathRate(const NetworkState& state, const PopNetworkState& pop) const {
  if (deathRate != NULL)
    return deathRate->eval(NULL, state, pop);
  else
    return 0.;
} 


void PopNetwork::initPopStates(PopNetworkState& initial_pop_state, RandomGenerator* randgen, unsigned int pop) {
  PopIStateGroup::initPopStates(this, initial_pop_state, randgen, pop);
}

void PopNetwork::display(std::ostream& os) const 
{
  Network::display(os);
  os << std::endl;
  if (deathRate != NULL) {
    os << "death {" << std::endl << "  rate = ";
    deathRate->display(os);
    os << ";" << std::endl << "}" << std::endl << std::endl;
  }
  
  for (auto rule: getDivisionRules())
  {
    rule->display(os);
    os << std::endl;
  }
}

PopNetwork::~PopNetwork()
{
  delete deathRate;
  for (auto * division_rule : divisionRules) {
    delete division_rule;
  }
  
  for (auto * pop_istate_group : *pop_istate_group_list) {
    delete pop_istate_group;
  }
  delete pop_istate_group_list;
  
}


PopNetwork::PopNetwork(const PopNetwork& pop_network) {
  *this = pop_network;
}

PopNetwork& PopNetwork::operator=(const PopNetwork& pop_network) {
  Network::operator=(pop_network);
  deathRate = pop_network.getDeathRate()->clone();
  for (auto division_rule: pop_network.getDivisionRules()) {
    divisionRules.push_back(division_rule);
  }
  return *this;
}
