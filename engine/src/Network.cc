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

#include <iostream>

#include "Network.h"
#include "parsers/BooleanGrammar.h"
#include "Utils.h"

#ifdef SBML_COMPAT
#include "sbml/SBMLParser.h"
#endif


extern FILE* ctbndlin;
extern void ctbndl_scan_expression(const char *);
extern int ctbndlparse();
extern int ctbndllex_destroy();

bool MaBoSS_quiet = false;

size_t Network::MAX_NODE_SIZE = 0;

Network::Network() : last_index(0U)
{
  istate_group_list = new std::vector<IStateGroup*>();
  symbol_table = new SymbolTable();
  set_current_network(NULL);
  set_pop_network(NULL);
}

int Network::parseExpression(const char* content, std::map<std::string, NodeIndex>* nodes_indexes){
  
  set_current_network(this);
  ctbndl_scan_expression(content);

  try 
  {
    int r = ctbndlparse();
    set_current_network(NULL);

    if (r) {
      ctbndllex_destroy();
      return 1;
    }
    compile(nodes_indexes);
    ctbndllex_destroy();
    return 0;
  }
  catch (const BNException&) 
  {
    set_current_network(NULL);
    ctbndllex_destroy();

    throw;
  }
}


Expression* Network::parseSingleExpression(const char* content, std::map<std::string, NodeIndex>* nodes_indexes)
{
  set_expression(NULL);
  const char * se = "SINGLE_EXPRESSION ";
  const char * ee = ";";
  std::string new_content = se;
  new_content += content;
  new_content += ee;
  Network::parseExpression(new_content.c_str(), nodes_indexes);
  
  return get_expression();
}

int Network::parse(const char* file, std::map<std::string, NodeIndex>* nodes_indexes, bool is_temp_file, bool useSBMLNames)
{
#ifdef SBML_COMPAT
  if (hasEnding(std::string(file), ".xml") || hasEnding(std::string(file), ".sbml")) {
    return this->parseSBML(file, nodes_indexes, useSBMLNames);
  }
#endif

  if (NULL != file) {
    ctbndlin = fopen(file, "r");
    if (ctbndlin == NULL) {
      throw BNException("network parsing: cannot open file:" + std::string(file) + " for reading");
    }
    if (is_temp_file) {
      unlink(file);
    }
  }

  set_current_network(this);

  try{
    int r = ctbndlparse();

    set_current_network(NULL);

    if (r) {
      if (NULL != file)
        fclose(ctbndlin);
      ctbndllex_destroy();

      return 1;
    }
    compile(nodes_indexes);

    if (NULL != file)
      fclose(ctbndlin);
    
    ctbndllex_destroy();

    return 0;
  }
  catch (const BNException&) 
  {  
    if (NULL != file)
      fclose(ctbndlin);
    
    set_current_network(NULL);
    ctbndllex_destroy();

    throw;
  }

}
#ifdef SBML_COMPAT

int Network::parseSBML(const char* file, std::map<std::string, NodeIndex>* nodes_indexes, bool useSBMLNames) 
{  
  SBMLParser* parser = new SBMLParser(this, file, useSBMLNames);
    
  parser->build();
  compile(nodes_indexes);
  parser->setIStates();
        
  return 0;
}

#endif

void Network::compile(std::map<std::string, NodeIndex>* nodes_indexes)
{
  MAP<std::string, Node*>::iterator begin = node_map.begin();

#if 0
  // checks for cycles
  // actually, not really pertinent...
  while (begin != node_map.end()) {
    Node* node = (*begin).second;
    if (!node->isInputNode()) {
      if (node->getLogicalInputExpr()->getRateUpExpression()->hasCycle(node)) {
	//std::cerr << "cycle detected for node " << node->getLabel() << '\n';
      }
    }
    ++begin;
  }
#endif

  // looks for input and non input nodes
  begin = node_map.begin();
  while (begin != node_map.end()) {
    Node* node = (*begin).second;
    if (!isNodeDefined(node->getLabel())) {
      throw BNException("node " + node->getLabel() + " used but not defined"); 
    }
    ++begin;
  }

  begin = node_map.begin();
  nodes.resize(node_map.size());
  while (begin != node_map.end()) {
    Node* node = (*begin).second;
    if (nodes_indexes != NULL) {
      node->setIndex((*nodes_indexes)[node->getLabel()]);
    }
    
    if (node->isInputNode()) {
      input_nodes.push_back(node);
    } else {
      non_input_nodes.push_back(node);
    }
    nodes[node->getIndex()] = node;
    ++begin;
  }
}

Node* Network::defineNode(const std::string& label, const std::string& description)
{
  if (isNodeDefined(label)) {
    throw BNException("node " + label + " already defined");
  }
  checkNewNode();
  Node* node = new Node(label, description, last_index++); // node factory
  setNodeAsDefined(label);
  node_map[label] = node;
  return node;
}

Node* Network::getNode(const std::string& label)
{
  if (node_map.find(label) == node_map.end()) {
    throw BNException("network: node " + label + " not defined");
  }
  return node_map[label];
}

void Network::initStates(NetworkState& initial_state, RandomGenerator * randgen)
{
  if (backward_istate) {
    for (const auto * node : nodes) {
      initial_state.setNodeState(node, node->getIState(this, randgen));
    }
  } else {
    IStateGroup::initStates(this, initial_state, randgen);
  }
}

void Network::display(std::ostream& os) const
{
  int nn = 0;
  for (const auto * node : nodes) {
    if (0 != nn) {
      os << '\n';
    }
    node->display(os);
    nn++;
  }
}

void Network::displayHeader(std::ostream& os) const
{
  int nn = 0;
  for (const auto * node : nodes) {
    os << (nn > 0 ? "\t" : "") << node->getLabel();
    nn++;
  }
  os << '\n';
}

Network::Network(const Network& network)
{
  *this = network;
}

Network& Network::operator=(const Network& network)
{
  node_map = network.node_map;
  last_index = network.last_index;
  input_nodes = network.input_nodes;
  non_input_nodes = network.non_input_nodes;
  nodes = network.nodes;
  symbol_table = network.symbol_table;
  return *this;
}

Network::~Network()
{
  delete symbol_table;
  
  for (auto * istate_group: *istate_group_list) {
    delete istate_group;
  }
  delete istate_group_list;
  
  for (auto & node : node_map) {
    delete node.second;
  }
}

void Network::cloneIStateGroup(std::vector<IStateGroup*>* _istate_group_list) 
{
  for (auto istate_group: *_istate_group_list) 
  {
    new IStateGroup(istate_group, this);
  }
}
