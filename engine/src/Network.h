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

#ifndef _NETWORK_H_
#define _NETWORK_H_

#include <string>
#include <vector>

#include "Node.h"
// #include "IStates.h"
#include "Symbols.h"

extern bool MaBoSS_quiet;
class IStateGroup;

// the boolean network (also used as a Node factory)
class Network {
  MAP<std::string, Node*> node_map;
  NodeIndex last_index;
  std::vector<Node*> input_nodes;
  std::vector<Node*> non_input_nodes;
  std::vector<Node*> nodes;

  MAP<std::string, bool> node_def_map;
  std::vector<IStateGroup*>* istate_group_list;
  SymbolTable* symbol_table;
  static size_t MAX_NODE_SIZE;

  // must be call before creating a new node
  void checkNewNode() {
    size_t size = node_map.size();
    if (size >= MAXNODES) {
      std::ostringstream ostr;
      ostr << "maximum node count exceeded: maximum allowed is " << MAXNODES;
      throw BNException(ostr.str());
    }
    if (size >= MAX_NODE_SIZE) {
      MAX_NODE_SIZE = size+1;
    }
  }

public:

  Network();

  Network(const Network& network);
  Network& operator=(const Network& network);

  int parse(const char* file = NULL, std::map<std::string, NodeIndex>* nodes_indexes = NULL, bool is_temp_file = false, bool useSBMLNames = false);
  int parseExpression(const char* content = NULL, std::map<std::string, NodeIndex>* nodes_indexes = NULL);
  Expression* parseSingleExpression(const char* content, std::map<std::string, NodeIndex>* nodes_indexes = NULL);
  
  #ifdef SBML_COMPAT
  int parseSBML(const char* file, std::map<std::string, NodeIndex>* nodes_indexes = NULL, bool useSBMLNames = false);
  #endif
  
  std::vector<IStateGroup*>* getIStateGroup() {
    return istate_group_list;
  }

  void cloneIStateGroup(std::vector<IStateGroup*>* _istate_group_list);

  SymbolTable* getSymbolTable() { 
    return symbol_table;
  };

  Node* defineNode(const std::string& label, const std::string& description = "");

  Node* getNode(const std::string& label);

  Node* getNode(NodeIndex node_idx) {
    assert(node_idx < getNodeCount());
    return nodes[node_idx];
  }

  Node* getOrMakeNode(const std::string& label) {
    if (node_map.find(label) != node_map.end()) {
      return node_map[label];
    }
    checkNewNode();
    Node* node = new Node(label, "", last_index++); // node factory
    node_map[label] = node;
    return node;
  }

  size_t getNodeCount() const {return node_map.size();}

  static size_t getMaxNodeSize() {
    //MAX_NODE_SIZE = 508; // for testing
    //static bool msg_displayed = false;
    static bool msg_displayed = true;
    if (!msg_displayed) {
      if (!MaBoSS_quiet) {
	std::cerr << "\nMaBoSS notice:\n";
	std::cerr << "  Using dynamic bitset implementation (any number of nodes): this version is not fully optimized and may use a large amount of memory\n";
	std::cerr << "  For this " << MAX_NODE_SIZE << " node network, preferably used ";
	if (MAX_NODE_SIZE <= 64) {
	  std::cerr << "the standard 'MaBoSS' program\n";
	} else {
	  std::cerr << "the static bitset implementation program 'MaBoSS_" << MAX_NODE_SIZE << "n' built using: make MAXNODES=" << MAX_NODE_SIZE << "\n";
	}
      }
      msg_displayed = true;
    }
    return MAX_NODE_SIZE;
  }

  void compile(std::map<std::string, NodeIndex>* nodes_indexes = NULL);

  // vector of nodes which do not depend on other nodes
  const std::vector<Node*>& getInputNodes() const {return input_nodes;}

  // vector of the other nodes
  const std::vector<Node*>& getNonInputNodes() const {return non_input_nodes;}

  // vector of all nodes
  const std::vector<Node*>& getNodes() const {return nodes;}

  std::string toString() const {
    std::ostringstream ostr;
    display(ostr);
    return ostr.str();
  }
  
  void initStates(NetworkState& initial_state, RandomGenerator* randgen);

  void displayHeader(std::ostream& os) const;

  virtual void display(std::ostream& os) const;

  void generateLogicalExpressions(std::ostream& os) const;

  bool isNodeDefined(const std::string& identifier) {
    return node_def_map.find(identifier) != node_def_map.end();
  }

  void setNodeAsDefined(const std::string& identifier) {
    node_def_map[identifier] = true;
  }

  void resetNodeDefinition() {
    node_def_map.clear();
  }

  void removeLastNode(const std::string& identifier) {
    /* This function was created to remove the last node which was just created. 
       It is only used when parsing a bnet file, to remove the "target, factors" which is part of the header
       Chances are that you should NEVER use it for another purpose.
    */
    if (node_map.find(identifier) != node_map.end()) {
      Node* to_delete = node_map[identifier];
      node_map.erase(identifier);
      if (to_delete->getIndex() == (last_index-1)) {
        last_index--;
      }
      delete to_delete;
      MAX_NODE_SIZE--;
    }
  }

  virtual bool isPopNetwork() { return false; }

  virtual ~Network();
};

#endif
