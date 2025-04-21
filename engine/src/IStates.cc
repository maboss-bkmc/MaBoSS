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
     IStates.cc

   Authors:
     Eric Viara <viara@sysra.com>
     Gautier Stoll <gautier.stoll@curie.fr>
     Vincent Noël <vincent.noel@curie.fr>
 
   Date:
     January-March 2011
     updated October 2014
*/

#include "IStates.h"
#include "PopNetwork.h"

void PopIStateGroup::epilogue(PopNetwork* network) 
{
  network->getPopIStateGroup()->push_back(this);
}

void IStateGroup::checkAndComplete(Network* network)
{
  std::map<std::string, bool> node_label_map;
  for (auto * istate_group : *(network->getIStateGroup())) {
    std::vector<const Node*>* nodes = istate_group->getNodes();

    for (const auto * node : *nodes) {
      if (node_label_map.find(node->getLabel()) != node_label_map.end()) {
	      throw BNException("duplicate node istate declaration: " + node->getLabel());
      }
      node_label_map[node->getLabel()] = true;
    }
  }

  const std::vector<Node*>& nodes = network->getNodes();
  for (const auto * node : nodes) {
    if (node_label_map.find(node->getLabel()) == node_label_map.end()) {
      new IStateGroup(network, node);
    }
  }

  // now complete missing nodes
}

void IStateGroup::initStates(Network* network, NetworkState& initial_state, RandomGenerator* randgen)
{
  for (auto * istate_group : *(network->getIStateGroup())) {
    
    std::vector<const Node*>* nodes = istate_group->getNodes();
    std::vector<ProbaIState*>* proba_istates = istate_group->getProbaIStates();

    if (proba_istates->size() == 1) {
      ProbaIState* proba_istate = (*proba_istates)[0];
      std::vector<double>* state_value_list = proba_istate->getStateValueList();
      size_t nn = 0;
      for (const auto & value : *state_value_list) {
        const Node* node = (*nodes)[nn++];
        initial_state.setNodeState(node, value != 0. ? true : false);
      }
    } else {
      double rand = randgen->generate();
      assert(rand >= 0. && rand <= 1.);
      size_t proba_istate_size = proba_istates->size();
      double proba_sum = 0;
      size_t jj = 0;
      for (auto * proba_istate : *proba_istates) {
        proba_sum += proba_istate->getProbaValue();
        //std::cerr << "rand: " << rand << " " << proba_sum << '\n';
        if (rand < proba_sum || jj == proba_istate_size - 1) {
          std::vector<double>* state_value_list = proba_istate->getStateValueList();
          size_t nn = 0;
          //std::cerr << "state #" << jj << " choosen\n";
          for (const auto & value : *state_value_list) {
            const Node* node = (*nodes)[nn++];
            initial_state.setNodeState(node, value != 0. ? true : false);
          }
          break;
        }
      }
    }
  }
}

void PopIStateGroup::initPopStates(PopNetwork* network, PopNetworkState& initial_state, RandomGenerator* randgen, unsigned int pop)
{
  PopNetwork* pop_network = static_cast<PopNetwork*>(network);
  initial_state.clear();
  if (pop_network->getPopIStateGroup()->size() > 0) 
  {
    // std::cout << "Creating a pop state from " << pop_network->getPopIStateGroup()->size() << " istate groups" << std::endl;
    for (auto * istate_group: *(pop_network->getPopIStateGroup())) 
    {  
      std::vector<const Node*>* nodes = istate_group->getNodes();
      std::vector<PopProbaIState*>* proba_istates = istate_group->getPopProbaIStates();  
    
      if (proba_istates->size() == 1)
      {
        PopProbaIState* proba_istate = (*proba_istates)[0];
        
        std::vector<PopIStateGroup::PopProbaIState::PopIStateGroupIndividual*>* individual_list = proba_istate->getIndividualList();
                
        for (auto * individual: *individual_list) 
        {   
          int i=0;
          NetworkState network_state;
          for (const auto & value: individual->getStateValueList()) {
            const Node* node = (*nodes)[i++];
            network_state.setNodeState(node, value != 0. ? true : false);
          }
          
          initial_state.addStatePop(network_state.getState(), individual->getPopSize());
        }
      } else {
        double rand = randgen->generate();
        assert(rand >= 0. && rand <= 1.);
        
        NetworkState network_state;
        unsigned int pop_size = 0;
        double proba_sum = 0;
        
        // std::cout << "Creating a random initial state from " << proba_istates->size() << " possibilities " << std::endl;
        for (auto * proba_istate: *proba_istates) 
        {
          proba_sum += proba_istate->getProbaValue();
          if (rand < proba_sum)
          {
            for (auto * individual: *(proba_istate->getIndividualList())) 
            {
              pop_size = individual->getPopSize();
              int i=0;
              for (const auto & value: individual->getStateValueList()) 
              {
                const Node* node = (*nodes)[i++];
                network_state.setNodeState(node, value != 0. ? true : false);
              }
              initial_state.addStatePop(network_state.getState(), pop_size); 

            }

            break; 
          }
        }
      }
    }
  } else {
    NetworkState state;
    IStateGroup::initStates(network, state, randgen);
    initial_state = PopNetworkState(state.getState(), pop);
          // std::cout << "State chosen : " << initial_state.getName(network) << std::endl << std::endl;

  }
}

void PopIStateGroup::display(PopNetwork* network, std::ostream& os)
{
  for (auto * popistate_group: *network->getPopIStateGroup()) {
    std::vector<const Node*>* nodes = popistate_group->getNodes();
    std::vector<PopProbaIState*>* proba_istates = popistate_group->getPopProbaIStates();  
    
    os << '[';
    size_t nn = 0;
    for (const auto * node : *nodes) {
      os << (nn > 0 ? ", " : "") << node->getLabel();
      nn++;
    }
    os << "].pop_istate = ";

    size_t jj = 0;
    
    for (auto * proba_istate: *proba_istates) 
    {
      os << (jj > 0 ? ", " : "") << proba_istate->getProbaValue() << " [";
      
      size_t kk = 0;
      for (auto * individual: *(proba_istate->getIndividualList())) 
      {
        os << (kk > 0 ? ", " : "") << "{[";
        size_t ll = 0;
        for (const auto & value: individual->getStateValueList()) 
        {
          os << (ll > 0 ? ", " : "") << ((int) value);
          ll++;
        }

        os << "]: " << individual->getPopSize() << "}";
        kk++;
      }


      os << "]";
      jj++;
    }

    os << ";\n";
  }
}

void IStateGroup::display(Network* network, std::ostream& os)
{
  for (auto * istate_group: *network->getIStateGroup()) {
    std::vector<const Node*>* nodes = istate_group->getNodes();
    std::vector<ProbaIState*>* proba_istates = istate_group->getProbaIStates();

    if (nodes->size() == 1 && proba_istates->size() == 1) {
      std::vector<double>* state_value_list = (*proba_istates)[0]->getStateValueList();
      os << (*nodes)[0]->getLabel() << ".istate = " << ((*state_value_list)[0] != 0. ? "1" : "0") << ";\n";
      continue;
    }
    
    if (nodes->size() == 1 && proba_istates->size() == 2 && (*proba_istates)[0]->getProbaValue() == 0.5 && (*proba_istates)[1]->getProbaValue() == 0.5)
    {
      continue;
    }
    
    os << '[';
    size_t nn = 0;
    for (const auto * node : *nodes) {
      os << (nn > 0 ? ", " : "") << node->getLabel();
      nn++;
    }
    os << "].istate = ";

    size_t jj = 0;
    for (auto * proba_istate: *proba_istates) {
      os << (jj > 0 ? ", " : "") << proba_istate->getProbaValue() << " [";
      std::vector<double>* state_value_list = proba_istate->getStateValueList();
      size_t ii = 0;
      for (auto & value : *state_value_list) {
        os << (ii > 0 ? ", " : "") << value;
        ii++;
      }
      os << "]";
      jj++;
    }

    os << ";\n";
  }
}

void IStateGroup::reset(Network * network) {
  network->getIStateGroup()->clear();
}
