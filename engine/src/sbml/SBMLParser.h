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
     SBMLParser.h
     
     Authors:
     Vincent Noel <vincent-noel@curie.fr>
     Marco Ruscone <marco.ruscone@curie.fr>
     
     Date:
     June 2021
*/

#if defined SBML_COMPAT && !defined _SBML_PARSER_H_
#define _SBML_PARSER_H_
#include "../Network.h"
#include <sbml/packages/qual/extension/QualModelPlugin.h>
#include <sbml/SBMLTypes.h>

LIBSBML_CPP_NAMESPACE_USE


class SBMLParser
{
    Network* network;
    bool useSBMLNames;
    Model* model;
    QualModelPlugin* qual_model;
    std::map<std::string, int> maxLevels;
    std::map<std::string, int> initialLevels;
    std::map<std::string, std::vector<std::string> > fixedNames;

    std::string getName(std::string id, int level);
    void parseDocument(SBMLDocument* document);
    void parseTransition(Transition* transition);
    Expression* parseASTNode(const ASTNode* tree); 
    void createNodes(std::vector<std::string> names, Expression* exp);

  public:
  
    
    SBMLParser(Network* network, const char* file, bool useSBMLNames);
    SBMLParser(Network* network, SBMLDocument* document, bool useSBMLNames);
    
    void build();
    void setIStates();
};

#endif