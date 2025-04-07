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
     SEDMLParser.h
     
     Authors:
     Vincent Noel <vincent-noel@curie.fr>
     
     Date:
     June 2025
*/
#include <sbml/packages/qual/sbml/QualitativeSpecies.h>
#include <sedml/SedChangeAttribute.h>
#include <sedml/SedChangeXML.h>
#include <sedml/SedRemoveXML.h>
#define SEDML_COMPAT 1
#if defined SEDML_COMPAT && !defined _SEDML_PARSER_H_
#define _SEDML_PARSER_H_
#include "BooleanNetwork.h"
#include "MaBEstEngine.h"
#include "SBMLParser.h"

// #include <sedml/SedReader.h>
// #include <sedml/SedDocument.h>
// #include <sedml/SedModel.h>
// #include <sedml/SedUniformTimeCourse.h>
// #include <sedml/SedTask.h>
// #include <sedml/SedReport.h>
#include <sedml/SedTypes.h>

#include <sbml/packages/qual/extension/QualModelPlugin.h>
#include <sbml/SBMLTypes.h>

#include <filesystem>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>


namespace fs = std::filesystem;


template<typename T, template<typename> class C>
C<T> operator +(const C<T> &A, const C<T> &B) {
  C<T> result;
  std::transform(std::begin(A), std::end(A), std::begin(B), std::back_inserter(result), std::plus<T>{});

  return result;
}

class SEDMLException: public BNException {
    using BNException::BNException;
};

class SedASTExpression
{
  public:
    SedASTExpression(){}
    
    virtual std::vector<double> eval(std::map<std::string, std::map<std::string, std::vector<double> > >& results) =0;
    
};

class SedASTVariable: public SedASTExpression
{
  public:
    std::string task;
    std::string name;
    SedASTVariable(const std::string _task, const std::string _name) : task(_task), name(_name) { }
    
    std::vector<double> eval(std::map<std::string, std::map<std::string, std::vector<double> > >& results)
    {
        return results[task][name];
    }
};

SedASTExpression* parseSedASTExpression(const ASTNode* tree, std::map<std::string, std::string>& variables)
{
    if (tree->getType() == AST_NAME) {
        
        return new SedASTVariable(variables[tree->getName()], tree->getName());    
        
    } else {
        return NULL;
    }
    
}

struct SedTargetNode {
    // std::vector<std::string> namespaces;
    std::string node;
    std::string selector_attribute;
    std::string selector_value;  
};

struct SedTarget {
    std::vector<SedTargetNode> nodes;
    std::string attribute;
};

static SedTargetNode parseTargetNode(std::string targetNode)
{
    SedTargetNode ret;
    size_t pos_left_bracket = targetNode.find("[");
    size_t pos_right_bracket = targetNode.find("]");
    
    if (pos_left_bracket != std::string::npos && pos_right_bracket != std::string::npos)
    {
        std::string name_target = targetNode.substr(pos_left_bracket+1, pos_right_bracket-1);
        size_t pos3 = name_target.find("=");
        ret.selector_attribute = name_target.substr(1, pos3-1);
        ret.selector_value = name_target.substr(pos3+2, name_target.size()-pos3-4);

        targetNode = targetNode.substr(0, pos_left_bracket);

    } 
    
    // size_t last_pos = targetNode.find(":");
    // size_t pos = targetNode.find(":",last_pos+1);
    // while (pos != std::string::npos) {
    //     ret.namespaces.push_back(targetNode.substr(last_pos+1,pos-last_pos-1));
        
    //     last_pos = pos;
    //     pos = targetNode.find(":", pos+1);
    // }
    ret.node = targetNode;
    
    // if (pos_left_bracket == std::string::npos && pos_right_bracket == std::string::npos)
    // {
    //     ret.node = targetNode;
    // }
    return ret;
     
}

static SedTarget parseTarget(const std::string target)
{
    SedTarget ret;
    std::string token;
    std::vector<std::string> tokens;
    size_t last_pos = target.find("/");
    size_t pos = target.find("/",last_pos+1);
    while (last_pos != std::string::npos) {
        token = target.substr(last_pos+1,pos-last_pos-1);
        tokens.push_back(token);
        
        last_pos = pos;
        pos = target.find("/", pos+1);
    }
    
    if (tokens[tokens.size()-1][0] == '@') {
        std::string t_attr = tokens[tokens.size()-1];
        ret.attribute = t_attr.substr(1, t_attr.size()-1);
        tokens.pop_back();
    }
    
    for (std::string token: tokens)
    {
        SedTargetNode tn = parseTargetNode(token);
        ret.nodes.push_back(tn);
    }
    
    if (RunConfig::getVerbose() >= 2) 
    {
        std::cout << "> Target " << target << ": " << std::endl;
        for (auto node: ret.nodes) {
            std::cout << ">> Node: " << node.node << std::endl;
            if (!node.selector_attribute.empty() && !node.selector_value.empty())
            {
                std::cout << ">>> selector_attribute: " << node.selector_attribute << std::endl;
                std::cout << ">>> selector_value: " << node.selector_value << std::endl;
            }
        }
        
        if (!ret.attribute.empty())
            std::cout << ">> Attribute: " << ret.attribute << std::endl; 
    }
    return ret;
}

static Node* parseNodeTarget(const std::string target, Network* network)
{
    
    std::vector<std::string> elements;
    std::string token;
    std::string type;
    std::string id;
    auto last_pos = target.find("/");
    auto pos = target.find("/",last_pos+1);
    while (pos != std::string::npos) {
        token = target.substr(last_pos,pos-last_pos);
        elements.push_back(token);
        
        last_pos = pos;
        pos = target.find("/", pos+1);
    }
    
    token = target.substr(last_pos,pos-last_pos);
    
    auto pos2 = token.find("[");
    auto name_target = token.substr(pos2+1, token.size()-pos2-3);
    token = token.substr(0, pos2);
    elements.push_back(token);

    auto pos3 = name_target.find("=");
    type = name_target.substr(1, pos3-1);
    id = name_target.substr(pos3+2, name_target.size()-pos3-2);

    return network->getNode(id);
}

static void patchModel(QualModelPlugin* model, SedChange* change)
{
    
    SedTarget target = parseTarget(change->getTarget());
    
    if (change->isSedChangeAttribute())
    {
        SedChangeAttribute* change_attribute = static_cast<SedChangeAttribute*>(change);
        if (target.nodes[0].node.compare("sbml:sbml") == 0 && target.nodes[1].node.compare("sbml:model") == 0)
        {
            if (
                target.nodes[2].node.compare("sbml:qual:listOfQualitativeSpecies") == 0 && 
                target.nodes[3].node.compare("sbml:qual:qualitativeSpecies") == 0
            )
            {
                QualitativeSpecies* species;
                if (target.nodes[3].selector_attribute.compare("id") == 0 || target.nodes[3].selector_attribute.compare("qual:id") == 0)
                {
                    species = static_cast<QualitativeSpecies*>(model->getListOfQualitativeSpecies()->getElementBySId(target.nodes[3].selector_value));
                    
                } else {
                    throw SEDMLException("Unknown selector " + target.nodes[3].selector_attribute);
                }
                
                if (target.attribute.compare("initialLevel") == 0) {
                    species->setInitialLevel(stoi(change_attribute->getNewValue()));
                }
            }
        }
    }    
    else if (change->isSedRemoveXML())
    {
        SedRemoveXML* remove_xml = static_cast<SedRemoveXML*>(change);
        
    } else if (change->isSedChangeXML()) 
    {
        SedChangeXML* change_xml = static_cast<SedChangeXML*>(change);
        
    }
}

static SBMLDocument* getDocument(fs::path path_model, SedListOfChanges* listOfChanges)
{
    SBMLDocument* document;
    SBMLReader reader;
    
    document = reader.readSBML(path_model.c_str());
    
    SBasePlugin* qual = document->getPlugin("qual");
    if (qual == NULL) {
        throw BNException("This SBML model is not a qualitative sbml");
    }
    
    Model* model = document->getModel();
    QualModelPlugin* qual_model = static_cast<QualModelPlugin*>(model->getPlugin("qual"));

    for (unsigned int i=0; i < listOfChanges->getNumChanges(); i++) {
        SedChange* change = listOfChanges->get(i);
        patchModel(qual_model, change);
    }
    return document;
}
class SEDMLParser
{
  SedDocument* doc;
  fs::path doc_path;
  std::map<std::string, Network*> networks;
  std::map<std::string, RunConfig*> default_configs;
  std::map<std::string, RunConfig*> configs;
  std::map<std::string, std::pair<Network*, RunConfig*> > tasks;
  std::map<std::string, SedASTExpression*> data_generators;
  std::map<std::string, std::map<std::string, std::string> > variables_by_data_generator;
  std::map<std::string, std::map<std::string, Expression*> > variables_by_task;
  std::map<std::string, MaBEstEngine*> results;
  std::map<std::string, std::map<std::string, std::vector<double> > > results_by_task;
  std::map<std::string, std::vector<double> > results_by_data_generator;
  public:
  
  SEDMLParser() {}
  
  void parse(std::string sedml_file) {
    
    doc = readSedMLFromFile(sedml_file.c_str());
    
    doc_path = fs::path(sedml_file);
    
    for (unsigned int i=0; i < doc->getListOfModels()->getNumModels(); i++)
    {
        
        SedModel* model = doc->getListOfModels()->get(i);
        
        
        
        fs::path model_source(model->getSource());
        fs::path path_model = doc_path.parent_path() / model_source;
        Network* network = new Network();
        SBMLDocument* document = getDocument(path_model, model->getListOfChanges());
        
        SBMLParser* parser = new SBMLParser(network, document, false);
        
        parser->build();
        network->compile(NULL);
        parser->setIStates();
        IStateGroup::checkAndComplete(network);

        // Set all nodes as internal, we will turn them on during the description of data generation
        for (auto* node: network->getNodes())
        {    
            node->isInternal(true);
        }
        
        networks[model->getId()] = network;
    }

    if (RunConfig::getVerbose() > 0)
    {
        std::cout << "> Loaded " << networks.size() << " models from sedml file" << std::endl;
    }
    
    for (unsigned int i=0; i < doc->getListOfSimulations()->getNumSimulations(); i++)
    {
        SedSimulation* simulation = doc->getListOfSimulations()->get(i);
        if (simulation->isSedUniformTimeCourse())
        {
            SedUniformTimeCourse* utc_simulation = static_cast<SedUniformTimeCourse*>(simulation);
            SedAlgorithm* algo = utc_simulation->getAlgorithm();
            if (algo->getKisaoID().compare("KISAO:0000019") == 0) 
            {
                throw SEDMLException("Simulation algorithm is not BKMC");
            }
            
            RunConfig* config = new RunConfig();
            config->setParameter("max_time", utc_simulation->getOutputEndTime());
            config->setParameter("time_tick", utc_simulation->getOutputEndTime()/utc_simulation->getNumberOfSteps());
            configs[utc_simulation->getId()] = config;
        }
    }
    
    if (RunConfig::getVerbose() > 0)
    {
        std::cout << "> Loaded " << configs.size() << " simulations from sedml file" << std::endl;
    }
    
    for (unsigned int i=0; i < doc->getListOfTasks()->getNumAbstractTasks(); i++)
    {
        SedAbstractTask* abstract_task = doc->getListOfTasks()->get(i);
        if (abstract_task->isSedTask()) {
            
            SedTask* task = static_cast<SedTask*>(abstract_task);
            std::string model_id = task->getModelReference();
            std::string simulation_id = task->getSimulationReference();
            
            if (networks.find(model_id) == networks.end())
                throw SEDMLException("Could not find model reference " + model_id);
            
            if (configs.find(simulation_id) == configs.end())
                throw SEDMLException("Could not find simulation reference " + simulation_id);
            
            tasks[task->getId()] = std::make_pair(networks[model_id], configs[simulation_id]);
        }
    }
    
    if (RunConfig::getVerbose() > 0)
    {
        std::cout << "> Loaded " << tasks.size() << " tasks from sedml file" << std::endl;
    }
   
    for (unsigned int i=0; i < doc->getListOfDataGenerators()->getNumDataGenerators(); i++)
    {
        SedDataGenerator* data_generator = doc->getListOfDataGenerators()->get(i);
        std::map<std::string, std::pair<std::string, Expression*> > variables;

        for (unsigned int j=0; j < data_generator->getListOfVariables()->getNumVariables(); j++)
        {
            SedVariable* variable = data_generator->getListOfVariables()->get(j);
            std::pair<Network*, RunConfig*> task = tasks[variable->getTaskReference()];
            if (!variable->getSymbol().empty()) {
        
                Expression* expr = new TimeExpression();
                variables_by_data_generator[data_generator->getId()][variable->getId()] = variable->getTaskReference();
                variables_by_task[variable->getTaskReference()][variable->getId()] = expr;
                
            } else if (!variable->getTarget().empty()) {
                SedTarget tar = parseTarget(variable->getTarget());
                Node* node_var = parseNodeTarget(variable->getTarget(), task.first);
                // It is used, so we turn it output
                node_var->isInternal(false);
                Expression* expr = new NodeExpression(node_var);
                variables_by_data_generator[data_generator->getId()][variable->getId()] = variable->getTaskReference();
                variables_by_task[variable->getTaskReference()][variable->getId()] = expr;
            }
        }
        
        data_generators[data_generator->getId()] = parseSedASTExpression(data_generator->getMath(), variables_by_data_generator[data_generator->getId()]);
    }
    
    if (RunConfig::getVerbose() > 0)
    {
        std::cout << "> Loaded " << data_generators.size() << " data generators from sedml file" << std::endl;
    }
    
    // Now we should have everything to run the simulations
    for (auto& task: tasks) {
        MaBEstEngine* simulation = new MaBEstEngine(
            task.second.first,
            task.second.second
        );
        
        if (RunConfig::getVerbose() > 0)
            std::cout << "> Running task " + task.first + "...";    
        
        simulation->run();
        
        if (RunConfig::getVerbose() > 0)
            std::cout << "Done" << std::endl;
            
        results[task.first] = simulation;
        
        if (RunConfig::getVerbose() > 0)
            std::cout << "> Getting " << variables_by_task[task.first].size() << " variables from task " + task.first + " results...";    
        
        results_by_task[task.first] = simulation->getMergedCumulator()->getVariablesTimecourse(variables_by_task[task.first]);
        
        if (RunConfig::getVerbose() > 0)
            std::cout << "Done" << std::endl;         
        
    }
    
    for (auto& data_generator: data_generators)
        results_by_data_generator[data_generator.first] = data_generator.second->eval(results_by_task);
    
    
    for (unsigned int i=0; i < doc->getListOfOutputs()->getNumOutputs(); i++)
    {
        SedOutput* output = doc->getListOfOutputs()->get(i);
        if (output->isSedReport())
        {
            SedReport* report = static_cast<SedReport*>(output);
            
                
            std::ofstream report_file;
            report_file.open(report->getName());
    
            for (unsigned int j=0; j < report->getListOfDataSets()->getNumDataSets(); j++)
            {
                SedDataSet* dataset = report->getListOfDataSets()->get(j);
                report_file << dataset->getLabel();
                
                if (j < report->getListOfDataSets()->getNumDataSets()-1)
                    report_file << ",";    
                
            }
            report_file << std::endl;
            
            size_t n_samples = results_by_data_generator[report->getListOfDataSets()->get(0)->getDataReference()].size();
            
            for (size_t t=0; t < n_samples; t++)
            {
                for (unsigned int j=0; j < report->getListOfDataSets()->getNumDataSets(); j++)
                {
                    SedDataSet* dataset = report->getListOfDataSets()->get(j);
                    report_file << results_by_data_generator[dataset->getDataReference()][t];
                    
                    if (j < report->getListOfDataSets()->getNumDataSets()-1)
                        report_file << ",";    

                }
                
                report_file << std::endl;

            }    
            
            report_file.close();
            
        } else {
            if (RunConfig::getVerbose() > 0)
            {
                std::cout << "Output " << output->getName() << " is not a report, so we ignore it" << std::endl;
            }
        }
    }
  }

    


};

#endif