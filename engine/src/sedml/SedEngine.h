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

#include <sbml/SBMLDocument.h>
#include <sbml/SBase.h>
#include <sbml/math/ASTNode.h>
#include <sbml/math/ASTNodeType.h>
#include <sbml/math/L3FormulaFormatter.h>
#include <sbml/packages/qual/sbml/QualitativeSpecies.h>
#include <sedml/SedChangeAttribute.h>
#include <sedml/SedChangeXML.h>
#include <sedml/SedListOfVariables.h>
#include <sedml/SedRemoveXML.h>
#include <sedml/SedRepeatedTask.h>
#include <sedml/SedSetValue.h>
#include <sedml/SedUniformRange.h>
#include <sedml/SedUniformTimeCourse.h>
#include <string>
#define SEDML_COMPAT 1
#if defined SEDML_COMPAT && !defined _SEDML_PARSER_H_
#define _SEDML_PARSER_H_
#include "../Network.h"
#include "../engines/MaBEstEngine.h"
#include "../sbml/SBMLParser.h"

#include "SedException.h"
#include "SedTarget.h"
#include "SedAstExpression.h"
#include "Report.h"
#include "Plot2D.h"
#include "XMLPatcher.h"

#include <sedml/SedTypes.h>

#include <sbml/packages/qual/extension/QualModelPlugin.h>
#include <sbml/SBMLTypes.h>

#include <filesystem>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>

LIBSEDML_CPP_NAMESPACE_USE

namespace fs = std::filesystem;


// template<typename T, template<typename> class C>
// C<T> operator +(const C<T> &A, const C<T> &B) {
//   C<T> result;
//   std::transform(std::begin(A), std::end(A), std::begin(B), std::back_inserter(result), std::plus<T>{});

//   return result;
// }

static ASTNode* find_states(ASTNode* math, SedListOfVariables* variables, std::map<std::string, std::pair<Network*, RunConfig*>> tasks, std::map<std::string, std::pair<Expression*, std::string> >& states_variables)
{
    
    if (math->getType() == AST_LOGICAL_AND) {
        bool all_nodes = true;
        std::set<std::string> found_tasks;
        for (unsigned int i=0; i < math->getNumChildren(); i++)
        {
            ASTNode* child = math->getChild(i);
            
            if (child->getType() == AST_LOGICAL_NOT && child->getNumChildren() == 1) {
                child = child->getChild(0);                    
            }
            
            if (child->getType() == AST_NAME)
            {
                SedVariable* var = variables->get(child->getName());
                found_tasks.insert(var->getTaskReference());
                
            } else all_nodes = false;
            
            
            
        }
        if (all_nodes && found_tasks.size() == 1)
        {
            std::pair<Network*, RunConfig*> task = tasks[*found_tasks.begin()];
            NetworkState state;
            for (unsigned int i=0; i < math->getNumChildren(); i++)
            {
                ASTNode* child = math->getChild(i);
                if (child->getType() == AST_NAME){
                    Node* node = task.first->getNode(child->getName());
                    state.setNodeState(node, true);
                } else if (child->getType() == AST_LOGICAL_NOT && child->getNumChildren() == 1 && child->getChild(0)->getType() == AST_NAME) {
                    Node* node = task.first->getNode(child->getChild(0)->getName());
                    state.setNodeState(node, false);
                }
            }
            Expression* state_expression = new StateExpression(state, task.first);
            std::string state_name = state.getName(task.first);
            states_variables[state_name] = std::make_pair(state_expression, *found_tasks.begin());
            ASTNode* new_math = new ASTNode(AST_FUNCTION);
            new_math->setName(state_name.c_str());
            return new_math;
        } else return math;
    } else return math;
}



static SBMLDocument* patchXMLModel(const char* path_model, SedListOfChanges* listOfChanges)
{
    XMLPatcher patcher(path_model);

    for (unsigned int i=0; i < listOfChanges->getNumChanges(); i++) {
        SedChange* change = listOfChanges->get(i);
        if (change->isSedChangeXML()) {
            SedChangeXML* change_xml = static_cast<SedChangeXML*>(change);
            patcher.changeXML(change_xml->getTarget(), change_xml->getNewXML()->toXMLString());
        }  
        else if (change->isSedRemoveXML()) {
            SedRemoveXML* remove_xml = static_cast<SedRemoveXML*>(change);
            patcher.removeXML(remove_xml->getTarget());
        }
        else if (change->isSedAddXML()) {
            SedAddXML* add_xml = static_cast<SedAddXML*>(change);
            patcher.addXML(add_xml->getTarget(), add_xml->getNewXML()->toXMLString());
        }
    }
    SBMLReader reader;
    std::string xml = patcher.getXML();
    
    if (RunConfig::getVerbose() >= 2)
      std::cout << xml << std::endl;
    
    return reader.readSBMLFromString(xml);
}
static void patchModel(QualModelPlugin* model, SedChange* change)
{
    
    SedTarget target = parseTarget(change->getTarget());

    if (change->isSedChangeAttribute())
    {
        SedChangeAttribute* change_attribute = static_cast<SedChangeAttribute*>(change);
        
        if (target.nodes[0].node.compare("sbml:sbml") == 0)
        {
            if (target.nodes[1].node.compare("sbml:model") == 0)
            {        
                if (target.nodes[2].node.compare("qual:listOfQualitativeSpecies") == 0)
                {
                    if (target.nodes[3].node.compare("qual:qualitativeSpecies") == 0)    
                    {
                        QualitativeSpecies* species;
                        if (target.nodes[3].selector_attribute.compare("qual:id") == 0)
                        {
                            species = static_cast<QualitativeSpecies*>(model->getListOfQualitativeSpecies()->getElementBySId(target.nodes[3].selector_value));
                            
                        } else {
                            throw SedException("Unknown selector " + target.nodes[3].selector_attribute);
                        }
                        
                        if (target.attribute.compare("qual:initialLevel") == 0) {
                            species->setInitialLevel(stoi(change_attribute->getNewValue()));
                        }
                    }
                }
            }
        }
    }
}

static SBMLDocument* getDocument(fs::path path_model, SedListOfChanges* listOfChanges)
{
    SBMLDocument* document;
    SBMLReader reader;
    
    document = patchXMLModel(path_model.c_str(), listOfChanges);
    
    
    // document = reader.readSBML(path_model.c_str());
    
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
    // std::cout << document->toSBML() << std::endl;   
    for (unsigned int i=0; i < document->getNumErrors(); i++)
    {
        std::cout << "Error " << i << ": " << document->getError(i)->getMessage() << std::endl;
    }
    return document;
}
class SedEngine
{
  SedDocument* doc;
  fs::path doc_path;
  std::map<std::string, std::pair<Network*, RunConfig*> > tasks;
  std::map<std::string, std::pair<Network*, RunConfig*> > repeated_tasks;
  std::map<std::string, SedASTExpression*> data_generators;
  std::set<std::string> used_tasks;
  std::map<std::string, std::map<std::string, Expression*> > variables_by_task;
  std::map<std::string, std::map<std::string, Expression*> > states_by_task;
  std::map<std::string, std::vector<double> > results_by_data_generator;
  std::vector<Report> reports;
  std::vector<Plot2D> plots;
  
  public:
  
  SedEngine() {}
  
  
  
  void parse(std::string sedml_file) {
    
    std::map<std::string, Network*> networks;
    std::map<std::string, RunConfig*> configs;
    std::map<std::string, std::map<std::string, std::string> > variables_by_data_generator;
    std::map<std::string, std::map<std::string, std::string> > states_by_data_generator;
    // std::map<std::string, std::map<std::string, std::string> > variables_by_data_generator;
    std::map<std::string, std::map<std::string, std::vector<double> > > reports;
    
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
            if (algo->getKisaoID().compare("KISAO:0000450") != 0
                && algo->getKisaoID().compare("KISAO:0000581") != 0
            ) 
            {
                throw SedException("MaBoSS can only simulate Asynchronous logical models (KISAO_0000450) or BKMC (KISAO_0000581)");
            }
            
            RunConfig* config = new RunConfig();
            
            if (algo->getKisaoID().compare("KISAO:0000450") == 0)
            {
                config->setParameter("discrete_time", 1);
            } else {
                config->setParameter("discrete_time", 0);
            }
            config->setParameter("max_time", utc_simulation->getOutputEndTime());
            config->setParameter("time_tick", utc_simulation->getOutputEndTime()/utc_simulation->getNumberOfSteps());
            
            for (unsigned int j=0; j < algo->getNumAlgorithmParameters(); j++)
            {
                SedAlgorithmParameter* param = algo->getAlgorithmParameter(j);
                if (param->getKisaoIDasInt() == 488)
                {
                    config->setParameter("seed_pseudorandom", std::stod(param->getValue()));
                    
                } else if (param->getKisaoIDasInt() == 498)
                {
                    config->setParameter("sample_count", std::stod(param->getValue()));
                    
                } else if (param->getKisaoIDasInt() == 529)
                {
                    config->setParameter("thread_count", stod(param->getValue()));
                } else 
                {
                    if (RunConfig::getVerbose() > 0)
                    {
                        std::cout << "Unknown parameter " << param->getKisaoID() << " = " << param->getValue() << std::endl;
                    }
                }
            }
            
            
            
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
                throw SedException("Could not find model reference " + model_id);
            
            if (configs.find(simulation_id) == configs.end())
                throw SedException("Could not find simulation reference " + simulation_id);
            
            tasks[task->getId()] = std::make_pair(networks[model_id], configs[simulation_id]);
        } else if (abstract_task->isSedRepeatedTask())
        {
            SedRepeatedTask* task = static_cast<SedRepeatedTask*>(abstract_task);
            
            bool modify_only_initial_levels = true;
            bool uniform_random_number = true;
            bool only_consume_mean = true;

            for (unsigned int j=0; j < task->getListOfTaskChanges()->getNumTaskChanges(); j++)
            {
                SedSetValue* set_value = static_cast<SedSetValue*>(task->getListOfTaskChanges()->get(j));
                SedTarget target = parseTarget(set_value->getTarget());
                
                modify_only_initial_levels &= (target.nodes[0].node.compare("sbml:sbml") == 0
                    && target.nodes[1].node.compare("sbml:model") == 0
                    && target.nodes[2].node.compare("qual:listOfQualitativeSpecies") == 0
                    && target.nodes[3].node.compare("qual:qualitativeSpecies") == 0
                    && target.attribute.compare("qual:initialLevel") == 0);
                
                ASTNode* math = set_value->getMath();
                uniform_random_number &= (
                    (
                        math->getType() == AST_RELATIONAL_LEQ
                        || math->getType() == AST_RELATIONAL_GEQ
                        || math->getType() == AST_RELATIONAL_LT
                        || math->getType() == AST_RELATIONAL_GT
                    ) 
                    && math->getNumChildren() == 2
                    && (
                        (math->getChild(0)->getType() == AST_FUNCTION && strcmp(math->getChild(0)->getName(),"uniform") == 0 && math->getChild(1)->getType() == AST_REAL)
                        || (math->getChild(1)->getType() == AST_FUNCTION && strcmp(math->getChild(1)->getName(),"uniform") == 0 && math->getChild(0)->getType() == AST_REAL)
                    )
                );
                
                for (unsigned int k=0; k < doc->getListOfDataGenerators()->getNumDataGenerators(); k++)
                {
                    SedDataGenerator* data_generator = doc->getListOfDataGenerators()->get(k);
                    for (unsigned int l=0; l < data_generator->getListOfVariables()->getNumVariables(); l++)
                    {
                        SedVariable* variable = data_generator->getListOfVariables()->get(l);
                        only_consume_mean &= (
                            variable->getDimensionTerm().compare("KISAO:0000825") == 0
                            && variable->getTaskReference().compare(task->getId()) == 0
                            && variable->getListOfAppliedDimensions()->getNumAppliedDimensions() == 1
                            // && variable->getListOfAppliedDimensions()->get(0)->getTarget().compare(task->getId()) == 0
                        );
                    }
                }
                
            }
            
            if (
                task->getListOfRanges()->getNumRanges() == 1
                && task->getListOfSubTasks()->getNumSubTasks() == 1
                && modify_only_initial_levels & uniform_random_number && only_consume_mean
            )
            {
                // Here we should do something ??
                std::cout << "Is it MaBoSS ??" << std::endl;
                SedSubTask* subtask = task->getListOfSubTasks()->get(0);
                SedTask* base_task = static_cast<SedTask*>(doc->getListOfTasks()->get(subtask->getTask()));
                std::string model_id = base_task->getModelReference();
                std::string simulation_id = base_task->getSimulationReference();
                    
                if (networks.find(model_id) == networks.end())
                    throw SedException("Could not find model reference " + model_id);
                
                if (configs.find(simulation_id) == configs.end())
                    throw SedException("Could not find simulation reference " + simulation_id);
                
                RunConfig* new_config = new RunConfig(*configs[simulation_id]);
                tasks[task->getId()] = std::make_pair(networks[model_id], new_config);
                
                SedRange* range = task->getListOfRanges()->get(0);
                if (range->isSedUniformRange()){
                    SedUniformRange* uniform_range = static_cast<SedUniformRange*>(range);
                    new_config->setParameter("sample_count", uniform_range->getNumberOfSteps());
                }
                
                for (unsigned int j=0; j < task->getListOfTaskChanges()->getNumTaskChanges(); j++)
                {
                    SedSetValue* set_value = static_cast<SedSetValue*>(task->getListOfTaskChanges()->get(j));
                    SedTarget target = parseTarget(set_value->getTarget());
                    Node* node_target = getTargetVariable(set_value->getTarget(), networks[model_id]);
                    
                    if (target.attribute.compare("qual:initialLevel") == 0)
                    {
                        ASTNode* math = set_value->getMath();
                        if (math->getType() == AST_RELATIONAL_GEQ || math->getType() == AST_RELATIONAL_GT)
                        {
                            if (math->getChild(0)->getType() == AST_FUNCTION)
                            {
                                IStateGroup::setNodeProba(networks[model_id], node_target, 1.0-math->getChild(1)->getReal());
                            }
                            else
                            {
                                IStateGroup::setNodeProba(networks[model_id], node_target, 1.0-math->getChild(0)->getReal());   
                            }
                            
                        } else if (math->getType() == AST_RELATIONAL_LEQ || math->getType() == AST_RELATIONAL_LT)
                        {
                            if (math->getChild(0)->getType() == AST_FUNCTION)
                            {
                                IStateGroup::setNodeProba(networks[model_id], node_target, math->getChild(1)->getReal());
                            }
                            else
                            {
                                IStateGroup::setNodeProba(networks[model_id], node_target, math->getChild(0)->getReal());   
                            }
                            
                        // } else {
                            // throw SedException("Unknown sedml expression " + SBML_formulaToL3String(set_value->getMath()));
                        }
                        
                    }
                }
                
            } else {
                std::cout << "Modify only initial levels: " << modify_only_initial_levels << std::endl;
                std::cout << "Uniform random number: " << uniform_random_number << std::endl;
            }
            
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

        std::set<std::string> tasks_dg;
        
        
        ASTNode* math = data_generator->getMath();
        std::map<std::string, std::pair<Expression*, std::string>> state_variables;
        math = find_states(math, data_generator->getListOfVariables(), tasks, state_variables);
        if (state_variables.size() > 0) {
            for (auto& state_variable: state_variables){
                
                states_by_data_generator[data_generator->getId()][state_variable.first] = state_variable.second.second;
                states_by_task[state_variable.second.second][state_variable.first] = state_variable.second.first;
            
            }
            data_generators[data_generator->getId()] = parseSedASTExpression(math, states_by_data_generator[data_generator->getId()]);
        } else {
            
            
            for (unsigned int j=0; j < data_generator->getListOfVariables()->getNumVariables(); j++)
            {
                SedVariable* variable = data_generator->getListOfVariables()->get(j);
                std::pair<Network*, RunConfig*> task;
                if (tasks.find(variable->getTaskReference()) != tasks.end())
                    task = tasks[variable->getTaskReference()];
                // else if (repeated_tasks.find(variable->getTaskReference()) != repeated_tasks.end())
                //     task = repeated_tasks[variable->getTaskReference()];
                else
                    throw SedException("Could not find task reference " + variable->getTaskReference());
                
                used_tasks.insert(variable->getTaskReference());
                // tasks_dg.insert(variable->getTaskReference());
                if (!variable->getSymbol().empty()) {
            
                    Expression* expr = new TimeExpression();
                    variables_by_data_generator[data_generator->getId()][variable->getId()] = variable->getTaskReference();
                    variables_by_task[variable->getTaskReference()][variable->getId()] = expr;
                    
                } else if (!variable->getTarget().empty()) {
                    SedTarget tar = parseTarget(variable->getTarget());
                    Node* node_var = getTargetVariable(variable->getTarget(), task.first);
                    // It is used, so we turn it output
                    node_var->isInternal(false);
                    Expression* expr = new NodeExpression(node_var);
                    variables_by_data_generator[data_generator->getId()][variable->getId()] = variable->getTaskReference();
                    variables_by_task[variable->getTaskReference()][variable->getId()] = expr;
                }
            }
            
            
            
            data_generators[data_generator->getId()] = parseSedASTExpression(data_generator->getMath(), variables_by_data_generator[data_generator->getId()]);
        }
    }
    
    if (RunConfig::getVerbose() > 0)
    {
        std::cout << "> Loaded " << data_generators.size() << " data generators from sedml file" << std::endl;
    }
  } 
  
  void run() 
  {  
    std::map<std::string, MaBEstEngine*> results;
    std::map<std::string, std::map<std::string, std::vector<double> > > results_by_task;
    std::map<std::string, std::map<std::string, std::vector<double> > > states_results_by_task;
  
    // Now we should have everything to run the simulations
    for (auto& used_task: used_tasks) {
        auto task = tasks[used_task];
        MaBEstEngine* simulation = new MaBEstEngine(
            task.first,
            task.second
        );
        
        if (RunConfig::getVerbose() > 0)
        {
            std::cout << "> Running task " + used_task + "...";    
            if (RunConfig::getVerbose() > 2)
            {
                std::cout << task.first->toString() << std::endl; 
                task.second->dump(task.first, std::cout, MaBEstEngine::VERSION);
            }
        }
        
        simulation->run();
        
        if (RunConfig::getVerbose() > 0)
            std::cout << "Done" << std::endl;
            
        results[used_task] = simulation;
        
        if (RunConfig::getVerbose() > 0)
            std::cout << "> Getting " << variables_by_task[used_task].size() << " variables from task " + used_task + " results...";    
                
        results_by_task[used_task] = simulation->getMergedCumulator()->getVariablesTimecourse(variables_by_task[used_task]);
        states_results_by_task[used_task] = simulation->getMergedCumulator()->getVariablesTimecourse(states_by_task[used_task]);
        // for (auto& data_generator: data_generators){
        //     results_by_data_generator[data_generator.first] = simulation->getMergedCumulator()->getExpressionsTimecourse(data_generator.second);
        // }
    
        if (RunConfig::getVerbose() > 0)
            std::cout << "Done" << std::endl;         
        
    }
    
    for (auto& data_generator: data_generators){
        results_by_data_generator[data_generator.first] = data_generator.second->eval(results_by_task, states_results_by_task);
    }
    
    for (unsigned int i=0; i < doc->getListOfOutputs()->getNumOutputs(); i++)
    {
        SedOutput* output = doc->getListOfOutputs()->get(i);
        if (output->isSedReport())
        {
            SedReport* sed_report = static_cast<SedReport*>(output);
            Report report(sed_report, results_by_data_generator);
            reports.push_back(report);
            
        } else if (output->isSedPlot2D())
        {
            SedPlot2D* sed_plot = static_cast<SedPlot2D*>(output);
            Plot2D plot(sed_plot, results_by_data_generator, doc->getListOfStyles());
            plots.push_back(plot);
            
        } else {
            if (RunConfig::getVerbose() > 0)
            {
                std::cout << "Output " << output->getName() << " is not a report nor a plot2D, so we ignore it" << std::endl;
            }
        }
    }
  }

  void writeReports()
  {
    for (auto& report: reports)
    {
        report.writeReport(fs::path(report.getName()));
    }
  }
  
  const std::vector<Report>& getReports() const
  {
      return reports;
  }
  
  const std::vector<Plot2D>& getPlots() const
  {
    return plots;
  }
};

#endif