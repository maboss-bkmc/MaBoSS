#include <stdlib.h>
#include <string>
#include <vector>

#include "../RunConfig.h"

#include "SedException.h"

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
    
    ret.node = targetNode;
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

static Node* getTargetVariable(const std::string string_target, Network* network)
{
    SedTarget target = parseTarget(string_target);
    
    if (
        target.nodes.size() == 4
        && target.nodes[0].node.compare("sbml:sbml") == 0
        && target.nodes[1].node.compare("sbml:model") == 0
        && target.nodes[2].node.compare("qual:listOfQualitativeSpecies") == 0
        && target.nodes[3].node.compare("qual:qualitativeSpecies") == 0
        && target.nodes[3].selector_attribute.compare("qual:id") == 0
    )
    {
        return network->getNode(target.nodes[3].selector_value);
        
    } else {
        throw SedException("Unknown target node " + string_target);
    }
}