#include <sbml/math/ASTNodeType.h>
#include <stdlib.h>
#include <vector>
#include <map>
#include <string> 
#include <sbml/math/ASTNode.h>

#include "../BooleanNetwork.h"

class SedASTExpression
{
  public:
    SedASTExpression(){}
    
    virtual std::vector<double> eval(std::map<std::string, std::map<std::string, std::vector<double> > >& results, std::map<std::string, std::map<std::string, std::vector<double> > >& state_results) =0;
    
};

class SedASTVariable: public SedASTExpression
{
  public:
    std::string task;
    std::string name;
    SedASTVariable(const std::string _task, const std::string _name) : task(_task), name(_name) { }
    
    std::vector<double> eval(std::map<std::string, std::map<std::string, std::vector<double> > >& results, std::map<std::string, std::map<std::string, std::vector<double> > >& state_results )
    {
        return results[task][name];
    }
};


class SedASTStateVariable: public SedASTExpression
{
  public:
    std::string task;
    std::string name;
    SedASTStateVariable(const std::string _task, const std::string _name) : task(_task), name(_name) { }
    
    std::vector<double> eval(std::map<std::string, std::map<std::string, std::vector<double> > >& results, std::map<std::string, std::map<std::string, std::vector<double> > >& state_results )
    {
        return state_results[task][name];
    }
};


static SedASTExpression* parseSedASTExpression(const ASTNode* tree, std::map<std::string, std::string>& variables)
{
    if (tree->getType() == AST_NAME) {
        
        return new SedASTVariable(variables[tree->getName()], tree->getName());    
        
    } else if (tree->getType() == AST_FUNCTION) {
        
        return new SedASTStateVariable(variables[tree->getName()], tree->getName());
        
        
    
    // } else if (tree->getType() == AST_LOGICAL_AND) {
        
    //     return new SedASTLogicalAnd(
    //         parseSedASTExpression(tree->getChild(0), variables),
    //         parseSedASTExpression(tree->getChild(1), variables)
    //     );
        
    // } else if (tree->getType() == AST_LOGICAL_NOT) {
        
    //     return new SedASTLogicalNot(
    //         parseSedASTExpression(tree->getChild(0), variables)
    //     );
    
    } else {
        return NULL;
    }
    
}

// static Expression* parseSedASTExpression(const ASTNode* tree, std::map<std::string, Expression*>& variables)
// {
//     if (tree->getType() == AST_NAME) {
        
//         return variables[tree->getName()];    
        
//     } else if (tree->getType() == AST_LOGICAL_AND) {
        
//         Expression * expr = parseSedASTExpression(tree->getChild(0), variables);
//         for (unsigned int i=1; i < tree->getNumChildren(); i++)
//         {
//             expr = new AndLogicalExpression(
//                 expr,
//                 parseSedASTExpression(tree->getChild(i), variables)
//             );
//         }
//         return expr;
        
//     } else if (tree->getType() == AST_LOGICAL_NOT) {
        
//         return new NotLogicalExpression(
//             parseSedASTExpression(tree->getChild(0), variables)
//         );
    
//     } else {
//         return NULL;
//     }
    
// }
