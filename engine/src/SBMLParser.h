#include "BooleanNetwork.h"
#include "BooleanGrammar.h"
#include <sbml/packages/qual/extension/QualModelPlugin.h>
#include <sbml/SBMLTypes.h>

class SBMLParser
{
  public:
  Network* network;
  Model* model;
  QualModelPlugin* qual_model;
  std::map<std::string, int> maxLevels;

  SBMLParser(Network* network, const char* file) : network(network) {
    
    SBMLDocument* document;
    SBMLReader reader;
    
    document = reader.readSBML(file);
    unsigned int errors = document->getNumErrors();
    
    if (errors > 0) {
        throw BNException("There are errors in the sbml file");  
    }
    
    SBasePlugin* qual = document->getPlugin("qual");
    if (qual == NULL) {
        throw BNException("This SBML model is not a qualitative sbml");
    }
    
    this->model = document->getModel();
    this->qual_model = static_cast<QualModelPlugin*>(model->getPlugin("qual"));
  
    for (unsigned int i=0; i < qual_model->getNumQualitativeSpecies(); i++) {
        QualitativeSpecies* specie = qual_model->getQualitativeSpecies(i);
        this->maxLevels[specie->getId()] = specie->getMaxLevel();
    }
  }
  
  void build() {
      
    
    for (unsigned int i=0; i < qual_model->getNumTransitions(); i++) {
        Transition* transition = qual_model->getTransition(i);
        parseTransition(transition);
    }
    
    for (unsigned int i=0; i < qual_model->getNumQualitativeSpecies(); i++) {
        QualitativeSpecies* specie = qual_model->getQualitativeSpecies(i);
        if (!this->network->isNodeDefined(specie->getId())) {
        
            NodeExpression* input_node = new NodeExpression(this->network->getOrMakeNode(specie->getId()));
            NodeDeclItem* decl_item = new NodeDeclItem("logic", input_node);
            std::vector<NodeDeclItem*>* decl_item_v = new std::vector<NodeDeclItem*>();
            decl_item_v->push_back(decl_item);

            NodeDecl* truc = new NodeDecl(specie->getId(), decl_item_v, this->network);

            for (std::vector<NodeDeclItem*>::iterator it = decl_item_v->begin(); it != decl_item_v->end(); ++it) {
                delete *it;
            }
            
            delete decl_item_v;
            delete truc;
        }
    }
  } 
  
  void parseTransition(Transition* transition) 
  {
    unsigned int num_outputs = transition->getNumOutputs();
    std::vector<std::string> t_outputs;
    int max_level = 0;
    
    for (unsigned int j=0; j < num_outputs; j++) {
        std::string name = transition->getOutput(j)->getQualitativeSpecies();
        t_outputs.push_back(name);
        max_level = std::max(max_level, maxLevels[name]);
        
        if (j < (transition->getNumOutputs()-1)) {
            std::cout << ",";
        }
    }
    
    std::vector<FunctionTerm*> fun_terms(max_level);
    DefaultTerm* def_term = NULL;
    int i_fun_term = -1;
    int nb_fun_term = 0;

    while(i_fun_term < ((int) transition->getNumFunctionTerms())) {

        if (i_fun_term == -1) {
            def_term = transition->getDefaultTerm();
            
        } else {
            fun_terms[transition->getFunctionTerm(i_fun_term)->getResultLevel()-1] = transition->getFunctionTerm(i_fun_term);
            nb_fun_term++;
        }
        
        i_fun_term++;
    }
    
    if (nb_fun_term == 0 && def_term == NULL) {
        throw BNException("Could not find the activating expression");
    }
    
    Expression* exp = NULL;
    
    if (def_term != NULL && (def_term->getResultLevel() > 1 || nb_fun_term == 0)){
        createNodes(t_outputs, new ConstantExpression((double) def_term->getResultLevel()));
    }
    else {
        for (int j=0; j < max_level; j++) {
            if (fun_terms[j] != NULL && fun_terms[j]->getMath() != NULL) {
                exp = parseASTNode(fun_terms[j]->getMath());
                if (j == 0) {
                    createNodes(t_outputs, exp);
                } else {
                    std::vector<std::string> new_outputs;
                    for (auto new_output: t_outputs) {
                        new_outputs.push_back(new_output + "_" + std::to_string(j+1));
                    }
                    
                    for(int k=1; k <= j; k++) {
                        Expression* lower_outputs = new NodeExpression(
                            this->network->getOrMakeNode(k==1?t_outputs[0]:(t_outputs[0] + "_" + std::to_string(k)))
                        );
                        for (size_t l=1; l < t_outputs.size(); l++) {
                            lower_outputs = new AndLogicalExpression(
                                lower_outputs,
                                new NodeExpression(
                                    this->network->getOrMakeNode(k==1?t_outputs[l]:(t_outputs[l] + "_" + std::to_string(k)))
                                )
                            );
                        }
                        
                        exp = new AndLogicalExpression(
                            exp,
                            lower_outputs
                        );
                    }
                    
                    createNodes(new_outputs, exp);
                }
            }  
        }
    
    } 
        
  }
  
  Expression* parseASTNode(const ASTNode* tree) 
  {
    std::string name;
    int value;
    
    switch(tree->getType()) {
        case AST_LOGICAL_AND:
        {
            AndLogicalExpression* children = new AndLogicalExpression(
                parseASTNode(tree->getChild(0)),
                parseASTNode(tree->getChild(1))
            );
            
            for (unsigned int n = 2; n < tree->getNumChildren(); n++) {
                children = new AndLogicalExpression(
                children, 
                parseASTNode(tree->getChild(n))
                );
            }
            
            return children;
        }
        
        case AST_LOGICAL_OR:
        {  
            OrLogicalExpression* children = new OrLogicalExpression(
                parseASTNode(tree->getChild(0)),
                parseASTNode(tree->getChild(1))
            );
            
            for (unsigned int n = 2; n < tree->getNumChildren(); n++) {
                children = new OrLogicalExpression(
                children, 
                parseASTNode(tree->getChild(n))
                );
            }
            
            return children;
        }
        
        case AST_LOGICAL_XOR:
        {
            XorLogicalExpression* children = new XorLogicalExpression(
                parseASTNode(tree->getChild(0)),
                parseASTNode(tree->getChild(1))
            );
            
            for (unsigned int n = 2; n < tree->getNumChildren(); n++) {
                children = new XorLogicalExpression(
                children, 
                parseASTNode(tree->getChild(n))
                );
            }
            
            return children;
        }
        

        case AST_LOGICAL_NOT:
            return new NotLogicalExpression(parseASTNode(tree->getChild(0)));

        case AST_RELATIONAL_EQ:
        {
        // This seems to be the standard pattern of GINsim ??
            
            if (tree->getChild(0)->getType() == AST_NAME && tree->getChild(1)->getType() == AST_INTEGER) {
                name = tree->getChild(0)->getName();
                value = tree->getChild(1)->getValue();
            } else if (tree->getChild(1)->getType() == AST_NAME && tree->getChild(0)->getType() == AST_INTEGER) {
                name = tree->getChild(1)->getName();
                value = tree->getChild(0)->getValue();
            } else throw BNException("Bad children for operator EQ");
            
            if (value == 0) {
                return new NotLogicalExpression(
                    new NodeExpression(
                        this->network->getOrMakeNode(name)
                    )
                );
            } else if (value == 1) {
                Expression* ret = new NodeExpression(
                    this->network->getOrMakeNode(name)
                );  
                for (int i=2; i <= this->maxLevels[name]; i++) {
                    ret = new AndLogicalExpression(
                        ret,
                        new NotLogicalExpression(
                            new NodeExpression(
                                this->network->getOrMakeNode(name + "_" + std::to_string(i))
                            )
                        )
                    );
                }
                return ret;
            
            } else {
                Expression* ret = new AndLogicalExpression(
                    new NodeExpression(
                        this->network->getOrMakeNode(name)
                    ),
                    new NodeExpression(
                        this->network->getOrMakeNode(name + "_2")
                    )
                );
                
                for (int i=2; i < value; i++) {
                    ret = new AndLogicalExpression(
                        ret,
                        new NodeExpression(
                            this->network->getOrMakeNode(name + "_" + std::to_string(i+1))
                        )
                    ); 
                }
                return ret;
            }
       
        }
        case AST_RELATIONAL_LEQ:
        // Here we have a multivalued model. The idea is to modify the formula, to replace it by a pure boolean one
        // Ex: Suppose we have A <= 1 , with max(A) = 1. It means that we can change it to : A | !A
        {
            if (tree->getChild(0)->getType() == AST_NAME && tree->getChild(1)->getType() == AST_INTEGER) {
                name = tree->getChild(0)->getName();
                value = tree->getChild(1)->getValue();
            } else if (tree->getChild(1)->getType() == AST_NAME && tree->getChild(0)->getType() == AST_INTEGER) {
                name = tree->getChild(1)->getName();
                value = tree->getChild(0)->getValue();
            } else throw BNException("Bad children for operator LEQ");
            
            if (value == 0) {
                // So equal to zero
                return new NotLogicalExpression(
                    new NodeExpression(
                    this->network->getOrMakeNode(name)
                    )
                );
            } else if (value == 1) {
                // // So A | !A
                // return new OrLogicalExpression(
                //     new NodeExpression(
                //         this->network->getOrMakeNode(name)
                //     ),
                //     new NotLogicalExpression(
                //     new NodeExpression(
                //         this->network->getOrMakeNode(name)
                //     )
                //     )
                // );
                // Actually, !A | A&!A_2
                Expression* part_1 = new NotLogicalExpression(
                    new NodeExpression(
                        this->network->getOrMakeNode(name)
                    )
                );
                Expression* part_2 = new NodeExpression(
                    this->network->getOrMakeNode(name)
                );
                for (int i=2; i <= this->maxLevels[name];i++) {
                    part_2 = new AndLogicalExpression(
                        part_2,
                        new NotLogicalExpression(
                            new NodeExpression(
                                this->network->getOrMakeNode(name + "_" + std::to_string(i))
                            )
                        )
                    );
                }
                return new OrLogicalExpression(part_1, part_2);
                
            } else {
                // Here we really are multi valued, but we just need to start from zero and go to the value
                // Ex : A <= 2 : !A | A | A_2
                Expression* ret = new OrLogicalExpression(
                    new NodeExpression(
                        this->network->getOrMakeNode(name)
                    ),
                    new NotLogicalExpression(
                        new NodeExpression(
                            this->network->getOrMakeNode(name)
                        )
                    )
                );
                
                for (int i=2; i <= value; i++) {
                    ret = new OrLogicalExpression(
                        ret, new NodeExpression(
                            this->network->getOrMakeNode(name + "_" + std::to_string(i))
                        )
                    );
                }
                return ret;
            }
           
        }
        
        case AST_RELATIONAL_GEQ:
        // Here we have a multivalued model. The idea is to modify the formula, to replace it by a pure boolean one
        // Ex: Suppose we have A >= 1 , with max(A) = 2. It means that we can change it to : A | A_2
        {
            if (tree->getChild(0)->getType() == AST_NAME && tree->getChild(1)->getType() == AST_INTEGER) {
                name = tree->getChild(0)->getName();
                value = tree->getChild(1)->getValue();
            } else if (tree->getChild(1)->getType() == AST_NAME && tree->getChild(0)->getType() == AST_INTEGER) {
                name = tree->getChild(1)->getName();
                value = tree->getChild(0)->getValue();
            } else throw BNException("Bad children for operator GEQ");
            
            if (value == 0) {
            // So equal to zero
                return new NotLogicalExpression(
                    new NodeExpression(
                    this->network->getOrMakeNode(name)
                    )
                );
            } else {
            
                if (value == 1 || maxLevels[name] <= 1) {
                    // Then it's just A
                    return new NodeExpression(
                        this->network->getOrMakeNode(name)
                    );
                    
                } else {
                    // // So A | A_2 | ... | A_n
                    // Expression* ret = new OrLogicalExpression(
                    //     new NodeExpression(
                    //         this->network->getOrMakeNode(name)
                    //     ),
                    //     new NodeExpression(
                    //         this->network->getOrMakeNode(name + "_2")
                    //     )
                    // );
                    
                    // for (int i=2; i < maxLevels[name]; i++) {
                    //     ret = new OrLogicalExpression(
                    //         ret, new NodeExpression(
                    //         this->network->getOrMakeNode(name + "_" + std::to_string(i))
                    //         )
                    //     );
                    // }
                    // return ret;   
                    
                    // Or actually, since you need A to have A_n, then for A > i you just need to have A_i    
                    return new NodeExpression(
                        this->network->getOrMakeNode(name + "_" + std::to_string(value))
                    );
                }
            }
        }
        default:
            std::cerr << "Unknown tag " << tree->getName() << std::endl;
            return NULL;
    }
  }
  
  void createNodes(std::vector<std::string> names, Expression* exp) 
  {
    for (auto name: names) {
        NodeDeclItem* decl_item = new NodeDeclItem("logic", exp);
        std::vector<NodeDeclItem*>* decl_item_v = new std::vector<NodeDeclItem*>();
        decl_item_v->push_back(decl_item);

        NodeDecl* truc = new NodeDecl(name, decl_item_v, this->network);

        for (std::vector<NodeDeclItem*>::iterator it = decl_item_v->begin(); it != decl_item_v->end(); ++it) {
            delete *it;
        }
        
        delete decl_item_v;
        delete truc;
        }
  }
};