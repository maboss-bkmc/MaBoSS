/* 
   MaBoSS (Markov Boolean Stochastic Simulator)
   Copyright (C) 2011-2018 Institut Curie, 26 rue d'Ulm, Paris, France
   
   MaBoSS is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.
   
   MaBoSS is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.
   
   You should have received a copy of the GNU Lesser General Public
   License along with this library; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA 
*/

/*
   Module:
     Function.h

   Authors:
     Eric Viara <viara@sysra.com>
     Gautier Stoll <gautier.stoll@curie.fr>
 
   Date:
   July 2018
*/

#ifndef _FUNCTION_H_
#define _FUNCTION_H_

#include <vector>
#include <map>
#include <math.h>

class ArgumentList;
class Expression;
class Node;
class NetworkState;

class Function {
  std::string funname;
  unsigned int min_args;
  unsigned int max_args;
  static std::map<std::string, Function*> func_map;

protected:
  Function(const std::string& funname, unsigned int min_args, unsigned int max_args = ~0U) : funname(funname), min_args(min_args), max_args(max_args == ~0U ? min_args : max_args) { 
    func_map[funname] = this;
  }

public:
  const std::string& getFunName() const {return funname;}
  unsigned int getMinArgs() const {return min_args;}
  unsigned int getMaxArgs() const {return max_args;}

  static Function* getFunction(const std::string& funname) {
    std::map<std::string, Function*>::iterator iter = func_map.find(funname);
    if (iter == func_map.end()) {
      return NULL;
    }
    return iter->second;
  }

  virtual bool isDeterministic() const {return true;}

  virtual std::string getDescription() const = 0;

  void check(ArgumentList* arg_list);

  virtual double eval(const Node* this_node, const NetworkState& network_state, ArgumentList* arg_list) = 0;

  static void displayFunctionDescriptions(std::ostream& os) {
    for (std::map<std::string, Function*>::iterator iter = func_map.begin(); iter != func_map.end(); ++iter) {
      os << "  " << iter->second->getDescription() << "\n\n";
    }
  }
};

//
// User function declarations
//

class LogFunction : public Function {

public:
  LogFunction() : Function("log", 1, 2) { }

  double eval(const Node* this_node, const NetworkState& network_state, ArgumentList* arg_list);

  std::string getDescription() const {
    return "double log(double VALUE[, double BASE=e])\n  computes the value of the natural logarithm of VALUE; uses BASE if set";
  }
};

class ExpFunction : public Function {

public:
  ExpFunction() : Function("exp", 1, 2) { }

  double eval(const Node* this_node, const NetworkState& network_state, ArgumentList* arg_list);

  std::string getDescription() const {
    return "double exp(double VALUE[, double BASE=e])\n  computes the base-e exponential of VALUE; uses BASE if set";
  }
};

#endif
