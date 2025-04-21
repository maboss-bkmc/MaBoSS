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

#ifndef _SYMBOLS_H_
#define _SYMBOLS_H_

#include <string>
#include <map>
#include <vector>
#include <cassert>

#include "BNException.h"

#define MAP std::map 

class SymbolExpression;

typedef unsigned int SymbolIndex;

// symbol entry (i.e. variables)
class Symbol {
  std::string symb;
  SymbolIndex symb_idx;

public:
  Symbol(const std::string& symb, SymbolIndex symb_idx) : symb(symb), symb_idx(symb_idx) { }
  const std::string& getName() const {return symb;}
  SymbolIndex getIndex() const {return symb_idx;}
};

//The symbol table
class SymbolTable {
  SymbolIndex last_symb_idx;
  MAP<std::string, Symbol*> symb_map;
  std::vector<double> symb_value;
  std::vector<bool> symb_def;
  std::map<SymbolIndex, bool> symb_dont_set;
  
  std::vector<SymbolExpression *> symbolExpressions;

public:
  SymbolTable() : last_symb_idx(0) { }
  
  const Symbol* getSymbol(const std::string& symb) {
    if (symb_map.find(symb) == symb_map.end()) {
      return NULL;
    }
    return symb_map[symb];
  }

  const Symbol* getOrMakeSymbol(const std::string& symb) {
    if (symb_map.find(symb) == symb_map.end()) {
      symb_def.push_back(false);
      symb_value.push_back(0.0);
      symb_map[symb] = new Symbol(symb, last_symb_idx++);
      assert(symb_value.size() == last_symb_idx);
      assert(symb_def.size() == last_symb_idx);
    }
    return symb_map[symb];
  }

  double getSymbolValue(const Symbol* symbol, bool check = true) const {
    SymbolIndex idx = symbol->getIndex();
    if (!symb_def[idx]) {
      if (check) {
	throw BNException("symbol " + symbol->getName() + " is not defined"); 
      }
      return 0.;
   }
    return symb_value[idx];
  }

  void defineUndefinedSymbols() {
    for (auto& symbol: symb_map) {
        symb_def[symbol.second->getIndex()] = true;
    }
  }
  size_t getSymbolCount() const {return symb_map.size();}

  void setSymbolValue(const Symbol* symbol, double value) {
    SymbolIndex idx = symbol->getIndex();
    if (symb_dont_set.find(idx) == symb_dont_set.end()) {
      symb_def[idx] = true;
      symb_value[idx] = value;
    }
  }

  void overrideSymbolValue(const Symbol* symbol, double value) {
    setSymbolValue(symbol, value);
    symb_dont_set[symbol->getIndex()] = true;
  }

  void display(std::ostream& os, bool check = true) const;
  void checkSymbols() const;

  std::vector<std::string> getSymbolsNames() const {
    std::vector<std::string> result;
    for (auto& symb : symb_map) {
      result.push_back(symb.first);
    }
    return result;
  }
  void reset();

  void addSymbolExpression(SymbolExpression * exp) {
    symbolExpressions.push_back(exp);
  }

  void unsetSymbolExpressions();

  ~SymbolTable() {
    for (auto& symbol : symb_map) {
      delete symbol.second;
    }
  }
};

#endif
