
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
     ProbTrajDisplayer.h

   Authors:
     Eric Viara <viara@sysra.com>
     Gautier Stoll <gautier.stoll@curie.fr>
     Vincent Noël <vincent.noel@curie.fr>
 
   Date:
     Decembre 2020
*/

#ifndef _POP_PROBTRAJ_DISPLAYER_H_
#define _POP_PROBTRAJ_DISPLAYER_H_

#include <iostream>
#include <iomanip>
#include "BooleanNetwork.h"
#include "Utils.h"
#include "ProbTrajDisplayer.h"

class CSVSimplePopProbTrajDisplayer final : public CSVProbTrajDisplayer<PopNetworkState>
{

public:
  std::ostream& os_simple_probtraj; 
  std::map<NetworkState, std::map<unsigned int, double> > distribs;

  CSVSimplePopProbTrajDisplayer(Network* network, std::ostream& os_probtraj, std::ostream& os_simple_probtraj, bool hexfloat = false) : CSVProbTrajDisplayer<PopNetworkState>(network, os_probtraj, hexfloat), os_simple_probtraj(os_simple_probtraj) { }
  
  void beginDisplay()
  {
    CSVProbTrajDisplayer<PopNetworkState>::beginDisplay();
    
    os_simple_probtraj << "Time\tTH" << (this->compute_errors ? "\tErrorTH" : "") << "\tH";
    for (unsigned int jj = 0; jj <= this->refnode_count; ++jj)
    {
      os_simple_probtraj << "\tHD=" << jj;
    }

    os_simple_probtraj << "\tPop\tVar\tH";
    for (unsigned int nn = 0; nn < this->max_simplecols; ++nn)
    {
      os_simple_probtraj << "\tState\tProba" << (this->compute_errors ? "\tErrorProba" : "");
    }

    os_simple_probtraj << '\n';

  }
  
  void beginTimeTickDisplay() {}
  void endTimeTickDisplay()
  {
    CSVProbTrajDisplayer<PopNetworkState>::endTimeTickDisplay();
    
    
    os_simple_probtraj << std::setprecision(4) << std::fixed << this->time_tick;
#ifdef HAS_STD_HEXFLOAT
    if (this->hexfloat)
    {
      os_simple_probtraj << std::hexfloat;
    }
#endif
    if (this->hexfloat)
    {
      os_simple_probtraj << '\t' << fmthexdouble(this->TH);
      os_simple_probtraj << '\t' << fmthexdouble(this->err_TH);
      os_simple_probtraj << '\t' << fmthexdouble(this->H);
    }
    else
    {
      os_simple_probtraj << '\t' << this->TH;
      os_simple_probtraj << '\t' << this->err_TH;
      os_simple_probtraj << '\t' << this->H;
    }

    for (unsigned int nn = 0; nn <= this->refnode_count; nn++)
    {
      os_simple_probtraj << '\t';
      if (this->hexfloat)
      {
        os_simple_probtraj << fmthexdouble(this->HD_v[nn]);
      }
      else
      {
        os_simple_probtraj << this->HD_v[nn];
      }
    }

    // Computing total population and state probabilities
    double pop = 0;
    std::map<NetworkState, double> network_state_probas;
    std::map<NetworkState, double> network_state_errors;
    std::map<unsigned int, double> pop_size_distrib;
    for (const typename ProbTrajDisplayer<PopNetworkState>::Proba &proba : this->proba_v)
    {
      pop += proba.proba * proba.state.count(NULL);
      for (const auto& network_state : proba.state.getMap())
      {
        if (network_state_probas.find(network_state.first) != network_state_probas.end())
        {
          network_state_probas[network_state.first] += proba.proba * network_state.second;
          if (this->compute_errors) {
            network_state_errors[network_state.first] += proba.err_proba;
          }
        }
        else
        {
          network_state_probas[network_state.first] = proba.proba * network_state.second;
          if (this->compute_errors) {
            network_state_errors[network_state.first] = proba.err_proba;
          }
        }
      }
      
      if (pop_size_distrib.find(proba.state.count(NULL)) != pop_size_distrib.end()) 
      { 
        pop_size_distrib[proba.state.count(NULL)] += proba.proba;
      }
      else 
      {
        pop_size_distrib[proba.state.count(NULL)] = proba.proba;
      }
    }

    double network_state_variance = - pop*pop;
    double network_state_entropy = 0;
    
    for (const auto &size_proba: pop_size_distrib) 
    {
      network_state_variance += size_proba.second * (size_proba.first * size_proba.first);
      network_state_entropy -= log2(size_proba.second)*size_proba.second;
    }
    
    // Total population
    if (this->hexfloat) {
      os_simple_probtraj << '\t' << fmthexdouble(pop);
      os_simple_probtraj << '\t' << fmthexdouble(network_state_variance);
      os_simple_probtraj << '\t' << fmthexdouble(network_state_entropy);
    } else {
      os_simple_probtraj << '\t' << pop;
      os_simple_probtraj << '\t' << network_state_variance;
      os_simple_probtraj << '\t' << network_state_entropy;
    }

    // Computing
    for (auto &network_state_proba : network_state_probas)
    {
      os_simple_probtraj << '\t';
      network_state_proba.first.displayOneLine(os_simple_probtraj, this->network);
      if (this->hexfloat)
      {
        os_simple_probtraj << '\t' << fmthexdouble(network_state_proba.second/pop);
        os_simple_probtraj << '\t' << fmthexdouble(network_state_errors[network_state_proba.first]);
      }
      else
      {
        os_simple_probtraj << '\t' << std::setprecision(6) << (network_state_proba.second/pop);
        os_simple_probtraj << '\t' << std::setprecision(6) << (network_state_errors[network_state_proba.first]);
      }
    }
    os_simple_probtraj << '\n';
  }
  void endDisplay() {}
};

class JSONSimpleProbTrajDisplayer final : public JSONProbTrajDisplayer<PopNetworkState> {


public:
  std::ostream& os_simple_probtraj;
  JSONSimpleProbTrajDisplayer(Network* network, std::ostream& os_probtraj, std::ostream& os_simple_probtraj, bool hexfloat = false) : JSONProbTrajDisplayer<PopNetworkState>(network, os_probtraj, hexfloat), os_simple_probtraj(os_simple_probtraj) { }

  void beginDisplay() {
    JSONProbTrajDisplayer<PopNetworkState>::beginDisplay();
    os_simple_probtraj << '[';
  }

  void beginTimeTickDisplay() {
    JSONProbTrajDisplayer<PopNetworkState>::beginTimeTickDisplay();

    if (this->current_line > 0) {
      os_simple_probtraj << ',';
    }
    os_simple_probtraj << '{';
  }

  void endTimeTickDisplay() {
    
    JSONProbTrajDisplayer<PopNetworkState>::endTimeTickDisplay();

    os_simple_probtraj << "\"tick\":" << std::setprecision(4) << std::fixed << this->time_tick << ",";
    if (this->hexfloat) {
      os_simple_probtraj << "\"TH\":" << fmthexdouble(this->TH, true) << ",";
      os_simple_probtraj << "\"ErrorTH\":"  << fmthexdouble(this->err_TH, true) << ",";
      os_simple_probtraj << "\"H\":" << fmthexdouble(this->H, true) << ",";
    } else {
      os_simple_probtraj << "\"TH\":" << this->TH << ",";
      os_simple_probtraj << "\"ErrorTH\":" << this->err_TH << ",";
      os_simple_probtraj << "\"H\":" << this->H << ",";
    }

    os_simple_probtraj << "\"HD\":[";
    for (unsigned int nn = 0; nn <= this->refnode_count; nn++) {
      if (this->hexfloat) {
        os_simple_probtraj << fmthexdouble(this->HD_v[nn], true);
      } else {
        os_simple_probtraj << this->HD_v[nn];
      }
      if (nn != this->refnode_count) {
        os_simple_probtraj << ",";
      }
    }
    os_simple_probtraj << "],";

    // Computing total population and state probabilities
    double pop = 0;
    std::map<NetworkState, double> network_state_probas;
    std::map<unsigned int, double> pop_size_distrib;
    for (const typename ProbTrajDisplayer<PopNetworkState>::Proba &proba : this->proba_v)
    {
      pop += proba.proba * proba.state.count(NULL);
      for (const auto &network_state : proba.state.getMap())
      {
        if (network_state_probas.find(network_state.first) != network_state_probas.end())
        {
          network_state_probas[network_state.first] += proba.proba * network_state.second;
        }
        else
        {
          network_state_probas[network_state.first] = proba.proba * network_state.second;
        }
      }
      
      if (pop_size_distrib.find(proba.state.count(NULL)) != pop_size_distrib.end()) 
      { 
        pop_size_distrib[proba.state.count(NULL)] += proba.proba;
      }
      else 
      {
        pop_size_distrib[proba.state.count(NULL)] = proba.proba;
      }
    }

    double network_state_variance = - pop*pop;
    double network_state_entropy = 0;
    
    for (const auto &size_proba: pop_size_distrib) 
    {
      network_state_variance += size_proba.second * (size_proba.first * size_proba.first);
      network_state_entropy -= log2(size_proba.second)*size_proba.second;
    }
    if (this->hexfloat){
      os_simple_probtraj << "\"pop\":" << fmthexdouble(pop, true) << ",";
      os_simple_probtraj << "\"var\":" << fmthexdouble(network_state_variance, true) << ",";
      os_simple_probtraj << "\"H\":" << fmthexdouble(network_state_entropy, true) << ",";
    } else {
      os_simple_probtraj << "\"pop\":" << pop << ",";
      os_simple_probtraj << "\"var\":" << network_state_variance << ",";
      os_simple_probtraj << "\"H\":" << network_state_entropy << ",";
    }

    os_simple_probtraj << "\"probas\":[";
    unsigned int idx = 0;
    for (auto &network_state_proba : network_state_probas) {
      os_simple_probtraj << "{\"state\":\"";
      network_state_proba.first.displayJSON(os_simple_probtraj, this->network);
      os_simple_probtraj << "\",";
      if (this->hexfloat) {
        os_simple_probtraj << "\"proba\":" << fmthexdouble(network_state_proba.second/pop, true) << ",";
        // os_probtraj << "\"err_proba\":" << fmthexdouble(proba.err_proba, true);
      } else {
        os_simple_probtraj << "\"proba\":" << std::setprecision(6) << (network_state_proba.second/pop) << ",";
        // os_probtraj << "\"err_proba\":" << proba.err_proba;
      }
      os_simple_probtraj << "}";
      if (idx < network_state_probas.size()-1) {
        os_simple_probtraj << ",";
      }
      idx++;
    }
    os_simple_probtraj << "]";
    os_simple_probtraj << '}';
  }
  void endDisplay() {
    JSONProbTrajDisplayer<PopNetworkState>::endDisplay();
    os_simple_probtraj << ']';
  }
};

#ifdef HDF5_COMPAT
class HDF5PopProbTrajDisplayer final : public HDF5ProbTrajDisplayer<PopNetworkState> {

  size_t dst_size;
  size_t * dst_offset;
  size_t * dst_sizes;
  double pop;
  double * simple_probas;

public:
  HDF5PopProbTrajDisplayer(Network* network, hid_t file) : HDF5ProbTrajDisplayer(network, file) { }

  void beginDisplay(){
    HDF5ProbTrajDisplayer<PopNetworkState>::beginDisplay();
    dst_size =  sizeof( double ) * (this->simple_states.size() + 1);
    dst_offset = (size_t*) malloc( sizeof( size_t ) * (this->simple_states.size() + 1) );
    dst_sizes = (size_t*) malloc( sizeof( size_t ) * (this->simple_states.size() + 1) );
    
    const char ** field_names = (const char**) calloc( (this->simple_states.size() + 1), sizeof( const char * ) );
    char ** column_names = (char**) calloc((this->simple_states.size() + 1), sizeof(char *));
    hid_t * field_type = (hid_t*) malloc( sizeof( hid_t ) * (this->simple_states.size() + 1) );
    
    dst_offset[0] = 0;
    dst_sizes[0] = sizeof( double );
    field_names[0] = "population";
    field_type[0] = H5T_NATIVE_DOUBLE;
    
    for (size_t i = 0; i < this->simple_states.size(); i++) {
      dst_offset[i+1] = (i+1) * sizeof( double );
      dst_sizes[i+1] = sizeof( double );
      std::string state_name = NetworkState(this->simple_states[i]).getName(this->network);
      column_names[i+1] = (char*) malloc( sizeof( char ) * (state_name.size() +1) );
      strcpy(column_names[i+1], state_name.c_str());
      field_names[i+1] = column_names[i+1];
      field_type[i+1] = H5T_NATIVE_DOUBLE;
    }
    
    hsize_t    chunk_size = 10;
    int        compress  = 0;
    int        *fill_data = NULL;
 
    H5TBmake_table( "simple_probas",file ,"simple_probas",this->simple_states.size()+1,0,
                         dst_size,field_names, dst_offset, field_type,
                         chunk_size, fill_data, compress, NULL  );
                         
    simple_probas = (double*) malloc(sizeof(double) * this->simple_states.size());
    
    free(column_names);
    free(field_names);
    free(field_type);
  }
  void beginTimeTickDisplay(){
    HDF5ProbTrajDisplayer<PopNetworkState>::beginTimeTickDisplay();
  }
  void endTimeTickDisplay(){
    HDF5ProbTrajDisplayer<PopNetworkState>::endTimeTickDisplay();
    for (size_t i = 0; i < this->simple_states.size(); i++) {
      simple_probas[i] = 0.0;
    }
    
    // Computing total population and state probabilities
    double pop = 0;
    std::map<NetworkState, double> network_state_probas;
    std::map<unsigned int, double> pop_size_distrib;
    for (const typename ProbTrajDisplayer<PopNetworkState>::Proba &proba : this->proba_v)
    {
      pop += proba.proba * proba.state.count(NULL);
      for (const auto &network_state : proba.state.getMap())
      {
        if (network_state_probas.find(network_state.first) != network_state_probas.end())
        {
          network_state_probas[network_state.first] += proba.proba * network_state.second;
        }
        else
        {
          network_state_probas[network_state.first] = proba.proba * network_state.second;
        }
      }
      
      if (pop_size_distrib.find(proba.state.count(NULL)) != pop_size_distrib.end()) 
      { 
        pop_size_distrib[proba.state.count(NULL)] += proba.proba;
      }
      else 
      {
        pop_size_distrib[proba.state.count(NULL)] = proba.proba;
      }
    }
    
    simple_probas[0] = pop;
    for (auto &network_state_proba : network_state_probas) {
      simple_probas[this->simple_state_to_index[network_state_proba.first.getState()]] = network_state_proba.second/pop;
    }
    H5TBappend_records(file, "simple_probas", 1, dst_size, dst_offset, dst_sizes, simple_probas);
  }
  
  void endDisplay(){
    HDF5ProbTrajDisplayer<PopNetworkState>::endDisplay();
    free(dst_offset);
    free(dst_sizes);
    free(simple_probas);
  }
};
#endif

#endif
