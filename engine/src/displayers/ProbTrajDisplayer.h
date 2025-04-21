
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

#ifndef _PROBTRAJ_DISPLAYER_H_
#define _PROBTRAJ_DISPLAYER_H_

#include <iostream>
#include "../Network.h"
#include "../Utils.h"
#include <iomanip>
#include <cstring>

#ifdef HDF5_COMPAT
#include <hdf5/serial/hdf5.h>
#include <hdf5/serial/hdf5_hl.h>
#endif

template <typename S>
class ProbTrajDisplayer {

public:
  Network* network;
  bool hexfloat;
  bool compute_errors;
  size_t maxrows;
  size_t maxcols;
  size_t max_simplecols;
  size_t refnode_count;
  std::vector<S> states;
  std::map<S, size_t> state_to_index;

  std::vector<NetworkState_Impl> simple_states;
  std::map<NetworkState_Impl, size_t> simple_state_to_index;

  size_t current_line;
  // current line
  double time_tick;
  double TH, err_TH, H;
  double* HD_v;
  
  struct Proba {
    S state;
    double proba;
    double err_proba;

    Proba(const S& _state, double _proba, double _err_proba) : state(_state), proba(_proba), err_proba(_err_proba) { }
  };

  std::vector<Proba> proba_v;

  ProbTrajDisplayer(Network* _network, bool _hexfloat = false) : network(_network), hexfloat(_hexfloat), current_line(0), HD_v(NULL) { }

// public:
  void begin(bool _compute_errors, size_t _maxrows, size_t _maxcols, size_t _max_simplecols, size_t _refnode_count, std::vector<S>& _states, std::vector<NetworkState_Impl>& _simple_states) {
    this->compute_errors = _compute_errors;
    this->maxrows = _maxrows;
    this->maxcols = _maxcols;
    this->max_simplecols = _max_simplecols;
    this->refnode_count = _refnode_count;
    this->HD_v = new double[_refnode_count+1];
    this->states = _states;
    this->simple_states = _simple_states;
    
    for (size_t i = 0; i < _states.size(); ++i) {
      state_to_index[_states[i]] = i;
    }
    
    for (size_t i = 0; i < _simple_states.size(); ++i) {
      simple_state_to_index[_simple_states[i]] = i;
    }

    beginDisplay();
  }

  void beginTimeTick(double _time_tick) {
    this->time_tick = _time_tick;
    proba_v.clear();
    beginTimeTickDisplay();
  }

  void setTH(double _TH) {
    this->TH = _TH;
  }

  void setErrorTH(double _err_TH) {
    this->err_TH = _err_TH;
  }

  void setH(double _H) {
    this->H = _H;
  }

  void setHD(unsigned int ind, double HD) {
    this->HD_v[ind] = HD;
  }

  void addProba(const S& state, double proba, double err_proba) {
    proba_v.push_back(Proba(state, proba, err_proba));
  }

  void endTimeTick() {
    endTimeTickDisplay();
    current_line++;
  }

  void end() {
    endDisplay();
  }

  virtual void beginDisplay() = 0;
  virtual void beginTimeTickDisplay() = 0;

  virtual void endTimeTickDisplay() = 0;
  virtual void endDisplay() = 0;

  virtual ~ProbTrajDisplayer() { delete[] HD_v; }
};


template <typename S>
class CSVProbTrajDisplayer : public ProbTrajDisplayer<S> {

  std::ostream& os_probtraj;

public:
  CSVProbTrajDisplayer(Network* network, std::ostream& os_probtraj, bool hexfloat = false) : ProbTrajDisplayer<S>(network, hexfloat), os_probtraj(os_probtraj) { }

  virtual void beginDisplay() {
    os_probtraj << "Time\tTH" << (this->compute_errors ? "\tErrorTH" : "") << "\tH";
    for (unsigned int jj = 0; jj <= this->refnode_count; ++jj) {
      os_probtraj << "\tHD=" << jj;
    }

    for (unsigned int nn = 0; nn < this->maxcols; ++nn) {
      os_probtraj << "\tState\tProba" << (this->compute_errors ? "\tErrorProba" : "");
    }

    os_probtraj << '\n';
  }
  virtual void beginTimeTickDisplay() {}
  virtual void endTimeTickDisplay() {
    os_probtraj << std::setprecision(4) << std::fixed << this->time_tick;
  #ifdef HAS_STD_HEXFLOAT
    if (this->hexfloat) {
      os_probtraj << std::hexfloat;
    }
  #endif
    if (this->hexfloat) {
      os_probtraj << '\t' << fmthexdouble(this->TH);
      os_probtraj << '\t' << fmthexdouble(this->err_TH);
      os_probtraj << '\t' << fmthexdouble(this->H);
    } else {
      os_probtraj << '\t' << this->TH;
      os_probtraj << '\t' << this->err_TH;
      os_probtraj << '\t' << this->H;
    }

    for (unsigned int nn = 0; nn <= this->refnode_count; nn++) {
      os_probtraj << '\t';
      if (this->hexfloat) {
        os_probtraj << fmthexdouble(this->HD_v[nn]);
      } else {
        os_probtraj << this->HD_v[nn];
      }
    }

    for (const typename ProbTrajDisplayer<S>::Proba &proba : this->proba_v) {
      os_probtraj << '\t';
      proba.state.displayOneLine(os_probtraj, this->network);
      if (this->hexfloat) {
        os_probtraj << '\t' << fmthexdouble(proba.proba);
        os_probtraj << '\t' << fmthexdouble(proba.err_proba);
      } else {
        os_probtraj << '\t' << std::setprecision(6) << proba.proba;
        os_probtraj << '\t' << proba.err_proba;
      }
    }
    os_probtraj << '\n';
  }
  virtual void endDisplay() {}
};

template <typename S>
class JSONProbTrajDisplayer : public ProbTrajDisplayer<S> {

  std::ostream& os_probtraj;

public:
  JSONProbTrajDisplayer(Network* _network, std::ostream& _os_probtraj, bool _hexfloat = false) : ProbTrajDisplayer<S>(_network, _hexfloat), os_probtraj(_os_probtraj) { }

  virtual void beginDisplay() {
    // void JSONProbTrajDisplayer<N, S>::beginDisplay() {
    os_probtraj << '[';
  }

  virtual void beginTimeTickDisplay() {
    if (this->current_line > 0) {
      os_probtraj << ',';
    }
    os_probtraj << '{';
  }
  
  virtual void endTimeTickDisplay() {
    os_probtraj << "\"tick\":" << std::setprecision(4) << std::fixed << this->time_tick << ",";
    if (this->hexfloat) {
      os_probtraj << "\"TH\":" << fmthexdouble(this->TH, true) << ",";
      os_probtraj << "\"ErrorTH\":"  << fmthexdouble(this->err_TH, true) << ",";
      os_probtraj << "\"H\":" << fmthexdouble(this->H, true) << ",";
    } else {
      os_probtraj << "\"TH\":" << this->TH << ",";
      os_probtraj << "\"ErrorTH\":" << this->err_TH << ",";
      os_probtraj << "\"H\":" << this->H << ",";
    }
    
    os_probtraj << "\"HD\":[";
    for (unsigned int nn = 0; nn <= this->refnode_count; nn++) {
      if (this->hexfloat) {
        os_probtraj << fmthexdouble(this->HD_v[nn], true);
      } else {
        os_probtraj << this->HD_v[nn];
      }
      if (nn != this->refnode_count) {
        os_probtraj << ",";
      }
    }
    os_probtraj << "],";

    os_probtraj << "\"probas\":[";
    unsigned int idx = 0;
    for (const typename ProbTrajDisplayer<S>::Proba &proba : this->proba_v) {
      os_probtraj << "{\"state\":\"";
      proba.state.displayJSON(os_probtraj, this->network);
      os_probtraj << "\",";
      if (this->hexfloat) {
        os_probtraj << "\"proba\":" << fmthexdouble(proba.proba, true) << ",";
        os_probtraj << "\"err_proba\":" << fmthexdouble(proba.err_proba, true);
      } else {
        os_probtraj << "\"proba\":" << std::setprecision(6) << proba.proba << ",";
        os_probtraj << "\"err_proba\":" << proba.err_proba;
      }
      os_probtraj << "}";
      if (idx < this->proba_v.size()-1) {
        os_probtraj << ",";
      }
      idx++;
    }
    os_probtraj << "]";
    os_probtraj << '}';
  }
  virtual void endDisplay() {
    os_probtraj << ']';
  }
};

#ifdef HDF5_COMPAT
template <typename S>
class HDF5ProbTrajDisplayer : public ProbTrajDisplayer<S> {

  size_t dst_size;
  size_t * dst_offset;
  size_t * dst_sizes;
  
public:
  
  hid_t file;
  double * probas;
  
  HDF5ProbTrajDisplayer(Network* network, hid_t& file) : ProbTrajDisplayer<S>(network, false), file(file) { 
  }

  virtual void beginDisplay(){
    dst_size =  sizeof( double ) * this->states.size();
    dst_offset = (size_t*) malloc( sizeof( size_t ) * this->states.size() );
    dst_sizes = (size_t*) malloc( sizeof( size_t ) * this->states.size() );
    
    const char ** field_names = (const char**) calloc( this->states.size(), sizeof( const char * ) );
    char ** column_names = (char**) calloc(this->states.size(), sizeof(char *));
    hid_t * field_type = (hid_t*) malloc( sizeof( hid_t ) * this->states.size() );
    
    for (size_t i = 0; i < this->states.size(); i++) {
      dst_offset[i] = i * sizeof( double );
      dst_sizes[i] = sizeof( double );
      std::string state_name = this->states[i].getName(this->network);
      column_names[i] = (char*) malloc( sizeof( char ) * (state_name.size() +1) );
      strcpy(column_names[i], state_name.c_str());
      field_names[i] = column_names[i];
      field_type[i] = H5T_NATIVE_DOUBLE;
    }
    
    hsize_t    chunk_size = this->maxrows;
    int        compress  = 1;
    int        *fill_data = NULL;
    
    H5TBmake_table( "probas",file ,"probas",this->states.size(),this->maxrows,
                         dst_size,field_names, dst_offset, field_type,
                         chunk_size, fill_data, compress, NULL  );
                         
    probas = (double*) malloc(sizeof(double) * this->states.size());
    
    free(column_names);
    free(field_names);
    free(field_type);
  }
  
  virtual void beginTimeTickDisplay(){   
  }
  
  virtual void endTimeTickDisplay(){

    for (size_t i = 0; i < this->states.size(); i++) {
      probas[i] = 0.0;
    }
    for (const typename ProbTrajDisplayer<S>::Proba &proba : this->proba_v) {
      probas[this->state_to_index[proba.state]] = proba.proba;
    }
    H5TBwrite_records(file, "probas", this->current_line, 1, dst_size, dst_offset, dst_sizes, probas);
  }
  virtual void endDisplay(){
    free(dst_offset);
    free(dst_sizes);
    free(probas);
  }

};
#endif

#endif
