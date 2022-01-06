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
     Cumulator.cc

   Authors:
     Eric Viara <viara@sysra.com>
     Gautier Stoll <gautier.stoll@curie.fr>
     Vincent Noël <vincent.noel@curie.fr>
 
   Date:
     January-March 2011
*/

#include "BooleanNetwork.h"
#include "Cumulator.h"
#include "RunConfig.h"
#include "ProbTrajDisplayer.h"
#include "StatDistDisplayer.h"
#include "Utils.h"
#include <sstream>
#include <iomanip>
#include <math.h>
#include <float.h>

double distance(const STATE_MAP<NetworkState_Impl, double>& proba_dist1, const STATE_MAP<NetworkState_Impl, double>& proba_dist2)
{
  return 0.;
}

void Cumulator::check() const
{
#ifndef NDEBUG
  // check that for each tick (except the last one), the sigma of each map == 1.
  for (int nn = 0; nn < max_tick_index; ++nn) {
    const CumulMap& mp = get_map(nn);
    CumulMap::Iterator iter = mp.iterator();
    double sum = 0.;
    while (iter.hasNext()) {
      TickValue tick_value;
      iter.next(tick_value);
      sum += tick_value.tm_slice;
    }
    sum /= time_tick*sample_count;
    assert(sum >= 1. - 0.0001 && sum <= 1. + 0.0001);
  }
#endif
}

void Cumulator::trajectoryEpilogue()
{
  if (sample_num >= statdist_trajcount) { 
    return; 
  }

  assert(sample_num < sample_count);

  ProbaDist::Iterator curtraj_proba_dist_iter = curtraj_proba_dist.iterator();

  double proba_max_time = 0.;

  while (curtraj_proba_dist_iter.hasNext()) {
    double tm_slice;
    curtraj_proba_dist_iter.next(tm_slice);
    proba_max_time += tm_slice;
  }

  //std::cout << "Trajepilogue #" << (sample_num+1) << " " << proba_max_time << '\n';
#ifndef NDEBUG
  double proba = 0;
#endif
  curtraj_proba_dist_iter.rewind();

  ProbaDist& proba_dist = proba_dist_v[sample_num++];
  while (curtraj_proba_dist_iter.hasNext()) {
    NetworkState_Impl state;
    double tm_slice;
    curtraj_proba_dist_iter.next(state, tm_slice);
    //assert(proba_dist.find(state) == proba_dist.end());
    double new_tm_slice = tm_slice / proba_max_time;
    proba_dist.set(state, new_tm_slice);
#ifndef NDEBUG
    proba += new_tm_slice;
#endif
  }

  assert(proba >= 0.9999 && proba <= 1.0001);
}

void Cumulator::computeMaxTickIndex()
{
  /*
  unsigned int tmp_tick_index = tick_index + !tick_completed;
  if (max_tick_index > tmp_tick_index) {
    max_tick_index = tmp_tick_index;
  }
  */
  if (max_tick_index > tick_index) {
    max_tick_index = tick_index;
  }
}

void Cumulator::epilogue(Network* network, const NetworkState& reference_state)
{
  computeMaxTickIndex();

  //check();

  // compute H (Entropy), TH (Transition entropy) and HD (Hamming distance)
  H_v.resize(max_tick_index);
  TH_v.resize(max_tick_index);

  maxcols = 0;
  double ratio = time_tick * sample_count;
  for (int nn = 0; nn < max_tick_index; ++nn) { // time tick
    const CumulMap& mp = get_map(nn);
    CumulMap::Iterator iter = mp.iterator();
    H_v[nn] = 0.;
    TH_v[nn] = 0.;
    while (iter.hasNext()) {
      TickValue tick_value;
      iter.next2(tick_value);
      double tm_slice = tick_value.tm_slice;
      double proba = tm_slice / ratio;      
      double TH = tick_value.TH / sample_count;
      H_v[nn] += -log2(proba) * proba;
      TH_v[nn] += TH;
    }
    TH_v[nn] /= time_tick;
    if (mp.size() > maxcols) {
      maxcols = mp.size();
    }
  }

  HD_v.resize(max_tick_index);

  for (int nn = 0; nn < max_tick_index; ++nn) { // time tick
    const HDCumulMap& hd_mp = get_hd_map(nn);
    HDCumulMap::Iterator iter = hd_mp.iterator();
    MAP<unsigned int, double>& hd_m = HD_v[nn];
    while (iter.hasNext()) {
      double tm_slice;
      const NetworkState_Impl &state = iter.next2(tm_slice);
      double proba = tm_slice / ratio;      
      int hd = reference_state.hamming(network, state);
      if (hd_m.find(hd) == hd_m.end()) {
	hd_m[hd] = proba;
      } else {
	hd_m[hd] += proba;
      }
    }
  }
}

void Cumulator::displayProbTraj(Network* network, unsigned int refnode_count, ProbTrajDisplayer* displayer) const
{
  displayer->begin(COMPUTE_ERRORS, maxcols, refnode_count);

  double time_tick2 = time_tick * time_tick;
  double ratio = time_tick*sample_count;
  for (int nn = 0; nn < max_tick_index; ++nn) {
    displayer->beginTimeTick(nn*time_tick);
    // TH
    const CumulMap& mp = get_map(nn);
    CumulMap::Iterator iter = mp.iterator();
    displayer->setTH(TH_v[nn]);

    // ErrorTH
    //assert((size_t)nn < TH_square_v.size());
    if (COMPUTE_ERRORS) {
      double TH_square = TH_square_v[nn];
      double TH = TH_v[nn]; // == TH
      double variance_TH = (TH_square / ((sample_count-1) * time_tick2)) - (TH*TH*sample_count/(sample_count-1));
      double err_TH;
      double variance_TH_sample_count = variance_TH/sample_count;
      //assert(variance_TH > 0.0);
      if (variance_TH_sample_count >= 0.0) {
	err_TH = sqrt(variance_TH_sample_count);
      } else {
	err_TH = 0.;
      }
      displayer->setErrorTH(err_TH);
    }

    // H
    displayer->setH(H_v[nn]);

    std::string zero_hexfloat = fmthexdouble(0.0);
    // HD
    const MAP<unsigned int, double>& hd_m = HD_v[nn];
    for (unsigned int hd = 0; hd <= refnode_count; ++hd) { 
      MAP<unsigned int, double>::const_iterator hd_m_iter = hd_m.find(hd);
      if (hd_m_iter != hd_m.end()) {
	displayer->setHD(hd, hd_m_iter->second);
      } else {
	displayer->setHD(hd, 0.);
      }
    }

    // Proba, ErrorProba
    while (iter.hasNext()) {
      TickValue tick_value;
      const NetworkState_Impl& state = iter.next2(tick_value);
      double proba = tick_value.tm_slice / ratio;      
      if (COMPUTE_ERRORS) {
	double tm_slice_square = tick_value.tm_slice_square;
	double variance_proba = (tm_slice_square / ((sample_count-1) * time_tick2)) - (proba*proba*sample_count/(sample_count-1));
	double err_proba;
	double variance_proba_sample_count = variance_proba/sample_count;
	if (variance_proba_sample_count >= DBL_MIN) {
	  err_proba = sqrt(variance_proba_sample_count);
	} else {
	  err_proba = 0.;
	}
	displayer->addProba(state, proba, err_proba);
      } else {
	displayer->addProba(state, proba, 0.);
      }
    }
    displayer->endTimeTick();
  }
  displayer->end();
}

void Cumulator::displayStatDist(Network* network, unsigned int refnode_count, StatDistDisplayer* displayer) const
{
  // should not be in cumulator, but somehwere in ProbaDist*

  // Probability distribution
  unsigned int statdist_traj_count = runconfig->getStatDistTrajCount();
  if (statdist_traj_count == 0) {
    return;
  }

  unsigned int max_size = 0;
  unsigned int cnt = 0;
  unsigned int proba_dist_size = proba_dist_v.size();
  for (unsigned int nn = 0; nn < proba_dist_size; ++nn) {
    const ProbaDist& proba_dist = proba_dist_v[nn];
    if (proba_dist.size() > max_size) {
      max_size = proba_dist.size();
    }
    cnt++;
    if (cnt > statdist_traj_count) {
      break;
    }
  }

  displayer->begin(max_size, statdist_traj_count);
  cnt = 0;
  displayer->beginStatDistDisplay();
  for (unsigned int nn = 0; nn < proba_dist_size; ++nn) {
    const ProbaDist& proba_dist = proba_dist_v[nn];
    displayer->beginStateProba(cnt+1);
    cnt++;

    proba_dist.display(displayer);
    displayer->endStateProba();
    if (cnt >= statdist_traj_count) {
      break;
    }
  }
  displayer->endStatDistDisplay();

  // should not be called from here, but from MaBestEngine
  ProbaDistClusterFactory* clusterFactory = new ProbaDistClusterFactory(proba_dist_v, statdist_traj_count);
  clusterFactory->makeClusters(runconfig);
  clusterFactory->display(displayer);
  clusterFactory->computeStationaryDistribution();
  clusterFactory->displayStationaryDistribution(displayer);
  displayer->end();
  delete clusterFactory;
}

void Cumulator::displayAsymptoticCSV(Network *network, unsigned int refnode_count, std::ostream &os_asymptprob, bool hexfloat, bool proba) const
{

  double ratio;
  if (proba)
  {
    ratio = time_tick * sample_count;
  }
  else
  {
    ratio = time_tick;
  }

  // Choosing the last tick
  int nn = max_tick_index - 1;

#ifdef HAS_STD_HEXFLOAT
  if (hexfloat)
  {
    os_asymptprob << std::hexfloat;
  }
#endif
  // TH
  const CumulMap &mp = get_map(nn);
  CumulMap::Iterator iter = mp.iterator();


  while (iter.hasNext())
  {
    TickValue tick_value;
    const NetworkState_Impl& state = iter.next2(tick_value);

    double proba = tick_value.tm_slice / ratio;
    if (proba)
    {
      if (hexfloat)
      {
        os_asymptprob << std::setprecision(6) << fmthexdouble(proba);
      }
      else
      {
        os_asymptprob << std::setprecision(6) << proba;
      }
    }
    else
    {
      int t_proba = static_cast<int>(round(proba));
      os_asymptprob << std::fixed << t_proba;
    }

    os_asymptprob << '\t';
    NetworkState network_state(state);
    network_state.displayOneLine(os_asymptprob, network);

    os_asymptprob << '\n';

  }
}

const std::map<double, STATE_MAP<NetworkState_Impl, double> > Cumulator::getStateDists() const
{
  std::map<double, STATE_MAP<NetworkState_Impl, double> > result;

  double ratio = time_tick*sample_count;
  for (int nn = 0; nn < max_tick_index; ++nn) {

    const CumulMap& mp = get_map(nn);
    CumulMap::Iterator iter = mp.iterator();

    STATE_MAP<NetworkState_Impl, double> t_result;

    while (iter.hasNext()) {
      TickValue tick_value;
      const NetworkState_Impl& state = iter.next2(tick_value);
      double proba = tick_value.tm_slice / ratio;      
      t_result[state] = proba;
    }

    result[((double) nn)*time_tick] = t_result;
  } 
  return result;
}


#ifdef PYTHON_API

std::set<NetworkState_Impl> Cumulator::getStates() const
{
  std::set<NetworkState_Impl> result_states;

  for (int nn=0; nn < getMaxTickIndex(); nn++) {
    const CumulMap& mp = get_map(nn);
    CumulMap::Iterator iter = mp.iterator();

    while (iter.hasNext()) {
      TickValue tick_value;
      const NetworkState_Impl& state = iter.next2(tick_value);
      result_states.insert(state);
    }
  }

  return result_states;
}

std::vector<NetworkState_Impl> Cumulator::getLastStates() const
{
  std::vector<NetworkState_Impl> result_states;

    const CumulMap& mp = get_map(getMaxTickIndex()-1);
    CumulMap::Iterator iter = mp.iterator();

    while (iter.hasNext()) {
      TickValue tick_value;
      const NetworkState_Impl& state = iter.next2(tick_value);
      result_states.push_back(state);
    }

  return result_states;
}


PyObject* Cumulator::getNumpyStatesDists(Network* network) const 
{
  std::set<NetworkState_Impl> result_states = getStates();
  
  npy_intp dims[2] = {(npy_intp) getMaxTickIndex(), (npy_intp) result_states.size()};
  PyArrayObject* result = (PyArrayObject *) PyArray_ZEROS(2,dims,NPY_DOUBLE, 0); 

  std::vector<NetworkState_Impl> list_states(result_states.begin(), result_states.end());
  std::map<NetworkState_Impl, unsigned int> pos_states;
  for(unsigned int i=0; i < list_states.size(); i++) {
    pos_states[list_states[i]] = i;
  }

  double ratio = time_tick*sample_count;

  for (int nn=0; nn < getMaxTickIndex(); nn++) {
    const CumulMap& mp = get_map(nn);
    CumulMap::Iterator iter = mp.iterator();

    while (iter.hasNext()) {
      TickValue tick_value;
      const NetworkState_Impl& state = iter.next2(tick_value);
      
      void* ptr = PyArray_GETPTR2(result, nn, pos_states[state]);
      PyArray_SETITEM(
        result, 
        (char*) ptr,
        PyFloat_FromDouble(tick_value.tm_slice / ratio)
      );
    }
  }
  PyObject* pylist_state = PyList_New(list_states.size());
  for (unsigned int i=0; i < list_states.size(); i++) {
    PyList_SetItem(
      pylist_state, i, 
      PyUnicode_FromString(NetworkState(list_states[i]).getName(network).c_str())
    );
  }

  PyObject* timepoints = PyList_New(getMaxTickIndex());
  for (int i=0; i < getMaxTickIndex(); i++) {
    PyList_SetItem(timepoints, i, PyFloat_FromDouble(((double) i) * time_tick));
  }

  return PyTuple_Pack(3, PyArray_Return(result), pylist_state, timepoints);
}


PyObject* Cumulator::getNumpyLastStatesDists(Network* network) const 
{
  std::vector<NetworkState_Impl> result_last_states = getLastStates();
  
  npy_intp dims[2] = {(npy_intp) 1, (npy_intp) result_last_states.size()};
  PyArrayObject* result = (PyArrayObject *) PyArray_ZEROS(2,dims,NPY_DOUBLE, 0); 

  std::map<NetworkState_Impl, unsigned int> pos_states;
  for(unsigned int i=0; i < result_last_states.size(); i++) {
    pos_states[result_last_states[i]] = i;
  }

  double ratio = time_tick*sample_count;

  const CumulMap& mp = get_map(getMaxTickIndex()-1);
  CumulMap::Iterator iter = mp.iterator();

  while (iter.hasNext()) {
    TickValue tick_value;
    const NetworkState_Impl& state = iter.next2(tick_value);
    
    void* ptr = PyArray_GETPTR2(result, 0, pos_states[state]);
    PyArray_SETITEM(
      result, 
      (char*) ptr,
      PyFloat_FromDouble(tick_value.tm_slice / ratio)
    );
  }

  PyObject* pylist_state = PyList_New(result_last_states.size());
  for (unsigned int i=0; i < result_last_states.size(); i++) {
    PyList_SetItem(
      pylist_state, i, 
      PyUnicode_FromString(NetworkState(result_last_states[i]).getName(network).c_str())
    );
  }

  PyObject* timepoints = PyList_New(1);
  PyList_SetItem(
    timepoints, 0, 
    PyFloat_FromDouble(
      (
        (double) (getMaxTickIndex()-1)
      )*time_tick
    )
  );

  return PyTuple_Pack(3, PyArray_Return(result), pylist_state, timepoints);
}




std::vector<Node*> Cumulator::getNodes(Network* network) const {
  std::vector<Node*> result_nodes;

  for (auto node: network->getNodes()) {
    if (!node->isInternal())
      result_nodes.push_back(node);
  }
  return result_nodes;
}

PyObject* Cumulator::getNumpyNodesDists(Network* network, std::vector<Node*> output_nodes) const 
{
  if (output_nodes.size() == 0){
    output_nodes = getNodes(network);
  }
  
  npy_intp dims[2] = {(npy_intp) getMaxTickIndex(), (npy_intp) output_nodes.size()};
  PyArrayObject* result = (PyArrayObject *) PyArray_ZEROS(2,dims,NPY_DOUBLE, 0); 

  std::map<Node*, unsigned int> pos_nodes;
  for(unsigned int i=0; i < output_nodes.size(); i++) {
    pos_nodes[output_nodes[i]] = i;
  }

  double ratio = time_tick*sample_count;

  for (int nn=0; nn < getMaxTickIndex(); nn++) {
    const CumulMap& mp = get_map(nn);
    CumulMap::Iterator iter = mp.iterator();

    while (iter.hasNext()) {
      TickValue tick_value;
      const NetworkState_Impl& state = iter.next2(tick_value);
      
      for (auto node: output_nodes) {
        
        if (((NetworkState) state).getNodeState(node)){
          void* ptr_val = PyArray_GETPTR2(result, nn, pos_nodes[node]);

          PyArray_SETITEM(
            result, 
            (char*) ptr_val,
            PyFloat_FromDouble(
              PyFloat_AsDouble(PyArray_GETITEM(result, (char*) ptr_val))
              + (tick_value.tm_slice / ratio)
            )
          );
        }
      }
    }
  }
  PyObject* pylist_nodes = PyList_New(output_nodes.size());
  for (unsigned int i=0; i < output_nodes.size(); i++) {
    PyList_SetItem(
      pylist_nodes, i, 
      PyUnicode_FromString(output_nodes[i]->getLabel().c_str())
    );
  }

  PyObject* timepoints = PyList_New(getMaxTickIndex());
  for (int i=0; i < getMaxTickIndex(); i++) {
    PyList_SetItem(timepoints, i, PyFloat_FromDouble(((double) i) * time_tick));
  }

  return PyTuple_Pack(3, PyArray_Return(result), pylist_nodes, timepoints);
}


PyObject* Cumulator::getNumpyLastNodesDists(Network* network, std::vector<Node*> output_nodes) const 
{
  if (output_nodes.size() == 0){
    output_nodes = getNodes(network);
  }
  
  npy_intp dims[2] = {(npy_intp) 1, (npy_intp) output_nodes.size()};
  PyArrayObject* result = (PyArrayObject *) PyArray_ZEROS(2,dims,NPY_DOUBLE, 0); 

  std::map<Node*, unsigned int> pos_nodes;
  for(unsigned int i=0; i < output_nodes.size(); i++) {
    pos_nodes[output_nodes[i]] = i;
  }

  double ratio = time_tick*sample_count;

  const CumulMap& mp = get_map(getMaxTickIndex()-1);
  CumulMap::Iterator iter = mp.iterator();

  while (iter.hasNext()) {
    TickValue tick_value;
    const NetworkState_Impl& state = iter.next2(tick_value);
    
    for (auto node: output_nodes) {
      
      if (((NetworkState) state).getNodeState(node)){
        void* ptr_val = PyArray_GETPTR2(result, 0, pos_nodes[node]);

        PyArray_SETITEM(
          result, 
          (char*) ptr_val,
          PyFloat_FromDouble(
            PyFloat_AsDouble(PyArray_GETITEM(result, (char*) ptr_val))
            + (tick_value.tm_slice / ratio)
          )
        );
      }
    }
  }
  PyObject* pylist_nodes = PyList_New(output_nodes.size());
  for (unsigned int i=0; i < output_nodes.size(); i++) {
    PyList_SetItem(
      pylist_nodes, i, 
      PyUnicode_FromString(output_nodes[i]->getLabel().c_str())
    );
  }
  PyObject* timepoints = PyList_New(1);
  PyList_SetItem(
    timepoints, 0, 
    PyFloat_FromDouble(
      (
        (double) (getMaxTickIndex()-1)
      )*time_tick
    )
  );
  
  return PyTuple_Pack(3, PyArray_Return(result), pylist_nodes, timepoints);
}


#endif

const STATE_MAP<NetworkState_Impl, double> Cumulator::getNthStateDist(int nn) const
{
  double ratio = time_tick*sample_count;

  const CumulMap& mp = get_map(nn);
  CumulMap::Iterator iter = mp.iterator();

  STATE_MAP<NetworkState_Impl, double> result;

  while (iter.hasNext()) {
    TickValue tick_value;
    const NetworkState_Impl& state = iter.next2(tick_value);
    double proba = tick_value.tm_slice / ratio;      
    result[state] = proba;
  }
 
  return result;
}
 
const STATE_MAP<NetworkState_Impl, double> Cumulator::getAsymptoticStateDist() const 
{ return getNthStateDist(getMaxTickIndex()-1); }

const double Cumulator::getFinalTime() const {
  return time_tick*(getMaxTickIndex()-1);
}
void Cumulator::add(unsigned int where, const CumulMap& add_cumul_map)
{
  CumulMap& to_cumul_map = get_map(where);

  CumulMap::Iterator iter = add_cumul_map.iterator();
  while (iter.hasNext()) {
    TickValue tick_value;
    const NetworkState_Impl& state = iter.next2(tick_value);
    to_cumul_map.add(state, tick_value);
  }
}

void Cumulator::add(unsigned int where, const HDCumulMap& add_hd_cumul_map)
{
  HDCumulMap& to_hd_cumul_map = get_hd_map(where);

  HDCumulMap::Iterator iter = add_hd_cumul_map.iterator();
  while (iter.hasNext()) {
    double tm_slice;
    const NetworkState_Impl& state = iter.next2(tm_slice);
    to_hd_cumul_map.add(state, tm_slice);
  }
}

struct MergeCumulatorWrapper {
  Cumulator* cumulator_1;
  Cumulator* cumulator_2;
  
  MergeCumulatorWrapper(Cumulator* cumulator_1, Cumulator* cumulator_2) :
    cumulator_1(cumulator_1), cumulator_2(cumulator_2) { }
};

void* Cumulator::threadMergeCumulatorWrapper(void *arg)
{
#ifdef USE_DYNAMIC_BITSET
  MBDynBitset::init_pthread();
#endif
  MergeCumulatorWrapper* warg = (MergeCumulatorWrapper*)arg;
  try {
    mergePairOfCumulators(warg->cumulator_1, warg->cumulator_2);
  } catch(const BNException& e) {
    std::cerr << e;
  }
#ifdef USE_DYNAMIC_BITSET
  MBDynBitset::end_pthread();
#endif
  return NULL;
}

Cumulator* Cumulator::mergeCumulatorsParallel(RunConfig* runconfig, std::vector<Cumulator*>& cumulator_v) {
  
  size_t size = cumulator_v.size();
  
  if (1 == size) {
    return cumulator_v[0];
  } else {
    
    unsigned int lvl=1;
    unsigned int max_lvl = ceil(log2(size));

    while(lvl <= max_lvl) {      
    
      unsigned int step_lvl = pow(2, lvl-1);
      unsigned int width_lvl = floor(size/(step_lvl*2)) + 1;
      pthread_t* tid = new pthread_t[width_lvl];
      unsigned int nb_threads = 0;
      std::vector<MergeCumulatorWrapper*> wargs;
      for(unsigned int i=0; i < size; i+=(step_lvl*2)) {
        
        if (i+step_lvl < size) {
          MergeCumulatorWrapper* warg = new MergeCumulatorWrapper(cumulator_v[i], cumulator_v[i+step_lvl]);
          pthread_create(&tid[nb_threads], NULL, Cumulator::threadMergeCumulatorWrapper, warg);
          nb_threads++;
          wargs.push_back(warg);
        } 
      }
      
      for(unsigned int i=0; i < nb_threads; i++) {   
          pthread_join(tid[i], NULL);
          
      }
      
      for (auto warg: wargs) {
        delete warg;
      }
      delete [] tid;
      lvl++;
    }
  }
  
  return cumulator_v[0];
}


void Cumulator::mergePairOfCumulators(Cumulator* cumulator_1, Cumulator* cumulator_2) {
    
  cumulator_1->sample_count += cumulator_2->sample_count;
  
  unsigned int rr = cumulator_1->proba_dist_v.size();
  cumulator_1->statdist_trajcount += cumulator_2->statdist_trajcount;
  cumulator_1->proba_dist_v.resize(cumulator_1->statdist_trajcount);
  
  cumulator_1->computeMaxTickIndex();
  cumulator_2->computeMaxTickIndex();
  if (cumulator_2->cumul_map_v.size() > cumulator_1->cumul_map_v.size()) {
    cumulator_1->cumul_map_v.resize(cumulator_2->cumul_map_v.size());
    cumulator_1->hd_cumul_map_v.resize(cumulator_2->cumul_map_v.size());
  }
  if (cumulator_2->max_tick_index > cumulator_1->max_tick_index) {
    cumulator_1->max_tick_index = cumulator_1->tick_index = cumulator_2->max_tick_index;
  }

  for (unsigned int nn = 0; nn < cumulator_2->cumul_map_v.size(); ++nn) {
    cumulator_1->add(nn, cumulator_2->cumul_map_v[nn]);
    cumulator_1->add(nn, cumulator_2->hd_cumul_map_v[nn]);
    cumulator_1->TH_square_v[nn] += cumulator_2->TH_square_v[nn];
  }
  unsigned int proba_dist_size = cumulator_2->proba_dist_v.size();
  for (unsigned int ii = 0; ii < proba_dist_size; ++ii) {
    assert(cumulator_1->proba_dist_v.size() > rr);
    cumulator_1->proba_dist_v[rr++] = cumulator_2->proba_dist_v[ii];
  }
  delete cumulator_2;
}
