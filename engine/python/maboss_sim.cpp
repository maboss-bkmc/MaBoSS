#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <set>
#include "src/BooleanNetwork.h"
#include "src/MaBEstEngine.h"
#include "src/FinalStateSimulationEngine.h"
#include "maboss_res.cpp"
#include "maboss_resfinal.cpp"
#include "maboss_commons.h"

typedef struct {
  PyObject_HEAD
  Network* network;
  RunConfig* runconfig;
} cMaBoSSSimObject;

static void cMaBoSSSim_dealloc(cMaBoSSSimObject *self)
{
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject * cMaBoSSSim_new(PyTypeObject* type, PyObject *args, PyObject* kwargs) 
{
  try {
    // Loading arguments
    char * network_file = NULL;
    char * config_file = NULL;
    char * network_str = NULL;
    char * config_str = NULL;
    static const char *kwargs_list[] = {"network", "config", "network_str", "config_str", NULL};
    if (!PyArg_ParseTupleAndKeywords(
      args, kwargs, "|ssss", const_cast<char **>(kwargs_list), 
      &network_file, &config_file, &network_str, &config_str
    ))
      return NULL;
      
    Network* network = nullptr;
    RunConfig* runconfig = nullptr;
    if (network_file != NULL && config_file != NULL) {
      // Loading bnd file
      network = new Network();
      network->parse(network_file);

      // Loading cfg file
      runconfig = new RunConfig();
      IStateGroup::reset(network);
      runconfig->parse(network, config_file);

    } 
    else if (network_str != NULL && config_str != NULL) {
      // Loading bnd file
      network = new Network();
      network->parseExpression((const char *) network_str);
      
      // Loading cfg file
      runconfig = new RunConfig();
      IStateGroup::reset(network);
      runconfig->parseExpression(network, config_str);

    }
    
    if (network != nullptr && runconfig != nullptr) {
      // Error checking
      IStateGroup::checkAndComplete(network);
      
      cMaBoSSSimObject* simulation;
      simulation = (cMaBoSSSimObject *) type->tp_alloc(type, 0);
      simulation->network = network;
      simulation->runconfig = runconfig;

      return (PyObject *) simulation;
    } else return Py_None;
  }
  catch (BNException& e) {
    PyErr_SetString(PyBNException, e.getMessage().c_str());
    return NULL;
  }
}

static PyObject* cMaBoSSSim_run(cMaBoSSSimObject* self, PyObject *args, PyObject* kwargs) {
  
  int only_last_state = 0;
  static const char *kwargs_list[] = {"only_last_state", NULL};
  if (!PyArg_ParseTupleAndKeywords(
    args, kwargs, "|i", const_cast<char **>(kwargs_list), 
    &only_last_state
  ))
    return NULL;
    
  bool b_only_last_state = PyObject_IsTrue(PyBool_FromLong(only_last_state));
  time_t start_time, end_time;

  RandomGenerator::resetGeneratedNumberCount();
  if (b_only_last_state) {
  
    FinalStateSimulationEngine* simulation = new FinalStateSimulationEngine(self->network, self->runconfig);
    time(&start_time);
    simulation->run(NULL);
    time(&end_time);
    cMaBoSSResultFinalObject* res = (cMaBoSSResultFinalObject*) PyObject_New(cMaBoSSResultFinalObject, &cMaBoSSResultFinal);
    res->network = self->network;
    res->runconfig = self->runconfig;
    res->engine = simulation;
    res->start_time = start_time;
    res->end_time = end_time;
    
    return (PyObject*) res;
  } else {

    MaBEstEngine* simulation = new MaBEstEngine(self->network, self->runconfig);
    time(&start_time);
    simulation->run(NULL);
    time(&end_time);
    
    cMaBoSSResultObject* res = (cMaBoSSResultObject*) PyObject_New(cMaBoSSResultObject, &cMaBoSSResult);
    res->network = self->network;
    res->runconfig = self->runconfig;
    res->engine = simulation;
    res->start_time = start_time;
    res->end_time = end_time;
    
    return (PyObject*) res;
  }
}

static PyMethodDef cMaBoSSSim_methods[] = {
    {"run", (PyCFunction) cMaBoSSSim_run, METH_VARARGS | METH_KEYWORDS, "runs the simulation"},
    {NULL}  /* Sentinel */
};

static PyTypeObject cMaBoSSSim = []{
    PyTypeObject sim{PyVarObject_HEAD_INIT(NULL, 0)};

    sim.tp_name = "cmaboss.cMaBoSSSimObject";
    sim.tp_basicsize = sizeof(cMaBoSSSimObject);
    sim.tp_itemsize = 0;
    sim.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    sim.tp_doc = "cMaBoSS Simulation object";
    sim.tp_new = cMaBoSSSim_new;
    sim.tp_dealloc = (destructor) cMaBoSSSim_dealloc;
    sim.tp_methods = cMaBoSSSim_methods;
    return sim;
}();
