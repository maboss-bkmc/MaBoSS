#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <set>
#include "src/BooleanNetwork.h"
#include "src/MaBEstEngine.h"

typedef struct {
  PyObject_HEAD
  Network* network;
  MaBEstEngine* simulation;
} cMaBoSSSimObject;

static void cMaBoSSSim_dealloc(cMaBoSSSimObject *self)
{
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject * cMaBoSSSim_new(PyTypeObject* type, PyObject *args, PyObject* kwargs) 
{
  try {
    // Loading arguments
    char * network_file;
    char * config_file;
    static const char *kwargs_list[] = {"network", "config", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|ss", const_cast<char **>(kwargs_list), &network_file, &config_file))
      return NULL;
      
    // Loading bnd file
    Network* network = new Network();
    network->parse(network_file);

    // Loading cfg file
    RunConfig* runconfig = RunConfig::getInstance();
    IStateGroup::reset(network);
    runconfig->parse(network, config_file);

    // Error checking
    IStateGroup::checkAndComplete(network);

    // Running the simulation
    MaBEstEngine* mabest = new MaBEstEngine(network, runconfig);
   
    cMaBoSSSimObject* simulation;
    simulation = (cMaBoSSSimObject *) type->tp_alloc(type, 0);
    simulation->network = network;
    simulation->simulation = mabest;

    return (PyObject *) simulation;
  }
  catch (BNException& e) {
    std::cout << e.getMessage() << std::endl;
    return Py_None;
  }
}

static PyObject* cMaBoSSSim_run(cMaBoSSSimObject* self, PyObject*Py_UNUSED) {
  self->simulation->run(NULL);
  return Py_None;
}

static PyObject* cMaBoSSSim_get_last_probtraj(cMaBoSSSimObject* self, PyObject* Py_UNUSED) {
  
  PyObject *dict = PyDict_New();
  
  const STATE_MAP<NetworkState_Impl, double> results = self->simulation->getAsymptoticStateDist();
  
  // Building the results as a python dict
  for (auto& result : results) {
    PyDict_SetItem(
      dict, 
      PyUnicode_FromString(NetworkState(result.first).getName(self->network).c_str()), 
      PyFloat_FromDouble(result.second)
    );
  }
  return (PyObject*) dict;
}

static PyObject* cMaBoSSSim_get_states(cMaBoSSSimObject* self, PyObject* Py_UNUSED) {

  std::map<double, STATE_MAP<NetworkState_Impl, double> > statedists = self->simulation->getStateDists();

  std::set<NetworkState_Impl> states;
  PyObject *timepoints = PyList_New(0);  
  
  // Building the results as a python dict
  for (auto& t_results : statedists) {
    PyList_Append(
      timepoints, 
      PyFloat_FromDouble(t_results.first)
    );

    for (auto& t_result : t_results.second) {
      states.insert(t_result.first);
    }
  }

  PyObject* pystates = PyList_New(0);
  for (auto& el: states) {
    PyList_Append(pystates, PyUnicode_FromString(NetworkState(el).getName(self->network).c_str()));
  }

  return PyTuple_Pack(2, timepoints, pystates);
}

static PyObject* cMaBoSSSim_get_raw_probtrajs(cMaBoSSSimObject* self, PyObject* Py_UNUSED) {

  std::map<double, STATE_MAP<NetworkState_Impl, double> > statedists = self->simulation->getStateDists();
  
  // Building the states set
  std::set<NetworkState_Impl> states;
  for (auto& t_results : statedists) {
    for (auto& t_result : t_results.second) {
      states.insert(t_result.first);
    }
  }

  // Building states index dict
  std::map<NetworkState_Impl, double> states_indexes;
  unsigned int i=0;
  for (auto& state: states) {
    states_indexes[state] = i;
    i++;
  }

  // Building array
  PyObject *timepoints = PyList_New(0);  

  for (auto& t_results: statedists) {
    PyObject* timepoint = PyList_New(states.size());
    for (unsigned int i=0; i < states.size(); i++) {
      PyList_SetItem(timepoint, i, PyFloat_FromDouble(0.0));
    }

    for (auto& t_result : t_results.second) {
      PyList_SetItem(timepoint, states_indexes[t_result.first], PyFloat_FromDouble(t_result.second));
    }

    PyList_Append(timepoints, timepoint);
  }

  return timepoints;
}

static PyMethodDef cMaBoSSSim_methods[] = {
    {"run", (PyCFunction) cMaBoSSSim_run, METH_NOARGS, "runs the simulation"},
    {"get_last_probtraj", (PyCFunction) cMaBoSSSim_get_last_probtraj, METH_NOARGS, "gets the last probtraj of the simulation"},
    {"get_states", (PyCFunction) cMaBoSSSim_get_states, METH_NOARGS, "gets the states visited by the simulation"},
    {"get_raw_probtrajs", (PyCFunction) cMaBoSSSim_get_raw_probtrajs, METH_NOARGS, "gets the raw states probability trajectories of the simulation"},
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
