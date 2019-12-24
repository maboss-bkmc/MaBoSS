#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <set>
#include "src/BooleanNetwork.h"
#include "src/MaBEstEngine.h"

typedef struct {
  PyObject_HEAD
  Network* network;
  MaBEstEngine* engine;
} cMaBoSSResultObject;

static void cMaBoSSResult_dealloc(cMaBoSSResultObject *self)
{
    free(self->engine);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject * cMaBoSSResult_new(PyTypeObject* type, PyObject *args, PyObject* kwargs) 
{
  cMaBoSSResultObject* res;
  res = (cMaBoSSResultObject *) type->tp_alloc(type, 0);

  return (PyObject*) res;
}

static PyObject* cMaBoSSResult_get_last_probtraj(cMaBoSSResultObject* self) {
  
  PyObject *dict = PyDict_New();
  
  // Building the results as a python dict
  for (auto& result : self->engine->getAsymptoticStateDist()) {
    PyDict_SetItem(
      dict, 
      PyUnicode_FromString(NetworkState(result.first).getName(self->network).c_str()), 
      PyFloat_FromDouble(result.second)
    );
  }
  return (PyObject*) dict;
}

static PyObject* cMaBoSSResult_get_states(cMaBoSSResultObject* self) {

  std::set<NetworkState_Impl> states;
  PyObject *timepoints = PyList_New(0);  
  
  // Building the results as a python dict
  for (auto& t_results : self->engine->getStateDists()) {
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

static PyObject* cMaBoSSResult_get_raw_probtrajs(cMaBoSSResultObject* self) {
  
  // Building the states set
  std::set<NetworkState_Impl> states;
  for (auto& t_results : self->engine->getStateDists()) {
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

  for (auto& t_results: self->engine->getStateDists()) {
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

static PyMethodDef cMaBoSSResult_methods[] = {
    {"get_last_probtraj", (PyCFunction) cMaBoSSResult_get_last_probtraj, METH_NOARGS, "gets the last probtraj of the simulation"},
    {"get_states", (PyCFunction) cMaBoSSResult_get_states, METH_NOARGS, "gets the states visited by the simulation"},
    {"get_raw_probtrajs", (PyCFunction) cMaBoSSResult_get_raw_probtrajs, METH_NOARGS, "gets the raw states probability trajectories of the simulation"},
    {NULL}  /* Sentinel */
};

static PyTypeObject cMaBoSSResult = []{
  PyTypeObject res{PyVarObject_HEAD_INIT(NULL, 0)};

  res.tp_name = "cmaboss.cMaBoSSResultObject";
  res.tp_basicsize = sizeof(cMaBoSSResultObject);
  res.tp_itemsize = 0;
  res.tp_flags = Py_TPFLAGS_DEFAULT;// | Py_TPFLAGS_BASETYPE;
  res.tp_doc = "cMaBoSSResultobject";
  res.tp_new = cMaBoSSResult_new;
  res.tp_dealloc = (destructor) cMaBoSSResult_dealloc;
  res.tp_methods = cMaBoSSResult_methods;
  return res;
}();
