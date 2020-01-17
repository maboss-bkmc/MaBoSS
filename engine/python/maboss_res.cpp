#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <fstream>
#include <stdlib.h>
#include <set>
#include "src/BooleanNetwork.h"
#include "src/MaBEstEngine.h"

typedef struct {
  PyObject_HEAD
  Network* network;
  RunConfig* runconfig;
  MaBEstEngine* engine;
  time_t start_time;
  time_t end_time;
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

static PyObject* cMaBoSSResult_get_fp_table(cMaBoSSResultObject* self) {

  PyObject *dict = PyDict_New();

  for (auto& result: self->engine->getFixPointsDists()) {
    PyObject *tuple = PyTuple_Pack(2, 
      PyFloat_FromDouble(result.second.second),
      PyUnicode_FromString(result.second.first.getName(self->network).c_str())
    );

    PyDict_SetItem(dict, PyLong_FromUnsignedLong(result.first), tuple);
  }

  return dict;
}

static PyObject* cMaBoSSResult_get_last_states_probtraj(cMaBoSSResultObject* self) {
  
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

static PyObject* cMaBoSSResult_get_final_time(cMaBoSSResultObject* self) {

  return PyFloat_FromDouble(self->engine->getFinalTime());

}

static PyObject* cMaBoSSResult_get_last_nodes_probtraj(cMaBoSSResultObject* self) {
  
  PyObject *dict = PyDict_New();
  
  // Building the results as a python dict
  for (auto& result : self->engine->getAsymptoticNodesDist()) {
    PyDict_SetItem(
      dict, 
      PyUnicode_FromString(result.first->getLabel().c_str()), 
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

static PyObject* cMaBoSSResult_get_nodes(cMaBoSSResultObject* self) {

  std::set<Node*> nodes;
  PyObject *timepoints = PyList_New(0);  
  
  // Building the results as a python dict
  for (auto& t_results : self->engine->getNodesDists()) {
    PyList_Append(
      timepoints,
      PyFloat_FromDouble(t_results.first)
    );

    for (auto& t_result : t_results.second) {
      nodes.insert(t_result.first);
    }
  }

  PyObject* pynodes = PyList_New(0);
  for (auto& el: nodes) {
    PyList_Append(pynodes, PyUnicode_FromString(el->getLabel().c_str()));
  }

  return PyTuple_Pack(2, timepoints, pynodes);
}

static PyObject* cMaBoSSResult_get_raw_nodes_probtrajs(cMaBoSSResultObject* self) {
  
  // Building the states set
  std::set<Node*> nodes;
  for (auto& t_results : self->engine->getNodesDists()) {
    for (auto& t_result : t_results.second) {
      nodes.insert(t_result.first);
    }
  }

  // Building states index dict
  std::map<Node*, double> nodes_indexes;
  unsigned int i=0;
  for (auto& node: nodes) {
    nodes_indexes[node] = i;
    i++;
  }

  // Building array
  PyObject *timepoints = PyList_New(0);  

  for (auto& t_results: self->engine->getNodesDists()) {
    PyObject* timepoint = PyList_New(nodes.size());
    for (unsigned int i=0; i < nodes.size(); i++) {
      PyList_SetItem(timepoint, i, PyFloat_FromDouble(0.0));
    }

    for (auto& t_result : t_results.second) {
      PyList_SetItem(timepoint, nodes_indexes[t_result.first], PyFloat_FromDouble(t_result.second));
    }

    PyList_Append(timepoints, timepoint);
  }

  return timepoints;
}

static PyObject* cMaBoSSResult_display_fp(cMaBoSSResultObject* self, PyObject *args) 
{
  char * filename = NULL;
  int hexfloat = 0;
  if (!PyArg_ParseTuple(args, "s|i", &filename, &hexfloat))
    return NULL;
    
  std::ostream* output_fp = new std::ofstream(filename);
  self->engine->displayFixpoints(*output_fp, (bool) hexfloat);
  ((std::ofstream*) output_fp)->close();
  delete output_fp;

  return Py_None;
}

static PyObject* cMaBoSSResult_display_probtraj(cMaBoSSResultObject* self, PyObject *args) 
{
  char * filename = NULL;
  int hexfloat = 0;
  if (!PyArg_ParseTuple(args, "s|i", &filename, &hexfloat))
    return NULL;
    
  std::ostream* output_probtraj = new std::ofstream(filename);
  self->engine->displayProbTraj(*output_probtraj, (bool) hexfloat);
  ((std::ofstream*) output_probtraj)->close();
  delete output_probtraj;

  return Py_None;
}

static PyObject* cMaBoSSResult_display_statdist(cMaBoSSResultObject* self, PyObject *args) 
{
  char * filename = NULL;
  int hexfloat = 0;
  if (!PyArg_ParseTuple(args, "s|i", &filename, &hexfloat))
    return NULL;
    
  std::ostream* output_statdist = new std::ofstream(filename);
  self->engine->displayStatDist(*output_statdist, (bool) hexfloat);
  ((std::ofstream*) output_statdist)->close();
  delete output_statdist;

  return Py_None;
}

static PyObject* cMaBoSSResult_display_run(cMaBoSSResultObject* self, PyObject* args) 
{
  char * filename = NULL;
  int hexfloat = 0;
  if (!PyArg_ParseTuple(args, "s|i", &filename, &hexfloat))
    return NULL;
    
  std::ostream* output_run = new std::ofstream(filename);
  self->runconfig->display(self->network, self->start_time, self->end_time,*(self->engine), *output_run);
  ((std::ofstream*) output_run)->close();
  delete output_run;

  return Py_None;
}

static PyMethodDef cMaBoSSResult_methods[] = {
    {"get_final_time", (PyCFunction) cMaBoSSResult_get_final_time, METH_NOARGS, "gets the final time of the simulation"},
    {"get_fp_table", (PyCFunction) cMaBoSSResult_get_fp_table, METH_NOARGS, "gets the fixpoints table"},
    {"get_last_states_probtraj", (PyCFunction) cMaBoSSResult_get_last_states_probtraj, METH_NOARGS, "gets the last probtraj of the simulation"},
    {"get_states", (PyCFunction) cMaBoSSResult_get_states, METH_NOARGS, "gets the states visited by the simulation"},
    {"get_raw_probtrajs", (PyCFunction) cMaBoSSResult_get_raw_probtrajs, METH_NOARGS, "gets the raw states probability trajectories of the simulation"},
    {"get_last_nodes_probtraj", (PyCFunction) cMaBoSSResult_get_last_nodes_probtraj, METH_NOARGS, "gets the last nodes probtraj of the simulation"},
    {"get_nodes", (PyCFunction) cMaBoSSResult_get_nodes, METH_NOARGS, "gets the nodes visited by the simulation"},
    {"get_raw_nodes_probtrajs", (PyCFunction) cMaBoSSResult_get_raw_nodes_probtrajs, METH_NOARGS, "gets the raw nodes probability trajectories of the simulation"},
    {"display_fp", (PyCFunction) cMaBoSSResult_display_fp, METH_VARARGS, "prints the fixpoints to a file"},
    {"display_probtraj", (PyCFunction) cMaBoSSResult_display_probtraj, METH_VARARGS, "prints the probtraj to a file"},
    {"display_statdist", (PyCFunction) cMaBoSSResult_display_statdist, METH_VARARGS, "prints the statdist to a file"},
    {"display_run", (PyCFunction) cMaBoSSResult_display_run, METH_VARARGS, "prints the run of the simulation to a file"},
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
