#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <fstream>
#include <stdlib.h>
#include <set>
#include "src/BooleanNetwork.h"
#include "src/FinalStateSimulationEngine.h"

typedef struct {
  PyObject_HEAD
  Network* network;
  RunConfig* runconfig;
  FinalStateSimulationEngine* engine;
  time_t start_time;
  time_t end_time;
} cMaBoSSResultFinalObject;

static void cMaBoSSResultFinal_dealloc(cMaBoSSResultFinalObject *self)
{
    free(self->engine);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject * cMaBoSSResultFinal_new(PyTypeObject* type, PyObject *args, PyObject* kwargs) 
{
  cMaBoSSResultFinalObject* res;
  res = (cMaBoSSResultFinalObject *) type->tp_alloc(type, 0);

  return (PyObject*) res;
}

static PyObject* cMaBoSSResultFinal_get_last_states_probtraj(cMaBoSSResultFinalObject* self) {
  
  PyObject *dict = PyDict_New();
  
  // Building the results as a python dict
  for (auto& result : self->engine->getFinalStates()) {
    PyDict_SetItem(
      dict, 
      PyUnicode_FromString(NetworkState(result.first).getName(self->network).c_str()), 
      PyFloat_FromDouble(result.second)
    );
  }
  return (PyObject*) dict;
}

static PyObject* cMaBoSSResultFinal_get_last_nodes_probtraj(cMaBoSSResultFinalObject* self) {
  
  PyObject *dict = PyDict_New();
  
  // Building the results as a python dict
  for (auto& result : self->engine->getFinalNodes()) {
    PyDict_SetItem(
      dict, 
      PyUnicode_FromString(result.first->getLabel().c_str()), 
      PyFloat_FromDouble(result.second)
    );
  }
  return (PyObject*) dict;
}
static PyObject* cMaBoSSResultFinal_display_final_states(cMaBoSSResultFinalObject* self, PyObject* args) {

  char * filename = NULL;
  int hexfloat = 0;
  if (!PyArg_ParseTuple(args, "s|i", &filename, &hexfloat))
    return NULL;

  std::ostream* output_final = new std::ofstream(filename);

  self->engine->displayFinal(*output_final, PyObject_IsTrue(PyBool_FromLong(hexfloat)));

  ((std::ofstream*) output_final)->close();
  delete output_final;

  return Py_None;
}

static PyObject* cMaBoSSResultFinal_get_final_time(cMaBoSSResultFinalObject* self) {
  return PyFloat_FromDouble(self->engine->getFinalTime());
}


static PyObject* cMaBoSSResultFinal_display_run(cMaBoSSResultFinalObject* self, PyObject* args) 
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

static PyMethodDef cMaBoSSResultFinal_methods[] = {
    {"get_final_time", (PyCFunction) cMaBoSSResultFinal_get_final_time, METH_NOARGS, "gets the final time of the simulation"},
    {"get_last_states_probtraj", (PyCFunction) cMaBoSSResultFinal_get_last_states_probtraj, METH_NOARGS, "gets the last probtraj of the simulation"},
    {"display_final_states", (PyCFunction) cMaBoSSResultFinal_display_final_states, METH_VARARGS, "display the final state"},
    {"get_last_nodes_probtraj", (PyCFunction) cMaBoSSResultFinal_get_last_nodes_probtraj, METH_NOARGS, "gets the last nodes probtraj of the simulation"},
    {"display_run", (PyCFunction) cMaBoSSResultFinal_display_run, METH_VARARGS, "prints the run of the simulation to a file"},
    {NULL}  /* Sentinel */
};

static PyTypeObject cMaBoSSResultFinal = []{
  PyTypeObject res{PyVarObject_HEAD_INIT(NULL, 0)};

  res.tp_name = "cmaboss.cMaBoSSResultFinalObject";
  res.tp_basicsize = sizeof(cMaBoSSResultFinalObject);
  res.tp_itemsize = 0;
  res.tp_flags = Py_TPFLAGS_DEFAULT;// | Py_TPFLAGS_BASETYPE;
  res.tp_doc = "cMaBoSSResultFinalObject";
  res.tp_new = cMaBoSSResultFinal_new;
  res.tp_dealloc = (destructor) cMaBoSSResultFinal_dealloc;
  res.tp_methods = cMaBoSSResultFinal_methods;
  return res;
}();
