#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include "../src/BooleanNetwork.h"
#include "../src/MaBEstEngine.h"

typedef struct {
  Network* network;
} PyMaBoSSNetwork;


static PyObject *load(PyObject* self, PyObject *args, PyObject* kwargs) 
{

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
  runconfig->parse(network, config_file);

  // Error checking
  IStateGroup::checkAndComplete(network);

  // Running the simulation
  MaBEstEngine mabest(network, runconfig);
  mabest.run(NULL);

  // Getting the results
  const STATE_MAP<NetworkState_Impl, double> results = mabest.getAsymptoticStateDist();

  // Building the results as a python dict
  PyObject* dict_result = PyDict_New();
  for (auto& result : results) {
    PyDict_SetItem(
      dict_result, 
      PyUnicode_FromString(NetworkState(result.first).getName(network).c_str()), 
      PyFloat_FromDouble(result.second)
    );
  }

  free(network);
  return dict_result;
}
/*  define functions in module */

static PyMethodDef cMaBoSS[] =
{
     {"load", (PyCFunction) load, METH_VARARGS | METH_KEYWORDS, "load"},
     {NULL, NULL, 0, NULL}
};

/* module initialization */
/* Python version 3*/
static struct PyModuleDef cMaBoSSDef =
{
    PyModuleDef_HEAD_INIT,
    "maboss_module", 
    "Some documentation",
    -1,
    cMaBoSS
};

PyMODINIT_FUNC
PyInit_maboss_module(void)
{
    return PyModule_Create(&cMaBoSSDef);
}
