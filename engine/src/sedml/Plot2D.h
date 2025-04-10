#include <sedml/SedListOfStyles.h>
#include <stdlib.h>
#include <vector>
#include <map>

#include <sedml/SedPlot2D.h>
#include <sedml/SedCurve.h>

#ifdef PYTHON_API
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL MABOSS_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#endif
LIBSEDML_CPP_NAMESPACE_USE

class Plot2D {
   
    std::map<std::string, std::pair<std::vector<double>, std::vector<double> > > data;
    std::map<std::string, std::string> labels;
    std::map<std::string, std::string> colors;
    std::map<std::string, std::pair<std::string, std::string> > axis_labels;
    std::string name;
    bool legend;
    size_t n_samples;
  public: 
    Plot2D(SedPlot2D* sed_plot, std::map<std::string, std::vector<double> >& results_by_data_generator, SedListOfStyles* styles)
    { 
        n_samples = 0;
        legend = sed_plot->isSetLegend() ? sed_plot->getLegend() : true;
        
        for (unsigned int j=0; j < sed_plot->getListOfCurves()->getNumCurves(); j++)
        {  
            SedCurve* curve = static_cast<SedCurve*>(sed_plot->getListOfCurves()->get(j));
            data[curve->getId()] = std::make_pair(
                results_by_data_generator[curve->getXDataReference()],
                results_by_data_generator[curve->getYDataReference()]
            );
            labels[curve->getId()] = curve->getName();
            
            if (curve->isSetStyle()){
              SedStyle* style = styles->get(curve->getStyle());
              SedLine* line_style = style->getLineStyle();
              colors[curve->getId()] = line_style->getColor();
            }
            
            n_samples = results_by_data_generator[curve->getXDataReference()].size();
        }
    }
    
#ifdef PYTHON_API

    PyObject* getPlotData() const
    {     
      npy_intp dims[1] = {(npy_intp) n_samples};
      
      PyObject* curves = PyDict_New();
      
      for (auto& curve: data) 
      {
        PyArrayObject* result = (PyArrayObject *) PyArray_ZEROS(1,dims,NPY_DOUBLE, 0); 
        PyObject* time = PyList_New(n_samples);
        for (size_t t=0; t < n_samples; t++) 
        {
          void* ptr = PyArray_GETPTR1(result, t);
          PyArray_SETITEM(
              result, 
              (char*) ptr,
              PyFloat_FromDouble(curve.second.second[t])
          );
          PyList_SetItem(time, t, PyFloat_FromDouble(curve.second.first[t]));
        }
        PyObject* py_name = PyUnicode_FromString(labels.at(curve.first).c_str());
        PyObject* py_color = PyUnicode_FromString(colors.at(curve.first).c_str());
        PyObject* py_legend = legend ? Py_True : Py_False;
        PyObject* curve_data = PyTuple_Pack(5, result, time, py_name, py_color, py_legend);
        PyDict_SetItemString(
          curves,
          curve.first.c_str(),
          curve_data
        );
      }
      
      return curves;

    }

#endif
};