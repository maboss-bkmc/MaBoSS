#include <map>
#include <fstream>
#include <filesystem>

#include <sedml/SedReport.h>

#include "SedException.h"

namespace fs = std::filesystem;
LIBSEDML_CPP_NAMESPACE_USE

class Report
{
    std::map<std::string, std::vector<double> > data;
    std::map<std::string, std::string> labels;
    std::string name;
    size_t n_samples;
    
  public:
    
    Report(SedReport* sed_report, std::map<std::string, std::vector<double> >& results_by_data_generator)
    {
        n_samples = 0;
        name = sed_report->getName();
        
        for (unsigned int j=0; j < sed_report->getListOfDataSets()->getNumDataSets(); j++)
        {  
            
            SedDataSet* dataset = sed_report->getListOfDataSets()->get(j);
            if (results_by_data_generator.find(dataset->getDataReference()) == results_by_data_generator.end())
            {
                throw SedException("Could not find data reference " + dataset->getDataReference());
            }
            data[dataset->getId()] = results_by_data_generator[dataset->getDataReference()];
            labels[dataset->getId()] = dataset->getLabel();
            if (n_samples == 0)
                n_samples = data[dataset->getId()].size();
        }
    }
    
    std::string getName()
    {
        return name;
    }
       
#ifdef PYTHON_API

PyObject* getReportData() const
{     
  npy_intp dims[2] = {(npy_intp) n_samples, (npy_intp) data.size()};
  PyArrayObject* result = (PyArrayObject *) PyArray_ZEROS(2,dims,NPY_DOUBLE, 0); 
  PyObject* py_labels = PyList_New(data.size());

  size_t i=0;
  for (auto& datum: data) 
  {
    for (size_t t=0; t < n_samples; t++) 
    {
      void* ptr = PyArray_GETPTR2(result, t, i);
      PyArray_SETITEM(
          result, 
          (char*) ptr,
          PyFloat_FromDouble(datum.second[t])
      );
    }
    PyList_SetItem(py_labels, i, PyUnicode_FromString(labels.at(datum.first).c_str()));

    i++;
  }
  
  return PyTuple_Pack(2, result, py_labels);

}

#endif
    void writeReport(fs::path filename)
    {
        std::ofstream report_file;
        report_file.open(filename.c_str());

        size_t i=0;
        for (const auto& label: labels)
        {
            report_file << label.second;
            if (i < labels.size()-1)
                report_file << ",";    
                
            i++;
        }
        report_file << std::endl;
                
        for (size_t t=0; t < n_samples; t++)
        {
            size_t i=0;
            for (const auto& datum: data)
            {
                report_file << datum.second[t];
                
                if (i < data.size()-1)
                    report_file << ",";    

                i++;
            }
            
            report_file << std::endl;
        }    
        
        report_file.close();
    }
};