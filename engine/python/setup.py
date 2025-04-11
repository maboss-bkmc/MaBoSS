from setuptools import setup, Extension, find_packages
import os, numpy

maboss_version = '1.0.0b26'

maboss_sources = [
   # PopMaBoSS
   "PopMaBEstEngine.cc", 
   
   # Core
   "BooleanNetwork.cc", "BooleanGrammar.cc", "RunConfigGrammar.cc", "Function.cc", "BuiltinFunctions.cc", 
   "RunConfig.cc", "LogicalExprGen.cc", "Utils.cc", "MBDynBitset.cc", "RandomGenerator.cc", "FixedPointDisplayer.cc", 

   # MaBoSS
   "MetaEngine.cc", "FixedPointEngine.cc", "ProbTrajEngine.cc",
   "FinalStateSimulationEngine.cc", "StochasticSimulationEngine.cc", "MaBEstEngine.cc", "EnsembleEngine.cc", 
   "ProbaDist.cc", "ObservedGraph.cc",
   "StatDistDisplayer.cc", "FinalStateDisplayer.cc",   
]

maboss_module_sources = [
   'cmaboss/maboss_node.cpp',
   'cmaboss/maboss_param.cpp',
   'cmaboss/maboss_cfg.cpp',
   'cmaboss/maboss_net.cpp',
   'cmaboss/maboss_sim.cpp',
   'cmaboss/maboss_res.cpp',
   'cmaboss/maboss_resfinal.cpp',
   'cmaboss/popmaboss_net.cpp',
   'cmaboss/popmaboss_sim.cpp',
   'cmaboss/popmaboss_res.cpp',
   'cmaboss/maboss_module.cpp', 
]

extra_compile_args = ['-std=c++11', '-DPYTHON_API']

libraries = []
include_dirs = [numpy.get_include()]

if (os.environ.get('SBML_COMPAT') is not None):
   extra_compile_args.append('-DSBML_COMPAT')
   libraries.append('sbml')
   
   
if (os.environ.get('SEDML_COMPAT') is not None):
   maboss_sources.append('sedml/XMLPatcher.cc')
   maboss_module_sources.append('cmaboss/sedml_sim.cpp')
   extra_compile_args.remove('-std=c++11')
   extra_compile_args.insert(0, '-std=c++17')
   extra_compile_args.append('-DSEDML_COMPAT')
   libraries += ['sedml', 'xml2']
   import lxml
   include_dirs += lxml.get_include()   
   
def getExtensionByMaxnodes(maxnodes=64):
   return Extension(
      "cmaboss.cmaboss" if maxnodes <= 64 else "cmaboss_%dn.cmaboss_%dn" % (maxnodes, maxnodes), 
      sources=maboss_module_sources + ["cmaboss/src/%s" % source for source in maboss_sources], 
      include_dirs=include_dirs,
      extra_compile_args=(extra_compile_args + (['-DMAXNODES=%d' % max(maxnodes, 64)])),
      libraries=libraries,
      language="c++"
   )

setup (name = 'cmaboss',
   version = maboss_version,
   author = "contact@vincent-noel.fr",
   description = """MaBoSS python bindings""",
   ext_modules = [
      getExtensionByMaxnodes(), 
      getExtensionByMaxnodes(128), 
      getExtensionByMaxnodes(256), 
      getExtensionByMaxnodes(512),
      getExtensionByMaxnodes(1024)
   ],
   install_requires = ["numpy", "lxml"],
   packages=find_packages()
)
