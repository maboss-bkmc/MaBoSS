from setuptools import setup, Extension
from sys import executable, argv
from os.path import join, dirname

maboss_version = '1.0.0-beta-6'

maboss_sources = [
   # Core
   "BooleanNetwork.cc", "BooleanGrammar.cc", "RunConfigGrammar.cc", "Function.cc", "BuiltinFunctions.cc", 
   "RunConfig.cc", "LogicalExprGen.cc", "Utils.cc", "MBDynBitset.cc", "RandomGenerator.cc", "FixedPointDisplayer.cc", 

   # MaBoSS
   "MetaEngine.cc", "FixedPointEngine.cc", "ProbTrajEngine.cc",
   "FinalStateSimulationEngine.cc", "StochasticSimulationEngine.cc", "MaBEstEngine.cc", "EnsembleEngine.cc", 
   "Cumulator.cc", "ProbaDist.cc", 
   "StatDistDisplayer.cc", "FinalStateDisplayer.cc",  
   
   # PopMaBoSS
   "PopMaBEstEngine.cc", "PopCumulator.cc", "PopProbaDist.cc", "PopProbTrajDisplayer.cc"
]

maboss_module_sources = [
   'maboss_module.cpp', 
]
extra_compile_args = ['-std=c++11', '-DPYTHON_API']

def getExtensionByMaxnodes(maxnodes=64):
   import numpy
   return Extension(
      "cmaboss" if maxnodes <= 64 else "cmaboss_%dn" % maxnodes, 
      sources=maboss_module_sources + ["src/%s" % source for source in maboss_sources], 
      include_dirs=[numpy.get_include()],
      extra_compile_args=(extra_compile_args + (['-DMAXNODES=%d' % max(maxnodes, 64)])),
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
   install_requires = ["numpy"]
)
