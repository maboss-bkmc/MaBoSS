from distutils.core import setup, Extension
from sys import executable, argv
from os.path import join, dirname

maboss_version = '3.0'

maboss_sources = [
   "MaBEstEngine.cc", "EnsembleEngine.cc", "Cumulator.cc", "ProbaDist.cc", 
   "BooleanNetwork.cc", "BooleanGrammar.cc", "RunConfigGrammar.cc", 
   "Function.cc", "BuiltinFunctions.cc", "RunConfig.cc", "LogicalExprGen.cc", "Utils.cc"
]

maboss_module_sources = ['maboss_module.cpp', 'maboss_sim.cpp']
extra_compile_args = ['-std=c++11']

def getExtensionByMaxnodes(maxnodes=64):
   if maxnodes <= 64:
      return Extension(
         'maboss_module', 
         sources= maboss_module_sources + ["src/%s" % source for source in maboss_sources], 
         extra_compile_args=extra_compile_args,
         language="c++"
      )
   else:
      return Extension(
         'maboss_module_%dn' % maxnodes, 
         sources=maboss_module_sources + ["src/%s" % source for source in maboss_sources], 
         extra_compile_args=extra_compile_args + ['-DMAXNODES=%d' % maxnodes],
         language="c++"
      )

setup (name = 'maboss_module',
   version = maboss_version,
   author = "contact@vincent-noel.fr",
   description = """MaBoSS python bindings""",
   ext_modules = [getExtensionByMaxnodes()],
)

setup (name = 'maboss_module_128n',
   version = maboss_version,
   author = "contact@vincent-noel.fr",
   description = """MaBoSS python bindings""",
   ext_modules = [getExtensionByMaxnodes(128)],
)

setup (name = 'maboss_module_256n',
   version = maboss_version,
   author = "contact@vincent-noel.fr",
   description = """MaBoSS python bindings""",
   ext_modules = [getExtensionByMaxnodes(256)],
)

setup (name = 'maboss_module_512',
   version = maboss_version,
   author = "contact@vincent-noel.fr",
   description = """MaBoSS python bindings""",
   ext_modules = [getExtensionByMaxnodes(512)],
)

setup (name = 'maboss_module_1024',
   version = maboss_version,
   author = "contact@vincent-noel.fr",
   description = """MaBoSS python bindings""",
   ext_modules = [getExtensionByMaxnodes(1024)],
)
