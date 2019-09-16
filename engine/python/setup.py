from distutils.core import setup, Extension

maboss_sources = [
   "MaBEstEngine.cc", "EnsembleEngine.cc", "Cumulator.cc", "ProbaDist.cc", 
   "BooleanNetwork.cc", "BooleanGrammar.cc", "RunConfigGrammar.cc", 
   "Function.cc", "BuiltinFunctions.cc", "RunConfig.cc", "LogicalExprGen.cc", "Utils.cc"
]



# define the extension module
maboss_module = Extension('maboss_module', sources=['maboss_module.cpp'] + ["../src/%s" % source for source in maboss_sources], language = "c++")

setup (name = 'maboss_module',
   version = '2.0',
   author = "contact@vincent-noel.fr",
   description = """MaBoSS python bindings""",
   ext_modules = [maboss_module],
   # py_modules = ["pymaboss"],
)

