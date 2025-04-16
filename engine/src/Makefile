#
# MaBoSS project Makefile 
#
# Institut Curie, France
#
# Eric Viara, Gautier Stoll February 2011
#

ifndef MAXNODES
MAXNODES = 64
endif

CXX_VERSION = c++11

ifeq ($(USE_DYNAMIC_BITSET_STD_ALLOC), 1)
INFIX := _dn
BITSET_FLAG := -DUSE_DYNAMIC_BITSET -DUSE_DYNAMIC_BITSET_STD_ALLOC
endif

ifeq ($(USE_DYNAMIC_BITSET), 1)
INFIX := _dn_exp
BITSET_FLAG := -DUSE_DYNAMIC_BITSET
endif

ifeq ($(USE_BITSET), 1)
INFIX := _bitset
BITSET_FLAG := -DUSE_BITSET
endif

ifeq ($(shell expr $(MAXNODES) '>' 64), 1)
INFIX := _$(MAXNODES)n
else
MAXNODES = 64
#MABOSS := MaBoSS
endif

ifeq ($(MPI_COMPAT), 1)
INFIX := $(INFIX).MPI
endif

UNAME_S := $(shell uname -s)

ifeq ($(OS),Windows_NT)
       WINDOWS_FLAG := -DWINDOWS
       MABOSS := MaBoSS$(INFIX).exe
       POPMABOSS := PopMaBoSS$(INFIX).exe
	   STATIC_LIBS := -lz -lbz2 -llzma -liconv -lws2_32
else
	LDL_FLAG := -ldl
	MABOSS := MaBoSS$(INFIX)
    POPMABOSS := PopMaBoSS$(INFIX)

	ifeq ($(UNAME_S),Darwin)
	   STATIC_LIBS := -lz -lbz2
	else
       STATIC_LIBS := -lz -lbz2 -llzma -licuuc -licudata
	endif
endif

ifdef SBML_COMPAT
SBML_FLAG = -DSBML_COMPAT
SBML_LIB = -lsbml
ifeq ($(UNAME_S),Darwin)
SBML_LIB_STATIC = -lsbml-static -lxml2
else
SBML_LIB_STATIC = -static -lsbml-static -lxml2
endif
endif

ifdef SEDML_COMPAT
LIBXML2_FLAG = $(shell xml2-config --cflags)
SEDML_FLAG = -DSEDML_COMPAT $(LIBXML2_FLAG)
SEDML_LIB = -lsedml-static -lnuml
SBML_LIB = -lsbml-static -lxml2 -lbz2 -lz
CXX_VERSION = c++17

SEDML_LIB_SRC := sedml/XMLPatcher.cc
SEDML_LIB_OBJS := sedml/XMLPatcher$(INFIX).o

endif

ifdef HDF5_COMPAT
HDF5_FLAG = -DHDF5_COMPAT
HDF5_LIB = -lhdf5_serial -lhdf5_serial_hl
endif

ifdef STD_THREAD
THREAD_FLAG = -DSTD_THREAD
endif

MABOSS_SERVER := MaBoSS$(INFIX)-server
MABOSS_CLIENT := MaBoSS-client

MABOSS_LIBNAME = MaBoSS$(INFIX)
POPMABOSS_LIBNAME = PopMaBoSS$(INFIX)

MABOSS_LIB = lib$(MABOSS_LIBNAME).so
MABOSS_ALIB = lib$(MABOSS_LIBNAME)-static.a
POPMABOSS_LIB = lib$(POPMABOSS_LIBNAME).so
POPMABOSS_ALIB = lib$(POPMABOSS_LIBNAME)-static.a


EXTRA_FLAGS = -std=$(CXX_VERSION) -Wpedantic

ifndef CC
CC = gcc
endif

ifndef CXX
CXX = g++
endif

ifdef MPI_COMPAT
MPI_FLAG = -DMPI_COMPAT
ifndef CC
CC = mpicc
endif
ifndef CXX
CXX = mpic++
endif
endif

ifdef DEBUG
DEBUG_FLAGS = -ggdb -D_GLIBCXX_DEBUG -DDEBUG
else
DEBUG_FLAGS = -DNDEBUG
endif

EXTRA_CXXOPT = -O2
EXTRA_CXXFLAGS = $(EXTRA_CXXOPT) -Wall -DMAXNODES=$(MAXNODES) $(MPI_FLAG) $(THREAD_FLAG) $(BITSET_FLAG) $(USER_FLAGS) $(EXTRA_FLAGS) -fpic ${DEBUG_FLAGS}

ifndef FLEX
FLEX = flex
endif

ifndef BISON
BISON = bison
endif 


CHECK_FLEX  = $(FLEX) --version > /dev/null
CHECK_BISON = $(BISON) --version > /dev/null

CORE_LIB_SRC := BooleanNetwork.cc BooleanGrammar.cc RunConfigGrammar.cc Function.cc BuiltinFunctions.cc RunConfig.cc LogicalExprGen.cc Utils.cc MBDynBitset.cc RandomGenerator.cc FixedPointDisplayer.cc ObservedGraph.cc
CORE_LIB_OBJS := BooleanNetwork$(INFIX).o parsers/BooleanGrammar$(INFIX).o parsers/RunConfigGrammar$(INFIX).o Function$(INFIX).o BuiltinFunctions$(INFIX).o RunConfig$(INFIX).o LogicalExprGen$(INFIX).o Utils$(INFIX).o MBDynBitset$(INFIX).o RandomGenerator$(INFIX).o  FixedPointDisplayer$(INFIX).o ObservedGraph$(INFIX).o
POP_LIB_SRC := PopMaBEstEngine.cc MetaEngine.cc
POP_LIB_OBJS := PopMaBEstEngine$(INFIX).o MetaEngine.o
LIB_SRC := FinalStateSimulationEngine.cc StochasticSimulationEngine.cc MetaEngine.cc FixedPointEngine.cc ProbTrajEngine.cc MaBEstEngine.cc EnsembleEngine.cc ProbaDist.cc StatDistDisplayer.cc FinalStateDisplayer.cc
LIB_OBJS := FinalStateSimulationEngine$(INFIX).o StochasticSimulationEngine$(INFIX).o MetaEngine$(INFIX).o FixedPointEngine$(INFIX).o ProbTrajEngine$(INFIX).o MaBEstEngine$(INFIX).o EnsembleEngine$(INFIX).o ProbaDist$(INFIX).o StatDistDisplayer$(INFIX).o FinalStateDisplayer$(INFIX).o

SERVER_OBJS := Server$(INFIX).o DataStreamer$(INFIX).o MaBoSS-RPC$(INFIX).o

# if USE_LIB is set, maboss library will be used for MaBoSS and MaBoSS-server programs => that will oblige the user to set LD_LIBRARY_PATH or DYLD_LIBRARY_PATH environment variable to execute MaBoSS or MaBoSS-server
#USE_LIB = 1

ifdef USE_LIB
MABOSS_OBJS = $(MABOSS_LIB)
POPMABOSS_OBJS = $(POPMABOSS_LIB)
else
MABOSS_OBJS = $(CORE_LIB_OBJS) $(LIB_OBJS) $(SEDML_LIB_OBJS) 
POPMABOSS_OBJS = $(CORE_LIB_OBJS) $(POP_LIB_OBJS)
endif

ifdef FUNC_MODULE
.PHONY: func_module
func_module: $(FUNC_MODULE)

$(FUNC_MODULE): $(FUNC_MODULE).o Function$(INFIX).o
	$(CXX) $(CXXFLAGS) $(EXTRA_CXXFLAGS) -shared -o lib$(FUNC_MODULE).so $(FUNC_MODULE).o Function$(INFIX).o -lpthread -L.

endif

.PHONY: maboss
maboss: $(MABOSS_LIB) $(MABOSS)
	mkdir -p ../pub
	cp $(MABOSS) ../pub

.PHONY: popmaboss
popmaboss: $(POPMABOSS_LIB) $(POPMABOSS)
	mkdir -p ../pub
	cp $(POPMABOSS) ../pub

.PHONY: client
client: $(MABOSS_CLIENT)
	mkdir -p ../pub
	cp $(MABOSS_CLIENT) ../pub

.PHONY: server
server: $(MABOSS_SERVER)
	mkdir -p ../pub
	cp $(MABOSS_SERVER) ../pub

.PHONY: grammars
grammars: parsers/BooleanGrammar.cc parsers/RunConfigGrammar.cc

ifndef CLIENT_SERVER
all: maboss popmaboss
else
all: maboss popmaboss client server
endif

$(MABOSS_ALIB): $(CORE_LIB_OBJS) $(LIB_OBJS)
	ar rcs $(MABOSS_ALIB) $+

$(MABOSS_LIB): $(CORE_LIB_OBJS) $(LIB_OBJS)
	$(CXX) -shared -o $(MABOSS_LIB) $+ -lpthread $(LDL_FLAG) $(LDFLAGS) $(SBML_LIB) $(SEDML_LIB) $(HDF5_LIB)

$(POPMABOSS_ALIB): $(CORE_LIB_OBJS) $(POP_LIB_OBJS)
	ar rcs $(POPMABOSS_ALIB) $+

$(POPMABOSS_LIB): $(CORE_LIB_OBJS) $(POP_LIB_OBJS)
	$(CXX) -shared -o $(POPMABOSS_LIB) $+ -lpthread $(LDL_FLAG) $(LDFLAGS) $(SBML_LIB) $(HDF5_LIB)

$(MABOSS): MaBoSS$(INFIX).o $(MABOSS_OBJS)
ifdef USE_LIB
	$(CXX) -o $(MABOSS) MaBoSS$(INFIX).o -l$(MABOSS_LIBNAME) -lpthread -L. $(LDL_FLAG) $(SBML_LIB) $(SEDML_LIB) $(HDF5_LIB) $(LDFLAGS) $(EXTRA_LDFLAGS)
else
	$(CXX) -o $(MABOSS) $+ -lpthread $(LDL_FLAG) $(SBML_LIB) $(SEDML_LIB) $(HDF5_LIB) $(LDFLAGS) $(EXTRA_LDFLAGS)
endif

$(POPMABOSS): PopMaBoSS$(INFIX).o $(POPMABOSS_OBJS)
ifdef USE_LIB
	$(CXX) -o $(POPMABOSS) PopMaBoSS$(INFIX).o -l$(POPMABOSS_LIBNAME) -lpthread -L. $(LDL_FLAG) $(SBML_LIB) $(HDF5_LIB) $(LDFLAGS) $(EXTRA_LDFLAGS)
else
	$(CXX) -o $(POPMABOSS) $+ -lpthread $(LDL_FLAG) $(SBML_LIB) $(HDF5_LIB) $(LDFLAGS) $(EXTRA_LDFLAGS)
endif

$(MABOSS_SERVER): MaBoSS$(INFIX)-server.o $(MABOSS_OBJS) $(SERVER_OBJS)
ifdef USE_LIB
	$(CXX) -o $(MABOSS_SERVER) MaBoSS$(INFIX)-server.o $(SERVER_OBJS) -l$(MABOSS_LIBNAME) -lpthread -L. $(LDL_FLAG) $(SBML_LIB) $(LDFLAGS) $(EXTRA_LDFLAGS)
else
	$(CXX) -o $(MABOSS_SERVER) $+ -lpthread $(LDL_FLAG) $(SBML_LIB) $(LDFLAGS) $(EXTRA_LDFLAGS)
endif

$(MABOSS_CLIENT): MaBoSS-client.o Client.o DataStreamer.o MaBoSS-RPC.o Utils$(INFIX).o
	$(CXX) -o $(MABOSS_CLIENT) $+ -lpthread $(LDL_FLAG) $(LDFLAGS) $(EXTRA_LDFLAGS)

%$(INFIX).o: %.cc
	$(CXX) $(CXXFLAGS) $(EXTRA_CXXFLAGS) $(WINDOWS_FLAG) $(SBML_FLAG) $(SEDML_FLAG) $(HDF5_FLAG) -c $*.cc -o $*$(INFIX).o

%$(INFIX)-server.o: %-server.cc
	$(CXX) $(CXXFLAGS) $(EXTRA_CXXFLAGS) $(SBML_FLAG) $(SEDML_FLAG) $(HDF5_FLAG) -c $*-server.cc -o $*$(INFIX)-server.o

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(EXTRA_CXXFLAGS) $(WINDOWS_FLAG) $(SBML_FLAG) $(SEDML_FLAG) $(HDF5_FLAG) -c $*.cc -o $*.o

parsers/BooleanGrammar.cc: parsers/lex.CTBNDL.cc parsers/BooleanGrammar.y
	@$(CHECK_BISON)
	$(BISON) -v parsers/BooleanGrammar.y --output=parsers/BooleanGrammar.cc

parsers/lex.CTBNDL.cc: parsers/BooleanGrammar.l
	@$(CHECK_FLEX)
	$(FLEX) --outfile=parsers/lex.CTBNDL.cc $+

parsers/RunConfigGrammar.cc: parsers/RunConfigGrammar.y parsers/lex.RC.cc
	@$(CHECK_BISON)
	$(BISON) -v parsers/RunConfigGrammar.y --output parsers/RunConfigGrammar.cc
	
parsers/lex.RC.cc: parsers/RunConfigGrammar.l
	@$(CHECK_FLEX)
	$(FLEX) --outfile=parsers/lex.RC.cc $+ 

.PHONY: mbdyn
mbdyn: test_MBDynBitset

test_MBDynBitset: test_MBDynBitset.o
	g++ -o $@ $+ $(EXTRA_FLAGS)

clean: cleanobj
	rm -f OpenMpExampleMutant OpenMpExample Test TestApi KMC $(MABOSS) $(POPMABOSS) $(MABOSS_CLIENT) $(MABOSS_SERVER) 
	rm -fr parsers/lex.CTBNDL.cc parsers/BooleanGrammar.cc parsers/BooleanGrammar.output 
	rm -fr parsers/lex.RC.cc parsers/RunConfigGrammar.cc parsers/RunConfigGrammar.output
	rm -f MaBoSS MaBoSS_dn MaBoSS_[0-9]*n MaBoSS_[0-9]*n-server MaBoSS_[0-9]*n-client
	rm -f PopMaBoSS PopMaBoSS_dn PopMaBoSS_[0-9]*n
	rm -f libMaBoSS.so libMaBoSS_[0-9]*n.so libPopMaBoSS.so libPopMaBoSS_[0-9]*n.so
	rm -f libMaBoSS-static.a libMaBoSS_[0-9]*n-static.a libPopMaBoSS-static.a libPopMaBoSS_[0-9]*n-static.a
	
cleanobj:
	rm -f *$(INFIX).o *$(INFIX).so *$(INFIX).dylib

tar: all
	$(MAKE) -f Makefile.maboss tar_perform

tar_perform: clean
	(cd ../..; basename=$$(basename $$(pwd)); cd ..; tar cvfz backup/$${basename}-$$(date +"20%y-%m-%d-%H%M").tgz $${basename}/engine/examples/*/*.bnd $${basename}/engine/examples/*/*.cfg $${basename}/engine/src/*.[hly] $${basename}/engine/src/*.cc $${basename}/engine/src/Makefile $${basename}/engine/src/Makefile.maboss $${basename}/engine/src/init-config.sh $${basename}/engine/tests/*.sh $${basename}/engine/tests/*.bnd $${basename}/engine/tests/*.cfg)

ifndef CLIENT_SERVER
install: $(MABOSS) $(POPMABOSS)
	mkdir -p ../pub
	cp $(MABOSS) $(POPMABOSS) ../pub
else
install: $(MABOSS) $(MABOSS_SERVER) $(MABOSS_CLIENT) $(POPMABOSS)
	mkdir -p ../pub
	cp $(MABOSS) $(MABOSS_SERVER) $(MABOSS_CLIENT) $(POPMABOSS) ../pub
endif

package:
	@(cd ..; dir=`pwd`; cd src; make -j 4 all; version=`./MaBoSS -version | awk '{print $$3}'`; make clean; echo $$version; mkdir -p /tmp/MaBoSS-$$version/package; cd ..; find README src binaries doc examples \( -name Makefile\* -o -name \*.[ylh] -o -name init-config.sh -o -name \*.cc -o -name MaBoSS-RefCard.pdf -o -name DescriptionOutputFile.pdf -o -name MaBoSS -o -name MaBoSS_*n -o -name MaBoSS_*n.exe -o -name MaBoSS*-* -o -name MaBoSS.exe -o -name \*.cfg -o -name \*.bnd -o -name README \) -print | cpio -pdmv /tmp/MaBoSS-$$version; cd /tmp; tar  cvfz /tmp/MaBoSS-$$version.tgz --exclude='*.svn*' MaBoSS-$$version; echo | rm -r /tmp/MaBoSS-$$version; echo; echo "Package: /tmp/MaBoSS-$$version.tgz")

install_lib: $(MABOSS_LIB) $(POPMABOSS_LIB)
	mkdir -p ../lib
	cp $(MABOSS_LIB) ../lib
	cp $(POPMABOSS_LIB) ../lib
	mkdir -p ../include/
	cp *.h ../include/

install_alib: $(MABOSS_ALIB) $(POPMABOSS_ALIB)
	mkdir -p ../lib
	cp $(MABOSS_ALIB) ../lib
	cp $(POPMABOSS_ALIB) ../lib
	mkdir -p ../include/
	cp *.h ../include/
	
install_static: MaBoSS$(INFIX).o $(MABOSS_OBJS) PopMaBoSS$(INFIX).o $(POPMABOSS_OBJS)
	$(CXX) -o $(MABOSS) MaBoSS$(INFIX).o $(MABOSS_OBJS) $(SBML_LIB_STATIC) $(STATIC_LIBS) -lpthread $(LDL_FLAG) -L.  $(LDFLAGS) $(EXTRA_LDFLAGS)
	$(CXX) -o $(POPMABOSS) PopMaBoSS$(INFIX).o $(POPMABOSS_OBJS) $(SBML_LIB_STATIC) $(STATIC_LIBS) -lpthread $(LDL_FLAG) -L. $(LDFLAGS) $(EXTRA_LDFLAGS)
	mkdir -p ../pub
	cp $(MABOSS) ../pub
	cp $(POPMABOSS) ../pub
