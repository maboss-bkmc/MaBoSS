# MaBoSS : Markovian Boolean Stochastic Simulator 
[![Build Status](https://travis-ci.org/sysbio-curie/MaBoSS-env-2.0.svg?branch=master)](https://travis-ci.org/sysbio-curie/MaBoSS-env-2.0) [![Anaconda-Server Badge](https://anaconda.org/sysbio-curie/maboss/badges/version.svg)](https://anaconda.org/sysbio-curie/maboss) [![PyPI version](https://badge.fury.io/py/cmaboss.svg)](https://badge.fury.io/py/cmaboss) <img align="right" height="100" src="https://maboss.curie.fr/images/maboss_logo.jpg">


MaBoSS is a C++ software for simulating continuous/discrete time Markov processes, applied on a Boolean network.

MaBoSS uses a specific language for associating transition rates to each node. Given some initial conditions, MaBoSS applies Monte-Carlo kinetic algorithm (or Gillespie algorithm) to the network to produce time trajectories. Time evolution of probabilities are estimated. In addition, global and semi-global characterizations of the whole system are computed. 

### Contact
Institut Curie 

26 rue d'Ulm 75248 PARIS CEDEX 05 

Contact: [maboss.bkmc@gmail.com](mailto://maboss.bkmc@gmail.com) 

Web Site: [https://maboss.curie.fr](https://maboss.curie.fr)

### Package Contents
MaBoSS-env-2.0 is composed of:
- MaBoSS engine 2.0: C++ core program simulating continuous/discrete time Markov processes, applied on a Boolean network.
- MaBoSS tools 2.0: perl and python scripts using MaBoSS engine 2.0
- cMaBoSS: python bindings using Python C API.

### Tested platforms
- Linux: Ubuntu (Ubuntu 4.3.2-1ubuntu11 and higher), RedHat and CentOS
- MacOS X x86
- Windows with cygwin

### Requirements
##### MaBoSS engine 2.0:

- gcc: version 4.0.1 or higher
- bison: version 2.3 or higher
- flex: version 2.5.35 or higher
- cygwin is needed on Windows

##### MaBoSS tools:

- perl
- python3
- python3 modules: matplotlib (matplotlib.cm, matplotlib.gridspec, matplotlib.patches, matplotlib.pylab), numpy, pandas, seaborn, xlsxwriter

To check requirements on a Unix platform (Linux, MacOS X), you can launch the script check-requirements.sh as follows:
./check-requirements

MaBoSS engine requirements are checked first.

The output must be:

Checking MaBoSS engine 2.0 requirements...

  flex: OK
  bison: OK
  gcc: OK
  g++: OK

MaBoSS engine 2.0 requirements: OK

If an error is displayed, you have to fix it, as neither the engine, nor the tools will be able to be launched.

MaBoSS tools requirements are then checked.

The most frequent errors are that the following python3 modules are missing: matplotlib, numpy, pandas, seaborn, xlsxwriter.

You must install the missing modules if you want to use: MBSSf_DrugSim.py, MBSS_PieChart.py, MBSS_PrepareProjectFilePieChart.py, MBSS_PrepareProjectFileTrajectoryFig.py or MBSS_TrajectoryFig.py.

### Binary Distribution

To avoid installing compilation tools, we provide binary versions for linux x86, MacOS X x86 and Windows x86:
- linux   : binaries/linux-x86/MaBoSS
- MacOS X : binaries/macos-x86/MaBoSS
- Windows : binaries/win-x86/MaBoSS.exe

Important notes on the Windows version:
- to execute MaBoSS.exe, cygwin must be installed (http://www.cygwin.com/)
- because of the cygwin emulation, the windows version is very slow (about 4 times slower than the linux and Mac OS versions). We urge you to run MaBoSS on linux or Mac OS X if possible.

All these binary versions are provided "as is", they may not work on your OS. In such a case, you need to compile MaBoSS.

If you want to use the binary version:

  1. your have first to test it:
  launch ./binaries/YOUR_OS/MaBoSS --version, for instance, on a Linux OS:
  ./binaries/linux-x86/MaBoSS --version
  if everything is ok, you should see:
  MaBoSS version 2.0 [networks up to 64 nodes]

  2. copy this binary to engine/pub, for instance, on a Linux OS:
  cp binaries/linux-x86/MaBoSS engine/pub/

  3. then, you can skip the following section "Engine Compilation"

### Engine Compilation

cd engine/src
make install

The executable file will be located in engine/pub and is named MaBoSS.

This compiled version supports up to 64 nodes per network.

If you need more nodes per network, you have to add an extra hint to the compilation command, for instance to compile a version supporting up to 100 nodes:
make MAXNODES=100 install

The executable file will also be located in engine/pub and will be named MaBoSS_100n.

Notes:
- if you manage only networks with up to 64 nodes, we recommend you to use the default compiled version as for networks with more than 64 nodes, the implementation is very different and will be slower.
- generally speaking, a version compiled with a given number of nodes will be slower and will use more memory than a version compiled with a lesser number of nodes.

### MaBoSS Engine Usage

cd engine/pub

./MaBoSS --version
MaBoSS version 2.0 [networks up to 64 nodes]

./MaBoSS_100n --version
MaBoSS version 2.0 [networks up to 100 nodes]

The usage is described in the reference card doc/MaBoSS-RefCard.pdf

### Environment

Go to the directory of MaBoSS-env-2.0 and perform:
source MaBoSS.env

Then, the MaBoSS engine and all the tools will be accessible from your environment:

Type MBSS_ followed by a Tab, this should be displayed:
MBSS_DrugSim.py                          MBSS_MutBndCfg.pl                        MBSS_PrepareProjectFile.sh
MBSS_FormatTable.pl                      MBSS_MutBnd.pl                           MBSS_PrepareProjectFileTrajectoryFig.py
MBSS_InitCondFromTrajectory.pl           MBSS_PieChart.py                         MBSS_SensitivityAnalysis.pl
MBSS_MultipleSim.py                      MBSS_PrepareProjectFilePieChart.py       MBSS_TrajectoryFig.py

Type MaB followed by a Tab, the MaBoSS engine program should be displayed:
MaBoSS

### Examples

The directory examples contains two examples:
- ToyModel
- p53_Mdm2

To test an example (for instance p53_Mdm2):
- change to the directory containing the model (examples/p53_Mdm2)
- ../../pub/MaBoSS -c p53_Mdm2_runcfg.cfg -o p53_Mdm2_out p53_Mdm2.bnd
- the files p53_Mdm2_out_probtraj.csv and p53_Mdm2_out_statdist.csv will be created.
The description of these files can be found at https://maboss.curie.fr/pub/DescriptionOutputFile.pdf

### Tutorials

The directory tutorial contains two tutorials (in pdf) based on an example. In order to follow a tutorial, the necessary files are provided

### License

The BSD 3-Clause License

### Copyright

BSD 3-Clause License (see https://opensource.org/licenses/BSD-3-Clause)  
                                                                         
Copyright (c) 2011-2020 Institut Curie, 26 rue d'Ulm, Paris, France      
All rights reserved.                                                     
                                                                         
Redistribution and use in source and binary forms, with or without       
modification, are permitted provided that the following conditions are   
met:                                                                     
                                                                         
1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.                    
                                                                         
2. Redistributions in binary form must reproduce the above copyright     
notice, this list of conditions and the following disclaimer in the      
documentation and/or other materials provided with the distribution.     
                                                                         
3. Neither the name of the copyright holder nor the names of its         
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.                      
                                                                         
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS      
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A          
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,      
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR       
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF   
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING     
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS       
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.             