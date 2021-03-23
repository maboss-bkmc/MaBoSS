/*
#############################################################################
#                                                                           #
# BSD 3-Clause License (see https://opensource.org/licenses/BSD-3-Clause)   #
#                                                                           #
# Copyright (c) 2011-2020 Institut Curie, 26 rue d'Ulm, Paris, France       #
# All rights reserved.                                                      #
#                                                                           #
# Redistribution and use in source and binary forms, with or without        #
# modification, are permitted provided that the following conditions are    #
# met:                                                                      #
#                                                                           #
# 1. Redistributions of source code must retain the above copyright notice, #
# this list of conditions and the following disclaimer.                     #
#                                                                           #
# 2. Redistributions in binary form must reproduce the above copyright      #
# notice, this list of conditions and the following disclaimer in the       #
# documentation and/or other materials provided with the distribution.      #
#                                                                           #
# 3. Neither the name of the copyright holder nor the names of its          #
# contributors may be used to endorse or promote products derived from this #
# software without specific prior written permission.                       #
#                                                                           #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED #
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A           #
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER #
# OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,  #
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,       #
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR        #
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    #
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      #
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              #
#                                                                           #
#############################################################################

   Module:
     MetaEngine.cc

   Authors:
     Eric Viara <viara@sysra.com>
     Gautier Stoll <gautier.stoll@curie.fr>
     Vincent Noël <vincent.noel@curie.fr>
 
   Date:
     January-March 2011
*/

#include "MetaEngine.h"
#include "Probe.h"
#include "Utils.h"
#ifndef WINDOWS
  #include <dlfcn.h>
#else
  #include <windows.h>
#endif

static const char* MABOSS_USER_FUNC_INIT = "maboss_user_func_init";

void MetaEngine::init()
{
  extern void builtin_functions_init();
  builtin_functions_init();
}

void MetaEngine::loadUserFuncs(const char* module)
{
  init();

#ifndef WINDOWS
  void* dl = dlopen(module, RTLD_LAZY);
#else
  void* dl = LoadLibrary(module);
#endif

  if (NULL == dl) {
#ifndef WINDOWS    
    std::cerr << dlerror() << std::endl;
#else
    std::cerr << GetLastError() << std::endl;
#endif
    exit(1);
  }

#ifndef WINDOWS
  void* sym = dlsym(dl, MABOSS_USER_FUNC_INIT);
#else
  typedef void (__cdecl *MYPROC)(std::map<std::string, Function*>*);
  MYPROC sym = (MYPROC) GetProcAddress((HINSTANCE) dl, MABOSS_USER_FUNC_INIT);
#endif

  if (sym == NULL) {
    std::cerr << "symbol " << MABOSS_USER_FUNC_INIT << "() not found in user func module: " << module << "\n";
    exit(1);
  }
  typedef void (*init_t)(std::map<std::string, Function*>*);
  init_t init_fun = (init_t)sym;
  init_fun(Function::getFuncMap());
}

// const std::map<unsigned int, std::pair<NetworkState, double> > MetaEngine::getFixPointsDists() const {
  
//   std::map<unsigned int, std::pair<NetworkState, double> > res;
//   if (0 == fixpoints.size()) {
//     return res;
//   }

//   STATE_MAP<NetworkState_Impl, unsigned int>::const_iterator begin = fixpoints.begin();
//   STATE_MAP<NetworkState_Impl, unsigned int>::const_iterator end = fixpoints.end();
  
//   for (unsigned int nn = 0; begin != end; ++nn) {
//     const NetworkState& network_state = (*begin).first;
//     res[nn] = std::make_pair(network_state,(double) (*begin).second / sample_count);
//     ++begin;
//   }
//   return res;
// }

// void MetaEngine::displayFixpoints(std::ostream& output_fp, bool hexfloat) const 
// {
//   output_fp << "Fixed Points (" << fixpoints.size() << ")\n";
//   if (0 == fixpoints.size()) {
//     return;
//   }

// #ifdef HAS_STD_HEXFLOAT
//   if (hexfloat) {
//     output_fp << std::hexfloat;
//   }
// #endif

//   STATE_MAP<NetworkState_Impl, unsigned int>::const_iterator begin = fixpoints.begin();
//   STATE_MAP<NetworkState_Impl, unsigned int>::const_iterator end = fixpoints.end();
  
//   output_fp << "FP\tProba\tState\t";
//   network->displayHeader(output_fp);
//   for (unsigned int nn = 0; begin != end; ++nn) {
//     const NetworkState& network_state = (*begin).first;
//     output_fp << "#" << (nn+1) << "\t";
//     if (hexfloat) {
//       output_fp << fmthexdouble((double)(*begin).second / sample_count) <<  "\t";
//     } else {
//       output_fp << ((double)(*begin).second / sample_count) <<  "\t";
//     }
//     network_state.displayOneLine(output_fp, network);
//     output_fp << '\t';
//     network_state.display(output_fp, network);
//     ++begin;
//   }
// }

// void MetaEngine::displayFixpoints(FixedPointDisplayer* displayer) const 
// {
//   displayer->begin(fixpoints.size());
//   /*
//   output_fp << "Fixed Points (" << fixpoints.size() << ")\n";
//   if (0 == fixpoints.size()) {
//     return;
//   }
//   */

//   STATE_MAP<NetworkState_Impl, unsigned int>::const_iterator begin = fixpoints.begin();
//   STATE_MAP<NetworkState_Impl, unsigned int>::const_iterator end = fixpoints.end();
  
//   //output_fp << "FP\tProba\tState\t";
//   //network->displayHeader(output_fp);
//   for (unsigned int nn = 0; begin != end; ++nn) {
//     const NetworkState& network_state = begin->first;
//     displayer->displayFixedPoint(nn+1, network_state, begin->second, sample_count);
//     /*
//     output_fp << "#" << (nn+1) << "\t";
//     if (hexfloat) {
//       output_fp << fmthexdouble((double)begin->second / sample_count) <<  "\t";
//     } else {
//       output_fp << ((double)begin->second / sample_count) <<  "\t";
//     }
//     network_state.displayOneLine(output_fp, network);
//     output_fp << '\t';
//     network_state.display(output_fp, network);
//     */
//     ++begin;
//   }
//   displayer->end();
// }