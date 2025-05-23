
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
     CustomPopProbTrajDisplayer.cc

   Authors:
     Eric Viara <viara@sysra.com>
     Gautier Stoll <gautier.stoll@curie.fr>
     Vincent Noël <vincent.noel@curie.fr>
 
   Date:
     Decembre 2020
*/

#include "CustomPopProbTrajDisplayer.h"
#include "../Utils.h"
#include <iomanip>
#include <cstring>

void CSVCustomPopProbTrajDisplayer::beginDisplay() 
{
  os_probtraj << "Time\tTH" << (this->compute_errors ? "\tErrorTH" : "") << "\tH";
  for (unsigned int jj = 0; jj <= this->refnode_count; ++jj) {
    os_probtraj << "\tHD=" << jj;
  }

  for (unsigned int nn = 0; nn < this->maxcols; ++nn) {
    os_probtraj << "\tState\tProba" << (this->compute_errors ? "\tErrorProba" : "");
  }

  os_probtraj << '\n';
}

void CSVCustomPopProbTrajDisplayer::endTimeTickDisplay() 
{
  os_probtraj << std::setprecision(4) << std::fixed << this->time_tick;
#ifdef HAS_STD_HEXFLOAT
  if (this->hexfloat) {
    os_probtraj << std::hexfloat;
  }
#endif
  if (this->hexfloat) {
    os_probtraj << '\t' << fmthexdouble(this->TH);
    os_probtraj << '\t' << fmthexdouble(this->err_TH);
    os_probtraj << '\t' << fmthexdouble(this->H);
  } else {
    os_probtraj << '\t' << this->TH;
    os_probtraj << '\t' << this->err_TH;
    os_probtraj << '\t' << this->H;
  }

  for (unsigned int nn = 0; nn <= this->refnode_count; nn++) {
    os_probtraj << '\t';
    if (this->hexfloat) {
      os_probtraj << fmthexdouble(this->HD_v[nn]);
    } else {
      os_probtraj << this->HD_v[nn];
    }
  }

  for (const typename ProbTrajDisplayer<PopSize>::Proba &proba : this->proba_v) {
    os_probtraj << '\t';
    proba.state.displayOneLine(os_probtraj, this->network);
    if (this->hexfloat) {
      os_probtraj << '\t' << fmthexdouble(proba.proba);
      os_probtraj << '\t' << fmthexdouble(proba.err_proba);
    } else {
      os_probtraj << '\t' << std::setprecision(6) << proba.proba;
      os_probtraj << '\t' << proba.err_proba;
    }
  }
  os_probtraj << '\n';
}