#!/bin/bash

# ===============================================================================
# Copyright (c)  2014 Geoscience Australia
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither Geoscience Australia nor the names of its contributors may be
#       used to endorse or promote products derived from this software
#       without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ===============================================================================

PBS_SCRIPT="$HOME/source/agdc-api/api-examples/source/main/python/wofs_multi_year.pbs.sh"

#qsub -v outputdir="/g/data/u46/sjo/wofs_2_year/output",xmin=110,xmax=119,ymin=-45,ymax=-10,acqmin=1987,acqmax=2015,satellites=LS8,mask_pqa_mask="PQ_MASK_SATURATION PQ_MASK_CONTIGUITY PQ_MASK_CLOUD" "${PBS_SCRIPT}"
#qsub -v outputdir="/g/data/u46/sjo/wofs_2_year/output",xmin=120,xmax=129,ymin=-45,ymax=-10,acqmin=1987,acqmax=2015,satellites=LS8,mask_pqa_mask="PQ_MASK_SATURATION PQ_MASK_CONTIGUITY PQ_MASK_CLOUD" "${PBS_SCRIPT}"
#qsub -v outputdir="/g/data/u46/sjo/wofs_2_year/output",xmin=130,xmax=139,ymin=-45,ymax=-10,acqmin=1987,acqmax=2015,satellites=LS8,mask_pqa_mask="PQ_MASK_SATURATION PQ_MASK_CONTIGUITY PQ_MASK_CLOUD" "${PBS_SCRIPT}"
#qsub -v outputdir="/g/data/u46/sjo/wofs_2_year/output",xmin=140,xmax=149,ymin=-45,ymax=-10,acqmin=1987,acqmax=2015,satellites=LS8,mask_pqa_mask="PQ_MASK_SATURATION PQ_MASK_CONTIGUITY PQ_MASK_CLOUD" "${PBS_SCRIPT}"
#qsub -v outputdir="/g/data/u46/sjo/wofs_2_year/output",xmin=150,xmax=155,ymin=-45,ymax=-10,acqmin=1987,acqmax=2015,satellites=LS8,mask_pqa_mask="PQ_MASK_SATURATION PQ_MASK_CONTIGUITY PQ_MASK_CLOUD" "${PBS_SCRIPT}"

qsub -v xmin=120,xmax=125,ymin=-20,ymax=-20,yearmin=2005,yearmax=2006,input="/g/data/u46/sjo/tmp/wofs_2_year/input",output="/g/data/u46/sjo/tmp/wofs_2_year/output" "${PBS_SCRIPT}"
