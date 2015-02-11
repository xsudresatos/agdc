#!/bin/bash
#PBS -N mosaic
#PBS -P u46
#PBS -q normal
#PBS -l ncpus=8,mem=16GB
#PBS -l wd
#PBS -l other=gdata1

export MODULEPATH=/projects/u46/opt/modules/modulefiles:$MODULEPATH

module unload python
module load python/2.7.6
module load psycopg2
module load gdal
module load luigi-mpi
module load enum34

export PYTHONPATH=$HOME/source/agdc-api/api-examples/source/main/python:$HOME/source/agdc-api/api/source/main/python:$HOME/tmp/enum34-1.0-py2.7.egg:$PYTHONPATH

#module unload python
#module load agdc-api

COMMAND="python $HOME/source/agdc-api/api-examples/source/main/python/landsat_mosaic.py --output-dir $outputdir --x-min $xmin --x-max $xmax --y-min $ymin --y-max $ymax --acq-min $acqmin --acq-max $acqmax"
[ -n "${satellites}" ] && COMMAND="${COMMAND}  --satellite $satellites"
[ "${mask_pqa_apply}" != "false" ] && COMMAND="${COMMAND} --mask-pqa-apply"
[ -n "${mask_pqa_mask}" ] && COMMAND="${COMMAND}  --mask-pqa-mask $mask_pqa_mask"

# MPI
mpirun -n 8 $COMMAND

# NO MPI
#$COMMAND --local-scheduler
