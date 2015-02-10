#!/bin/bash
#PBS -N wofs_2_year
#PBS -P u46
#PBS -q normal
#PBS -l ncpus=1,mem=2GB
#PBS -l wd
#PBS -l other=gdata1

export MODULEPATH=/projects/u46/opt/modules/modulefiles:$MODULEPATH

module unload python
module load python/2.7.6
module load psycopg2
module load gdal
module load luigi-mpi
module load enum34

export PYTHONPATH=$HOME/source/agdc-api/api-examples/source/main/python:$HOME/source/agdc-api/api/source/main/python:$PYTHONPATH

#module unload python
#module load agdc-api

COMMAND="python $HOME/source/agdc-api/api-examples/source/main/python/wofs_multi_year.py"

# MPI
#mpirun -n 8 $COMMAND

# NO MPI
$COMMAND --local-scheduler
