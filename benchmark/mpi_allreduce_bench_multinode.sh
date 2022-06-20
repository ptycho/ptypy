#!/bin/bash

workdir=~/work/ptypy
export PYTHONPATH=$workdir
cd $workdir/benchmark

numproc=$1

HOSTLIST=$(cat ${PE_HOSTFILE} | awk '{print $1}' | tr "\n" ",") # comma separated list of hosts
# echo "THE HOSTLIST IS $HOSTLIST"
# NUMCORES=$(cat ${PE_HOSTFILE} | awk 'NR==1{print $2}') # to be overridden soon, but is just the number of cores per host
NUMCORES=4
# echo "THE number of cores per node is $NUMCORES" 
HOST_LIST_WITH_CORES=${HOSTLIST//,/:$NUMCORES,} # puts in the number of cores where the comma would be
HOST_LIST_WITH_CORES=${HOST_LIST_WITH_CORES%?} # gets rid of the trailing comma
# echo "THE HOST LIST WITH CORES IS $HOST_LIST_WITH_CORES" 

mpirun -np $numproc --host ${HOST_LIST_WITH_CORES} python mpi_allreduce_speed.py