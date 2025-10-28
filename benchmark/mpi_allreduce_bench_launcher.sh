#!/bin/bash

for p in {2..32}
do 
    qsub -pe openmpi ${p}0 -l exclusive,gpu=4,gpu_arch=pascal \
      -P ptychography -o mpi.${p}.out -j y mpi_allreduce_bench_multinode.sh ${p} ; 
done

