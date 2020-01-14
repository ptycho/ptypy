#!/bin/bash

# Runs MPI all reduce benchmark with variable number of processes

echo "Processes,i08,i13,i14_1,i14_2"
for p in {2..16}
do
    echo -n $p
    mpirun -np $p python mpi_allreduce_speed.py | \
     awk -F, '$0 ~ /^i[0-9]/ {printf(",%s", $2)} END {print ""}'
done