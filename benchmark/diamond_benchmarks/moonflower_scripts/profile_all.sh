#!/bin/bash

# exit on errors
set -e   

# Find folder of benchmarks
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $SCRIPTDIR/../../..  # ptypy folder

# scripts to run + output folder
scripts="i08 i13 i14_1 i14_2"
profdir=/dls/tmp/${USER}/nvprof


mkdir -p ${profdir}


# run all scripts
for script in $scripts
do
    rm -f ${profdir}/${script}.*.nvprof
    mpirun -np 4 \
       nvprof -f -o ${profdir}/${script}.%q{OMPI_COMM_WORLD_RANK}.nvprof \
       python benchmark/diamond_benchmarks/moonflower_scripts/${script}.py \
       2>&1 | tee ${profdir}/${script}.log
done

# Output summary
for script in $scripts
do
    totaltime=$(awk '$0 ~ /Elapsed Compute Time:/ {print $4}' ${profdir}/${script}.log)
    echo $script Time: $totaltime
done
