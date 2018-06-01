#!/bin/bash

set -e
set -o pipefail
IFS=$'\n'

SRC_DIR=${HOME}/Projects/sensorimotor-dev
mkdir -p ${HOME}/working


run_cmd()
{

    local wdir=$1

    screen -dmS $(basename $wdir)
    screen -S $(basename $wdir) -X stuff "cd ${HOME}/working/$wdir\n"
    screen -S $(basename $wdir) -X stuff "${SRC_DIR}/run/g5k_batteries.sh -t 10000 -n 100 -b -P params\n"

}



# START SIMULATIONS
for i in $(seq 0 $(( ${#params[@]} - 1 ))); do
    for ((individual = 0; individual < INDIVIDUALS; individual++)); do
        wdir=${dirs[$i]}_${i}_${individual}

        echo "storing in ${HOME}/working/$wdir"
        echo

        mkdir -p ${HOME}/working/$wdir
        echo "${params[i]}" > ${HOME}/working/${wdir}/params

        run_cmd $wdir

        individual=$((individual+1))
    done
done

