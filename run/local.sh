#!/bin/bash
##
## Copyright (c) 2018 Francesco Mannella. 
## 
## This file is part of sensorimotor-contingencies
## (see https://github.com/GOAL-Robots/CNRUPD_010618_sensorimotorcontingencies).
## 
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to deal
## in the Software without restriction, including without limitation the rights
## to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
## copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
## 
## The above copyright notice and this permission notice shall be included in all
## copies or substantial portions of the Software.
## 
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
## SOFTWARE.
##

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

