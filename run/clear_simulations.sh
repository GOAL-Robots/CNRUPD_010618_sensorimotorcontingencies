#!/usr/bin/env bash
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
 
IFS=$'\n'

INFO_FILE=${HOME}/.grid_deploy/info 

LABEL=$1

[[ -z $LABEL ]] && (echo "must give a label" && exit) 

# CLEAR NON RUNNING SIMULATIOMS
echo "clear non-running simulations"
running_jobs="$(cat $INFO_FILE)"
[ ! -z "$running_jobs" ] && \
    running_jobs="$(echo "$running_jobs"| awk '{print $2}' | perl -p -e "s/\n/ /" )"

echo "running_jobs: $running_jobs"
for job in $(ls ${HOME}/.${LABEL}/ | grep "[0-9]\+"); do
    if [ -z "$(echo "$running_jobs" | grep "\<$(basename $job)\>")" ]; then
        echo "cleared job $job"
        rm -fr ${HOME}/.${LABEL}/$job
    fi
done

if [ ! -z "$(cat $INFO_FILE| grep "\<$LABEL\>")" ]; then
    echo "simulation already running!"
    exit 1
fi

