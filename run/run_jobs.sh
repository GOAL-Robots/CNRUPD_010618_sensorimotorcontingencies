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

# CLEAR NON RUNNING SIMULATIOMS
LABEL=sensorimotor

RUN_DIR=$(echo $0 | sed -e"s/\/$(basename $0)$//")
RUN_DIR=$(realpath $RUN_DIR)

$RUN_DIR/clear_simulations.sh $LABEL
[ $? -ne 0 ] && echo "no further simulation started." && exit 0


SCREEN=/usr/bin/screen

LOG_DIR=${HOME}/.${LABEL}
mkdir -p $LOG_DIR

# control for the existence of a correct session
screen_session="$(${SCREEN} -ls | grep "\<sm\>"| awk '{print $1}')"
[ -z "${screen_session}" ] && eval "${SCREEN} -dmS sm"

# run command in the current window
eval "${SCREEN} -S sm \
    -X stuff \"cd ${LOG_DIR}\n\""
eval "${SCREEN} -S sm \
    -X stuff \"${RUN_DIR}/g5k.sh &> ${LOG_DIR}/log_jobs\n\""

