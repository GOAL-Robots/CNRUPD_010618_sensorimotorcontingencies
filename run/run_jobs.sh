#!/bin/bash

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

