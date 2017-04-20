#!/bin/bash


# CLEAR NON RUNNING SIMULATIOMS

RUN_DIR=$(echo $0 | sed -e"s/\/$(basename $0)$//")

$RUN_DIR/clear_simulations.sh
[ $? -ne 0 ] && echo "no further simulation started." && exit 0


SCREEN=/usr/bin/screen

LOG_DIR=${HOME}/.sensorimotor
[ ! -d $LOG_DIR ] && mkdir $LOG_DIR
clear_simulations &> ${LOG_DIR}/clear_log

# control for the existence of a correct session
screen_session="$(${SCREEN} -ls | grep "\<sm\>"| awk '{print $1}')"
[ -z "${screen_session}" ] && eval "${SCREEN} -dmS sm"

# run command in the current window
eval "${SCREEN} -S sm \
    -X stuff \"cd ${LOG_DIR}\n\""
eval "${SCREEN} -S sm \
    -X stuff \"${RUN_DIR}/g5k.sh &> ${LOG_DIR}/log_jobs\n\""

