#!/bin/bash

SCREEN=/usr/bin/screen

LOG_DIR=${HOME}/.sensorimotor
[ ! -d $LOG_DIR ] && mkdir $LOG_DIR

# control for the existence of a correct session
screen_session="$(${SCREEN} -ls | grep "\<sm\>"| awk '{print $1}')"
[ -z "${screen_session}" ] && eval "${SCREEN} -dmS sm"

# run command in the current window
eval "${SCREEN} -S sm \
    -X stuff \"cd ${LOG_DIR}\n\""
eval "${SCREEN} -S sm \
    -X stuff \"/home/fmannella/working/sensorimotor-development/run/g5k.sh &> ${LOG_DIR}/log_jobs\n\""

