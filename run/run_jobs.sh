#!/bin/bash

SCREEN=/usr/bin/screen

# control for the existence of a correct session
screen_session="$(${SCREEN} -ls | grep "\<sm\>"| awk '{print $1}')"
[ -z "${screen_session}" ] && eval "${SCREEN} -dmS sm"

# run command in the current window
eval "${SCREEN} -S sm \
    -X stuff \"/home/fmannella/working/sensorimotor-development/run/g5k.sh &> ${HOME}/log_jobs\n\""

