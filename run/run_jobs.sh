#!/bin/bash


# control for the existence of a correct screen session
screen_session="$(${SCREEN} -ls | grep "\<sm\>"| awk '{print $1}')"
[ -z "${screen_session}" ] && \
   ${SCREEN} -dmS sm

# clean previous windows
WINS="$(screen -S sm -Q windows |sed -e"s/\s\+\([0-9]\)/\n\1/g")"


for win_id in $(echo "$WINS"| sed -e"s/^\([0-9]\+\)\s\+.*/\1/"); do
    echo $win_id
    eval "screen -S sm -p $win_id -X kill"
done


screen_window="sm_$(date +"%Y%m%d%H%M")"

screen -S sm -p ${screen_window}  -X screen

# run command in the current window
screen -S sm -p ${screen_window} \
    -X stuff "/home/fmannella/working/sensorimotor-development/run/g5k.sh &> ${HOME}/log_jobs\n"

