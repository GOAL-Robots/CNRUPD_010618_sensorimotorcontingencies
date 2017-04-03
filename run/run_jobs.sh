#!/bin/bash

SCREEN=/usr/bin/screen

# control path variable in screen
[ ! -e ${HOME}/.screenrc ] && echo "setenv PATH $PATH" > ${HOME}/.screenrc

# control for the existence of a correct screen session
screen_session="$(${SCREEN} -ls | grep "\<sm\>"| awk '{print $1}')"
[ -z "${screen_session}" ] && \
   ${SCREEN} -dmS sm

# control the existence of the requested window
screen_window="sm_$(date +"%Y%m%d%H%M")"
declare -a windows_list=($(screen -S ${screen_window} -Q windows |\
    sed -e"s/\(\(\s\+\)*\)\([0-9]\+\s\+\)/\n/g; s/^\n//g"))
[[ ! " ${windows_list[@]} " == *" ${screen_window} "* ]] &&
   ${SCREEN} -S sm -X screen -t ${screen_window}
${SCREEN} -S sm -p ${screen_window}  -X stuff "\n"

# run command in the current window
${SCREEN} -S sm -p ${screen_window} \
    -X stuff "/home/fmannella/working/sensorimotor-development/run/g5k.sh &> ${HOME}/log_jobs\n"

