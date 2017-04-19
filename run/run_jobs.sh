#!/bin/bash


# CLEAR NON RUNNING SIMULATIOMS
clear_simulations()
{
    echo "clear non-running simulations"
    running_jobs=
    [ ! -z "$(ls -a ${HOME} | grep "\.G_" )" ] &&
        running_jobs="$(basename -a $(ls -a ${HOME}| grep "\.G_")|sed -e"s/\.G_//")"
    for job in $(ls ${HOME}/.sensorimotor/ | grep "[0-9]\+"); do
        if [ -z "$(echo "$running_jobs" | grep "\<$(basename $job)\>")" ]; then
            rm -fr $job
        else
            for sim_info_dir in ${dirs[@]}; do
                if [ ! -z "$(cat  $job|grep $sim_info_dir)" ]; then 
                    echo "simulation already running!"
                    exit 1
                fi
            done
        fi
    done
}

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
    -X stuff \"/home/fmannella/working/sensorimotor-development/run/g5k.sh &> ${LOG_DIR}/log_jobs\n\""

