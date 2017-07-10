#!/bin/bash

GRID_INFO=${HOME}/.grid_deploy/info


# CLEAR NON RUNNING SIMULATIOMS
echo "clear non-running simulations"

running_jobs=
[ ! -z "$(cat $GRID_INFO)" ] && \
    running_jobs="$(cat $GRID_INFO|grep sensorimotor|awk '{print $2}')"

if [ -z $running_jobs ]; then

    for job in $(ls ${HOME}/.sensorimotor/ | grep "[0-9]\+"); do
        echo "cleared job $job"
        rm -fr $job
    done
    
else

    for job in $(ls ${HOME}/.sensorimotor/ | grep "[0-9]\+"); do
        if [ -z "$(echo "$running_jobs" | grep "\<$(basename $job)\>")" ]; then
            echo "cleared job $job"
            rm -fr $job
        fi
    done

    echo "simulation already running!"
    exit 1
fi

