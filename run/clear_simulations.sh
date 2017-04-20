#!/bin/bash


# CLEAR NON RUNNING SIMULATIOMS
echo "clear non-running simulations"
running_jobs=
[ ! -z "$(ls -a ${HOME} | grep "\.G_" )" ] &&
    running_jobs="$(basename -a $(ls -a ${HOME}| grep "\.G_")|sed -e"s/\.G_//")"
echo "running_jobs: $running_jobs"
for job in $(ls ${HOME}/.sensorimotor/ | grep "[0-9]\+"); do
    if [ -z "$(echo "$running_jobs" | grep "\<$(basename $job)\>")" ]; then
        echo "cleared job $job"
        rm -fr $job
    else
        echo "simulation already running!"
        exit 1
    fi
done

