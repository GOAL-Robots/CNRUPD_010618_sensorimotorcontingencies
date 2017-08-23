#!/usr/bin/env bash
 
IFS=$'\n'

INFO_FILE=${HOME}/.grid_deploy/info 

LABEL=$1

[[ -z $LABEL ]] && (echo "must give a label" && exit) 

# CLEAR NON RUNNING SIMULATIOMS
echo "clear non-running simulations"
running_jobs="$(cat $INFO_FILE)"
[ ! -z "$running_jobs" ] && \
    running_jobs="$(echo "$running_jobs"| awk '{print $2}' | perl -p -e "s/\n/ /" )"

echo "running_jobs: $running_jobs"
for job in $(ls ${HOME}/.${LABEL}/ | grep "[0-9]\+"); do
    if [ -z "$(echo "$running_jobs" | grep "\<$(basename $job)\>")" ]; then
        echo "cleared job $job"
        rm -fr ${HOME}/.${LABEL}/$job
    fi
done

if [ ! -z "$(cat $INFO_FILE| grep "\<$LABEL\>")" ]; then
    echo "simulation already running!"
    exit 1
fi

