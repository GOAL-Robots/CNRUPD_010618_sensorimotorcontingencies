#!/bin/bash

set -e
set -o pipefail
IFS=$'\n'

source ${HOME}/g5kutils/setwalltime.sh
${HOME}/g5kutils/clear.sh
[ ! -d ${HOME}/.sensorimotor_data ] && mkdir ${HOME}/.sensorimotor_data

LABEL=sensorimotor_data
GRID_INFO=${HOME}/.grid_deploy/info
N_MACHINES=1
MIN_CORES=4
MIN_RAM=16
WALLTIME=$(max_walltime)

# FIND RESOURCES 
rm -fr log_resources
${HOME}/g5kutils/autodeploy.sh -c $MIN_CORES -n $N_MACHINES -r $MIN_RAM -w $WALLTIME -l $LABEL 

# exit if no resources 
[ -z $(cat $GRID_INFO|grep $LABEL| grep deploy) ] && echo "no deployment made." && exit 1 

# get job_id
JOB_ID=$(cat $GRID_INFO|grep $LABEL|awk '{print $2}')

# save it in log dir
echo -n ''> ${HOME}/.sensorimotor_data/$JOB_ID

# set array of node names
declare -a nodes=($(cat ~/.G_${JOB_ID}/NODES|uniq))

# save simulation status in log dir 
echo "${dirs[@]}" > ${HOME}/.sensorimotor_data/$JOB_ID 


