#!/bin/bash
##
## Copyright (c) 2018 Francesco Mannella. 
## 
## This file is part of sensorimotor-contingencies
## (see https://github.com/GOAL-Robots/CNRUPD_010618_sensorimotorcontingencies).
## 
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to deal
## in the Software without restriction, including without limitation the rights
## to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
## copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
## 
## The above copyright notice and this permission notice shall be included in all
## copies or substantial portions of the Software.
## 
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
## SOFTWARE.
##

set -e
set -o pipefail
IFS=$'\n'
current_host="$(hostname | sed -e"s/\..*//; s/\<f\(.*\)/\1/")"

source ${HOME}/g5kutils/setwalltime.sh
${HOME}/g5kutils/clear.sh
[ ! -d ${HOME}/.sensorimotor ] && mkdir ${HOME}/.sensorimotor

LABEL=sensorimotor
GRID_INFO=${HOME}/.grid_deploy/info
N_MACHINES=4
MIN_CORES=4
MIN_RAM=20
WALLTIME=$(max_walltime)


# exit if no resources 
[[ ! -z $(cat $GRID_INFO | grep " $LABEL ") ]] && \
    echo "other simulatino in deployment, stoppping" && exit 1 

declare -a dirs=(sm_singleecho_25g_p1 sm_singleecho_25g_p2)
declare -a params
params[0]="
GOAL_NUMBER = 25
body_simulator_num_touch_sensors = 30
body_simulator_substep_min_angle = 0.5 
body_simulator_substeps = 10
body_simulator_touch_epsilon = 0.05
body_simulator_touch_grow = False
body_simulator_touch_sigma = 0.06
body_simulator_touch_th = 0.1
gm_goalrep_lr = 0.25
gm_single_kohonen = True
gm_single_kohonen_neigh_bl = 0.01
gm_single_kohonen_neigh_scale = 0.99
gp_eta = 0.25
gs_eta = 4.0
gs_eta_decay = True
gs_goal_learn_start = 10
gs_goal_window = 100
gs_match_decay = 0.9
gs_multiple_echo = False
gs_n_echo_units = 200
gs_reset_window = 10
gs_sm_temp = 0.01
simulation_im_decay = 0.2
HERE_PARAMS
"
params[1]="${params[0]}"

# FIND RESOURCES 
rm -fr log_resources
${HOME}/g5kutils/autodeploy.sh -c $MIN_CORES -n $N_MACHINES -r $MIN_RAM -w $WALLTIME -l $LABEL 

# exit if no resources 
[[ -z $(cat $GRID_INFO|grep " $LABEL " | grep deploy) ]] && echo "no deployment made." && exit 1 

# get job_id
JOB_ID=$(cat $GRID_INFO|grep  " $LABEL " | awk '{print $2}')

# save it in log dir
echo -n ''> ${HOME}/.sensorimotor/$JOB_ID

# set array of node names
declare -a nodes=($(cat ~/.G_${JOB_ID}/NODES|uniq))

# save simulation status in log dir 
echo "making ${dirs[@]}" > ${HOME}/.sensorimotor/$JOB_ID 

run_cmd()
{

    wdir=$1
    node=$2

    # make run file and store it within the working dir 

    run="
    cd ~/working/$wdir
    \${HOME}/working/sensorimotor-development/run/g5k_batteries.sh -t 10000 -n 100 -b -P params
    "

    echo "$run" > ${HOME}/working/${wdir}/run

    chmod +x ${HOME}/working/${wdir}/run

    # build the command to run in the machine
    CMD="
    [ -z \"\$(mount | grep working )\" ] && \
        (sshfs -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 \
        $current_host:\${HOME}/working \${HOME}/working)

    screen -dmS $(basename $wdir)
    screen -S $(basename $wdir) -X exec bash -c \"\${HOME}/working/$wdir/run; bash\"
    "

    # execute the command within the machine
    ssh $node "$CMD"

}



# START SIMULATIONS
for i in $(seq 0 $(( ${#params[@]} - 1 ))); do
    individual=1
    for node in ${nodes[@]}; do
        wdir=${dirs[$i]}_${i}_${individual}

        echo "using node $node"
        echo "storing in ${HOME}/working/$wdir"
        echo

        if [ ! -d ${HOME}/working/$wdir ]; then
            mkdir ${HOME}/working/$wdir
        fi

        echo "${params[i]}" > ${HOME}/working/${wdir}/params


        run_cmd $wdir $node


        individual=$((individual+1))

    done
done
