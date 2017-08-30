#!/bin/bash

set -e
set -o pipefail
IFS=$'\n'

source ${HOME}/g5kutils/setwalltime.sh
${HOME}/g5kutils/clear.sh
[ ! -d ${HOME}/.sensorimotor ] && mkdir ${HOME}/.sensorimotor

LABEL=sensorimotor
GRID_INFO=${HOME}/.grid_deploy/info
N_MACHINES=4
MIN_CORES=4
MIN_RAM=40
WALLTIME=$(max_walltime)

declare -a dirs=(sm_singleecho_25g_p1 sm_singleecho_25g_p2)
declare -a params
params[0]="
simulation_im_decay = 0.2
body_simulator_touch_sigma = 0.2
body_simulator_num_touch_sensors = 30
gs_eta = 4.0
gs_n_echo_units = 200
gs_match_decay = 0.9
gs_sm_temp = 0.01
gs_goal_window = 100
gs_goal_learn_start = 10
gs_reset_window = 10
gs_multiple_echo = False
gp_eta = 0.35
gm_goalrep_lr = 0.25
gm_single_kohonen = True
gm_single_kohonen_neigh_bl = 0.01
gm_single_kohonen_neigh_scale = 0.99
GOAL_NUMBER = 25
body_simulator_touch_grow = False
HERE_PARAMS
"

params[1]="
simulation_im_decay = 0.2
body_simulator_touch_sigma = 0.2
body_simulator_num_touch_sensors = 30
gs_eta = 4.0
gs_n_echo_units = 200
gs_match_decay = 0.9
gs_sm_temp = 0.01
gs_goal_window = 100
gs_goal_learn_start = 10
gs_reset_window = 10
gs_multiple_echo = False
gp_eta = 0.35
gm_goalrep_lr = 0.25
gm_single_kohonen = True
gm_single_kohonen_neigh_bl = 0.01
gm_single_kohonen_neigh_scale = 0.99
GOAL_NUMBER = 25
body_simulator_touch_grow = True
HERE_PARAMS
"

# FIND RESOURCES 
rm -fr log_resources
${HOME}/g5kutils/autodeploy.sh -c $MIN_CORES -n $N_MACHINES -r $MIN_RAM -w $WALLTIME -l $LABEL 

# exit if no resources 
[[ -z $(cat $GRID_INFO|grep "\<$LABEL\>" | grep deploy) ]] && echo "no deployment made." && exit 1 

# get job_id
JOB_ID=$(cat $GRID_INFO|grep  "\<${LABEL}\>" | awk '{print $2}')

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
    \${HOME}/working/sensorimotor-development/run/g5k_batteries.sh -t 50000 -n 1000 -b -P params
    "

    echo "$run" > ${HOME}/working/${wdir}/run

    chmod +x ${HOME}/working/${wdir}/run

    # build the command to run in the machine
    CMD="
    [ -z \"\$(mount | grep working )\" ] && \
        (sshfs -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 \
        rennes:\${HOME}/working \${HOME}/working)

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
