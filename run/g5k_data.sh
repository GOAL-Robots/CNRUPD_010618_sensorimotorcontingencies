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
[[ -z $(cat $GRID_INFO | grep " $LABEL " | grep deploy) ]] && echo "no deployment made." && exit 1 

# get job_id
JOB_ID=$(cat $GRID_INFO | grep " $LABEL " | awk '{print $2}')

# save it in log dir
echo -n ''> ${HOME}/.sensorimotor_data/$JOB_ID

# set array of node names
declare -a nodes=($(cat ~/.G_${JOB_ID}/NODES | uniq))

# save simulation status in log dir 
echo "${dirs[@]}" > ${HOME}/.sensorimotor_data/$JOB_ID 

run_cmd()
{

    wdir=$1
    node=$2

    # make run file and store it within the working dir 
    
    run="

    cd ~/working/$wdir

    sudo service apache2 restart
    pip install pyqtgraph
    mkdir -p \${HOME}/public_html

    cp kill_g.sh  \${HOME}/public_html
    cp make_g.sh  \${HOME}/public_html
    cd \${HOME}/public_html
    ./make_g.sh &> log & 

    "
    echo "$run" > ${HOME}/working/${wdir}/run

    chmod +x ${HOME}/working/${wdir}/run

    # build the command to run in the machine
    CMD="

    [[ -z \"\$(mount | grep working )\" ]] && \
        (sshfs -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 \
        rennes:\${HOME}/working \${HOME}/working)

    screen -dmS $(basename $wdir)
    screen -S $(basename $wdir) -X exec bash -c \"\${HOME}/working/$wdir/run; bash\"
    
    "

    # execute the command within the machine
    ssh $node "$CMD"

}

# START NODES
for node in ${nodes[@]}; do
    wdir=sm_data

    echo "using node $node"
    echo "storing in ${HOME}/working/$wdir"
    echo

    if [ ! -d ${HOME}/working/$wdir ]; then
        mkdir ${HOME}/working/$wdir
    fi

    run_cmd $wdir $node
done
