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
declare -a dirs=(sm_singleecho_25g)
#declare -a dirs=(test)
declare -a params

params[0]=$(cat<<HERE_PARAMS
gs_n_echo_units = 800
gs_multiple_echo = False
GOAL_NUMBER = 25
HERE_PARAMS
)

# FIND RESOURCES 
rm -fr log_resources
${HOME}/g5kutils/autodeploy.sh -c $MIN_CORES -n $N_MACHINES -r $MIN_RAM -w $WALLTIME -l $LABEL 

# exit if no resources 
[ -z $(cat $GRID_INFO|grep $LABEL| grep deploy) ] && echo "no deployment made." && exit 1 

# get job_id
JOB_ID=$(cat $GRID_INFO|grep $LABEL|awk '{print $2}')

# save it in log dir
echo -n ''> ${HOME}/.sensorimotor/$JOB_ID

# set array of node names
declare -a nodes=($(cat ~/.G_${JOB_ID}/NODES|uniq))

# save simulation status in log dir 
echo "${dirs[@]}" > ${HOME}/.sensorimotor/$JOB_ID 

run_cmd()
{

wdir=$1
node=$2

# make run file and store it within the working dir 
cat << EOS > ${HOME}/working/${wdir}/run
cd ~/working/$wdir
\${HOME}/working/sensorimotor-development/run/g5k_batteries.sh -t 50000 -n 1000 -b -P params
EOS

chmod +x ${HOME}/working/${wdir}/run

# build the command to run in the machine
CMD=$(cat<<EOS
[ -z "\$(mount | grep working )" ] && \
    (sshfs -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 \
    rennes:\${HOME}/working \${HOME}/working)

screen -dmS $(basename $wdir)
screen -S $(basename $wdir) -X exec bash -c "\${HOME}/working/$wdir/run; bash"
EOS
)

# execute the command within the machine
ssh $node "$CMD"

}



# START SIMULATIONS
for i in $(seq 0 0); do
    individual=1
    for node in ${nodes[@]}; do
        wdir=${dirs[$i]}_$individual

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
