#!/bin/bash

set -e
IFS=$'\n'

source ${HOME}/g5kutils/setwalltime.sh
${HOME}/g5kutils/clear.sh
[ ! -d ${HOME}/.sensorimotor ] && mkdir ${HOME}/.sensorimotor

N_MACHINES=2
MIN_CORES=30
MIN_RAM=100
WALLTIME=$(max_walltime)
declare -a dirs=(sm_singleecho_25g)
declare -a params

params[0]=$(cat<<HERE_PARAMS
simulation_competence_improvement_prop = 1.0
simulation_incompetence_prop = 0.0
gs_n_echo_units = 800
gs_multiple_echo = False
GOAL_NUMBER = 25
HERE_PARAMS
)

# CLEAR NON RUNNING SIMULATIOMS
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

# FIND RESOURCES 
cluster=$(${HOME}/g5kutils/select_res.sh -c $MIN_CORES -n $N_MACHINES -r $MIN_RAM -t $WALLTIME | grep resource: | tail -n 1 | awk '{print $3}')

[ -z "$cluster" ] && echo "no reservation avaliable" && exit 1

echo
echo "deployng $N_MACHINES machines on $cluster:"
echo
echo

${HOME}/g5kutils/deploy.sh -t $WALLTIME -u $cluster -n $N_MACHINES -f | tee log_deploy 2>&1

JOB_ID="$(cat log_deploy|grep JOB_ID| sed  -e"s/.*:\s\+\([0-9]\+\)\s*$/\1/")"

if [ -z "$JOB_ID" ]; then
    echo "no reservation avaliable"
    exit 1
fi

declare -a nodes=($(cat ~/.G_${JOB_ID}/NODES|uniq))

# SAVE SIMULATION STATUS IN INFO DIR 
echo "${dirs[@]}" > ${HOME}/.sensorimotor/$JOB_ID 

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

#-HERE---------------------------------------------------------------------
cat << EOS > ${HOME}/working/${wdir}/run
cd ~/working/$wdir
\${HOME}/working/sensorimotor-development/run/g5k_batteries.sh -t 50000 \\
-n 1000 -b -P params
EOS
#-HERE-END-----------------------------------------------------------------

        chmod +x ${HOME}/working/${wdir}/run

#-HERE---------------------------------------------------------------------
CMD=$(cat<<EOS
[ -z "\$(mount | grep working )" ] && \
    (sshfs -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 \
    rennes:\${HOME}/working \${HOME}/working)

screen -dmS $(basename $wdir)
screen -S $(basename $wdir) -X exec bash -c "\${HOME}/working/$wdir/run; bash"
EOS
)
#-HERE-END-----------------------------------------------------------------o

        ssh $node "$CMD"
        individual=$((individual+1))

    done
done
