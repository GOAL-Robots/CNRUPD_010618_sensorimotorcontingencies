#!/bin/bash

source ${HOME}/g5kutils/setwalltime.sh

IFS=$'\n'
N_MACHINES=4
MIN_CORES=20
MIN_RAM=100
WALLTIME=$(max_walltime)

cluster=$(${HOME}/g5kutils/select_res.sh -c $MIN_CORES -n $N_MACHINES -r $MIN_RAM -t $WALLTIME | grep resource: |  head -n 1 | awk '{print $3}')

[ -z "$cluster" ] && echo "no reservation avaliable" && exit 1

echo
echo "deployng $N_MACHINES machines on $cluster:"
echo
echo

${HOME}/g5kutils/deploy.sh -t $WALLTIME -u $cluster -n $N_MACHINES -f

if [ -z "$(ls ~/.G_*)" ]; then
    echo "no reservation avaliable"
    exit 1
fi

declare -a nodes=($(cat ~/.G_*/NODES|uniq))
declare -a dirs=(sm_t0p{50,25,00})

params[0]=$(cat<<HERE_PARAMS
simulation_competence_improvement_prop = 1.0
simulation_incompetence_prop = 0.0
gs_prediction_temperature = 0.0
GOAL_NUMBER = 9
HERE_PARAMS
)

params[1]=$(cat<<HERE_PARAMS
simulation_competence_improvement_prop = 1.0
simulation_incompetence_prop = 0.0
gs_prediction_temperature = 0.25
GOAL_NUMBER = 9
HERE_PARAMS
)


params[1]=$(cat<<HERE_PARAMS
simulation_competence_improvement_prop = 1.0
simulation_incompetence_prop = 0.0
gs_prediction_temperature = 0.5
GOAL_NUMBER = 9
HERE_PARAMS
)

echo $params

for i in $(seq 0 2); do
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
