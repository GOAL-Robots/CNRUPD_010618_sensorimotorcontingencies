#!/bin/bash

IFS=$'\n'
N_MACHINES=2
MIN_CORES=10
MIN_RAM=32
WALLTIME=9:0:0


cluster=$(${HOME}/g5kutils/select_res.sh -c $MIN_CORES -n $N_MACHINES -r $MIN_RAM -t $WALLTIME | grep resource: |  head -n 1 | awk '{print $3}') 

[ -z "$cluster" ] && echo "no reservation avaliable" && exit 1 

echo 
echo "deployng $N_MACHINES machines on $cluster:"
echo
echo

${HOME}/g5kutils/deploy.sh -t 9:0:0 -u $cluster -n $N_MACHINES -f

if [ -z "$(ls ~/.G_*)" ]; then
    echo "no reservation avaliable"
    exit 1
fi

declare -a nodes=($(cat ~/.G_*/NODES|uniq))
declare -a dirs=(sm_c1i0_n sm_c1i0p2_n)

params[0]=$(cat<<HERE_PARAMS
simulation_competence_improvement_prop = 1.0 
simulation_incompetence_prop = 0.0 
GOAL_NUMBER = 9 
HERE_PARAMS
)

params[1]=$(cat<<HERE_PARAMS
simulation_competence_improvement_prop = 1.0 
simulation_incompetence_prop = 0.2 
GOAL_NUMBER = 9 
HERE_PARAMS
)

echo $params

for i in $(seq 0 2); do
   

    node=${nodes[$i]}
    wdir=${dirs[$i]}
    
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
        -l mixed-2 -n 1000 -c -b -P params
EOS
#-HERE-END-----------------------------------------------------------------

    chmod +x ${HOME}/working/${wdir}/run 

#-HERE---------------------------------------------------------------------
    CMD=$(cat<<EOS
[ -z "\$(mount | grep working )" ] && \
    (sshfs -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 \
    rennes:\${HOME}/working \${HOME}/working)

screen -dmS sm 
screen -S sm -X exec bash -c "\${HOME}/working/$wdir/run; bash"
EOS
    )
#-HERE-END-----------------------------------------------------------------


    ssh $node "$CMD"

done

