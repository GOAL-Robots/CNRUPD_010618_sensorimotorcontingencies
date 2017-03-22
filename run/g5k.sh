#!/bin/bash

# IFS=$'\n'
# N_MACHINES=3
# MIN_CORES=10
# MIN_RAM=32
# 
# RES="$(${HOME}/g5kutils/avaliable_resources.sh 2>/dev/null)"
# 
# cluster=$(
# for res in $(echo "$RES"); do 
#     n_nodes=$(echo $res | awk '{print $3}')
#     n_cores=$(echo $res | awk '{print $5}')
#     ram=$(echo $res | awk '{print $7}')
# 
#     if [ $n_nodes -ge $N_MACHINES ] && [ $n_cores -ge $MIN_CORES ] && [ $ram -ge $MIN_RAM ]; then
#         echo $res
#     fi
# done | head -n 1 | awk '{print $2}'
# )
# 
# [ -z "$cluster" ] && echo "no reservation avaliable" && exit 1 
# 
# ${HOME}/g5kutils/deploy.sh -t 4:0:0 -u $cluster -n $N_MACHINES -f

if [ -z "$(ls ~/.G_*)" ]; then
    echo "no reservation avaliable"
    exit 1
fi

declare -a nodes=($(cat ~/.G_*/NODES|uniq))
declare -a dirs=(sm_c0i1 sm_c1i0p2 sm_c1i0p5)

params[0]=$(cat<<HERE_PARAMS
simulation_competence_improvement_prop = 0.0 
simulation_incompetence_prop = 1.0 
GOAL_NUMBER = 9 
HERE_PARAMS
)

params[2]=$(cat<<HERE_PARAMS
simulation_competence_improvement_prop = 1.0 
simulation_incompetence_prop = 0.5 
GOAL_NUMBER = 9 
HERE_PARAMS
)

params[3]=$(cat<<HERE_PARAMS
simulation_competence_improvement_prop = 1.0 
simulation_incompetence_prop = 0.2 
GOAL_NUMBER = 9 
HERE_PARAMS
)

for i in $(seq 0 2); do
   

    node=${nodes[$i]}
    wdir=${dirs[$i]}
    
    echo "using node $node"
    echo "storing in ${HOME}/working/$wdir"    
    echo

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

