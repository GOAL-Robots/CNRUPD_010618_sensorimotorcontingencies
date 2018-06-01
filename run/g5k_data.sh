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
    sudo sed -i -e\"s/\(ps:alpha.*-dGraphicsAlphaBits=%u\)/\1 -dGraphicsAlphaBits=1/\" /etc/ImageMagick-6/delegates.xml

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
        $current_host:\${HOME}/working \${HOME}/working)

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
