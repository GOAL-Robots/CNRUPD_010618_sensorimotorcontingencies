#!/bin/bash
n=2
dirs="$(seq $n)"
for d in $dirs; do 
    mkdir -p $d 
    mkdir -p $d/{sim,data}
done


cp ../6/sim/params .
#sed -i -e"s/\(gs_eta_decay =\).*$/\1 True/" params
find | grep "\<sim$" | xargs -L 1 cp params

main=~/Projects/sensorimotor-dev-custom-sensors

curr=$(pwd)


for d in $dirs; do

    screen -mdS sc$d -t sim
    sleep .2
    screen -S sc$d -X screen -t data
    sleep 1
     
    screen -S sc$d -p sim -X stuff "cd $curr/$d/sim\n"
    sleep .2
    screen -S sc$d -p sim -X stuff "$main/run/g5k_batteries.sh -b -t 10000 -n 45 -d $main/simulation\n" 
    sleep 1
    
    screen -S sc$d -p data -X stuff "cd $curr/$d/data\n"
    sleep .2
    screen -S sc$d -p data -X stuff "$main/run/elaborate_data.sh -c -s -g -l 20 -d ../sim\n"
    sleep 2


done
