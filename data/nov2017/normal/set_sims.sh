#!/bin/bash
strt=1
end=2
dirs="$(seq $strt $end)"
for d in $dirs; do 
    mkdir -p $d 
    mkdir -p $d/{sim,data}
done


sed -i -e"s/\(gs_eta_decay =\).*$/\1 True/" params
sed -i -e"s/\(gm_goalrep_lr =\).*$/\1 0.40/" params
find | grep "\<sim$" | xargs -L 1 cp params

main=~/Projects/sensorimotor-dev

curr=$(pwd)

name=new

for d in $dirs; do

    screen -mdS $name$d -t sim
    sleep .2
    screen -S $name$d -X screen -t data
    sleep 1
     
    screen -S $name$d -p sim -X stuff "cd $curr/$d/sim\n"
    sleep .2
    screen -S $name$d -p sim -X stuff "$main/run/g5k_batteries.sh -P params -b -t 10000 -n 30 -d $main/simulation\n" 
    sleep 1
    
    screen -S $name$d -p data -X stuff "cd $curr/$d/data\n"
    sleep .2
    screen -S $name$d -p data -X stuff "$main/run/elaborate_data.sh -c -s -g -l 20 -d ../sim\n"
    sleep 2


done
