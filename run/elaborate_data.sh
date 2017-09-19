#!/bin/bash


IFS=$'\n'

# Manage arguments
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

usage()
{
cat << EOF

usage: $0 options

This script builds the online plots

OPTIONS:
echo $DIR
echo $LOCAL_DIR
exit
   -d --dir PATH    where to find data
   -g --graph       make graphs   
   -c -local        local directory
   -b -blocks       storage blocks to recover (from the begin of simulation)
   -s -weights      include weights data
   -w --www         open browser
   -l --loop SEC    run recursivelly to follow online course
   -h --help        show this help

EOF
}

EXE=$0
WWW=false
CURR=$(pwd)
DIR=
LOOP=
VISUALIZE=false
GRAPHS=false
BLOCKS=all
LOCAL=false
WEIGHTS=false
BLOCKS=all

# getopt
GOTEMP="$(getopt -o "d:gcb:wl:sh" -l "dir:,graphs,blocks:,local,www,loop,weights,help"  -n '' -- "$@")"

if ! [ "$(echo -n $GOTEMP |sed -e"s/\-\-.*$//")" ]; then
    usage; exit;
fi

eval set -- "$GOTEMP"

while true ;
do
    case "$1" in
        -d | --dir)
            DIR="$2"
            shift 2;;
        -g | --graphs)
            GRAPHS=true
            shift;;
        -b | --blocks)
            BLOCKS="$2"
            shift 2;;
        -c | --local)
            LOCAL=true
            shift;;
        -w | --www)
            WWW=true
            shift;;
        -l | --loop)
            LOOP=$2
            shift 2;;
        -s | --weights)
            WEIGHTS=true
            shift;;
        -h | --help)
            echo "on help"
            usage; exit;
            shift;
            break;;
        --) shift ;
            break ;;
    esac
done

if [ -z "$DIR" ]; then
    usage; exit;
fi

manage_path()
{
    path=$1
    path="${path/#\~/$HOME}"
    path=$(realpath $path)
    echo -n $path
}

start_elab()
{
    echo $DIR
    BASE=$(echo $EXE|sed -e"s/\/run\/$(basename $EXE)//")
    BASE=$(manage_path $BASE)
    DIR=$(manage_path $DIR)

    CURR=$(pwd)
    LOCAL_DIR=/tmp/$(basename "$DIR")_plots
    [ $LOCAL == true ] && LOCAL_DIR=$CURR

    [ ! -d "$LOCAL_DIR" ] && mkdir $LOCAL_DIR
    rm -fr $LOCAL_DIR/*

    echo "data dir: $DIR"
    echo "source dir: $BASE"
    echo "out dir: $LOCAL_DIR"
    if [ $GRAPHS == true ]; then

        plots_file="

        <!DOCTYPE html>
        <html>
        <head>
        <meta charset=\"UTF-8\">
        <title> $(basename $DIR) </title>
        </head>

        <body>
        <h1>$(basename $DIR)</h1>
        <table style=\"width:100%\">
        <tr>
        <td><img src=\"means_all.png\"  width=\"100%\"></td>
        <td><img src=\"g_means.png\"  width=\"100%\"></td>
        </tr>
        <tr>
        <td><img src=\"sensors_per_goal.png\"   width=\"100%\"></td>
        <td><img src=\"touches.png\"  width=\"100%\"></td>
        </tr>
        <tr>
        <td><img src=\"sensors.png\"   width=\"100%\"></td>
        <td></td>
        </tr>
        <tr>
        <td><img src=\"weights.png\"   width=\"100%\"></td>
        <td></td>
        </tr>
        <tr>
        <td><img src=\"weights_grid.png\"   width=\"100%\"></td>
        <td><img src=\"positions_grid.png\"   width=\"100%\"></td>
        </tr>
        </table>

        </body>

        </html>
        
        " 
        
        echo "$plots_file" > $LOCAL_DIR/plots.html
    fi

    if [ $WWW == true ] && [ $GRAPHS == true ]; then
        x-www-browser $LOCAL_DIR/plots.html &
        sleep 2
    fi
}

run()
{
    
    TMP_DIR="$(mktemp -d)"
    cd $TMP_DIR
    echo "collect data..."
    cat $(find $DIR | grep cont) > $TMP_DIR/log_cont_sensors
    
    SELECT_BLOCKS=
    [[ $BLOCKS != all ]] && SELECT_BLOCKS=" | grep store | sort -n |head -$BLOCKS "

	echo "		block option: $SELECT_BLOCKS"

    cat $(find $DIR | eval "grep predictions $SELECT_BLOCKS" ) | \
    	sed -e"s/\s\+/ /g; s/[^[:print:]]//g" | \
    	sort -k 1 -n | sed -e "s/^/SIM 1 /" | \
    	sed -e"s/\s\+/ /g" > $TMP_DIR/all_predictions
    	
    cat $(find $DIR | eval "grep log_sensors $SELECT_BLOCKS" ) | \
    	sed -e"s/\s\+/ /g; s/[^[:print:]]//g" | \
    	sort -k 1 -n | sed -e "s/^/SIM 1 /" | \
    	sed -e"s/\s\+/ /g" > $TMP_DIR/all_sensors
    	
    cat $(find $DIR | eval "grep log_weights $SELECT_BLOCKS" ) | \
    	sed -e"s/\s\+/ /g; s/[^[:print:]]//g" | \
    	sort -k 1 -n | sed -e "s/^/SIM 1 /" | \
    	sed -e"s/\s\+/ /g" > $TMP_DIR/all_weights
    	
     cat $(find $DIR | eval "grep log_trials $SELECT_BLOCKS" ) | \
    	sed -e"s/\s\+/ /g; s/[^[:print:]]//g" | \
    	sort -k 1 -n | sed -e "s/^/SIM 1 /" | \
    	sed -e"s/\s\+/ /g" > $TMP_DIR/all_trials  
    	 	
    if [[ $BLOCKS != all ]]; then
	    if [[ -d "${DIR}/main_data/store" ]]; then
	        last_dumped=$(find $DIR | eval "grep dump | grep store $SELECT_BLOCKS | tail -1")
	    	[[ -f "$last_dumped" ]] && \
	    			cp $last_dumped $TMP_DIR/dumped_robot
	    fi
    else
	    if [[ -d "${DIR}/main_data/store" ]]; then
	        last_dumped=$(find $DIR | eval "grep dump | grep store| sort | tail -1")
	    	cp $last_dumped $TMP_DIR/dumped_robot
	    fi
	fi
    
    if [[ -f dumped_robot ]]; then
    	echo "run Python script..."
        cd ${BASE}/simulation/src
        python get_data.py -s "$TMP_DIR" &> $TMP_DIR/get_data_log        
        cd -
    fi
    
    if [[  $GRAPHS == true ]]; then

        	
        echo "run R scripts..."
        R CMD BATCH ${BASE}/rscripts/analyze_touches.R
        R CMD BATCH ${BASE}/rscripts/analyze_predictions.R
        R CMD BATCH ${BASE}/rscripts/analyze_predictions_final.R
        R CMD BATCH ${BASE}/rscripts/analyze_sensors.R 
        R CMD BATCH ${BASE}/rscripts/analyze_weights.R 
        R CMD BATCH ${BASE}/rscripts/analyze_pred_history.R 
        
        echo "convert images to png..."
        for f in *.pdf; do
            echo "converting $f ..."
           convert -density 300 -trim $f -quality 100 $(echo $f|sed -e"s/\.pdf/.png/")
        done
        echo "done"
    fi

    cp $TMP_DIR/* $LOCAL_DIR
    rm -fr $TMP_DIR
    cd $CURR
}

start_elab
run

if [ ! -z "$LOOP" ]; then
    for((;;)); do sleep $LOOP; run; done
fi
