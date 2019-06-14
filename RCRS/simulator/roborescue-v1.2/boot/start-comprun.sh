#! /bin/bash

. functions.sh

processArgs $*

# Delete old logs
rm -f $LOGDIR/*.log

#startGIS
startKernel --nomenu --autorun
startSims --nogui

execute sample "../rcrs-adf-sample/start.sh -1 -1 -1 -1 -1 -1 localhost"
#execute sample "./sampleagent.sh"

echo "Start your agents"
waitFor $LOGDIR/kernel.log "Kernel has shut down" 30


sleep 5
kill $PIDS
./kill.sh
