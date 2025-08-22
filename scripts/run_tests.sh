#!/bin/bash

return_code=0

check()
{
    if [ $? = 0 ]; then
    	echo "$1 OK"
    else
	    echo "$1 ERR"
        return_code=1
    fi
}

cd engine/tests/maboss/

bash ./test-cellcycle.sh
check "cellcycle"
bash ./test-ensemble.sh
check "ensemble"
bash ./test-ewing.sh
check "ewing"
bash ./test-bnet.sh
check "bnet"
bash ./test-prngs.sh
check "prngs"
bash ./test-popmaboss.sh
check popmaboss
bash ./test-observed_graph.sh
check observed_graph
bash ./test-schedule.sh
check "schedule"

if [[ -n $RUNNER_OS ]] && [[ $RUNNER_OS != "Windows" ]]; then
    bash ./test-server.sh
    check "server"
    bash ./test-rngs.sh    
    check "rngs"
fi

cd ../../..

exit $return_code