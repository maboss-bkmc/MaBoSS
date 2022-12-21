#!/bin/bash

cd engine/tests/

bash ./test-cellcycle.sh
bash ./test-ensemble.sh
bash ./test-ewing.sh
bash ./test-sbml.sh
bash ./test-prngs.sh
bash ./test-user_func.sh

if [[ -n $RUNNER_OS ]] && [[ $RUNNER_OS != "Windows" ]]; then
    bash ./test-server.sh
    bash ./test-rngs.sh    
fi

cd ../..