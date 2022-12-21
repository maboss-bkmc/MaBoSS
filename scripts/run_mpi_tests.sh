#!/bin/bash

cd engine/tests

bash ./test-mpi-cellcycle.sh 2
bash ./test-mpi-cellcycle.sh 4
bash ./test-mpi-ewing.sh 2
bash ./test-mpi-ewing.sh 4
bash ./test-mpi-ensemble.sh 2
bash ./test-mpi-ensemble.sh 4

cd ../..