#!/bin/bash

cd engine/tests

python3 -m unittest test
python3 -m unittest test_128n

cd ../..