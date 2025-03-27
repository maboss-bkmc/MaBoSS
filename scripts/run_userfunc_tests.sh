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

if [[ $1 == "cmake" ]]; then
    cmake -B build -S . \
        -DCMAKE_INSTALL_PREFIX=engine \
        -DCMAKE_BUILD_TYPE=Release \
        -DUSERFUNC=./engine/tests/user_func/user_func.cc
    cmake --build build
    cmake --install build
else 
    cd engine
    cp tests/user_func/user_func.cc src
    cd src
    make clean
    make FUNC_MODULE=user_func
    if [ $? != 0 ]; then exit 1; fi
    mkdir -p ../lib
    cp libuser_func.so ../lib
    cd ../..
    
fi


cd engine/tests/
bash ./test-user_func.sh ../lib/libuser_func.so
check "user_func"

cd ../..

exit $return_code