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
    cd engine/tests/
    if [[ $2 == "win" ]]; then
        bash ./test-user_func.sh ../bin/user_func.dll
        check "user_func" 
    else
        if [[ $2 == "mac" ]]; then
            bash ./test-user_func.sh ../lib/libuser_func.dylib
            check "user_func"    
        else
            bash ./test-user_func.sh ../lib/libuser_func.so
            check "user_func"
        fi
    fi
    
else 
    cd engine
    cp tests/user_func/user_func.cc src
    cd src
    make clean
    make FUNC_MODULE=user_func
    if [ $? != 0 ]; then exit 1; fi
    mkdir -p ../lib
    cp libuser_func.so ../lib
    cd ../tests/
    bash ./test-user_func.sh ../lib/libuser_func.so
    check "user_func"
    
fi



cd ../..

exit $return_code