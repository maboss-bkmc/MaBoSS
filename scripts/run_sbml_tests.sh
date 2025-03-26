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

cd engine/tests/

bash ./test-sbml.sh
check "sbml"

cd ../..

exit $return_code