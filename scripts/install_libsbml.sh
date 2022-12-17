#!/bin/bash


if [ -z "${BUILD_PATH}" ]
then 
BUILD_PATH=/tmp/
fi

cd ${BUILD_PATH}/libSBML-5.19.0-Source/build
make install