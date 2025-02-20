#!/bin/bash

if [ -z "${INSTALL_PATH}" ]
then 
INSTALL_PATH=/usr/local/
fi


if [ -z "${BUILD_PATH}" ]
then 
BUILD_PATH=/tmp/
fi

if [ -z "${CXX}" ]
then 
CXX=g++
fi

if [ -z "${CC}" ]
then 
CC=gcc
fi

if [ -z "${LIBXML_LIBDIR}" ]
then 
LIBXML_LIBDIR=/usr/lib
fi

if [ -z "${LIBXML_INCLUDEDIR}" ]
then 
LIBXML_INCLUDEDIR=/usr/include/libxml2
fi

mkdir -p ${BUILD_PATH}
cd ${BUILD_PATH}
wget https://github.com/sbmlteam/libsbml/archive/refs/tags/v5.20.4.tar.gz
tar -zxf v5.20.4.tar.gz
cd libsbml-5.20.4
mkdir build
cd build

if [[ -n $RUNNER_OS ]] && [[ $RUNNER_OS == "Windows" ]]; then

cmake -G"MSYS Makefiles" \
-DCMAKE_INSTALL_PREFIX=${INSTALL_PATH} \
-DCMAKE_INSTALL_LIBDIR=${INSTALL_PATH}/lib \
-DCMAKE_CXX_COMPILER=${CXX} \
-DCMAKE_C_COMPILER=${CC} \
-DCMAKE_CXX_STANDARD_LIBRARIES=-lxml2 \
-DWITH_SWIG=OFF \
-DLIBXML_LIBRARY=${LIBXML_LIBDIR} \
-DLIBXML_INCLUDE_DIR=${LIBXML_INCLUDEDIR} \
-DENABLE_COMP=ON \
-DENABLE_FBC=ON \
-DENABLE_GROUPS=ON \
-DENABLE_LAYOUT=ON \
-DENABLE_MULTI=ON \
-DENABLE_QUAL=ON \
-DENABLE_RENDER=ON \
-DENABLE_DISTRIB=ON \
-DWITH_CPP_NAMESPACE=ON \
..
else
cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH} \
-DCMAKE_INSTALL_LIBDIR=${INSTALL_PATH}/lib \
-DCMAKE_CXX_COMPILER=${CXX} \
-DCMAKE_C_COMPILER=${CC} \
-DCMAKE_CXX_STANDARD_LIBRARIES=-lxml2 \
-DWITH_SWIG=OFF \
-DLIBXML_LIBRARY=${LIBXML_LIBDIR} \
-DLIBXML_INCLUDE_DIR=${LIBXML_INCLUDEDIR} \
-DENABLE_COMP=ON \
-DENABLE_FBC=ON \
-DENABLE_GROUPS=ON \
-DENABLE_LAYOUT=ON \
-DENABLE_MULTI=ON \
-DENABLE_QUAL=ON \
-DENABLE_RENDER=ON \
-DENABLE_DISTRIB=ON \
-DWITH_CPP_NAMESPACE=ON \
..

fi


if [ -z "${CPU_COUNT}" ]
then
	make
else
    echo "Building libsbml in parallel on ${CPU_COUNT} cores"
	make -j "${CPU_COUNT}"
fi
