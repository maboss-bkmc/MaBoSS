FROM ubuntu:24.04

# Preventing python3-tk from asking timezone
ENV DEBIAN_FRONTEND=noninteractive

# Installing dependencies
RUN apt-get -qq update
RUN apt-get install -yq git flex bison gcc g++ make cmake wget pkg-config \
    && apt clean -y \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --recurse-submodules https://github.com/fbergmann/libSEDML.git /opt/libSEDML


RUN cd /opt/libSEDML/submodules/libSBML-dependencies \
    && cmake -B build -S . \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr \
        -DWITH_LIBXML=ON \
        -DWITH_ZLIB=ON \
        -DWITH_ICONV=ON \
        -DWITH_BZIP2=ON \
        -DLIBXML2_WITH_ZLIB=ON \
    && cmake --build build --parallel 16 \
    && cmake --install build
    

RUN cd /opt/libSEDML/submodules/libSBML \
    && git checkout 7c26715 \
    && cmake -B build -S . -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr \
        -DWITH_STABLE_PACKAGES=ON \
        -DWITH_CPP_NAMESPACE=ON \
        -DWITH_ZLIB=ON \
        -DLIBSBML_SKIP_SHARED_LIBRARY=ON \
    && cmake --build build --parallel 16 \
    && cmake --install build

RUN cd /opt/libSEDML/submodules/NuML/libnuml \
    && cmake -B build -S . \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr \
        -DWITH_CPP_NAMESPACE=ON \
        -DLIBNUML_SKIP_SHARED_LIBRARY=ON \
    && cmake --build build --parallel 16 \
    && cmake --install build
    
RUN cd /opt/libSEDML/ \
    && cmake -B build -S . \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr \
        -DWITH_CPP_NAMESPACE=ON \
        -DEXTRA_LIBS="/usr/lib/x86_64-linux-gnu/libxml2.a;/usr/lib/libbz2.a;/lib/libzlib.a" \
        -DBUILD_TESTS=OFF \
        -DLIBSEDML_SKIP_SHARED_LIBRARY=ON \
    && cmake --build build --parallel 16 \
    && cmake --install build


# Installing MaBoSS
ADD engine/src /opt/MaBoSS/engine/src
ADD CMakeLists.txt /opt/MaBoSS/CMakeLists.txt
ADD cmake /opt/MaBoSS/cmake
RUN cd /opt/MaBoSS \
    && cmake -B build -S . \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/ \
        -DSEDML=1 -DSBML=1 \
        -DBUILD_SERVER=1 \
    && cmake --build build --parallel 16 \
    && cmake --install build --component executables \
    && cmake -B build -S . \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/ \
        -DMAXNODES=128 \
        -DSEDML=1 -DSBML=1 \
        -DBUILD_SERVER=1 \
    && cmake --build build --parallel 16 \
    && cmake --install build --component executables \
    && cmake -B build -S . \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/ \
        -DMAXNODES=256 \
        -DSEDML=1 -DSBML=1 \
        -DBUILD_SERVER=1 \
    && cmake --build build --parallel 16 \
    && cmake --install build --component executables \
    && cmake -B build -S . \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/ \
        -DMAXNODES=512 \
        -DSEDML=1 -DSBML=1 \
        -DBUILD_SERVER=1 \
    && cmake --build build --parallel 16 \
    && cmake --install build --component executables \
    && cmake -B build -S . \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/ \
        -DMAXNODES=1024 \
        -DSEDML=1 -DSBML=1 \
        -DBUILD_SERVER=1 \
    && cmake --build build --parallel 16 \
    && cmake --install build  --component executables

RUN rm -fr /opt/libSEDML \
    && rm -fr /opt/MaBoSS/ 
    
# Exposing port 7777
EXPOSE 7777
EXPOSE 7778
EXPOSE 7779
EXPOSE 7780
EXPOSE 7781

# Running MaBoSS server
CMD MaBoSS-server -q --host 0.0.0.0 --port 7777 \
 & MaBoSS_128n-server -q --host 0.0.0.0 --port 7778 \
 & MaBoSS_256n-server -q --host 0.0.0.0 --port 7779 \
 & MaBoSS_512n-server -q --host 0.0.0.0 --port 7780 \
 & MaBoSS_1024n-server -q --host 0.0.0.0 --port 7781
