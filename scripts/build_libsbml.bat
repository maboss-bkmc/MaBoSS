@echo on

curl -L https://github.com/sbmlteam/libsbml/archive/refs/tags/v5.20.4.tar.gz --output v5.20.4.tar.gz
7z e v5.20.4.tar.gz  && 7z x v5.20.4.tar

cmake -B libsbml-5.20.4/build -S libsbml-5.20.4 -G"Ninja" ^
		-DCMAKE_INSTALL_PREFIX="%LIBSBML_INSTALL_PREFIX%" ^
		-DCMAKE_BUILD_TYPE=Release ^
		-DCMAKE_C_COMPILER="%CC%" ^
		-DCMAKE_CXX_COMPILER="%CXX%" ^
        -DLIBXML_LIBRARY="%LIBXML_LIBRARY%" ^
        -DLIBXML_INCLUDE_DIR="%LIBXML_INCLUDE_DIR%" ^
		-DWITH_SWIG=OFF ^
        -DENABLE_QUAL=ON ^
        -DWITH_CPP_NAMESPACE=ON ^
        -DBUILD_SHARED_LIBS=ON
		
cmake --build libsbml-5.20.4/build --verbose --parallel "%CPU_COUNT%"
