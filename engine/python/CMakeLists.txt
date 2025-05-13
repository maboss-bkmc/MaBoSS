cmake_minimum_required(VERSION 3.15...3.30)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C CXX)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/Modules")

find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)

add_definitions(-DSTD_THREAD=1)
add_definitions(-DPYTHON_API=1)
    
include(../src/CMakeLists.txt)
include(src/CMakeLists.txt)

set(cmaboss_modules cmaboss;cmaboss_128n;cmaboss_256n;cmaboss_512n;cmaboss_1024n;cmaboss_dn)
foreach(mod ${cmaboss_modules})
    python_add_library(${mod} MODULE WITH_SOABI ${cmaboss_files} ${src_files})
    target_include_directories(
        ${mod} PUBLIC
        ${Python_NumPy_INCLUDE_DIRS}
        ${CMAKE_CURRENT_LIST_DIR}/../src/
        ${CMAKE_CURRENT_LIST_DIR}/../src/parsers
    )    
endforeach()

target_compile_definitions(cmaboss PUBLIC -DMAXNODES=64)
target_compile_definitions(cmaboss_128n PUBLIC -DMAXNODES=128)
target_compile_definitions(cmaboss_256n PUBLIC -DMAXNODES=256)
target_compile_definitions(cmaboss_512n PUBLIC -DMAXNODES=512)
target_compile_definitions(cmaboss_1024n PUBLIC -DMAXNODES=1024)
target_compile_definitions(cmaboss_dn PUBLIC -DUSE_DYNAMIC_BITSET=1 -DUSE_DYNAMIC_BITSET_STD_ALLOC=1)

if(SBML)
    find_package(LIBSBML REQUIRED)
    if(LIBSBML_FOUND)
        foreach(mod ${cmaboss_modules})
            target_compile_definitions(${mod} PUBLIC SBML_COMPAT=1)
            target_include_directories(${mod} PUBLIC ${LIBSBML_INCLUDE_DIR})
            target_link_libraries(${mod} PRIVATE ${LIBSBML_LIBRARY})
        endforeach()
    endif()

    find_package(LIBXML REQUIRED)
    if (LIBXML_FOUND)
        foreach(mod ${cmaboss_modules})
            target_link_libraries(${mod} PRIVATE ${LIBXML_LIBRARY})
        endforeach()
    endif()
      
    find_package(ZLIB)
    if (ZLIB_FOUND)
        foreach(mod ${cmaboss_modules})
            target_link_libraries(${mod} PRIVATE ${ZLIB_LIBRARY})
        endforeach()
    endif()

    find_package(BZ2)
    if (BZ2_FOUND)    
        foreach(mod ${cmaboss_modules})
            target_link_libraries(${mod} PRIVATE ${LIBBZ_LIBRARY})
        endforeach()
    endif()

endif()

install(TARGETS cmaboss  DESTINATION ${SKBUILD_PROJECT_NAME})

install(DIRECTORY cmaboss_128n DESTINATION ${SKBUILD_PROJECT_NAME}/..)
install(TARGETS cmaboss_128n DESTINATION ${SKBUILD_PROJECT_NAME}/../cmaboss_128n)

install(DIRECTORY cmaboss_256n DESTINATION ${SKBUILD_PROJECT_NAME}/..)
install(TARGETS cmaboss_256n DESTINATION ${SKBUILD_PROJECT_NAME}/../cmaboss_256n)

install(DIRECTORY cmaboss_512n DESTINATION ${SKBUILD_PROJECT_NAME}/..)
install(TARGETS cmaboss_512n DESTINATION ${SKBUILD_PROJECT_NAME}/../cmaboss_512n)

install(DIRECTORY cmaboss_1024n DESTINATION ${SKBUILD_PROJECT_NAME}/..)
install(TARGETS cmaboss_1024n DESTINATION ${SKBUILD_PROJECT_NAME}/../cmaboss_1024n)

install(DIRECTORY cmaboss_dn DESTINATION ${SKBUILD_PROJECT_NAME}/..)
install(TARGETS cmaboss_dn DESTINATION ${SKBUILD_PROJECT_NAME}/../cmaboss_dn)
