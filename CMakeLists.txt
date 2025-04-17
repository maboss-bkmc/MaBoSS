cmake_minimum_required(VERSION 3.10.0)
project(MaBoSS VERSION 2.6.2)

if (SEDML)
  set(CMAKE_C_STANDARD 17)
  set(CMAKE_C_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
else()
  set(CMAKE_C_STANDARD 11)
  set(CMAKE_C_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(MSVC)
  # set(C_FLAGS ${C_FLAGS} /W4 /bigobj /wd4100 /wd4996)
  # Remove /W3, which is added by default, since we set /W4. Some build
  # generators with MSVC complain about both /W3 and /W4 being specified.
  foreach(flag_var  CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
    if(${flag_var} MATCHES "/W3")
      string(REGEX REPLACE "/W3" "" ${flag_var} "${${flag_var}}")
    endif()
  endforeach()
  add_compile_options(/W4 /bigobj /wd4100 /wd4996)
else()
# add_compile_options(-Wall -Wextra -pedantic -O2)
# add_compile_options(-Wall -Wextra -O2)
add_compile_options(-Wall -O2)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

# Global settings : bitset data structure
if(NOT MAXNODES)
set(MAXNODES 64)
endif()

if(MPI)
set(MPI_SUFFIX ".MPI")
endif()

if(DYNBITSET)
  if (DYNBITSET_STD_ALLOC)
    set(DYNBITSET_FLAG USE_DYNAMIC_BITSET=1 USE_DYNAMIC_BITSET_STD_ALLOC=1)
  else()
    set(DYNBITSET_FLAG USE_DYNAMIC_BITSET=1)
  endif()
endif()

# Userfunc is a special case, where we ONLY want to compile this
if (USERFUNC)
  add_library(user_func SHARED ${USERFUNC} ${CMAKE_SOURCE_DIR}/engine/src/Function.cc)
  target_include_directories(user_func PUBLIC ${CMAKE_SOURCE_DIR}/engine/src)
  target_compile_definitions(user_func PUBLIC MAXNODES=${MAXNODES} ${DYNBITSET_FLAG})

  install( 
    TARGETS user_func
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT user_func
  )
  
# If not userfunc, then the rest
else()

  # setting executable names
  if (DYNBITSET)
      set(MABOSS_NAME MaBoSS_dn${MPI_SUFFIX})
      set(MABOSS_LIBNAME libMaBoSS_dn${MPI_SUFFIX})
      set(POPMABOSS_NAME PopMaBoSS_dn${MPI_SUFFIX})
      set(MABOSS_SERVER_NAME MaBoSS_dn-server${MPI_SUFFIX})
  elseif(MAXNODES EQUAL 64)
      set(MABOSS_NAME MaBoSS${MPI_SUFFIX})
      set(MABOSS_LIBNAME libMaBoSS${MPI_SUFFIX})
      set(POPMABOSS_NAME PopMaBoSS${MPI_SUFFIX})
      set(MABOSS_SERVER_NAME MaBoSS-server${MPI_SUFFIX})
  else()
      set(MABOSS_NAME MaBoSS_${MAXNODES}n${MPI_SUFFIX})
      set(MABOSS_LIBNAME libMaBoSS_${MAXNODES}n${MPI_SUFFIX})
      set(POPMABOSS_NAME PopMaBoSS_${MAXNODES}n${MPI_SUFFIX})
      set(MABOSS_SERVER_NAME MaBoSS_${MAXNODES}n-server${MPI_SUFFIX})
  endif()
  set(MABOSS_CLIENT_NAME MaBoSS-client)
  
  set(CMAKE_SHARED_LIBRARY_PREFIX "")
  set(CMAKE_STATIC_LIBRARY_PREFIX "")
  
  # configuring executables
  add_executable(${MABOSS_NAME} engine/src/MaBoSS.cc)
  add_executable(${POPMABOSS_NAME} engine/src/PopMaBoSS.cc)

  
  add_definitions(-DMAXNODES=${MAXNODES})
  
  include(engine/src/CMakeLists.txt)
   
  add_library(${MABOSS_LIBNAME} STATIC ${src_files} ${BISON_ctbndl_parser_OUTPUTS} ${BISON_rc_parser_OUTPUTS})
  target_compile_definitions(${MABOSS_LIBNAME} PUBLIC MAXNODES=${MAXNODES} ${DYNBITSET_FLAG})

  if(MPI)
      find_package(MPI)
      if(MPI_CXX_FOUND)
          target_compile_definitions(${MABOSS_LIBNAME} PUBLIC MPI_COMPAT=1)
          target_include_directories(${MABOSS_LIBNAME} PUBLIC ${MPI_CXX_INCLUDE_PATH})
          target_link_libraries(${MABOSS_LIBNAME} MPI::MPI_CXX)
      endif()
  endif()

  if(STD_THREAD)
    target_compile_definitions(${MABOSS_LIBNAME} PUBLIC STD_THREAD=1)
  else()
    if (NOT MSVC)
      target_link_libraries(${MABOSS_LIBNAME} -lpthread)
    endif()
  endif()
  # find_package(HDF5)
  # if(HDF5_FOUND)
  #     target_compile_definitions(${MABOSS_LIBNAME} PUBLIC HDF5_COMPAT=1)
  #     target_include_directories(${MABOSS_LIBNAME} PUBLIC ${HDF5_INCLUDE_DIRS})
  #     target_link_libraries(${MABOSS_LIBNAME} ${HDF5_LIBRARIES})
  # endif()

  
  if(SBML)
      
      find_package(LIBSBML REQUIRED)
      if(LIBSBML_FOUND)
          target_compile_definitions(${MABOSS_LIBNAME} PUBLIC SBML_COMPAT=1)
          target_include_directories(${MABOSS_LIBNAME} PUBLIC ${LIBSBML_INCLUDE_DIR})
          target_link_libraries(${MABOSS_LIBNAME} ${LIBSBML_LIBRARY})
          if (MSVC)
            target_link_libraries(${MABOSS_LIBNAME} -lz -lbz2 -lxml2 -lm)
          endif()
      endif()
      
      if(SEDML)
          
          find_package(LIBSEDML REQUIRED)
          if (LIBSEDML_FOUND)
              target_compile_definitions(${MABOSS_LIBNAME} PUBLIC SBML_COMPAT=1 SEDML_COMPAT=1)
              target_include_directories(${MABOSS_LIBNAME} PUBLIC ${LIBSEDML_INCLUDE_DIR})
              target_link_libraries(${MABOSS_LIBNAME} ${LIBSEDML_LIBRARY})
          endif()
          
          find_package(LIBNUML REQUIRED)
          if (LIBNUML_FOUND)
              # target_include_directories(${MABOSS_LIBNAME} PUBLIC ${LIBNUML_INCLUDE_DIR})
              target_link_libraries(${MABOSS_LIBNAME} ${LIBNUML_LIBRARY})
          endif()
      endif()
      
      find_package(LIBXML REQUIRED)
      if (LIBXML_FOUND)
          target_include_directories(${MABOSS_LIBNAME} PUBLIC ${LIBXML_INCLUDE_DIR})
          target_link_libraries(${MABOSS_LIBNAME} ${LIBXML_LIBRARY})
      endif()
      
      find_package(ZLIB)
      if (ZLIB_FOUND)
          # target_include_directories(${MABOSS_LIBNAME} PUBLIC ${ZLIB_INCLUDE_DIR})
          target_link_libraries(${MABOSS_LIBNAME} ${ZLIB_LIBRARY})
      endif()
  
      find_package(BZ2)
      if (BZ2_FOUND)
          # target_include_directories(${MABOSS_LIBNAME} PUBLIC ${LIBBZ2_INCLUDE_DIR})
          target_link_libraries(${MABOSS_LIBNAME} ${LIBBZ_LIBRARY})
      endif()
      
  endif()
  
  target_include_directories(${MABOSS_LIBNAME} PUBLIC ${CMAKE_SOURCE_DIR}/engine/src)
  target_include_directories(${MABOSS_LIBNAME} PUBLIC ${CMAKE_SOURCE_DIR}/engine/src/parsers)
  
  target_link_libraries(${MABOSS_NAME} ${MABOSS_LIBNAME})
  target_link_libraries(${POPMABOSS_NAME} ${MABOSS_LIBNAME})

  list(APPEND list_targets ${MABOSS_NAME} ${POPMABOSS_NAME})


  if (NOT MSVC)
  set(CMAKE_CXX_FLAGS_DEBUG "-g  -D_DEBUG -D_GLIBCXX_DEBUG")
  endif() 


  install(
    TARGETS ${list_targets}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT executables
  )

  install( 
    TARGETS ${MABOSS_LIBNAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT libs
  )

endif()



###############################################################################
#
# Print Summary
#

if (NOT HOSTNAME)
 site_name(HOSTNAME)
endif()

message(STATUS "
----------------------------------------------------------------------
MaBoSS version ${PROJECT_VERSION}
----------------------------------------------------------------------

   More information and the latest version are available online at
   https://github.com/sysbio-curie/MaBoSS

   Please report problems using the issue tracker at
   https://github.com/sysbio-curie/MaBoSS/issues

   To contact the developers directly, email vincent.noel@curie.fr

   Configured on host '${HOSTNAME}'
     host type                     = ${CMAKE_SYSTEM_NAME}
     host operating system         = ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION}
     host CPU architecture         = ${CMAKE_SYSTEM_PROCESSOR}

   General build flags:
     CC                            = ${CMAKE_C_COMPILER}
     CXX                           = ${CMAKE_CXX_COMPILER}
     CPPFLAGS                      = ${BUILD_DEFINITIONS}
     CFLAGS                        = ${CMAKE_C_FLAGS}
     CXXFLAGS                      = ${CMAKE_CXX_FLAGS}
     LDFLAGS                       = ${CMAKE_EXE_LINKER_FLAGS}
 ")

if(SBML)
message(STATUS "
   ZLIB library configuration:
     ZLIB library                  = ${ZLIB_LIBRARY}
     ZLIB include                  = -I${ZLIB_INCLUDE_DIR}
     
   BZ2 library configuration:
     BZ2 library                   = ${LIBBZ_LIBRARY}
     BZ2 include                   = -I${LIBBZ_INCLUDE_DIR}
     
   XML library configuration:
     XML library                   = ${LIBXML_LIBRARY}
     XML include                   = -I${LIBXML_INCLUDE_DIR}  
     
   SBML library configuration:
     SBML library                  = ${LIBSBML_LIBRARY}
     SBML include                  = -I${LIBSBML_INCLUDE_DIR}
 ")
if (SEDML)
message(STATUS "
   SEDML library configuration:
     SEDML library                  = ${LIBSEDML_LIBRARY}
     SEDML include                  = -I${LIBSEDML_INCLUDE_DIR}
     
   NUML library configuration:
     NUML library                  = ${LIBNUML_LIBRARY}
     NUML include                  = -I${LIBNUML_INCLUDE_DIR}
 ")
endif()
endif()

message(STATUS "
  Other libSEDML configuration settings:
     Installation $prefix          = ${CMAKE_INSTALL_PREFIX}")



if(UNIX)
    if(APPLE)
        message(STATUS "  Value of $DYLD_LIBRARY_PATH   = $ENV{DYLD_LIBRARY_PATH}")
    else()
        message(STATUS "  Value of $LD_LIBRARY_PATH     = $ENV{LD_LIBRARY_PATH}")
    endif()
endif()

message(STATUS "
----------------------------------------------------------------------")