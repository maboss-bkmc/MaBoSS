
include(${CMAKE_CURRENT_LIST_DIR}/displayers/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/engines/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/parsers/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/rpc/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/sedml/CMakeLists.txt)

list(APPEND src_files
  ${CMAKE_CURRENT_LIST_DIR}/BNException.cc
  ${CMAKE_CURRENT_LIST_DIR}/Expressions.cc
  ${CMAKE_CURRENT_LIST_DIR}/IStates.cc
  ${CMAKE_CURRENT_LIST_DIR}/Symbols.cc
  ${CMAKE_CURRENT_LIST_DIR}/NetworkState.cc
  ${CMAKE_CURRENT_LIST_DIR}/Network.cc 
  ${CMAKE_CURRENT_LIST_DIR}/Node.cc
  ${CMAKE_CURRENT_LIST_DIR}/PopNetwork.cc 
  ${CMAKE_CURRENT_LIST_DIR}/BuiltinFunctions.cc
  ${CMAKE_CURRENT_LIST_DIR}/Function.cc 
  ${CMAKE_CURRENT_LIST_DIR}/LogicalExprGen.cc
  ${CMAKE_CURRENT_LIST_DIR}/ObservedGraph.cc
  ${CMAKE_CURRENT_LIST_DIR}/ProbaDist.cc
  ${CMAKE_CURRENT_LIST_DIR}/RandomGenerator.cc
  ${CMAKE_CURRENT_LIST_DIR}/RunConfig.cc
  ${CMAKE_CURRENT_LIST_DIR}/Utils.cc
)

if (NOT MSVC)
  list(APPEND src_files
    ${CMAKE_CURRENT_LIST_DIR}/MBDynBitset.cc
  )
endif()
