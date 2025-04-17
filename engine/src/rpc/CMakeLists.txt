if(NOT MSVC)  
    if (BUILD_CLIENT OR BUILD_SERVER)
        list(APPEND src_files 
            ${CMAKE_CURRENT_LIST_DIR}/MaBoSS-RPC.cc 
            ${CMAKE_CURRENT_LIST_DIR}/DataStreamer.cc 
        )
    endif()

    if (BUILD_SERVER)
        list(APPEND src_files ${CMAKE_CURRENT_LIST_DIR}/Server.cc)
        add_executable(${MABOSS_SERVER_NAME} ${CMAKE_CURRENT_LIST_DIR}/MaBoSS-server.cc)
        target_link_libraries(${MABOSS_SERVER_NAME} ${MABOSS_LIBNAME})
        list(APPEND list_targets ${MABOSS_SERVER_NAME})

    endif()

    if (BUILD_CLIENT)
        list(APPEND src_files ${CMAKE_CURRENT_LIST_DIR}/Client.cc)
        add_executable(${MABOSS_CLIENT_NAME} ${CMAKE_CURRENT_LIST_DIR}/MaBoSS-client.cc)
        target_link_libraries(${MABOSS_CLIENT_NAME} ${MABOSS_LIBNAME})
        list(APPEND list_targets ${MABOSS_CLIENT_NAME})
    endif()
    
endif()
