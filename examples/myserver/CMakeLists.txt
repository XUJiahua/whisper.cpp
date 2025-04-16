set(TARGET myserver)
add_executable(${TARGET} server.cpp)

include(DefaultTargetOptions)

target_link_libraries(${TARGET} PRIVATE common json_cpp httplib whisper ${CMAKE_THREAD_LIBS_INIT})

if (WIN32)
    target_link_libraries(${TARGET} PRIVATE ws2_32)
endif()
