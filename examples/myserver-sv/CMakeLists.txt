set(TARGET myserver-sv)
add_executable(${TARGET} server.cc)

include(DefaultTargetOptions)

target_link_libraries(${TARGET} PRIVATE common json_cpp httplib sense-voice-core ggml ${CMAKE_THREAD_LIBS_INIT})

if (WIN32)
    target_link_libraries(${TARGET} PRIVATE ws2_32)
endif()
