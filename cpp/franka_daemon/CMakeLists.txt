cmake_minimum_required(VERSION 3.10)
project(franka_daemon)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Franka 0.15 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PkgConfig REQUIRED)

# Find ZeroMQ
pkg_check_modules(ZMQ REQUIRED libzmq)

# Find MessagePack
find_path(MSGPACK_INCLUDE_DIR msgpack.hpp
    HINTS /usr/include /usr/local/include
)

if(NOT MSGPACK_INCLUDE_DIR)
    message(FATAL_ERROR "MessagePack C++ headers not found. Install with: apt-get install libmsgpack-dev")
endif()

# FrankaDaemon executable
add_executable(franka_daemon
    franka_daemon.cpp
)

target_include_directories(franka_daemon PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${ZMQ_INCLUDE_DIRS}
    ${MSGPACK_INCLUDE_DIR}
)

target_link_libraries(franka_daemon PRIVATE
    Franka::Franka
    ${ZMQ_LIBRARIES}
    pthread
)

# Install target
install(TARGETS franka_daemon
    RUNTIME DESTINATION bin
)

# Print configuration
message(STATUS "")
message(STATUS "=== FrankaDaemon Build Configuration ===")
message(STATUS "  Franka: ${Franka_DIR}")
message(STATUS "  ZeroMQ: ${ZMQ_LIBRARIES}")
message(STATUS "  MessagePack: ${MSGPACK_INCLUDE_DIR}")
message(STATUS "============================================")
message(STATUS "")
