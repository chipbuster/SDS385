cmake_minimum_required(VERSION 3.2)
project(SGD)

# Add custom module search files (This one for homebrew cmake)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "/usr/local/share/cmake/Modules/")

# Find required libraries
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

# Add appropriate flags for compilation
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -Wall -Wextra" )
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

# Mark Eigen headers as system to suppress warnings from them
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR}) 

add_executable(testreader read_svmlight.cpp test_funcs.cpp)
