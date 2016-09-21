cmake_minimum_required(VERSION 3.2)
project(SGD)

####### Add custom module search files (This one for homebrew cmake) for OSX
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "/usr/local/share/cmake/Modules/")

# Find required libraries
find_package(Eigen3 REQUIRED)
find_package(OpenMP)

#### Options for program
# Use double for floating point?
option(USE_DOUBLES "Use 64-bit floating point. More accurate, but uses over 20GB of RAM" OFF)


# Add openmp flags if the compiler supports it
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -DUSE_OPENMP")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -DUSE_OPENMP")
endif()

# Process double flags, warning the user about memory consumption
if(USE_DOUBLES)
  message(WARNING "You have chosen to use double precision. If your computer does not\
  have at least 22GB of RAM, your computer may crash when you\
  try to run this program. If this was an accident, remove your\
  entire build directory and re-run CMake, or run\
  `cmake <path-to-build> -DUSER_DOUBLES=OFF`" )
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DUSE_DOUBLES")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_DOUBLES")
endif()

# Add appropriate flags for compilation
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -fno-builtin")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -fno-builtin" )
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

# Add CMake debugging and release flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${CMAKE_CXX_FLAGS} -g -Og")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS}\
-Ofast -mavx -march=native -flto -DNDEBUG")

# Mark Eigen headers as system to suppress warnings from them
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})
include_directories(${SGD_SOURCE_DIR}/include)

add_executable(testreader read_svmlight.cpp test_funcs.cpp)
set_property(TARGET testreader PROPERTY CXX_STANDARD 11)


