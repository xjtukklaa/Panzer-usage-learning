# example cmakelists file
cmake_minimum_required(VERSION 3.20)
project(MyTrilinos)
# change Trilinos_ROOT to your local trilinos path
set(Trilinos_ROOT "path/to/Trilinos")
set(CMAKE_PREFIX_PATH ${Trilinos_ROOT}/lib/cmake/Trilinos)
set(CMAKE_CXX_COMPILER ${Trilinos_ROOT}/bin/nvcc_wrapper)
# change executable name 
set(TARGET "main")

file(GLOB_RECURSE TARGET_SRC  "src/*.cpp")
file(GLOB_RECURSE TARGET_INC  "include/*.hpp")
set(TARGET_SRC ${TARGET_SRC}  ${TARGET_INC})

set(CMAKE_EXE_LINKER_FLAGS "-fopenmp -lmpi")

find_package(Trilinos REQUIRED)
find_package(MPI REQUIRED)

add_executable(${TARGET} ${TARGET_SRC})

target_link_libraries(${TARGET} Trilinos::all_libs)
target_link_libraries(${TARGET} MPI::all_libs)