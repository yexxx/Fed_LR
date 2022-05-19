cmake_minimum_required(VERSION 3.1)
project("Logistic Regression")
set(CMAKE_CXX_STANDARD 11)

find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

find_package(Boost COMPONENTS system filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

include_directories(${OPENMP_INCLUDES})

link_directories(${OPENMP_LIBRARIES})

add_executable(fl_seal_line fl_seal_line.cpp)

find_package (SEAL)
target_link_libraries(fl_seal_line SEAL::seal)
target_link_libraries(fl_seal_line ${Boost_LIBRARIES})