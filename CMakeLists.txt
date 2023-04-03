cmake_minimum_required(VERSION 3.10)
project(2DShallowWater)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3")

find_package(MPI REQUIRED)
find_package(BLAS REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)

include_directories(${MPI_INCLUDE_PATH} ${Boost_INCLUDE_DIRS})

add_executable(test1 main.cpp Comm.cpp ShallowWater.cpp)
add_executable(test2 main.cpp Comm.cpp ShallowWater.cpp)
add_executable(test3 main.cpp Comm.cpp ShallowWater.cpp)
add_executable(test4 main.cpp Comm.cpp ShallowWater.cpp)

target_compile_definitions(test1 PRIVATE DEFAULT_DT=0.1 DEFAULT_T=80 DEFAULT_NX=100 DEFAULT_NY=100 DEFAULT_IC=1 DEFAULT_METHOD='l')
target_compile_definitions(test2 PRIVATE DEFAULT_DT=0.1 DEFAULT_T=80 DEFAULT_NX=100 DEFAULT_NY=100 DEFAULT_IC=2 DEFAULT_METHOD='b')
target_compile_definitions(test3 PRIVATE DEFAULT_DT=0.1 DEFAULT_T=80 DEFAULT_NX=100 DEFAULT_NY=100 DEFAULT_IC=3 DEFAULT_METHOD='l')
target_compile_definitions(test4 PRIVATE DEFAULT_DT=0.1 DEFAULT_T=80 DEFAULT_NX=100 DEFAULT_NY=100 DEFAULT_IC=4 DEFAULT_METHOD='b')



target_link_libraries(test1 ${MPI_LIBRARIES} ${BLAS_LIBRARIES} ${Boost_LIBRARIES})
target_link_libraries(test2 ${MPI_LIBRARIES} ${BLAS_LIBRARIES} ${Boost_LIBRARIES})
target_link_libraries(test3 ${MPI_LIBRARIES} ${BLAS_LIBRARIES} ${Boost_LIBRARIES})
target_link_libraries(test4 ${MPI_LIBRARIES} ${BLAS_LIBRARIES} ${Boost_LIBRARIES})

if(MPI_COMPILE_FLAGS)
  set_target_properties(test1 PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
  set_target_properties(test2 PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
  set_target_properties(test3 PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
  set_target_properties(test4 PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(test1 PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
  set_target_properties(test2 PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
  set_target_properties(test3 PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
  set_target_properties(test4 PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()
