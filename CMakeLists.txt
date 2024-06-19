cmake_minimum_required(VERSION 3.0.0)
project(multicore_final_project VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(CUDA REQUIRED)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

CUDA_ADD_EXECUTABLE(multicore_final_project main_cuda.cu)

target_link_libraries( multicore_final_project ${OpenCV_LIBS} )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
set(CMAKE_BUILD_TYPE Debug)

include(CPack)