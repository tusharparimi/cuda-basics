cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(rgb2gray LANGUAGES CXX CUDA)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} rgb2gray.cu)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

