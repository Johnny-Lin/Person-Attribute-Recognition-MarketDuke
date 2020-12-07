cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(main)
set(CMAKE_PREFIX_PATH "./libtorch")

find_package(Torch REQUIRED)
find_package(OpenCV QUIET)

add_executable(${PROJECT_NAME} "main.cpp")
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)
