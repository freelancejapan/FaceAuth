cmake_minimum_required(VERSION 3.15)
project(FaceAuth)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)
set(USE_AVX_INSTRUCTIONS 1)
set(USE_SSE4_INSTRUCTIONS 1)

add_subdirectory(/home/hjd/Documents/Tools/dlib-master dlib_build)

find_package(realsense2 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(FaceAuth main.cpp cv-helpers.hpp filter.hpp)
add_executable(FaceInfoMgr faceInfoMgr.cpp cv-helpers.hpp filter.hpp )

target_link_libraries(FaceAuth dlib::dlib)
target_link_libraries(FaceAuth ${OpenCV_LIBS} )
target_link_libraries(FaceAuth ${realsense2_LIBRARY})

target_link_libraries(FaceInfoMgr dlib::dlib)
target_link_libraries(FaceInfoMgr ${OpenCV_LIBS} )
target_link_libraries(FaceInfoMgr ${realsense2_LIBRARY})