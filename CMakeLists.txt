cmake_minimum_required(VERSION 2.8)
project(DynaProp)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
  # SET(CMAKE_BUILD_TYPE Debug)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3  ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 ")

# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O0 -march=native ")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O0 -march=native")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# change it to your dir where opencv is built
set(OpenCV_DIR ~/lib_sources/opencv-2.4.11/release)

message("OpenCV_DIR: " ${OpenCV_DIR})
find_package(OpenCV 2.4.11 QUIET)
if(NOT OpenCV_FOUND)
    message("OpenCV > 2.4.11 not found.")
    find_package(OpenCV 3.0 QUIET)
    if(NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV > 3.0 not found.")
    endif()
endif()

# find_package(OpenCV 3.0 QUIET)
#if(NOT OpenCV_FOUND)
#   find_package(OpenCV 2.4.3 QUIET)
#   if(NOT OpenCV_FOUND)
#      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
#   endif()
#endif()

find_package(Qt5Widgets REQUIRED)
find_package(Qt5Concurrent REQUIRED)
find_package(Qt5OpenGL REQUIRED)
find_package(Qt5Test REQUIRED)

# change to your own PYTHON dir
#set( PYTHON_INCLUDE_DIR /usr/include/python2.7/ )
#set( PYTHON_LIBRARY /usr/lib/x86_64-linux-gnu/libpython2.7.so)
set( PYTHON_INCLUDE_DIR /home/yxz/anaconda3/envs/dyna/include/python2.7 )
set( PYTHON_LIBRARY /home/yxz/anaconda3/envs/dyna/lib/libpython2.7.so)
#message("PYTHON LIBS DIRECTORY: " ${PYTHON_LIBRARY})
#find_package(PythonLibs REQUIRED)
#if (NOT PythonLibs_FOUND)
#    message(FATAL_ERROR "PYTHON LIBS not found.")
#else()
#    message("PYTHON LIBS were found!")
#    message("PYTHON LIBS DIRECTORY: " ${PYTHON_LIBRARY})
#endif()

find_package(Boost REQUIRED COMPONENTS thread)
if(Boost_FOUND)
    message("Boost was found!")
    message("Boost Headers DIRECTORY: " ${Boost_INCLUDE_DIRS})
    message("Boost LIBS DIRECTORY: " ${Boost_LIBRARY_DIRS})
    message("Found Libraries: " ${Boost_LIBRARIES})
endif()

find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)

# set(PYTHON_INCLUDE_DIRS  ~/anaconda3/envs/dyna/lib/python2.7/site-packages/numpy/core/include/numpy)
# set(PYTHON_INCLUDE_DIRS  /usr/lib/python2.7/dist-packages/numpy/core/include/numpy)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${PYTHON_INCLUDE_DIR}
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
${Boost_INCLUDE_DIRS}
)
include_directories(/home/yxz/anaconda3/envs/dyna/include/python2.7)
include_directories(/home/yxz/anaconda3/envs/dyna/lib/python2.7/site-packages/numpy/core/include/numpy)
include_directories(/home/yxz/anaconda3/envs/dyna/lib/python2.7/site-packages)
# /usr/include/python2.7/
# /usr/lib/python2.7/dist-packages/numpy/core/include/numpy/
# ~/anaconda3/envs/dyna/include/python2.7
# ~/anaconda3/envs/dyna/lib/python2.7/site-packages/numpy/core/include/numpy
# ~/anaconda3/envs/dyna/lib/python2.7/site-packages
message("PROJECT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
src/System.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/PnPsolver.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Initializer.cc
src/Viewer.cc
src/Conversion.cc
src/MaskNet.cc
src/Geometry.cc
src/MaskPropagation.cc
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
~/anaconda3/envs/dyna/lib/libpython2.7.so
${Boost_LIBRARIES}
)

# Build examples

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)

add_executable(rgbd_tum
Examples/RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum ${PROJECT_NAME})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

