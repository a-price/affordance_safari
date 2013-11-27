cmake_minimum_required(VERSION 2.8.3)
project(affordance_safari)

SET(CMAKE_BUILD_TYPE "Release")
#SET(CMAKE_BUILD_TYPE "Debug")

# Add additional CMake files
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

## System dependencies are found with CMake's conventions
IF(WIN32)
	set(OpenCV_DIR "C:/OpenLibraries/opencv/build")#/x64/vc11/lib")
	set(OpenCV_INCLUDE_DIR "C:/OpenLibraries/opencv/build/include")
	set(BUILD_FLAGS "-std=c++0x")
ELSE(WIN32)
	set(BUILD_FLAGS "-std=c++0x -Wreturn-type")
ENDIF(WIN32)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})
link_libraries(${OpenCV_LIBS})

find_package(CppUnit)
if(${CppUnit})
	include_directories(${CPPUNIT_INCLUDE_DIR})
	link_libraries(${CPPUNIT_LIBRARIES})
endif(${CppUnit})

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include)

## Declare a cpp library
add_executable(safari src/affordance_safari.cpp)
set_target_properties(safari PROPERTIES COMPILE_FLAGS ${BUILD_FLAGS})
