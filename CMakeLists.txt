cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 11)

project(orb_streamer)

find_package(PkgConfig)
pkg_check_modules(PC_ZeroMQ QUIET zmq)

find_path(ZeroMQ_INCLUDE_DIR
	NAMES zmq.hpp
	PATHS ${PC_ZeroMQ_INCLUDE_DIRS}
	)

find_library(ZeroMQ_LIBRARY
	NAMES zmq
	PATHS ${PC_ZeroMQ_LIBRARY_DIRS}
	)

find_package(OpenCV REQUIRED)
find_package(Threads)

include_directories(
	${PROJECT_SOURCE_DIR}
	${PROJECT_SOURCE_DIR}/include
)

# add_library(${PROJECT_NAME} SHARED
# src/bufferedORBNetStream.cpp
# src/Benchmark.cpp
# include/bufferedORBNetStream.h
# include/Benchmark.h
# )

add_executable(${PROJECT_NAME} main.cpp src/bufferedORBNetStream.cpp src/Benchmark.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC ${ZeroMQ_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} PUBLIC ${ZeroMQ_LIBRARY} opencv_cudaimgproc opencv_cudafeatures2d opencv_imgproc opencv_core opencv_imgcodecs opencv_features2d opencv_videoio ${CMAKE_THREAD_LIBS_INIT})
