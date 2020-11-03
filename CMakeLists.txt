project(trtwrapper_proj)
cmake_minimum_required(VERSION 3.5.0)

set(TARGET_NAME "trtwrapper")
set(EXEC_NAME "inference_demo")

add_definitions(-DTFWRAPPER_LIBRARY)
#add_definitions(-DTFDEBUG)

# list(APPEND CMAKE_PREFIX_PATH "/opt/tensorflow_cc/lib/cmake")
# find_package(TensorflowCC COMPONENTS Shared REQUIRED)
# find_package(TensorRT)

set(SOURCE_NAMES
        trt_segmentation.cpp
        trt_classification_inferencer.cpp
        trt_cnn_inferencer.cpp
        trt_detection_inferencer.cpp
        logger.cpp
        main.cpp)

set(HEADERS_NAMES
        trt_segmentation.h
        trt_buffers.h
        trt_half.h
        trt_common.h
        trt_classification_inferencer.h
        trt_cnn_inferencer.h
        trt_detection_inferencer.h
        logger.h
        logging.h)


foreach (SOURCE ${SOURCE_NAMES})
    set(SOURCES ${SOURCES} ${SOURCE})
endforeach (SOURCE)

foreach (HEADER ${HEADERS_NAMES})
    set(HEADERS ${HEADERS} ${PROJECT_SOURCE_DIR}/${HEADER})
    #   set(HEADERS ${HEADERS} ${PROJECT_SOURCE_DIR}/include/${TARGET_NAME}/${HEADER})
    #   set(HEADERS ${HEADERS} /media/xavier_home/pmvl/catkin_ws/src/algorithms/include/${TARGET_NAME}/${HEADER})
endforeach (HEADER)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
#set(CMAKE_CXX_FLAGS “${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-parameter -Werror -Wno-deprecated-declarations”)

include_directories(${TARGET_NAME}
        ${PROJECT_SOURCE_DIR}/include/)
include_directories(${EXEC_NAME}
        ${PROJECT_SOURCE_DIR}/include/)
#include_directories(/media/xavier_home/pmvl/catkin_ws/src/algorithms/include)

add_library(${TARGET_NAME} STATIC ${SOURCES} ${HEADERS})
add_executable(${EXEC_NAME} ${SOURCES} ${HEADERS})

message("TRT libs:" ${TensorRT_LIBRARIES})

target_link_libraries(${TARGET_NAME}
        PUBLIC
        ${TensorRT_LIBRARIES} ${OpenCV_LIBS} ${CUDA_LIBRARIES} nvonnxparser
        )

target_link_libraries(${EXEC_NAME}
        PUBLIC
        ${TensorRT_LIBRARIES} ${OpenCV_LIBS} ${CUDA_LIBRARIES} nvonnxparser nvinfer nvinfer_plugin nvparsers
        )
#set_property(${TARGET_NAME} trtwrapper_proj PROPERTY CXX_STANDARD 14)
target_include_directories(${TARGET_NAME}
        PUBLIC
        ${PROJECT_SOURCE_DIR}/include/${TARGET_NAME}
        )
target_include_directories(${EXEC_NAME}
        PUBLIC
        ${PROJECT_SOURCE_DIR}/include/${TARGET_NAME}
        )


