cmake_minimum_required(VERSION 3.10)
project(image_preprocessor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # Let clangd resolve includes

# 1. OpenCV Requirement
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 2. Triton Backend SDK (Include Third Party)
set(TRITON_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party")
set(TRITON_BACKEND_DIR "${TRITON_ROOT}/backend")
set(TRITON_COMMON_DIR  "${TRITON_ROOT}/common")
set(TRITON_CORE_DIR    "${TRITON_ROOT}/core")

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    "${TRITON_BACKEND_DIR}/include"
    "${TRITON_COMMON_DIR}/include"
    "${TRITON_CORE_DIR}/include"
)

# rapidjson: prefer third_party/, fall back to system path (e.g. apt rapidjson-dev)
if(EXISTS "${TRITON_ROOT}/rapidjson/include")
    include_directories("${TRITON_ROOT}/rapidjson/include")
else()
    message(STATUS "rapidjson not in third_party, assuming system install (apt rapidjson-dev)")
endif()

# 3. Source Files
set(SOURCES 
    src/model.cc
    "${TRITON_BACKEND_DIR}/src/backend_model.cc"
    "${TRITON_BACKEND_DIR}/src/backend_model_instance.cc"
    "${TRITON_BACKEND_DIR}/src/backend_common.cc" # Also recommended
)

# 4. Build as Shared Library (for Triton Backend usage)
add_library(triton_image_preprocessor SHARED ${SOURCES})
target_link_libraries(triton_image_preprocessor PRIVATE ${OpenCV_LIBS})

# 5. Output Location
set_target_properties(triton_image_preprocessor PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    PREFIX ""
)

message(STATUS "OpenCV found in: ${OpenCV_DIR}")
message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")
