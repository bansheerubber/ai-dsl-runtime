cmake_minimum_required(VERSION 3.5)

project(ai_dsl_runtime)

set(Torch_DIR ./libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_subdirectory(src)

add_executable(ai_dsl_runtime ${SRC_DSL} ${SRC_API} ${SRC_PPO})
target_link_libraries(ai_dsl_runtime "${TORCH_LIBRARIES}")
target_compile_options(ai_dsl_runtime PUBLIC -O2 -Wall -std=c++20 -Werror=return-type)

add_library(airt SHARED ${SRC_DSL} ${SRC_API} ${SRC_PPO})
target_link_libraries(airt "${TORCH_LIBRARIES}")
target_compile_options(airt PUBLIC -O2 -Wall -std=c++20 -Werror=return-type)

# message(STATUS ${SRC_API})
