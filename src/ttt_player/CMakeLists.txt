cmake_minimum_required(VERSION 3.8)
project(ttt_player)

# -------------------------- 全局编译配置 --------------------------
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ROS 2 通常要求 C++17，确保与 PyTorch 编译标准一致
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -------------------------- ROS 2 依赖 --------------------------
find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(ttt_interfaces REQUIRED)
find_package(ament_index_cpp REQUIRED)

# -------------------------- PyTorch 配置（可选） --------------------------
option(TTT_PLAYER_WITH_TORCH "Build with PyTorch (libtorch) support" ON)

set(TTT_PLAYER_HAS_TORCH OFF)
if(TTT_PLAYER_WITH_TORCH)
  find_package(Python3 QUIET COMPONENTS Interpreter)

  set(CMAKE_DISABLE_FIND_PACKAGE_CUDA ON CACHE BOOL "Completely disable CUDA find_package" FORCE)
  set(USE_CUDA OFF CACHE BOOL "Disable CUDA" FORCE)
  set(USE_CUDNN OFF CACHE BOOL "Disable cuDNN" FORCE)
  set(CMAKE_CUDA_COMPILER "NOTFOUND" CACHE STRING "Force no CUDA compiler" FORCE)
  set(TORCH_CUDA_ARCH_LIST "" CACHE STRING "Clear CUDA arch list" FORCE)
  unset(CUDA_TOOLKIT_ROOT_DIR CACHE)

  if(Python3_Interpreter_FOUND)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path, end='')"
      OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
      RESULT_VARIABLE TORCH_PATH_RESULT
    )
    if(TORCH_PATH_RESULT EQUAL 0 AND NOT "${TORCH_CMAKE_PREFIX_PATH}" STREQUAL "")
      list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})
    endif()
  endif()

  find_package(Torch QUIET)
  if(Torch_FOUND)
    set(TTT_PLAYER_HAS_TORCH ON)
  else()
    message(WARNING "PyTorch (Torch) not found. Building ttt_player without torch support.")
  endif()
endif()

# -------------------------- 目标构建 --------------------------
add_executable(student_player_node src/student_player_node.cpp)

# 链接 ROS 2 依赖
ament_target_dependencies(student_player_node
  rclcpp
  geometry_msgs
  moveit_msgs
  sensor_msgs
  trajectory_msgs
  ttt_interfaces
  ament_index_cpp
)

if(TTT_PLAYER_HAS_TORCH)
  target_compile_definitions(student_player_node PRIVATE TTT_PLAYER_HAS_TORCH=1)

  if(DEFINED TORCH_CXX_FLAGS AND NOT "${TORCH_CXX_FLAGS}" STREQUAL "")
    separate_arguments(TORCH_CXX_FLAGS_LIST UNIX_COMMAND "${TORCH_CXX_FLAGS}")
    target_compile_options(student_player_node PRIVATE ${TORCH_CXX_FLAGS_LIST})
  endif()

  if(TARGET Torch::Torch)
    target_link_libraries(student_player_node Torch::Torch)
  else()
    if(DEFINED TORCH_INCLUDE_DIRS)
      target_include_directories(student_player_node PRIVATE ${TORCH_INCLUDE_DIRS})
    endif()
    target_link_libraries(student_player_node ${TORCH_LIBRARIES})
  endif()
endif()

if(TTT_PLAYER_HAS_TORCH AND DEFINED TORCH_INSTALL_PREFIX)
  set_target_properties(student_player_node PROPERTIES
    BUILD_WITH_INSTALL_RPATH ON
    BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  )
endif()

# -------------------------- macOS 专属链接修复 --------------------------
# 解决 macOS 上动态库 (@rpath) 找不到的问题
if(APPLE)
  if(TTT_PLAYER_HAS_TORCH AND DEFINED TORCH_INSTALL_PREFIX)
    set_target_properties(student_player_node PROPERTIES
      MACOSX_RPATH ON
      BUILD_WITH_INSTALL_RPATH ON
      INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    )
  endif()
endif()

# -------------------------- 安装规则 --------------------------
install(TARGETS
  student_player_node
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES
  ttt_rl_model.pt
  DESTINATION share/${PROJECT_NAME}/resource
)

ament_package()
