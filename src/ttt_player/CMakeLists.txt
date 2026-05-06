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

# -------------------------- PyTorch 配置 (macOS 专属) --------------------------
# 1. 强制禁用 CUDA (放在最前面，彻底阻断查找)
set(CMAKE_DISABLE_FIND_PACKAGE_CUDA ON CACHE BOOL "Completely disable CUDA find_package" FORCE)
set(USE_CUDA OFF CACHE BOOL "Disable CUDA" FORCE)
set(USE_CUDNN OFF CACHE BOOL "Disable cuDNN" FORCE)
set(CMAKE_CUDA_COMPILER "NOTFOUND" CACHE STRING "Force no CUDA compiler" FORCE)
set(TORCH_CUDA_ARCH_LIST "" CACHE STRING "Clear CUDA arch list" FORCE)
# 防止任何 CUDA 相关变量被缓存
unset(CUDA_TOOLKIT_ROOT_DIR CACHE)

# 2. 安全获取 PyTorch CMake 路径 (增加错误检查)
execute_process(
    COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path, end='')"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    RESULT_VARIABLE TORCH_PATH_RESULT
)
if(NOT TORCH_PATH_RESULT EQUAL 0)
    message(FATAL_ERROR "无法获取 PyTorch 路径！请确认已通过 pip3 install torch 安装正确版本。")
endif()
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})

# 3. 查找 PyTorch (QUIET 模式减少无关日志)
find_package(Torch REQUIRED QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

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

# 链接 PyTorch 库
target_link_libraries(student_player_node "${TORCH_LIBRARIES}")

# -------------------------- macOS 专属链接修复 --------------------------
# 解决 macOS 上动态库 (@rpath) 找不到的问题
if(APPLE)
    set_target_properties(student_player_node PROPERTIES
        MACOSX_RPATH ON
        BUILD_WITH_INSTALL_RPATH ON
        INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    )
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