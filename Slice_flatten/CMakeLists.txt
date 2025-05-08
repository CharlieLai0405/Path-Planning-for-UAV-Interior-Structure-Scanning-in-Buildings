cmake_minimum_required(VERSION 3.5)
project(uniform_sampling_example)

# 找到 PCL 套件
find_package(PCL 1.8 REQUIRED)

# 包含 PCL 的目錄
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# 創建執行檔
add_executable(slice slice.cpp)

# 將 PCL 庫鏈接到執行檔
target_link_libraries(slice ${PCL_LIBRARIES})
