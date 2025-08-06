# Minimum required CMake version
cmake_minimum_required(VERSION 3.10)

# Project name
project(NeuralNet)

# # Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(Eigen3_DIR "include/eigen-3.4.0/cmake")
# Find Eigen3
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
# include(FetchContent)
# FetchContent_Declare(
#   eigen
#   GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
#   GIT_TAG        3.4.0
# )
# FetchContent_MakeAvailable(eigen)

# Add the executable
add_executable(main src/main.cpp)

# Link Eigen to the project
target_link_libraries(main Eigen3::Eigen)

# Include Eigen headers
target_include_directories(main PRIVATE ${Eigen3_INCLUDE_DIRS})