cmake_minimum_required(VERSION 3.15)
project(PlasmaLeakageMC)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# PYTHON & PYBIND11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# FILE SOURCE C++
set(CORE_SOURCES
    plasma_leakage.cpp
)

# File Binding
set(BINDING_SOURCES
    bindings.cpp
)

# CREATE A PYTHON MODULE
pybind11_add_module(plasma_leakage
    ${BINDING_SOURCES}
    ${CORE_SOURCES}
)

# CONFIGURATION INCLUDE FOLDER
target_include_directories(plasma_leakage PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Ensure the Python module has no library prefix on Unix-like systems
set_target_properties(plasma_leakage PROPERTIES
    PREFIX ""
)

# Install targets to the Python site-packages directory
install(TARGETS plasma_leakage
    LIBRARY DESTINATION ${Python3_SITEARCH}
    RUNTIME DESTINATION ${Python3_SITEARCH}
)