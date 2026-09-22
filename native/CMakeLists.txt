cmake_minimum_required(VERSION 3.15)
project(frames2py_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find pybind11 -- try cmake config first, then fall back to pip install.
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    find_package(pybind11 CONFIG REQUIRED PATHS ${pybind11_DIR})
endif()

pybind11_add_module(_frames2py_native
    src/module.cpp
    src/kernels/event_count.cpp
    src/kernels/polarity.cpp
    src/kernels/time_surface.cpp
)

target_include_directories(_frames2py_native PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Enable optimisations for release builds.
target_compile_options(_frames2py_native PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native>
)

# Install into the Python package.
install(TARGETS _frames2py_native
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
)
