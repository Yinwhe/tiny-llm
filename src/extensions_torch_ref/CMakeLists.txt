cmake_minimum_required(VERSION 3.27)

project(_ext LANGUAGES CXX)

# ----------------------------- Setup -----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build extensions as a shared library" ON)
include(CheckLanguage)

# ----------------------------- Dependencies -----------------------------
find_package(
  Python 3.8
  COMPONENTS Interpreter Development.Module
  REQUIRED)

find_package(Torch REQUIRED)

find_library(
  TORCH_PYTHON_LIBRARY
  torch_python
  PATHS ${TORCH_INSTALL_PREFIX}/lib
  NO_DEFAULT_PATH
  REQUIRED)

if(TORCH_CXX_FLAGS)
  separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
endif()

# ----------------------------- Extensions -----------------------------

# Add library
add_library(tiny_llm_ext_torch_ref)

# Add sources
target_sources(
  tiny_llm_ext_torch_ref
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/src/quantized_matmul.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/utils.cpp
)

# Add include headers
target_include_directories(
  tiny_llm_ext_torch_ref
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}
  ${CMAKE_CURRENT_LIST_DIR}/src
  ${Python_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
)

# Link to torch
target_link_libraries(tiny_llm_ext_torch_ref PUBLIC ${TORCH_LIBRARIES})
target_compile_options(tiny_llm_ext_torch_ref PRIVATE ${TORCH_CXX_FLAGS_LIST})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # so that clangd can pick this up

# ----------------------------- CUDA -----------------------------
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  target_sources(
    tiny_llm_ext_torch_ref
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/src/quantized_matmul.cu
  )

  set_target_properties(tiny_llm_ext_torch_ref PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_compile_definitions(tiny_llm_ext_torch_ref PUBLIC _CUDA_)
endif()

# ----------------------------- Python Bindings -----------------------------
Python_add_library(_ext MODULE WITH_SOABI ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp)
target_include_directories(_ext PRIVATE ${Python_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
target_link_libraries(
  _ext
  PRIVATE
  Python::Module
  tiny_llm_ext_torch_ref
  ${TORCH_LIBRARIES}
  ${TORCH_PYTHON_LIBRARY}
)
target_compile_definitions(_ext PRIVATE TORCH_EXTENSION_NAME=_ext)
target_compile_options(_ext PRIVATE ${TORCH_CXX_FLAGS_LIST})

if(DEFINED TORCH_INSTALL_PREFIX)
  set_target_properties(_ext PROPERTIES BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib")
endif()

if(BUILD_SHARED_LIBS)
  target_link_options(_ext PRIVATE -Wl,-rpath,$ORIGIN)
endif()
