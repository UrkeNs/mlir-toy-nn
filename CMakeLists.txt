cmake_minimum_required(VERSION 3.20)
project(toy_nn_ch7 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# These will be given from the command line:
#   -DMLIR_DIR=/home/uros/Desktop/llvm-project/build-clang/lib/cmake/mlir
#   -DLLVM_DIR=/home/uros/Desktop/llvm-project/build-clang/lib/cmake/llvm
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Found LLVM in: ${LLVM_DIR}")
message(STATUS "Found MLIR in: ${MLIR_DIR}")

# Make LLVM/MLIR CMake modules (AddMLIR, AddLLVM, TableGen, etc.) visible
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

include(AddLLVM)
include(AddMLIR)
include(TableGen)
include(HandleLLVMOptions)

# ------------------------------------------------------------
# Tell CMake where the MLIR *source* tree is, for .td includes.
# ------------------------------------------------------------
set(MLIR_SRC_DIR "/home/uros/Desktop/llvm-project/mlir")

# ------------------------------------------------------------
# Use exported include dirs from LLVM/MLIR (source + *build*).
# This is what gives you build-clang/include/llvm/Config/...
# ------------------------------------------------------------
include_directories(SYSTEM
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)

# ============================================================
# Original chapter logic (slightly tweaked to be standalone)
# ============================================================

# This chapter depends on JIT support enabled.
if(NOT MLIR_ENABLE_EXECUTION_ENGINE)
  message(FATAL_ERROR "MLIR built without execution engine; Ch7 requires it")
endif()

include_directories(include)
add_subdirectory(include)

set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  OrcJIT
  )

# TableGen for ToyCombine.td: add include dirs so .td includes resolve.
set(LLVM_TARGET_DEFINITIONS mlir/ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters
  -I ${CMAKE_CURRENT_SOURCE_DIR}
  -I ${CMAKE_CURRENT_SOURCE_DIR}/mlir
  -I ${CMAKE_CURRENT_SOURCE_DIR}/include
  -I ${MLIR_SRC_DIR}/include
)
add_public_tablegen_target(ToyCh7CombineIncGen)


# We define toyc-ch7 explicitly instead of using add_toy_chapter.
add_executable(toyc-ch7
  toyc.cpp
  parser/AST.cpp
  mlir/MLIRGen.cpp
  mlir/Dialect.cpp 
  mlir/LowerToAffineLoops.cpp
  mlir/LowerToLLVM.cpp
  mlir/LowerToyStructToLLVM.cpp
  mlir/ShapeInferencePass.cpp
  mlir/ToyCombine.cpp
)

add_dependencies(toyc-ch7
  ToyCh7ShapeInferenceInterfaceIncGen
  ToyCh7OpsIncGen
  ToyCh7CombineIncGen
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)

target_link_libraries(toyc-ch7
  PRIVATE
    MLIRAnalysis
    MLIRBuiltinToLLVMIRTranslation
    MLIRCallInterfaces
    MLIRCastInterfaces
    MLIRExecutionEngine
    MLIRFunctionInterfaces
    MLIRIR
    MLIRLLVMCommonConversion
    MLIRLLVMToLLVMIRTranslation
    MLIRMemRefDialect
    MLIRParser
    MLIRPass
    MLIRRegisterAllDialects
    MLIRRegisterAllExtensions
    MLIRRegisterAllPasses
    MLIRSideEffectInterfaces
    MLIRTargetLLVMIRExport
    MLIRTransforms
    MLIRArithDialect      
    MLIRMathDialect 
)

