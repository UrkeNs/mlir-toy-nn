# Ops and Dialect TableGen for Toy Ch7

# --------------------------------------------------
# Ops + Dialect
# --------------------------------------------------
set(LLVM_TARGET_DEFINITIONS Ops.td)

# Ops.h.inc: op declarations
mlir_tablegen(Ops.h.inc -gen-op-decls
  -I ${CMAKE_CURRENT_SOURCE_DIR}
  -I ${CMAKE_CURRENT_SOURCE_DIR}/..
  -I ${MLIR_SRC_DIR}/include
)

# Ops.cpp.inc: op definitions
mlir_tablegen(Ops.cpp.inc -gen-op-defs
  -I ${CMAKE_CURRENT_SOURCE_DIR}
  -I ${CMAKE_CURRENT_SOURCE_DIR}/..
  -I ${MLIR_SRC_DIR}/include
)

# Dialect.h.inc: dialect declarations
mlir_tablegen(Dialect.h.inc -gen-dialect-decls
  -I ${CMAKE_CURRENT_SOURCE_DIR}
  -I ${CMAKE_CURRENT_SOURCE_DIR}/..
  -I ${MLIR_SRC_DIR}/include
)

# Dialect.cpp.inc: dialect definitions
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs
  -I ${CMAKE_CURRENT_SOURCE_DIR}
  -I ${CMAKE_CURRENT_SOURCE_DIR}/..
  -I ${MLIR_SRC_DIR}/include
)

add_public_tablegen_target(ToyCh7OpsIncGen)

# --------------------------------------------------
# Shape inference interface
# This must match ShapeInferenceInterface.h includes:
#   "toy/ShapeInferenceOpInterfaces.h.inc"
# --------------------------------------------------
set(LLVM_TARGET_DEFINITIONS ShapeInferenceInterface.td)

mlir_tablegen(ShapeInferenceOpInterfaces.h.inc -gen-op-interface-decls
  -I ${CMAKE_CURRENT_SOURCE_DIR}
  -I ${CMAKE_CURRENT_SOURCE_DIR}/..
  -I ${MLIR_SRC_DIR}/include
)

mlir_tablegen(ShapeInferenceOpInterfaces.cpp.inc -gen-op-interface-defs
  -I ${CMAKE_CURRENT_SOURCE_DIR}
  -I ${CMAKE_CURRENT_SOURCE_DIR}/..
  -I ${MLIR_SRC_DIR}/include
)

add_public_tablegen_target(ToyCh7ShapeInferenceInterfaceIncGen)

