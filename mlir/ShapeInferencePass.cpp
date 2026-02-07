//===- ShapeInferencePass.cpp - Shape Inference ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a Function level pass performing interprocedural
// propagation of array shapes through function specialization.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "toy/Dialect.h"
#include "toy/Passes.h"
#include "toy/ShapeInferenceInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

#define DEBUG_TYPE "shape-inference"

using namespace mlir;
using namespace toy;

/// Include the auto-generated definitions for the shape inference interfaces.
#include "toy/ShapeInferenceOpInterfaces.cpp.inc"

namespace {

/// The ShapeInferencePass is a pass that performs intra-procedural
/// shape inference.
///
/// Algorithm:
///  1) Build a worklist containing all the operations that return an *unranked*
///     tensor. These are the operations that need shape inference.
///  2) Iterate on the worklist:
///     a) pick the next operation whose operands are already inferred,
///     b) remove it from the worklist,
///     c) if it implements the ShapeInference interface, invoke inferShapes().
///        Otherwise, error (because it still produces an unranked result).
///  3) If the worklist is empty, the algorithm succeeded.
struct ShapeInferencePass
    : public mlir::PassWrapper<ShapeInferencePass,
                               OperationPass<toy::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShapeInferencePass)
  StringRef getArgument() const override { return "toy-shape-inference"; }

  void runOnOperation() override {
    auto f = getOperation();

    // Populate the worklist with operations that truly need inference:
    // those producing UnrankedTensorType results.
    llvm::SmallPtrSet<mlir::Operation *, 16> opWorklist;
    f.walk([&](mlir::Operation *op) {
      if (returnsDynamicShape(op))
        opWorklist.insert(op);
    });

    // Iterate until fixpoint or we get stuck.
    while (!opWorklist.empty()) {
      // Find an op whose operands are all inferred (i.e., ranked tensors).
      auto nextop = llvm::find_if(opWorklist, allOperandsInferred);
      if (nextop == opWorklist.end())
        break;

      Operation *op = *nextop;
      opWorklist.erase(op);

      LDBG() << "Inferring shape for: " << *op;

      // If the op implements the interface, ask it to infer shapes.
      if (auto shapeOp = dyn_cast<mlir::toy::ShapeInference>(op)) {
        shapeOp.inferShapes();
        continue; // IMPORTANT: do NOT return; keep processing the worklist
      }

      // Otherwise, only fail if it still produces unranked tensor results.
      bool hasUnrankedResult = false;
      for (mlir::Type t : op->getResultTypes()) {
        if (llvm::isa<mlir::UnrankedTensorType>(t)) {
          hasUnrankedResult = true;
          break;
        }
      }

      if (hasUnrankedResult) {
        op->emitError("unable to infer shape of operation without shape "
                      "inference interface");
        signalPassFailure();
        return;
      }

      // No unranked results => nothing to infer => keep going.
    }

    // If the worklist isn't empty, we got stuck (cycle or missing inference).
    if (!opWorklist.empty()) {
      f.emitError("Shape inference failed, ")
          << opWorklist.size()
          << " operations couldn't be inferred\n";
      signalPassFailure();
      return;
    }
  }

  /// Return true if the given op has all tensor operands inferred (ranked).
  /// Non-tensor operands are ignored.
  static bool allOperandsInferred(Operation *op) {
    for (Type operandType : op->getOperandTypes()) {
      if (llvm::isa<TensorType>(operandType) &&
          !llvm::isa<RankedTensorType>(operandType))
        return false;
    }
    return true;
  }

  /// Return true if the given op produces *unranked* tensor results.
  /// Ranked tensors with dynamic dims (e.g., tensor<?x?xf64>) are considered
  /// "inferred enough" for this pass.
  static bool returnsDynamicShape(Operation *op) {
    return llvm::any_of(op->getResultTypes(), [](Type resultType) {
      return llvm::isa<UnrankedTensorType>(resultType);
    });
  }
};

} // namespace

/// Create a Shape Inference pass.
std::unique_ptr<mlir::Pass> mlir::toy::createShapeInferencePass() {
  return std::make_unique<ShapeInferencePass>();
}
