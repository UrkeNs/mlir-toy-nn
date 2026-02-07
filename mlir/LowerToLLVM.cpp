//====- LowerToLLVM.cpp - Lowering from Toy+Affine+Std to LLVM ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements full lowering of Toy operations to LLVM MLIR dialect.
// 'toy.print' is lowered to a loop nest that calls `printf` on each element of
// the input array. The file also sets up the ToyToLLVMLoweringPass. This pass
// lowers the combination of Arithmetic + Affine + SCF + Func dialects to the
// LLVM one:
//
//                         Affine --
//                                  |
//                                  v
//                       Arithmetic + Func --> LLVM (Dialect)
//                                  ^
//                                  |
//     'toy.print' --> Loop (SCF) --
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "toy/Dialect.h"
#include "toy/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include <memory>
#include <utility>

using namespace mlir;

//===----------------------------------------------------------------------===//
// ToyToLLVM Conversion Patterns
//===----------------------------------------------------------------------===//

namespace {

  static Value gepToElement(Location loc,
    ConversionPatternRewriter &rewriter,
    MemRefDescriptor desc,
    MemRefType memRefType,
    Value i,              // index
    Value j = Value()) {  // optional index
      MLIRContext *ctx = rewriter.getContext();

      // LLVM types
      auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(ctx);
      auto llvmI64Ty = rewriter.getI64Type();
      auto llvmF64Ty = rewriter.getF64Type(); // This becomes LLVM double later

      // Base pointer (aligned)
      Value basePtr = desc.alignedPtr(rewriter, loc);

      // offset is i64 (in elements)
      Value linear = desc.offset(rewriter, loc); // i64

      // Cast induction vars (index -> i64)
      Value i64 = rewriter.create<arith::IndexCastOp>(loc, llvmI64Ty, i);

      // linear += i * stride(0)
      Value s0 = desc.stride(rewriter, loc, 0);        // i64
      Value term0 = rewriter.create<arith::MulIOp>(loc, i64, s0);
      linear = rewriter.create<arith::AddIOp>(loc, linear, term0);

      // rank-2: linear += j * stride(1)
      if (memRefType.getRank() == 2) {
        Value j64 = rewriter.create<arith::IndexCastOp>(loc, llvmI64Ty, j);
        Value s1 = desc.stride(rewriter, loc, 1);
        Value term1 = rewriter.create<arith::MulIOp>(loc, j64, s1);
        linear = rewriter.create<arith::AddIOp>(loc, linear, term1);
      }

      // GEP basePtr + linear (in elements of f64)
      // NOTE: In LLVM dialect, GEPOp takes element type + indices.
      Value elemPtr = rewriter.create<mlir::LLVM::GEPOp>(
      loc,
      llvmPtrTy,
      llvmF64Ty,     // element type
      basePtr,
      ArrayRef<Value>{linear});

      return elemPtr;
}
/// Lowers `toy.print` to a loop nest calling `printf` on each of the individual
/// elements of the array.
class PrintOpLowering : public OpConversionPattern<toy::PrintOp> {
  public:
    using OpConversionPattern<toy::PrintOp>::OpConversionPattern;
  
    LogicalResult matchAndRewrite(toy::PrintOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

        Location loc = op.getLoc();
        MLIRContext *context = rewriter.getContext();
        ModuleOp module = op->getParentOfType<ModuleOp>();

// Original type (still memref) is on the op.
        auto memRefType = llvm::dyn_cast<MemRefType>(op.getInput().getType());
        if (!memRefType)
          return rewriter.notifyMatchFailure(op, "expected memref operand");

// In the LLVM conversion, the adaptor input is an LLVM memref descriptor.
        Value descValue = adaptor.getInput();
        MemRefDescriptor desc(descValue);

// printf plumbing
        auto printfRef = getOrInsertPrintf(rewriter, module);
        Value formatSpecifierCst = getOrCreateGlobalString(
        loc, rewriter, "frmt_spec", StringRef("%f \0", 4), module);
        Value newLineCst = getOrCreateGlobalString(
        loc, rewriter, "nl", StringRef("\n\0", 2), module);

// Loop bounds:
// - if static, use constants
// - if dynamic, read from descriptor.size(dim)
        auto idxTy = rewriter.getIndexType();
        auto i64Ty = rewriter.getI64Type();

        auto getUpperBoundIndex = [&](unsigned dim) -> Value {
        if (memRefType.hasStaticShape() && memRefType.getShape()[dim] != ShapedType::kDynamic) {
          return rewriter.create<arith::ConstantIndexOp>(loc, memRefType.getShape()[dim]);
        }
// desc.size(dim) is i64; cast to index for scf.for bound
        Value sz64 = desc.size(rewriter, loc, dim);
        return rewriter.create<arith::IndexCastOp>(loc, idxTy, sz64);
        };

        unsigned rank = memRefType.getRank();
        if (rank != 0 && rank != 1 && rank != 2)
          return rewriter.notifyMatchFailure(op, "print supports rank-0, rank-1 or rank-2 memrefs");

        Value lb = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);

        auto emitPrintValue = [&](Value elem) {
          rewriter.create<LLVM::CallOp>(
          loc, getPrintfType(context), printfRef,
          ArrayRef<Value>({formatSpecifierCst, elem}));
        };

        if(rank == 0){
          // For rank-0 memref, the element is at [].
          // Use the descriptor to get the data pointer and load one f64.
          Value elemPtr = desc.alignedPtr(rewriter, loc); // or desc.dataPtr(...) depending on your helper
          Value elem = rewriter.create<LLVM::LoadOp>(loc, rewriter.getF64Type(), elemPtr);
        
          emitPrintValue(elem);
          rewriter.create<LLVM::CallOp>(
              loc, getPrintfType(context), printfRef, ArrayRef<Value>{newLineCst});
        
          rewriter.eraseOp(op);
          return success();
        }

        if (rank == 1) {
          Value ub0 = getUpperBoundIndex(0);

          scf::ForOp iLoop = rewriter.create<scf::ForOp>(loc, lb, ub0, step);
          {
            OpBuilder::InsertionGuard g(rewriter);
            rewriter.setInsertionPointToStart(iLoop.getBody());

            Value i = iLoop.getInductionVar();
            Value i64 = rewriter.create<arith::IndexCastOp>(loc, i64Ty, i);

            // Element pointer via descriptor helper
            Value elemPtr = gepToElement(loc, rewriter, desc, memRefType, i);
            Value elem = rewriter.create<LLVM::LoadOp>(loc, rewriter.getF64Type(), elemPtr);

            emitPrintValue(elem);
          }
          rewriter.create<LLVM::CallOp>(loc, getPrintfType(context), printfRef, newLineCst);

          rewriter.eraseOp(op);

          return success();
        }

        // rank == 2
        Value ub0 = getUpperBoundIndex(0);
        Value ub1 = getUpperBoundIndex(1);

        scf::ForOp iLoop = rewriter.create<scf::ForOp>(loc, lb, ub0, step);
        {
          OpBuilder::InsertionGuard g(rewriter);
          rewriter.setInsertionPointToStart(iLoop.getBody());
          Value i = iLoop.getInductionVar();

        scf::ForOp jLoop = rewriter.create<scf::ForOp>(loc, lb, ub1, step);
        {
          OpBuilder::InsertionGuard g2(rewriter);
          rewriter.setInsertionPointToStart(jLoop.getBody());
          Value j = jLoop.getInductionVar();

          Value i64 = rewriter.create<arith::IndexCastOp>(loc, i64Ty, i);
          Value j64 = rewriter.create<arith::IndexCastOp>(loc, i64Ty, j);

          Value elemPtr = gepToElement(loc, rewriter, desc, memRefType, i, j);
          Value elem = rewriter.create<LLVM::LoadOp>(loc, rewriter.getF64Type(), elemPtr);

          emitPrintValue(elem);
        }

        // newline after each row
        rewriter.create<LLVM::CallOp>(loc, getPrintfType(context), printfRef, newLineCst);
        }

        rewriter.eraseOp(op);
        return success();
      
  }

  
  private:
    static LLVM::LLVMFunctionType getPrintfType(MLIRContext *context) {
      auto i32 = IntegerType::get(context, 32);
      auto ptr = LLVM::LLVMPointerType::get(context);
      return LLVM::LLVMFunctionType::get(i32, ptr, /*isVarArg=*/true);
    }
  
    static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter, ModuleOp module) {
      auto *context = module.getContext();
      if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
        return SymbolRefAttr::get(context, "printf");
  
      PatternRewriter::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      LLVM::LLVMFuncOp::create(rewriter, module.getLoc(), "printf", getPrintfType(context));
      return SymbolRefAttr::get(context, "printf");
    }
  
    static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                         StringRef name, StringRef value, ModuleOp module) {
      LLVM::GlobalOp global = module.lookupSymbol<LLVM::GlobalOp>(name);
      if (!global) {
        OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPointToStart(module.getBody());
        auto type = LLVM::LLVMArrayType::get(IntegerType::get(builder.getContext(), 8),
                                             value.size());
        global = LLVM::GlobalOp::create(builder, loc, type, /*isConstant=*/true,
                                        LLVM::Linkage::Internal, name,
                                        builder.getStringAttr(value),
                                        /*alignment=*/0);
      }
  
      Value globalPtr = LLVM::AddressOfOp::create(builder, loc, global);
      Value cst0 = LLVM::ConstantOp::create(builder, loc, builder.getI64Type(),
                                            builder.getI64IntegerAttr(0));
      return LLVM::GEPOp::create(builder, loc, LLVM::LLVMPointerType::get(builder.getContext()),
                                 global.getType(), globalPtr, ArrayRef<Value>({cst0, cst0}));
    }
  };
  
} // namespace

//===----------------------------------------------------------------------===//
// ToyToLLVMLoweringPass
//===----------------------------------------------------------------------===//

namespace {
struct ToyToLLVMLoweringPass
    : public PassWrapper<ToyToLLVMLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyToLLVMLoweringPass)
  StringRef getArgument() const override { return "toy-to-llvm"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<cf::ControlFlowDialect>();
  registry.insert<arith::ArithDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<memref::MemRefDialect>();
  registry.insert<math::MathDialect>();
  }
  void runOnOperation() final;
};
} // namespace

void ToyToLLVMLoweringPass::runOnOperation() {
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering. For this lowering, we are only targeting
  // the LLVM dialect.
  LLVMConversionTarget target(getContext());
  target.addLegalOp<ModuleOp>();
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalOp<scf::YieldOp>();


  // During this lowering, we will also be lowering the MemRef types, that are
  // currently being operated on, to a representation in LLVM. To perform this
  // conversion we use a TypeConverter as part of the lowering. This converter
  // details how one type maps to another. This is necessary now that we will be
  // doing more complicated lowerings, involving loop region arguments.
  LLVMTypeConverter typeConverter(&getContext());


  // Now that the conversion target has been defined, we need to provide the
  // patterns used for lowering. At this point of the compilation process, we
  // have a combination of `toy`, `affine`, and `std` operations. Luckily, there
  // are already exists a set of patterns to transform `affine` and `std`
  // dialects. These patterns lowering in multiple stages, relying on transitive
  // lowerings. Transitive lowering, or A->B->C lowering, is when multiple
  // patterns must be applied to fully transform an illegal operation into a
  // set of legal ones.
  RewritePatternSet patterns(&getContext());

  mlir::toy::populateToyStructToLLVMConversionPatterns(typeConverter, patterns);
  populateAffineToStdConversionPatterns(patterns);
  populateSCFToControlFlowConversionPatterns(patterns);
  mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
  cf::populateAssertToLLVMConversionPattern(typeConverter, patterns);
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);
  populateMathToLLVMConversionPatterns(typeConverter, patterns);

 

  // The only remaining operation to lower from the `toy` dialect, is the
  // PrintOp.
  patterns.add<PrintOpLowering>(typeConverter, &getContext());

  // We want to completely lower to LLVM, so we use a `FullConversion`. This
  // ensures that only legal operations will remain after the conversion.
  auto module = getOperation();
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

/// Create a pass for lowering operations the remaining `Toy` operations, as
/// well as `Affine` and `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> mlir::toy::createLowerToLLVMPass() {
  return std::make_unique<ToyToLLVMLoweringPass>();
}
