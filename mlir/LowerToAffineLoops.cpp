//====- LowerToAffineLoops.cpp - Partial lowering from Toy to Affine+Std --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a partial lowering of Toy operations to a combination of
// affine loops, memref operations and standard operations. This lowering
// expects that all calls have been inlined, and all shapes have been resolved.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "toy/Dialect.h"
#include "toy/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"


#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

using namespace mlir;

//===----------------------------------------------------------------------===//
// ToyToAffine Conversion Patterns
//===----------------------------------------------------------------------===//

/// Convert the given RankedTensorType into the corresponding MemRefType.
static MemRefType convertTensorToMemRef(RankedTensorType type) {
  return MemRefType::get(type.getShape(), type.getElementType());
}

/// Insert an allocation and deallocation for the given MemRefType.
static Value insertAllocAndDealloc(MemRefType type, Location loc,
                                   PatternRewriter &rewriter) {
  auto alloc = memref::AllocOp::create(rewriter, loc, type);
  auto dealloc = memref::DeallocOp::create(rewriter, loc, alloc);

  // Make sure to allocate at the beginning of the block.
  auto *parentBlock = alloc->getBlock();
  if (parentBlock->empty()) {
    // If the block is empty, just leave dealloc right after alloc for now
    // (or better: place it later once you know the terminator exists).
  } else {
    dealloc->moveBefore(&parentBlock->back());
  }
  alloc->moveBefore(&parentBlock->front());
  // Make sure to deallocate this alloc at the end of the block. This is fine
  // as toy functions have no control flow.
  
  dealloc->moveBefore(&parentBlock->back());
  return alloc;
}

/// This defines the function type used to process an iteration of a lowered
/// loop. It takes as input an OpBuilder and the range of loop induction
/// variables for the iteration. It returns a value to store at the current
/// index of the iteration.
using LoopIterationFn =
    function_ref<Value(OpBuilder &rewriter, ValueRange loopIvs)>;

static void lowerOpToLoops(Operation *op, PatternRewriter &rewriter,
                           LoopIterationFn processIteration) {
  auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));
  auto loc = op->getLoc();

  // Insert an allocation and deallocation for the result of this operation.
  auto memRefType = convertTensorToMemRef(tensorType);
  auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

  // Create a nest of affine loops, with one loop per dimension of the shape.
  // The buildAffineLoopNest function takes a callback that is used to construct
  // the body of the innermost loop given a builder, a location and a range of
  // loop induction variables.
  SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value=*/0);
  SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);
  affine::buildAffineLoopNest(
      rewriter, loc, lowerBounds, tensorType.getShape(), steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        // Call the processing function with the rewriter
        // and the loop induction variables. This function will return the value
        // to store at the current index.
        Value valueToStore = processIteration(nestedBuilder, ivs);
        affine::AffineStoreOp::create(nestedBuilder, loc, valueToStore, alloc,
                                      ivs);
      });

  // Replace this operation with the generated alloc.
  rewriter.replaceOp(op, alloc);
}

namespace {
//===----------------------------------------------------------------------===//
// ToyToAffine Conversion Patterns: Binary operations
//===----------------------------------------------------------------------===//

template <typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : public OpConversionPattern<BinaryOp> {
  using OpConversionPattern<BinaryOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<BinaryOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(BinaryOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(op, rewriter, [&](OpBuilder &builder, ValueRange loopIvs) {
      // Generate loads for the element of 'lhs' and 'rhs' at the
      // inner loop.
      auto loadedLhs =
          affine::AffineLoadOp::create(builder, loc, adaptor.getLhs(), loopIvs);
      auto loadedRhs =
          affine::AffineLoadOp::create(builder, loc, adaptor.getRhs(), loopIvs);

      // Create the binary operation performed on the loaded
      // values.
      return LoweredBinaryOp::create(builder, loc, loadedLhs, loadedRhs);
    });
    return success();
  }
};
using AddOpLowering = BinaryOpLowering<toy::AddOp, arith::AddFOp>;
using MulOpLowering = BinaryOpLowering<toy::MulOp, arith::MulFOp>;

//===----------------------------------------------------------------------===//
// ToyToAffine Conversion Patterns: Constant operations
//===----------------------------------------------------------------------===//

struct ConstantOpLowering : public OpConversionPattern<toy::ConstantOp> {
  using OpConversionPattern<toy::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    DenseElementsAttr constantValue = op.getValue();
    Location loc = op.getLoc();

    // When lowering the constant operation, we allocate and assign the constant
    // values to a corresponding memref allocation.
    auto tensorType = llvm::cast<RankedTensorType>(op.getType());
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // We will be generating constant indices up-to the largest dimension.
    // Create these constants up-front to avoid large amounts of redundant
    // operations.
    auto valueShape = memRefType.getShape();
    SmallVector<Value, 8> constantIndices;

    if (!valueShape.empty()) {
      for (auto i : llvm::seq<int64_t>(0, *llvm::max_element(valueShape)))
        constantIndices.push_back(
            arith::ConstantIndexOp::create(rewriter, loc, i));
    } else {
      // This is the case of a tensor of rank 0.
      constantIndices.push_back(
          arith::ConstantIndexOp::create(rewriter, loc, 0));
    }

    // The constant operation represents a multi-dimensional constant, so we
    // will need to generate a store for each of the elements. The following
    // functor recursively walks the dimensions of the constant shape,
    // generating a store when the recursion hits the base case.
    SmallVector<Value, 2> indices;
    auto valueIt = constantValue.value_begin<FloatAttr>();
    std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
      // The last dimension is the base case of the recursion, at this point
      // we store the element at the given index.
      if (dimension == valueShape.size()) {
        affine::AffineStoreOp::create(
            rewriter, loc, arith::ConstantOp::create(rewriter, loc, *valueIt++),
            alloc, llvm::ArrayRef(indices));
        return;
      }

      // Otherwise, iterate over the current dimension and add the indices to
      // the list.
      for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
        indices.push_back(constantIndices[i]);
        storeElements(dimension + 1);
        indices.pop_back();
      }
    };

    // Start the element storing recursion from the first dimension.
    storeElements(/*dimension=*/0);

    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine Conversion Patterns: Func operations
//===----------------------------------------------------------------------===//

struct LowerToyFuncByName : public mlir::ConversionPattern {
  LowerToyFuncByName(mlir::TypeConverter &tc, mlir::MLIRContext *ctx)
      : mlir::ConversionPattern(tc, "toy.func", /*benefit=*/1000, ctx) {}

      mlir::LogicalResult matchAndRewrite(mlir::Operation *op,
        mlir::ArrayRef<mlir::Value> /*operands*/,
        mlir::ConversionPatternRewriter &rewriter) const override {

        // Must be toy.func
        if (op->getNumRegions() != 1) {
        op->emitOpError("expected 1 region");
        return mlir::failure();
        }

        auto nameAttr = op->getAttrOfType<mlir::StringAttr>("sym_name");
        if (!nameAttr) {
        op->emitOpError("missing 'sym_name' attribute");
        return mlir::failure();
        }

        if (nameAttr.getValue() != "main")
        return rewriter.notifyMatchFailure(op, "only lowers @main");

        mlir::Region &body = op->getRegion(0);
        if (body.empty()) {
        op->emitOpError("function body region is empty");
        return mlir::failure();
        }

        mlir::Block &oldEntry = body.front();

        auto *tc = getTypeConverter();
        if (!tc) {
        op->emitOpError("missing TypeConverter");
        return mlir::failure();
        }

        // Convert arg types based on the entry block arguments (NOT function_type attr).
        mlir::TypeConverter::SignatureConversion sigConv(oldEntry.getNumArguments());
        SmallVector<mlir::Type, 4> convertedInputs;
        convertedInputs.reserve(oldEntry.getNumArguments());

        for (unsigned i = 0; i < oldEntry.getNumArguments(); ++i) {
        mlir::Type oldTy = oldEntry.getArgument(i).getType();
        mlir::Type newTy = tc->convertType(oldTy);
        if (!newTy) {
        op->emitOpError() << "failed to convert arg " << i << " type " << oldTy;
        return mlir::failure();
        }
        sigConv.addInputs(i, newTy);
        convertedInputs.push_back(newTy);

        }        

        auto newFuncType = rewriter.getFunctionType(convertedInputs, /*results=*/{});
        rewriter.setInsertionPoint(op);
        auto newFunc = rewriter.create<mlir::func::FuncOp>(op->getLoc(), nameAttr.getValue(), newFuncType);
        rewriter.inlineRegionBefore(body, newFunc.getBody(), newFunc.getBody().end());

        // Convert block argument types inside the moved region
        if (failed(rewriter.convertRegionTypes(&newFunc.getBody(), *tc, &sigConv))) {
          op->emitOpError("failed to convert region types");
          return mlir::failure();
        }

        // Replace toy.return with func.return in the entry block
        mlir::Block &entry = newFunc.getBody().front();
          if (!entry.mightHaveTerminator()) {
          op->emitOpError("entry block missing terminator after move");
          return mlir::failure();
        }

        rewriter.eraseOp(op);
        return mlir::success();
        }

};


//===----------------------------------------------------------------------===//
// ToyToAffine Conversion Patterns: Print operations
//===----------------------------------------------------------------------===//

struct PrintOpLowering : public OpConversionPattern<toy::PrintOp> {
  using OpConversionPattern<toy::PrintOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // We don't lower "toy.print" in this pass, but we need to update its
    // operands.
    rewriter.modifyOpInPlace(op,
                             [&] { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine Conversion Patterns: Return operations
//===----------------------------------------------------------------------===//

struct ReturnOpLowering : public OpConversionPattern<toy::ReturnOp> {
  using OpConversionPattern<toy::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // During this lowering, we expect that all function calls have been
    // inlined.
    if (op.hasOperand())
      return failure();

    // We lower "toy.return" directly to "func.return".
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine Conversion Patterns: Transpose operations
//===----------------------------------------------------------------------===//

struct TransposeOpLowering : public OpConversionPattern<toy::TransposeOp> {
  using OpConversionPattern<toy::TransposeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(op, rewriter, [&](OpBuilder &builder, ValueRange loopIvs) {
      Value input = adaptor.getInput();

      // Transpose the elements by generating a load from the
      // reverse indices.
      SmallVector<Value, 2> reverseIvs(llvm::reverse(loopIvs));
      return affine::AffineLoadOp::create(builder, loc, input, reverseIvs);
    });
    return success();
  }
};


/*struct ModelOpLowering : public OpConversionPattern<toy::ModelOp> {
  using OpConversionPattern<toy::ModelOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::ModelOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Input should already be converted to memref<...>
    Value xMemref = adaptor.getX();
    auto xType = llvm::dyn_cast<mlir::MemRefType>(xMemref.getType());
    if (!xType)
      return rewriter.notifyMatchFailure(op, "expected memref type for x");

    // Convert result tensor type → result memref type
    auto resultType = llvm::dyn_cast<mlir::MemRefType>(
      getTypeConverter()->convertType(op.getResult().getType()));
      if (!resultType)
        return rewriter.notifyMatchFailure(op, "could not convert result to memref");

    // Allocate output buffer
    Value result =
        rewriter.create<memref::AllocOp>(loc, resultType);

    // Element type (f32 or f64)
    Type elemType = xType.getElementType();

    // Constants w = 1.0, b = 0.5
    Value wConst = rewriter.create<arith::ConstantOp>(
        loc, elemType, rewriter.getFloatAttr(elemType, 1.0));
    Value bConst = rewriter.create<arith::ConstantOp>(
        loc, elemType, rewriter.getFloatAttr(elemType, 0.5));

    //
    // ───────────────────────────────────────────
    // Generate nested affine.for loops manually
    // ───────────────────────────────────────────
    //
    // For rank 2 (most common in toy):
    //
    // affine.for i = 0 to dim0 {
    //   affine.for j = 0 to dim1 {
    //     %x = affine.load xMemref[i, j]
    //     %z  = mulf %x, w
    //     %z2 = addf %z, b
    //     %y  = math.tanh %z2
    //     affine.store %y, result[i, j]
    //   }
    // }
    //

    int64_t rank = xType.getRank();
    if (rank != 2)
      return failure();  // keep simple for now

    int64_t dim0 = xType.getDimSize(0);
    int64_t dim1 = xType.getDimSize(1);

    auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // OUTER LOOP
    auto outer = rewriter.create<affine::AffineForOp>(loc, 0, dim0, 1);
    
    Block *outerBody = outer.getBody();
    OpBuilder outerBuilder = OpBuilder::atBlockTerminator(outerBody);

    // INNER LOOP
    auto inner = outerBuilder.create<affine::AffineForOp>(loc, 0, dim1, 1);

    Block *innerBody = inner.getBody();
    OpBuilder loopBuilder = OpBuilder::atBlockTerminator(innerBody);

    // Induction vars: %i, %j
    Value i = outer.getInductionVar();
    Value j = inner.getInductionVar();

    // Load x[i, j]
    Value xVal = loopBuilder.create<affine::AffineLoadOp>(loc, xMemref,
                                                          ValueRange{i, j});

    // z = x * w
    Value z = loopBuilder.create<arith::MulFOp>(loc, xVal, wConst);

    // z2 = z + b
    Value z2 = loopBuilder.create<arith::AddFOp>(loc, z, bConst);

    // activation: tanh(z2)
    Value y = loopBuilder.create<math::TanhOp>(loc, z2);

    // store result[i, j]
    loopBuilder.create<affine::AffineStoreOp>(loc, y, result,
                                              ValueRange{i, j});

    // Replace original op
    rewriter.replaceOp(op, result);

    return success();
  }
};*/

static mlir::Type getLoweredModelType(mlir::MLIRContext *ctx) {
  auto metaTy   = mlir::MemRefType::get({4}, mlir::IntegerType::get(ctx, 64));
  auto paramsTy = mlir::MemRefType::get({mlir::ShapedType::kDynamic}, mlir::Float64Type::get(ctx));                                        
  auto paramsMapTy = mlir::MemRefType::get({mlir::ShapedType::kDynamic}, mlir::IndexType::get(ctx));
                                    
  return mlir::toy::StructType::get({metaTy, paramsTy, paramsMapTy});
}

struct LowerCreateModel : OpConversionPattern<mlir::toy::CreateModelOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::toy::CreateModelOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    // Types
    auto metaTy = MemRefType::get({4}, rewriter.getI64Type());
    auto paramsTy = MemRefType::get({ShapedType::kDynamic}, rewriter.getF64Type());
    auto paramsMapTy = MemRefType::get({ShapedType::kDynamic}, rewriter.getIndexType());

    // Allocate meta
    Value meta = rewriter.create<memref::AllocOp>(loc, metaTy);

    // Index constants for meta indexing
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value c2 = rewriter.create<arith::ConstantIndexOp>(loc, 2);
    Value c3 = rewriter.create<arith::ConstantIndexOp>(loc, 3);

    // i64 constants from attributes
    Value in  = rewriter.create<arith::ConstantIntOp>(loc, op.getInput(), 64);
    Value dep = rewriter.create<arith::ConstantIntOp>(loc, op.getDepth(), 64);
    Value wid = rewriter.create<arith::ConstantIntOp>(loc, op.getWidth(), 64);
    Value out = rewriter.create<arith::ConstantIntOp>(loc, op.getOutput(), 64);

    // Store meta
    rewriter.create<memref::StoreOp>(loc, in,  meta, ValueRange{c0});
    rewriter.create<memref::StoreOp>(loc, dep, meta, ValueRange{c1});
    rewriter.create<memref::StoreOp>(loc, wid, meta, ValueRange{c2});
    rewriter.create<memref::StoreOp>(loc, out, meta, ValueRange{c3});

    // ---- Compute total params size -----------------------------------------
    // total = (in*wid + wid) + (dep-2)*(wid*wid + wid) + (wid*out + out)
    Value one64 = rewriter.create<arith::ConstantIntOp>(loc, 1, 64);
    Value two64 = rewriter.create<arith::ConstantIntOp>(loc, 2, 64);

    Value depMinus2 = rewriter.create<arith::SubIOp>(loc, dep, two64);

    Value inWid  = rewriter.create<arith::MulIOp>(loc, in, wid);
    Value widWid = rewriter.create<arith::MulIOp>(loc, wid, wid);
    Value widOut = rewriter.create<arith::MulIOp>(loc, wid, out);

    Value first    = rewriter.create<arith::AddIOp>(loc, inWid, wid);           // w0 + b0
    Value midChunk = rewriter.create<arith::AddIOp>(loc, widWid, wid);          // wi + bi for hidden
    Value mids     = rewriter.create<arith::MulIOp>(loc, depMinus2, midChunk);  // (dep-2) * chunk
    Value last     = rewriter.create<arith::AddIOp>(loc, widOut, out);          // wLast + bLast

    Value total64 = rewriter.create<arith::AddIOp>(
        loc, rewriter.create<arith::AddIOp>(loc, first, mids), last);

    Value totalIdx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), total64);

    // Allocate params
    Value params = rewriter.create<memref::AllocOp>(loc, paramsTy, ValueRange{totalIdx});

    // ---- Allocate paramsMap: length = 2*depth ------------------------------
    Value depIdx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), dep);
    Value twoIdx = rewriter.create<arith::ConstantIndexOp>(loc, 2);
    Value mapLen = rewriter.create<arith::MulIOp>(loc, depIdx, twoIdx);

    Value paramsMap = rewriter.create<memref::AllocOp>(loc, paramsMapTy, ValueRange{mapLen});

    // ---- Populate paramsMap ------------------------------------------------
    // paramsMap[2*l] = wBase
    // paramsMap[2*l+1] = bBase
    //
    // offset starts at 0 and accumulates (wSize + bSize) each layer.
    Value offset0 = c0;
    scf::ForOp mapLoop = rewriter.create<scf::ForOp>(loc, c0, depIdx, c1, ValueRange{offset0});

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mapLoop.getBody());

      Value l = mapLoop.getInductionVar();
      Value offset = mapLoop.getRegionIterArg(0);

      // idxW = 2*l, idxB = 2*l + 1
      Value idxW = rewriter.create<arith::MulIOp>(loc, l, twoIdx);
      Value idxB = rewriter.create<arith::AddIOp>(loc, idxW, c1);

      // Cast dimensions to index
      Value inIdx  = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), in);
      Value widIdx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), wid);
      Value outIdx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), out);

      Value lastLayer = rewriter.create<arith::SubIOp>(loc, depIdx, c1);

      // Build an scf.if that returns (wSize, bSize) as index values.
      auto idxTy = rewriter.getIndexType();
      Value isFirst = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, l, c0);

      scf::IfOp sizeIfFirst = rewriter.create<scf::IfOp>(
          loc, TypeRange{idxTy, idxTy}, isFirst, /*withElse=*/true);

      // then: first layer
      {
        OpBuilder thenB = sizeIfFirst.getThenBodyBuilder();
        Value wSize = thenB.create<arith::MulIOp>(loc, inIdx, widIdx);
        Value bSize = widIdx;
        thenB.create<scf::YieldOp>(loc, ValueRange{wSize, bSize});
      }

      // else: last layer ? hidden layer
      {
        OpBuilder elseB = sizeIfFirst.getElseBodyBuilder();
        Value isLast = elseB.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, l, lastLayer);

        scf::IfOp sizeIfLast = elseB.create<scf::IfOp>(
            loc, TypeRange{idxTy, idxTy}, isLast, /*withElse=*/true);

        // then: last layer
        {
          OpBuilder thenB = sizeIfLast.getThenBodyBuilder();
          Value wSize = thenB.create<arith::MulIOp>(loc, widIdx, outIdx);
          Value bSize = outIdx;
          thenB.create<scf::YieldOp>(loc, ValueRange{wSize, bSize});
        }

        // else: hidden layer
        {
          OpBuilder elseB2 = sizeIfLast.getElseBodyBuilder();
          Value wSize = elseB2.create<arith::MulIOp>(loc, widIdx, widIdx);
          Value bSize = widIdx;
          elseB2.create<scf::YieldOp>(loc, ValueRange{wSize, bSize});
        }

        // yield the chosen pair out of the outer else
        Value wSize = sizeIfLast.getResult(0);
        Value bSize = sizeIfLast.getResult(1);
        elseB.create<scf::YieldOp>(loc, ValueRange{wSize, bSize});
      }

      Value wSize = sizeIfFirst.getResult(0);
      Value bSize = sizeIfFirst.getResult(1);

      // Store bases
      rewriter.create<memref::StoreOp>(loc, offset, paramsMap, ValueRange{idxW});
      Value bBase = rewriter.create<arith::AddIOp>(loc, offset, wSize);
      rewriter.create<memref::StoreOp>(loc, bBase, paramsMap, ValueRange{idxB});

      // nextOffset = offset + (wSize + bSize)
      Value nextOffset = rewriter.create<arith::AddIOp>(
          loc, offset, rewriter.create<arith::AddIOp>(loc, wSize, bSize));

      rewriter.create<scf::YieldOp>(loc, ValueRange{nextOffset});
    }

    // ---- Initialize params using paramsMap --------------------------------
    // Weights: deterministic pseudo-random
    // Biases:  0.0

    auto f64Ty = rewriter.getF64Type();

    Value zeroF64 = rewriter.create<arith::ConstantOp>(
      loc, f64Ty, rewriter.getFloatAttr(f64Ty, 0.0));

    Value scale = rewriter.create<arith::ConstantOp>(
      loc, f64Ty, rewriter.getFloatAttr(f64Ty, 0.01));


    Value modBase = rewriter.create<arith::ConstantIndexOp>(loc, 13);
    Value center  = rewriter.create<arith::ConstantIndexOp>(loc, 6);

    scf::ForOp layerInit = rewriter.create<scf::ForOp>(loc, c0, depIdx, c1);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(layerInit.getBody());

      Value l = layerInit.getInductionVar();

      Value idxW = rewriter.create<arith::MulIOp>(loc, l, twoIdx);
      Value idxB = rewriter.create<arith::AddIOp>(loc, idxW, c1);

      Value wBase = rewriter.create<memref::LoadOp>(loc, paramsMap, ValueRange{idxW});
      Value bBase = rewriter.create<memref::LoadOp>(loc, paramsMap, ValueRange{idxB});

      // nextWBase = (l == last) ? totalIdx : paramsMap[2*(l+1)]
      Value lastLayer = rewriter.create<arith::SubIOp>(loc, depIdx, c1);
      Value isLast = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, l, lastLayer);

      scf::IfOp nextBaseIf = rewriter.create<scf::IfOp>(
          loc, TypeRange{rewriter.getIndexType()}, isLast, /*withElse=*/true);

      // then: nextWBase = totalIdx
      {
        OpBuilder thenB = nextBaseIf.getThenBodyBuilder();
        thenB.create<scf::YieldOp>(loc, ValueRange{totalIdx});
      }
      // else: nextWBase = paramsMap[2*(l+1)]
      {
        OpBuilder elseB = nextBaseIf.getElseBodyBuilder();
        Value lp1 = elseB.create<arith::AddIOp>(loc, l, c1);
        Value nextIdxW = elseB.create<arith::MulIOp>(loc, lp1, twoIdx);
        Value nextWBase = elseB.create<memref::LoadOp>(loc, paramsMap, ValueRange{nextIdxW});
        elseB.create<scf::YieldOp>(loc, ValueRange{nextWBase});
      }

      Value nextWBase = nextBaseIf.getResult(0);

      // lengths
      Value wLen = rewriter.create<arith::SubIOp>(loc, bBase, wBase);
      Value bLen = rewriter.create<arith::SubIOp>(loc, nextWBase, bBase);

      // ---- init weights: for i in [0, wLen)
      scf::ForOp wLoop = rewriter.create<scf::ForOp>(loc, c0, wLen, c1);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(wLoop.getBody());
        Value i = wLoop.getInductionVar();

        Value idx = rewriter.create<arith::AddIOp>(loc, wBase, i);

        // pseudo-random-ish: 0.01 * ( (idx % 13) - 6 )
        Value m = rewriter.create<arith::RemUIOp>(loc, idx, modBase);
        Value centered = rewriter.create<arith::SubIOp>(loc, m, center);

        // index -> i64 -> f64
        Value centeredI64 = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), centered);
        Value centeredF64 = rewriter.create<arith::SIToFPOp>(loc, rewriter.getF64Type(), centeredI64);

        Value val = rewriter.create<arith::MulFOp>(loc, centeredF64, scale);

        rewriter.create<memref::StoreOp>(loc, val, params, ValueRange{idx});
        
      }

      // ---- init biases: for i in [0, bLen)
      scf::ForOp bLoop = rewriter.create<scf::ForOp>(loc, c0, bLen, c1);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(bLoop.getBody());
        Value i = bLoop.getInductionVar();

        Value idx = rewriter.create<arith::AddIOp>(loc, bBase, i);
        rewriter.create<memref::StoreOp>(loc, zeroF64, params, ValueRange{idx});
        
      }
    }

    // ---- Pack model struct -------------------------------------------------
    Type loweredModelTy = getLoweredModelType(ctx);

    auto model = rewriter.create<mlir::toy::MakeStructOp>(
        loc,
        mlir::TypeRange{loweredModelTy},
        mlir::ValueRange{meta, params, paramsMap});

    rewriter.replaceOp(op, model.getResult());
    return success();
  }
};


/*
      x-input

      std::vector<double> cur(xIn, xIn + in);
      std::vector<double> next;

      for(int layer = 0; layer < depth; layer++){
        int bBase = paramsMap[i * 2 + 1];
        int wBase = paramsMap[i * 2];
        
        int inDim, outDim
        if(layer == 0){
          inDim = in
          outDim = width;
        }else if(layer == depth - 1){
          inDim = width;
          outDim = out;
        }else{
          inDim = width;
          outDim = width;
        }

        

        next.assign(outDim, 0.0);

        for (int row = 0; row < outDim; ++row) {
          double acc = params[bBase + row];                 // bias[row]
          for (int col = 0; col < inDim; ++col) {
            double w = params[wBase + row*inDim + col];     // W[row,col]
            acc += w * cur[col];                            // x[col]
          }

          if (layer != depth - 1) acc = std::tanh(acc);
          next[row] = acc;

        }

        cur.swap(next);
      }
    
      for (int i = 0; i < out; ++i) yOut[i] = cur[i];
    */

struct LowerPredictOp : OpConversionPattern<mlir::toy::PredictOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::toy::PredictOp op, OpAdaptor adaptor,
  ConversionPatternRewriter &rewriter) const override {
  
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value modelValue = adaptor.getModel(); 

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value c2 = rewriter.create<arith::ConstantIndexOp>(loc, 2);
    Value c3 = rewriter.create<arith::ConstantIndexOp>(loc, 3);


    Value inputValues = adaptor.getX(); // x
    auto mt = dyn_cast<MemRefType>(inputValues.getType());
    if (!mt || (mt.getRank() != 1 && mt.getRank() != 2)) {
      return rewriter.notifyMatchFailure(op, "predict expects rank-1 or rank-2 input memref");
    }

    Value cur = inputValues;   // default: rank-1 uses as-is
    Value xLen;               // index length of cur

    Value ownsCur = rewriter.create<arith::ConstantIntOp>(loc, 0, 1);

    // ---- rank-1: just read length
    if (mt.getRank() == 1) {
      // static? or dynamic?
      if (mt.hasStaticShape()) {
        int64_t n = mt.getShape()[0];
        xLen = rewriter.create<arith::ConstantIndexOp>(loc, n);
      } else {
        xLen = rewriter.create<memref::DimOp>(loc, inputValues, 0);
      }
    }

    // ---- rank-2: flatten to 1D
    if (mt.getRank() == 2) {
      // rows/cols as index
      Value rowsIdx, colsIdx;
      if (mt.hasStaticShape()) {
        int64_t rows = mt.getShape()[0];
        int64_t cols = mt.getShape()[1];
        rowsIdx = rewriter.create<arith::ConstantIndexOp>(loc, rows);
        colsIdx = rewriter.create<arith::ConstantIndexOp>(loc, cols);
      } else {
        rowsIdx = rewriter.create<memref::DimOp>(loc, inputValues, 0);
        colsIdx = rewriter.create<memref::DimOp>(loc, inputValues, 1);
      }

      // flatSize = rows * cols
      Value flatSize = rewriter.create<arith::MulIOp>(loc, rowsIdx, colsIdx);

      // alloc flatX : memref<?xf64>
      auto flatTy = MemRefType::get({ShapedType::kDynamic}, rewriter.getF64Type());
      Value flatX = rewriter.create<memref::AllocOp>(loc, flatTy, ValueRange{flatSize});

      // copy: flatX[r*cols+c] = inputValues[r,c]
      scf::ForOp rLoop = rewriter.create<scf::ForOp>(loc, c0, rowsIdx, c1);
      {
        OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPointToStart(rLoop.getBody());
        Value r = rLoop.getInductionVar();

        scf::ForOp cLoop = rewriter.create<scf::ForOp>(loc, c0, colsIdx, c1);
        {
          OpBuilder::InsertionGuard g2(rewriter);
          rewriter.setInsertionPointToStart(cLoop.getBody());
          Value c = cLoop.getInductionVar();

          Value rMulCols = rewriter.create<arith::MulIOp>(loc, r, colsIdx);
          Value linear   = rewriter.create<arith::AddIOp>(loc, rMulCols, c);

          Value elem = rewriter.create<memref::LoadOp>(loc, inputValues, ValueRange{r, c});
          rewriter.create<memref::StoreOp>(loc, elem, flatX, ValueRange{linear});

          
        }

        
      }

      // update cur + xLen to flattened view
      cur = flatX;
      xLen = flatSize;
      ownsCur = rewriter.create<arith::ConstantIntOp>(loc, 1, 1);
    }

    auto metaTy = mlir::MemRefType::get({4}, rewriter.getI64Type());
    Value meta = rewriter.create<toy::StructAccessOp>(
    loc, metaTy, modelValue, rewriter.getI64IntegerAttr(0));
    
    auto paramsTy = mlir::MemRefType::get({mlir::ShapedType::kDynamic}, rewriter.getF64Type());
    Value params = rewriter.create<toy::StructAccessOp>(loc, paramsTy, modelValue, rewriter.getI64IntegerAttr(1));

    auto mapTy = mlir::MemRefType::get({mlir::ShapedType::kDynamic}, rewriter.getIndexType());
    Value paramsMap = rewriter.create<toy::StructAccessOp>(loc, mapTy, modelValue, rewriter.getI64IntegerAttr(2));

    //load input layer
    Value in = rewriter.create<memref::LoadOp>(loc, meta, ValueRange{c0});
    Value dep = rewriter.create<memref::LoadOp>(loc, meta, ValueRange{c1});
    Value wid = rewriter.create<memref::LoadOp>(loc, meta, ValueRange{c2});
    Value out = rewriter.create<memref::LoadOp>(loc, meta, ValueRange{c3});

    Value inIdx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), in);
    Value depIdx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), dep);
    Value widIdx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), wid);
    Value outIdx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), out);
    
    auto i1Ty = rewriter.getI1Type();

    scf::ForOp layerLoop = rewriter.create<scf::ForOp>(
    loc, c0, depIdx, c1,
    ValueRange{cur, ownsCur},
    [&](OpBuilder &b, Location loc, Value l, ValueRange iterArgs) {

      Value curArg  = iterArgs[0]; // memref<?xf64>
      Value ownsArg = iterArgs[1]; // i1

      // ---- compute inDim/outDim etc USING b.create
      Value idxW = b.create<arith::MulIOp>(loc, l, c2);
      Value idxB = b.create<arith::AddIOp>(loc, idxW, c1);

      Value wBase = b.create<memref::LoadOp>(loc, paramsMap, ValueRange{idxW});
      Value bBase = b.create<memref::LoadOp>(loc, paramsMap, ValueRange{idxB});

      Value isFirst = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, l, c0);

      Value lastLayer = b.create<arith::SubIOp>(loc, depIdx, c1);
      Value isLastLayer = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, l, lastLayer);

      Value inDim  = b.create<arith::SelectOp>(loc, isFirst, inIdx, widIdx);
      Value outDim = b.create<arith::SelectOp>(loc, isLastLayer, outIdx, widIdx);

      auto nextTy = MemRefType::get({ShapedType::kDynamic}, b.getF64Type());
      Value next = b.create<memref::AllocOp>(loc, nextTy, ValueRange{outDim});

      // ---- row loop
      b.create<scf::ForOp>(loc, c0, outDim, c1, ValueRange{},
        [&](OpBuilder &rb, Location loc, Value row, ValueRange) {

          Value biasIdx = rb.create<arith::AddIOp>(loc, bBase, row);
          Value bias = rb.create<memref::LoadOp>(loc, params, ValueRange{biasIdx});

          scf::ForOp colLoop = rb.create<scf::ForOp>(
            loc, c0, inDim, c1, ValueRange{bias},
            [&](OpBuilder &cb, Location loc, Value col, ValueRange it) {
              Value acc = it[0];

              Value rowInDim = cb.create<arith::MulIOp>(loc, row, inDim);
              Value base = cb.create<arith::AddIOp>(loc, wBase, rowInDim);
              Value weightIdx = cb.create<arith::AddIOp>(loc, base, col);

              Value w = cb.create<memref::LoadOp>(loc, params, ValueRange{weightIdx});
              Value x = cb.create<memref::LoadOp>(loc, curArg, ValueRange{col});
              Value prod = cb.create<arith::MulFOp>(loc, w, x);
              Value accNext = cb.create<arith::AddFOp>(loc, acc, prod);

              cb.create<scf::YieldOp>(loc, accNext);
            });

          Value accFinal = colLoop.getResult(0);

          scf::IfOp actIf = rb.create<scf::IfOp>(
              loc, TypeRange{rb.getF64Type()}, isLastLayer, /*withElseRegion=*/true);

          {
            OpBuilder thenB = actIf.getThenBodyBuilder();
            thenB.create<scf::YieldOp>(loc, accFinal);
          }
          {
            OpBuilder elseB = actIf.getElseBodyBuilder();
            Value act = elseB.create<math::TanhOp>(loc, accFinal);
            elseB.create<scf::YieldOp>(loc, act);
          }

          rb.create<memref::StoreOp>(loc, actIf.getResult(0), next, ValueRange{row});
          rb.create<scf::YieldOp>(loc);
        });

      // ---- free previous buffer if we owned it
      b.create<scf::IfOp>(loc, ownsArg,
        [&](OpBuilder &tb, Location loc) {
          tb.create<memref::DeallocOp>(loc, curArg);
          tb.create<scf::YieldOp>(loc);
        },
        [&](OpBuilder &eb, Location loc) {
          eb.create<scf::YieldOp>(loc);
        });

      Value trueI1 = b.create<arith::ConstantIntOp>(loc, 1, 1);
      b.create<scf::YieldOp>(loc, ValueRange{next, trueI1});
    });
    
    Value curFinal = layerLoop.getResult(0);
    Value ownsFinal = layerLoop.getResult(1);

    auto resTy = mlir::MemRefType::get({1, mlir::ShapedType::kDynamic}, rewriter.getF64Type());
    Value result2D = rewriter.create<memref::AllocOp>(loc, resTy, ValueRange{outIdx});
    
    rewriter.create<scf::ForOp>(loc, c0, outIdx, c1, ValueRange{},
      [&](OpBuilder &b, Location loc, Value j, ValueRange) {
        Value v = b.create<memref::LoadOp>(loc, curFinal, ValueRange{j});
        b.create<memref::StoreOp>(loc, v, result2D, ValueRange{c0, j});
        b.create<scf::YieldOp>(loc);
      });

    rewriter.create<scf::IfOp>(loc, ownsFinal,
      [&](OpBuilder &b, Location loc) {
        b.create<memref::DeallocOp>(loc, curFinal);
        b.create<scf::YieldOp>(loc);
      },
      [&](OpBuilder &b, Location loc) {
        b.create<scf::YieldOp>(loc);
      });

      
      //Value yTensor = rewriter.create<bufferization::ToTensorOp>(loc, op.getType(), curFinal);
    rewriter.replaceOp(op, result2D);      
    return success();
  };


};// namespace

/*
Model train(Model model, Tensor2D dataset, int64_t epochs) {
  int in  = model.meta[0];
  int dep = model.meta[1];
  int wid = model.meta[2];
  int out = model.meta[3];

  int B = dataset.dim0; // batch size

  int maxDim = max(in, max(wid, out));

  // Allocate once (memrefs in MLIR lowering)
  TrainBuffers buf;
  buf.acts      = alloc<double>((dep + 1) * maxDim);
  buf.delta     = alloc<double>(maxDim);
  buf.deltaPrev = alloc<double>(maxDim);
  buf.grad      = alloc<double>(model.params.size);
  
  for (int64_t e = 0; e < epochs; ++e) {

    // ------------------------
    // (0) zero gradients
    // ------------------------
    for (int p = 0; p < model.params.size; ++p)
      buf.grad[p] = 0.0;

    // ------------------------
    // (1) loop over batch: forward + backward accumulate
    // ------------------------
    for (int b = 0; b < B; ++b) {

      // ===== (1A) Forward pass =====
      // store input x into acts at layer 0
      for (int i = 0; i < in; ++i)
        buf.acts[0*maxDim + i] = dataset[b][i];

      // layer-by-layer forward (duplicate your predict loops)
      for (int l = 0; l < dep; ++l) {
        int inDim  = (l == 0) ? in  : wid;
        int outDim = (l == dep-1) ? out : wid;

        index wBase = model.paramsMap[2*l];
        index bBase = model.paramsMap[2*l + 1];

        for (int row = 0; row < outDim; ++row) {
          double acc = model.params[bBase + row]; // bias

          for (int col = 0; col < inDim; ++col) {
            index wIdx = wBase + row*inDim + col;
            double w   = model.params[wIdx];
            double a   = buf.acts[l*maxDim + col];
            acc += w * a;
          }

          // activation: tanh except last layer
          double aOut = (l == dep-1) ? acc : tanh(acc);

          buf.acts[(l+1)*maxDim + row] = aOut;
        }
      }

      // ===== (1B) Backward pass =====

      // --- output delta (size out), MSE with linear output ---
      // delta_out[i] = (2/out) * (yhat[i] - y[i])
      for (int i = 0; i < out; ++i) {
        double yhat = buf.acts[dep*maxDim + i];
        double y    = dataset[b][in + i];
        buf.delta[i] = (2.0 / double(out)) * (yhat - y);
      }

      // --- walk layers backward ---
      for (int l = dep-1; l >= 0; --l) {
        int inDim  = (l == 0) ? in  : wid;
        int outDim = (l == dep-1) ? out : wid;

        index wBase = model.paramsMap[2*l];
        index bBase = model.paramsMap[2*l + 1];

        // (i) accumulate gradients for this layer using current delta
        // db[row] += delta[row]
        // dW[row,col] += delta[row] * a_prev[col]
        for (int row = 0; row < outDim; ++row) {
          double d = buf.delta[row];

          buf.grad[bBase + row] += d;

          for (int col = 0; col < inDim; ++col) {
            index wIdx = wBase + row*inDim + col;
            double aPrev = buf.acts[l*maxDim + col];
            buf.grad[wIdx] += d * aPrev;
          }
        }

        // (ii) compute delta for previous layer (unless l==0)
        if (l > 0) {
          // deltaPrev[k] = sum_row W[row,k] * delta[row]
          // then * tanh'(z_{l-1})  BUT we use activation:
          // tanh' = 1 - a^2 where a is acts[l, k]
          for (int k = 0; k < inDim; ++k) {
            double sum = 0.0;
            for (int row = 0; row < outDim; ++row) {
              index wIdx = wBase + row*inDim + k;   // W[row,k]
              double w   = model.params[wIdx];
              sum += w * buf.delta[row];
            }

            // l-1 layer output activation is acts[l, k] (because layer l-1 produced acts[l])
            double a = buf.acts[l*maxDim + k];
            double deriv = 1.0 - a*a; // tanh'(z) using activation

            buf.deltaPrev[k] = sum * deriv;
          }

          // swap delta buffers
          for (int k = 0; k < inDim; ++k)
            buf.delta[k] = buf.deltaPrev[k];
        }
      }
    }

    // ------------------------
    // (2) Parameter update (BGD)
    // ------------------------
    double invB = 1.0 / double(B);
    for (int p = 0; p < model.params.size; ++p) {
      double g = buf.grad[p] * invB;           // average gradient
      model.params[p] -= LR * g;               // gradient descent step
    }
  }

  return model;
}

*/
static Value buildMax(Value a, Value b,
  PatternRewriter &rewriter,
  Location loc) {
Value cmp = rewriter.create<arith::CmpIOp>(
loc, arith::CmpIPredicate::slt, a, b);
return rewriter.create<arith::SelectOp>(loc, cmp, b, a);
}

struct LowerTrainOp : OpConversionPattern<mlir::toy::TrainOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::toy::TrainOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const override {

      Location loc = op.getLoc();
      auto *ctx = rewriter.getContext();

      Value modelValue = adaptor.getModel(); 
      Value dataset = adaptor.getDataset();
      auto lrAttr = adaptor.getLearningRateAttr();
      Value lrVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), lrAttr);

      auto memRefTy = llvm::dyn_cast<MemRefType>(dataset.getType());
      if (!memRefTy) {
        op.emitOpError() << "expected dataset to be a memref, got " << dataset.getType();
        return failure();
      }
  
      if (memRefTy.getRank() != 2) {
        llvm::errs() << ">>> LowerTrainOp FAIL at check 2\n";
        return rewriter.notifyMatchFailure(op, "dataset memref must be rank-2");
      }
    
      IntegerAttr epochsAttr = adaptor.getEpochsAttr();
      Value epochsI64 = rewriter.create<arith::ConstantIntOp>(loc, epochsAttr.getInt(), 64);
      Value epochsIdx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), epochsI64);
      

      Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      Value c2 = rewriter.create<arith::ConstantIndexOp>(loc, 2);
      Value c3 = rewriter.create<arith::ConstantIndexOp>(loc, 3);
      

      auto metaTy = mlir::MemRefType::get({4}, rewriter.getI64Type());
      Value meta = rewriter.create<toy::StructAccessOp>(loc, metaTy, modelValue, rewriter.getI64IntegerAttr(0));
            
      auto paramsTy = MemRefType::get({ShapedType::kDynamic}, rewriter.getF64Type());
      Value params = rewriter.create<toy::StructAccessOp>(loc, paramsTy, modelValue, rewriter.getI64IntegerAttr(1));

      auto paramsMapTy = MemRefType::get({ShapedType::kDynamic}, rewriter.getIndexType());
      Value paramsMap = rewriter.create<toy::StructAccessOp>(loc, paramsMapTy, modelValue, rewriter.getI64IntegerAttr(2));

      Value in = rewriter.create<memref::LoadOp>(loc, meta, ValueRange{c0});
      Value dep = rewriter.create<memref::LoadOp>(loc, meta, ValueRange{c1});
      Value wid = rewriter.create<memref::LoadOp>(loc, meta, ValueRange{c2});
      Value out = rewriter.create<memref::LoadOp>(loc, meta, ValueRange{c3});

      Value inIdx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), in);
      Value depIdx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), dep);
      Value widIdx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), wid);
      Value outIdx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), out);
     
      Value maxInWid = buildMax(inIdx, widIdx, rewriter, loc);
      Value maxDim = buildMax(maxInWid, outIdx, rewriter, loc);

      auto memVecTy = MemRefType::get({ShapedType::kDynamic}, rewriter.getF64Type());

      Value depIdxPlusOne = rewriter.create<arith::AddIOp>(loc, depIdx, c1);
      Value activationsLength = rewriter.create<arith::MulIOp>(loc, depIdxPlusOne, maxDim);
      Value activations = rewriter.create<memref::AllocOp>(loc, memVecTy, ValueRange{activationsLength});
      
      Value delta = rewriter.create<memref::AllocOp>(loc, memVecTy, ValueRange{maxDim});
      Value deltaPrev = rewriter.create<memref::AllocOp>(loc, memVecTy, ValueRange{maxDim});
      
      Value numParams = rewriter.create<memref::DimOp>(loc, params, 0);
      Value grad = rewriter.create<memref::AllocOp>(loc, memVecTy, ValueRange{numParams});

      Value BIdx = rewriter.create<memref::DimOp>(loc, dataset, 0);

      auto f64Ty = rewriter.getF64Type();
      auto scalarTy = mlir::MemRefType::get({}, f64Ty); // rank-0 memref
      Value mseAcc = rewriter.create<memref::AllocOp>(loc, scalarTy);

      

      scf::ForOp epochLoop = rewriter.create<scf::ForOp>(loc, c0, epochsIdx, c1);
      {
        OpBuilder::InsertionGuard guard1(rewriter);
        rewriter.setInsertionPointToStart(epochLoop.getBody());

        Value zero = rewriter.create<arith::ConstantOp>(
          loc, f64Ty, rewriter.getFloatAttr(f64Ty, 0.0));
          rewriter.create<memref::StoreOp>(loc, zero, mseAcc, ValueRange{});

        Value idxI64 = rewriter.create<arith::IndexCastOp>(
          loc,
          rewriter.getI64Type(),
          epochLoop.getInductionVar()
      );

        Value idxF64 = rewriter.create<arith::SIToFPOp>(
          loc,
          rewriter.getF64Type(),
          idxI64
        );
        
        Value mem = rewriter.create<memref::AllocOp>(
          loc,
          MemRefType::get({}, rewriter.getF64Type()) // rank-0 memref
        );
      
        rewriter.create<memref::StoreOp>(loc, idxF64, mem);
        rewriter.create<toy::PrintOp>(loc, mem);
        rewriter.create<memref::DeallocOp>(loc, mem);
        scf::ForOp gradientResetLoop = rewriter.create<scf::ForOp>(loc, c0, numParams, c1);
        {
          OpBuilder::InsertionGuard guard2(rewriter);
          rewriter.setInsertionPointToStart(gradientResetLoop.getBody());
          Value iterator = gradientResetLoop.getInductionVar(); 
          auto f64Ty = rewriter.getF64Type();
          auto zeroAttr = rewriter.getFloatAttr(f64Ty, 0.0);
          Value zero = rewriter.create<arith::ConstantOp>(loc, f64Ty, zeroAttr);
          rewriter.create<memref::StoreOp>(loc, zero, grad, ValueRange{iterator});
        }      

        scf::ForOp batchLoop = rewriter.create<scf::ForOp>(loc, c0, BIdx, c1);
        {
          OpBuilder::InsertionGuard guardB(rewriter);
          rewriter.setInsertionPointToStart(batchLoop.getBody());

          Value b = batchLoop.getInductionVar();

          // ------------------------------------------------------------
          // (A) "Load inputs" into activations for layer 0
          //     activations[0*maxDim + i] = x[i]
          // ------------------------------------------------------------
          scf::ForOp loadX = rewriter.create<scf::ForOp>(loc, c0, inIdx, c1);
          {
            OpBuilder::InsertionGuard guardX(rewriter);
            rewriter.setInsertionPointToStart(loadX.getBody());

            Value i = loadX.getInductionVar();

            
            // x = dataset[b, i]
            Value x = rewriter.create<memref::LoadOp>(loc, dataset, ValueRange{b, i});
            
            // activations[0*maxDim + i] == activations[i] = x
            rewriter.create<memref::StoreOp>(loc, x, activations, ValueRange{i});
          }

          // ------------------------------------------------------------
          // (B) Forward pass over layers
          //     writes activations for each layer into the big buffer
          // ------------------------------------------------------------
          scf::ForOp layerLoop = rewriter.create<scf::ForOp>(loc, c0, depIdx, c1);
          {
            OpBuilder::InsertionGuard guardL(rewriter);
            rewriter.setInsertionPointToStart(layerLoop.getBody());

            Value l = layerLoop.getInductionVar();

            // paramsMap indexing: [2*l] = wBase, [2*l+1] = bBase
            Value idxW = rewriter.create<arith::MulIOp>(loc, l, c2);
            Value idxB = rewriter.create<arith::AddIOp>(loc, idxW, c1);

            Value wBase = rewriter.create<memref::LoadOp>(loc, paramsMap, ValueRange{idxW});
            Value bBase = rewriter.create<memref::LoadOp>(loc, paramsMap, ValueRange{idxB});

            Value isFirst = rewriter.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::eq, l, c0);

            Value lastLayer = rewriter.create<arith::SubIOp>(loc, depIdx, c1);
            Value isLastLayer = rewriter.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::eq, l, lastLayer);

            // inDim = (l==0) ? in : wid
            Value inDim = rewriter.create<arith::SelectOp>(loc, isFirst, inIdx, widIdx);
            // outDim = (l==last) ? out : wid
            Value outDim = rewriter.create<arith::SelectOp>(loc, isLastLayer, outIdx, widIdx);

            // base offsets into the flat activations buffer
            // prevBase = l * maxDim
            // nextBase = (l+1) * maxDim
            Value prevBase = rewriter.create<arith::MulIOp>(loc, l, maxDim);
            Value lPlus1   = rewriter.create<arith::AddIOp>(loc, l, c1);
            Value nextBase = rewriter.create<arith::MulIOp>(loc, lPlus1, maxDim);

            scf::ForOp rowLoop = rewriter.create<scf::ForOp>(loc, c0, outDim, c1);
            {
              OpBuilder::InsertionGuard guardR(rewriter);
              rewriter.setInsertionPointToStart(rowLoop.getBody());

              Value row = rowLoop.getInductionVar();

              // acc = bias
              Value biasIdx = rewriter.create<arith::AddIOp>(loc, bBase, row);
              Value bias = rewriter.create<memref::LoadOp>(loc, params, ValueRange{biasIdx});
              Value acc0 = bias;

              // acc += sum_j w[row,j] * aPrev[j]
              scf::ForOp colLoop = rewriter.create<scf::ForOp>(
                loc, c0, inDim, c1, ValueRange{acc0},
                [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
                  // iv = col
                  Value col = iv;
                  Value acc = iterArgs[0];
            
                  // weightIdx = wBase + row*inDim + col
                  Value rowInDim = b.create<arith::MulIOp>(loc, row, inDim);
                  Value wOff     = b.create<arith::AddIOp>(loc, wBase, rowInDim);
                  Value weightIdx= b.create<arith::AddIOp>(loc, wOff, col);
                  Value w        = b.create<memref::LoadOp>(loc, params, ValueRange{weightIdx});
            
                  // x = activations[prevBase + col]
                  Value xIdx = b.create<arith::AddIOp>(loc, prevBase, col);
                  Value x    = b.create<memref::LoadOp>(loc, activations, ValueRange{xIdx});
            
                  Value prod    = b.create<arith::MulFOp>(loc, w, x);
                  Value accNext = b.create<arith::AddFOp>(loc, acc, prod);
            
                  b.create<scf::YieldOp>(loc, accNext);
                });

              Value accFinal = colLoop.getResult(0);

              
              Value isLast = isLastLayer; 
              scf::IfOp actIf = rewriter.create<scf::IfOp>(
                  loc, TypeRange{rewriter.getF64Type()}, isLast, /*withElse=*/true);
              {
                OpBuilder thenB = actIf.getThenBodyBuilder();
                thenB.create<scf::YieldOp>(loc, ValueRange{accFinal});
              }
              {
                OpBuilder elseB = actIf.getElseBodyBuilder();
                Value act = elseB.create<math::TanhOp>(loc, accFinal);
                elseB.create<scf::YieldOp>(loc, ValueRange{act});
              }
              Value aOut = actIf.getResult(0);

              // activations[nextBase + row] = aOut
              Value outIdxLin = rewriter.create<arith::AddIOp>(loc, nextBase, row);
              rewriter.create<memref::StoreOp>(loc, aOut, activations, ValueRange{outIdxLin});

            }

          }

          // for (int i = 0; i < out; ++i) {
          //   double yhat = buf.acts[dep*maxDim + i];
          //   double y    = dataset[b][in + i];
          //   buf.delta[i] = (2.0 / double(out)) * (yhat - y);
          // }

          scf::ForOp deltaLoop = rewriter.create<scf::ForOp>(loc, c0, outIdx, c1);
          {
            OpBuilder::InsertionGuard guardD(rewriter);
            rewriter.setInsertionPointToStart(deltaLoop.getBody());

            Value deltaIterator = deltaLoop.getInductionVar();

            Value depMulMaxDim = rewriter.create<arith::MulIOp>(loc, depIdx, maxDim);
            Value yHatPosition = rewriter.create<arith::AddIOp>(loc, depMulMaxDim, deltaIterator);
            Value yHat = rewriter.create<memref::LoadOp>(loc, activations, ValueRange{yHatPosition});
            Value yDatasetPostion = rewriter.create<arith::AddIOp>(loc, inIdx, deltaIterator);
            Value y = rewriter.create<memref::LoadOp>(loc, dataset, ValueRange{b, yDatasetPostion});
            Value ySub = rewriter.create<arith::SubFOp>(loc, yHat, y);
            Value outF64 = rewriter.create<arith::SIToFPOp>(loc, rewriter.getF64Type(), out);
            Value two = rewriter.create<arith::ConstantOp>(loc, rewriter.getF64Type(),
            rewriter.getFloatAttr(rewriter.getF64Type(), 2.0));           
            Value fraction = rewriter.create<arith::DivFOp>(loc, two, outF64);
            Value computedDelta = rewriter.create<arith::MulFOp>(loc, fraction, ySub);

            rewriter.create<memref::StoreOp>(loc, computedDelta, delta, ValueRange{deltaIterator});

            Value sq = rewriter.create<arith::MulFOp>(loc, ySub, ySub);

            // mseAcc += sq
            Value cur = rewriter.create<memref::LoadOp>(loc, mseAcc, ValueRange{});
            Value next = rewriter.create<arith::AddFOp>(loc, cur, sq);
            rewriter.create<memref::StoreOp>(loc, next, mseAcc, ValueRange{});                        
          }

          Value negativeBound = rewriter.create<arith::ConstantIndexOp>(loc, -1);
          Value decrement   = rewriter.create<arith::ConstantIndexOp>(loc, -1);
          Value depIdxMinusOne = rewriter.create<arith::SubIOp>(loc, depIdx, c1);
          Value c4 = rewriter.create<arith::ConstantIndexOp>(loc, -1);
          scf::ForOp walkLayersBackwardsLoop = rewriter.create<scf::ForOp>(loc, c0, depIdx, c1);
          {
            OpBuilder::InsertionGuard guardWalkLayersBack(rewriter);
            rewriter.setInsertionPointToStart(walkLayersBackwardsLoop.getBody());
            
            Value t = walkLayersBackwardsLoop.getInductionVar();                 //
            Value layer = rewriter.create<arith::SubIOp>(loc, depIdxMinusOne, t);

            Value isFirstLayer = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, layer, c0);
            Value isLastLayer = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, layer, depIdxMinusOne);

            //   int inDim  = (l == 0) ? in  : wid;
            Value inDim = rewriter.create<arith::SelectOp>(loc, isFirstLayer, inIdx, widIdx);
            //   int outDim = (l == dep-1) ? out : wid;
            Value outDim = rewriter.create<arith::SelectOp>(loc, isLastLayer, outIdx, widIdx);

            Value idxW = rewriter.create<arith::MulIOp>(loc, layer, c2);
            Value idxB = rewriter.create<arith::AddIOp>(loc, idxW, c1);
            
            //index wBase = model.paramsMap[2*l];
            Value wBase = rewriter.create<memref::LoadOp>(loc, paramsMap, ValueRange{idxW});
            //index bBase = model.paramsMap[2*l + 1];
            Value bBase = rewriter.create<memref::LoadOp>(loc, paramsMap, ValueRange{idxB});
            //PROVERI OVAJ DEO
            scf::ForOp gradientRowLoop = rewriter.create<scf::ForOp>(loc, c0, outDim, c1);
            {
              OpBuilder::InsertionGuard guardGradientRowLoop(rewriter);
              rewriter.setInsertionPointToStart(gradientRowLoop.getBody());

              Value row = gradientRowLoop.getInductionVar();

              Value bBasePlusRow = rewriter.create<arith::AddIOp>(loc, bBase, row);
              Value d = rewriter.create<memref::LoadOp>(loc, delta, ValueRange{row});
              Value currGrad = rewriter.create<memref::LoadOp>(loc, grad, ValueRange{bBasePlusRow});
              Value updatedGrad = rewriter.create<arith::AddFOp>(loc, currGrad, d);
                           
              rewriter.create<memref::StoreOp>(loc, updatedGrad, grad, ValueRange{bBasePlusRow});              

              scf::ForOp gradientColLoop = rewriter.create<scf::ForOp>(loc, c0, inDim, c1);
              {
                OpBuilder::InsertionGuard guardgradientColLoop(rewriter);
                rewriter.setInsertionPointToStart(gradientColLoop.getBody());

                Value col = gradientColLoop.getInductionVar();

                Value rowMulInDim = rewriter.create<arith::MulIOp>(loc, row, inDim);
                Value rowMulInDimPlusRow = rewriter.create<arith::AddIOp>(loc, rowMulInDim, col);
                Value wIdx = rewriter.create<arith::AddIOp>(loc, rowMulInDimPlusRow, wBase);

                Value layerMulMaxDim = rewriter.create<arith::MulIOp>(loc, layer, maxDim);
                Value actsIdx = rewriter.create<arith::AddIOp>(loc, layerMulMaxDim, col);
                Value aPrev = rewriter.create<memref::LoadOp>(loc, activations, ValueRange{actsIdx});

                Value prod = rewriter.create<arith::MulFOp>(loc, d, aPrev);
                Value gw = rewriter.create<memref::LoadOp>(loc, grad, ValueRange{wIdx});
                Value gwNext = rewriter.create<arith::AddFOp>(loc, gw, prod);
                rewriter.create<memref::StoreOp>(loc, gwNext, grad, ValueRange{wIdx});

              }

            }
            //Proveri ovaj deo
            Value isNotFirst = rewriter.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::sgt, layer, c0);
          
            scf::IfOp ifNotFirst =rewriter.create<scf::IfOp>(loc, isNotFirst, /*withElse=*/false);
            {
              OpBuilder::InsertionGuard guardIf(rewriter);
              rewriter.setInsertionPointToStart(ifNotFirst.thenBlock());
              
              scf::ForOp  calculatePrevDelta = rewriter.create<scf::ForOp>(loc, c0, inDim, c1);
              {
                OpBuilder::InsertionGuard guardPrevDeltaLoop(rewriter);
                rewriter.setInsertionPointToStart(calculatePrevDelta.getBody());


                Value k = calculatePrevDelta.getInductionVar();
                
                auto f64 = rewriter.getF64Type();
                Value zero = rewriter.create<arith::ConstantOp>(
                loc, f64, rewriter.getFloatAttr(f64, 0.0));
                scf::ForOp sumLoop = rewriter.create<scf::ForOp>(
                  loc, c0, outDim, c1, ValueRange{zero},
                  [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
                    // iv = row
                    Value row = iv;
                    Value sum = iterArgs[0];
              
                    // weightIdx = wBase + row*inDim + k
                    Value rowInDim  = b.create<arith::MulIOp>(loc, row, inDim);
                    Value wOff      = b.create<arith::AddIOp>(loc, wBase, rowInDim);
                    Value weightIdx = b.create<arith::AddIOp>(loc, wOff, k);
                    Value w         = b.create<memref::LoadOp>(loc, params, ValueRange{weightIdx});
              
                    Value d = b.create<memref::LoadOp>(loc, delta, ValueRange{row});
              
                    // sumNext = sum + (w * d)
                    Value prod    = b.create<arith::MulFOp>(loc, w, d);
                    Value sumNext = b.create<arith::AddFOp>(loc, sum, prod);
              
                    b.create<scf::YieldOp>(loc, sumNext);
                  });
                Value sumFinal = sumLoop.getResult(0);
                Value layerMulMax = rewriter.create<arith::MulIOp>(loc, layer, maxDim);
                Value aIdx    = rewriter.create<arith::AddIOp>(loc, layerMulMax, k);
                Value a = rewriter.create<memref::LoadOp>(loc, activations, ValueRange{aIdx});
                  // one = 1.0 : f64
                Value one = rewriter.create<arith::ConstantOp>(loc, f64, rewriter.getFloatAttr(f64, 1.0));


                Value a2 = rewriter.create<arith::MulFOp>(loc, a, a);
                Value deriv = rewriter.create<arith::SubFOp>(loc, one, a2);

                  // deltaPrevVal = sum * deriv
                Value deltaPrevVal = rewriter.create<arith::MulFOp>(loc, sumFinal, deriv);

                  // deltaPrev[k] = deltaPrevVal
                rewriter.create<memref::StoreOp>(loc, deltaPrevVal, deltaPrev, ValueRange{k});
              }

              scf::ForOp swapDeltaLoop = rewriter.create<scf::ForOp>(loc, c0, inDim, c1);
              {
                OpBuilder::InsertionGuard guardSwap(rewriter);
                rewriter.setInsertionPointToStart(swapDeltaLoop.getBody());

                Value k = swapDeltaLoop.getInductionVar();

                // tmp = deltaPrev[k]
                Value tmp = rewriter.create<memref::LoadOp>(
                    loc, deltaPrev, ValueRange{k});

                // delta[k] = tmp
                rewriter.create<memref::StoreOp>(
                    loc, tmp, delta, ValueRange{k});


              }

            }            

          }
          
        }

        auto f64 = rewriter.getF64Type();
        Value one = rewriter.create<arith::ConstantOp>(loc, f64, rewriter.getFloatAttr(f64, 1.0));
      
        // cast B (index) -> f64
        Value BF = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), BIdx);
        BF = rewriter.create<arith::SIToFPOp>(loc, f64, BF);
        
        Value invB = rewriter.create<arith::DivFOp>(loc, one, BF);        
        
        // for p in [0..numParams)
        scf::ForOp paramLoop =rewriter.create<scf::ForOp>(loc, c0, numParams, c1);
        {
          OpBuilder::InsertionGuard guardP(rewriter);
          rewriter.setInsertionPointToStart(paramLoop.getBody());
        
          Value p = paramLoop.getInductionVar();
        
          // g = grad[p] * invB
          Value gp = rewriter.create<memref::LoadOp>(loc, grad, ValueRange{p});
          Value gAvg = rewriter.create<arith::MulFOp>(loc, gp, invB);
        
          // params[p] -= lr * gAvg
          Value pp = rewriter.create<memref::LoadOp>(loc, params, ValueRange{p});
          Value step = rewriter.create<arith::MulFOp>(loc, lrVal, gAvg);
          Value newP = rewriter.create<arith::SubFOp>(loc, pp, step);

        
          rewriter.create<memref::StoreOp>(loc, newP, params, ValueRange{p});
        
         
        }

      


        Value batchI64 = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), BIdx);
        Value batchF64 = rewriter.create<arith::SIToFPOp>(loc, f64Ty, batchI64);
        Value outF64 = rewriter.create<arith::SIToFPOp>(loc, rewriter.getF64Type(), out);
        Value denom    = rewriter.create<arith::MulFOp>(loc, batchF64, outF64);

        Value sse = rewriter.create<memref::LoadOp>(loc, mseAcc, ValueRange{});
        Value mse = rewriter.create<arith::DivFOp>(loc, sse, denom);

        Value rmse = rewriter.create<math::SqrtOp>(loc, mse);

        auto f64Ty = rewriter.getF64Type();
        auto scalarMemrefTy = mlir::MemRefType::get({}, f64Ty);

        Value rmseBuf = rewriter.create<memref::AllocaOp>(loc, scalarMemrefTy);
        rewriter.create<memref::StoreOp>(loc, rmse, rmseBuf, ValueRange{});

        rewriter.create<toy::PrintOp>(loc, rmseBuf);
        
      }

      Value loweredModel = modelValue; 


      Type origResTy = op.getResult().getType(); // !toy.model
      Value back = rewriter.create<UnrealizedConversionCastOp>(loc, origResTy, loweredModel).getResult(0);


      rewriter.create<memref::DeallocOp>(loc, activations);
      rewriter.create<memref::DeallocOp>(loc, grad);
      rewriter.create<memref::DeallocOp>(loc, delta);
      rewriter.create<memref::DeallocOp>(loc, deltaPrev);
      rewriter.create<memref::DeallocOp>(loc, mseAcc);
      rewriter.replaceOp(op, back);
      return success();
 
    }
};

//===----------------------------------------------------------------------===//
// ToyToAffineLoweringPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine loops of the toy operations that are
/// computationally intensive (like matmul for example...) while keeping the
/// rest of the code in the Toy dialect.
namespace {
struct ToyToAffineLoweringPass
    : public PassWrapper<ToyToAffineLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyToAffineLoweringPass)
  StringRef getArgument() const override { return "toy-to-affine"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, func::FuncDialect,
                    memref::MemRefDialect, math::MathDialect, scf::SCFDialect>();
  }
  void runOnOperation() final;
};
} // namespace

void ToyToAffineLoweringPass::runOnOperation() {
  MLIRContext &ctx = getContext();

  // --- Conversion target ----------------------------------------------------
  
  TypeConverter typeConverter;

  // One conversion that handles everything (old MLIR friendly).
  typeConverter.addConversion([&](mlir::Type t) -> mlir::Type {
    // tensor -> memref (make ALL dims dynamic, keep rank)
    if (auto rt = llvm::dyn_cast<mlir::RankedTensorType>(t)) {
      SmallVector<int64_t> dynShape(rt.getRank(), mlir::ShapedType::kDynamic);
      return mlir::MemRefType::get(dynShape, rt.getElementType());
    }
  
    if (auto ut = llvm::dyn_cast<mlir::UnrankedTensorType>(t)) {
      return mlir::UnrankedMemRefType::get(ut.getElementType(), /*memorySpace=*/0);
    }
  
    if (llvm::isa<mlir::toy::ModelType>(t)) {
      return getLoweredModelType(t.getContext());
    }
  
    return t;
  });

  typeConverter.addTargetMaterialization(
    [&](mlir::OpBuilder &b, mlir::Type resultType, mlir::ValueRange inputs,
        mlir::Location loc) -> mlir::Value {
      if (inputs.size() != 1) return mlir::Value();
  
      auto srcTy = llvm::dyn_cast<mlir::MemRefType>(inputs[0].getType());
      auto dstTy = llvm::dyn_cast<mlir::MemRefType>(resultType);
      if (!srcTy || !dstTy) return mlir::Value();
  
      if (!mlir::memref::CastOp::areCastCompatible(srcTy, dstTy))
        return mlir::Value();
  
      return b.create<mlir::memref::CastOp>(loc, dstTy, inputs[0]).getResult();
    });
  
  typeConverter.addSourceMaterialization(
    [&](mlir::OpBuilder &b, mlir::Type resultType, mlir::ValueRange inputs,
        mlir::Location loc) -> mlir::Value {
      if (inputs.size() != 1) return mlir::Value();
  
      auto srcTy = llvm::dyn_cast<mlir::MemRefType>(inputs[0].getType());
      auto dstTy = llvm::dyn_cast<mlir::MemRefType>(resultType);
      if (!srcTy || !dstTy) return mlir::Value();
  
      if (!mlir::memref::CastOp::areCastCompatible(srcTy, dstTy))
        return mlir::Value();
  
      return b.create<mlir::memref::CastOp>(loc, dstTy, inputs[0]).getResult();
    });

  ConversionTarget target(ctx);

  target.addLegalDialect<affine::AffineDialect, 
                         BuiltinDialect,
                         arith::ArithDialect,
                         memref::MemRefDialect, 
                         math::MathDialect, 
                         scf::SCFDialect,
                         func::FuncDialect>();

  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return typeConverter.isSignatureLegal(op.getFunctionType()) &&
    typeConverter.isLegal(&op.getBody());
  });

  target.addLegalOp<mlir::UnrealizedConversionCastOp, func::FuncOp, func::ReturnOp>();

  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return typeConverter.isSignatureLegal(op.getFunctionType()) &&
           typeConverter.isLegal(&op.getBody());
  });

  target.addIllegalOp<mlir::toy::TrainOp>();
  // All Toy ops must be eliminated, except toy.print (under a condition).
  target.addIllegalDialect<toy::ToyDialect>();

  // toy.print is only legal once its operands are no longer tensors.
   target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op) {
      return llvm::none_of(op->getOperandTypes(),
      [](Type type) { return llvm::isa<TensorType>(type); });
   });

  target.addDynamicallyLegalOp<toy::MakeStructOp>([](toy::MakeStructOp op) {
    return llvm::all_of(op.getInputs(), [](Value v) {
      return llvm::isa<MemRefType, UnrankedMemRefType>(v.getType());
    });
  });

  target.addDynamicallyLegalOp<toy::StructAccessOp>([](toy::StructAccessOp op) {
    // Not legal if it still operates on the high-level model type.
    if (llvm::isa<toy::ModelType>(op.getInput().getType()))
      return false;
  
    // After conversion, accessing a field should yield a lowered type:
    // - MemRefType / UnrankedMemRefType for buffers
    // - IndexType for maps (if you use index)
    // - IntegerType for meta loads (if you ever access meta directly)
    Type rTy = op.getResult().getType();
    return llvm::isa<mlir::MemRefType,
                     mlir::UnrankedMemRefType,
                     mlir::IndexType,
                     mlir::IntegerType,
                     mlir::FloatType>(rTy);
  });

  // --- TypeConverter: tensor<...> -> memref<...> ---------------------------
    

  


  
  // --- Patterns -------------------------------------------------------------
  RewritePatternSet patterns(&ctx);

  // Most patterns don’t *use* the converter explicitly, but it’s fine to
  // construct them with it; ModelOpLowering *does* rely on getTypeConverter().
  patterns.add(std::make_unique<LowerToyFuncByName>(typeConverter, &ctx));
  
  patterns.add<AddOpLowering,
               ConstantOpLowering,
               MulOpLowering,
               PrintOpLowering,
               ReturnOpLowering,
               TransposeOpLowering,
               LowerCreateModel,
               LowerPredictOp,
               LowerTrainOp>(typeConverter, &ctx);
              

  // --- Apply conversion -----------------------------------------------------
  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))){
    llvm::errs() << "\n=== Partial conversion FAILED. Dumping IR ===\n";
    getOperation()->dump();
    signalPassFailure();
    return;
    }
  }
}
/// Create a pass for lowering operations in the `Affine` and `Std` dialects,
/// for a subset of the Toy IR (e.g. matmul).
std::unique_ptr<Pass> mlir::toy::createLowerToAffinePass() {
  return std::make_unique<ToyToAffineLoweringPass>();
}
