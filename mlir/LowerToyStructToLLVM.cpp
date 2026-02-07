//====- LowerToyStructToLLVM.cpp - Lower toy struct ops to LLVM -----------===//
//
// Lowers:
//   - toy::StructType        -> !llvm.struct<...>
//   - toy.make_struct        -> llvm.undef + llvm.insertvalue
//   - toy.struct_access      -> llvm.extractvalue
//
//===----------------------------------------------------------------------===//

#include "toy/Dialect.h"
#include "toy/Passes.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

static void addToyStructTypeConversion(LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion([&](toy::StructType t) -> Type {
    SmallVector<Type> llvmFields;
    llvmFields.reserve(t.getElementTypes().size());

    for (Type fieldTy : t.getElementTypes()) {
      Type lowered = typeConverter.convertType(fieldTy);
      if (!lowered)
        return Type(); // signal failure
      llvmFields.push_back(lowered);
    }

    return LLVM::LLVMStructType::getLiteral(t.getContext(), llvmFields);
  });
}

struct MakeStructOpLowering : public OpConversionPattern<toy::MakeStructOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(toy::MakeStructOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Type lowered = getTypeConverter()->convertType(op.getResult().getType());
    if (!lowered)
      return rewriter.notifyMatchFailure(op, "failed to convert result struct type");

    auto structTy = llvm::dyn_cast<LLVM::LLVMStructType>(lowered);
    if (!structTy)
      return rewriter.notifyMatchFailure(op, "lowered type is not LLVMStructType");

    Value agg = rewriter.create<LLVM::UndefOp>(loc, structTy);

    for (auto it : llvm::enumerate(adaptor.getInputs())) {
      int64_t i = it.index();
      Value v = it.value();

      // LLVM insert/extract indices are DenseI64ArrayAttr in current MLIR APIs.
      auto idxAttr = rewriter.getDenseI64ArrayAttr({i});

      agg = rewriter.create<LLVM::InsertValueOp>(
          loc,
          structTy,   // result type
          agg,        // aggregate
          v,          // value to insert
          idxAttr     // position
      );
    }

    rewriter.replaceOp(op, agg);
    return success();
  }
};

struct StructAccessOpLowering : public OpConversionPattern<toy::StructAccessOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(toy::StructAccessOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Prefer converting the op result type (more robust than reading struct body).
    Type resTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!resTy)
      return rewriter.notifyMatchFailure(op, "failed to convert result type");

    int64_t idx = op.getIndex(); // from I64Attr:$index
    auto idxAttr = rewriter.getDenseI64ArrayAttr({idx});

    Value elem = rewriter.create<LLVM::ExtractValueOp>(
        loc,
        resTy,                // result type
        adaptor.getInput(),   // aggregate
        idxAttr               // position
    );

    rewriter.replaceOp(op, elem);
    return success();
  }
};

} // namespace

void mlir::toy::populateToyStructToLLVMConversionPatterns(
    mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns) {
  addToyStructTypeConversion(typeConverter);

  patterns.add<MakeStructOpLowering, StructAccessOpLowering>(
      typeConverter, patterns.getContext());
}
