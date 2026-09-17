#!/usr/bin/env bash
# apply_patches5.sh — apply retdec upgrade patches (round 5)
# Usage: cd <patches5-dir> && ./apply_patches5.sh <retdec-root>
set -euo pipefail

ROOT="${1:?Usage: $0 <retdec-root>}"
PATCHES="$(cd "$(dirname "$0")" && pwd)"

[[ -d "$ROOT" ]] || { echo "ERROR: $ROOT not found"; exit 1; }

echo "==> RetDec root: $ROOT"
echo "==> Patches:     $PATCHES"
echo ""

BIN2_CMAKE="$ROOT/src/bin2llvmir/CMakeLists.txt"
HLL_CMAKE="$ROOT/src/llvmir2hll/CMakeLists.txt"

#─────────────────────────────────────────────────
# Patch 1 — llvm_intrinsic_converter extension
#─────────────────────────────────────────────────
echo "[1/4] LLVM Intrinsic Converter Extension (20+ new mappings)"

cp "$PATCHES/intrinsic_conv_ext/llvm_intrinsic_converter_ext.cpp" \
   "$ROOT/src/llvmir2hll/llvm/llvm_intrinsic_converter_ext.cpp"
cp "$PATCHES/intrinsic_conv_ext/llvm_intrinsic_converter_ext.h" \
   "$ROOT/include/retdec/llvmir2hll/llvm/llvm_intrinsic_converter_ext.h"

# Add to CMakeLists
if ! grep -q "llvm_intrinsic_converter_ext" "$HLL_CMAKE"; then
    sed -i 's|llvm/llvm_intrinsic_converter.cpp|llvm/llvm_intrinsic_converter.cpp\n\tllvm/llvm_intrinsic_converter_ext.cpp|' "$HLL_CMAKE"
    echo "    Added to $HLL_CMAKE"
fi

# Patch llvm_intrinsic_converter.cpp to call getExtendedIntrinsicName
CONV="$ROOT/src/llvmir2hll/llvm/llvm_intrinsic_converter.cpp"
if ! grep -q "getExtendedIntrinsicName" "$CONV"; then
    sed -i '1s|^|#include "retdec/llvmir2hll/llvm/llvm_intrinsic_converter_ext.h"\n|' "$CONV"

    python3 - "$CONV" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path) as f:
    src = f.read()

# Extend convertIntrinsicFuncName to call extension at the end
old = (
    '\t// llvm.copysign\n'
    '\telse if (startsWith(funcName, "llvm.copysign")) {\n'
    '\t\trenameFloatIntrinsicFunc(func, "copysign");\n'
    '\t}\n'
    '}\n'
)
new = (
    '\t// llvm.copysign\n'
    '\telse if (startsWith(funcName, "llvm.copysign")) {\n'
    '\t\trenameFloatIntrinsicFunc(func, "copysign");\n'
    '\t}\n'
    '\t// Extended mappings (llvm_intrinsic_converter_ext.cpp)\n'
    '\telse {\n'
    '\t\tauto extName = getExtendedIntrinsicName(funcName);\n'
    '\t\tif (!extName.empty() && extName != "<strip>") {\n'
    '\t\t\trenameIntrinsicFunc(func, extName);\n'
    '\t\t}\n'
    '\t}\n'
    '}\n'
)
if old in src:
    src = src.replace(old, new, 1)
    with open(path, 'w') as f:
        f.write(src)
    print("    Patched convertIntrinsicFuncName with extension")
else:
    print("    WARNING: copysign block not found verbatim")
PYEOF

    # Patch visit(CallStmt) to strip metadata intrinsics
    python3 - "$CONV" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path) as f:
    src = f.read()

# Original: only strips llvm.ctpop.* void calls.
# We extend it to also strip metadata intrinsics (lifetime, dbg, etc.)
old = (
    'void LLVMIntrinsicsOptimizer::visit(ShPtr<CallStmt> stmt) {\n'
    '\tShPtr<Function> calledFunc(getCalledFunc(stmt->getCall()));\n'
    '\tif (!calledFunc || calledFunc->isDefinition() ||\n'
    '\t\t\t!startsWith(calledFunc->getInitialName(), "llvm.ctpop.")) {\n'
    '\t\tFuncOptimizer::visit(stmt);\n'
    '\t\treturn;\n'
    '\t}\n'
    '\n'
    '\t// It is a call to llvm.ctpop.*. Remove it.\n'
)
new = (
    'void LLVMIntrinsicsOptimizer::visit(ShPtr<CallStmt> stmt) {\n'
    '\tShPtr<Function> calledFunc(getCalledFunc(stmt->getCall()));\n'
    '\tif (!calledFunc || calledFunc->isDefinition()) {\n'
    '\t\tFuncOptimizer::visit(stmt);\n'
    '\t\treturn;\n'
    '\t}\n'
    '\tconst auto& iname = calledFunc->getInitialName();\n'
    '\tbool shouldStrip = startsWith(iname, "llvm.ctpop.")\n'
    '\t    || isStrippableLLVMIntrinsic(iname);\n'
    '\tif (!shouldStrip) {\n'
    '\t\tFuncOptimizer::visit(stmt);\n'
    '\t\treturn;\n'
    '\t}\n'
    '\t// Strip this no-op intrinsic call.\n'
)
if old in src:
    src = src.replace(old, new, 1)
    with open(path, 'w') as f:
        f.write(src)
    print("    Patched visit(CallStmt) to strip metadata intrinsics")
else:
    print("    WARNING: visit(CallStmt) block not found verbatim in llvm_intrinsics_optimizer; trying converter")
    # Note: LLVMIntrinsicConverter and LLVMIntrinsicsOptimizer are different files
    # The CallStmt visitor is in llvm_intrinsics_optimizer, not the converter
PYEOF
fi

# Also patch the optimizer (different file)
OPT="$ROOT/src/llvmir2hll/optimizer/optimizers/llvm_intrinsics_optimizer.cpp"
if ! grep -q "isStrippableLLVMIntrinsic" "$OPT"; then
    sed -i '1s|^|#include "retdec/llvmir2hll/llvm/llvm_intrinsic_converter_ext.h"\n|' "$OPT"
    python3 - "$OPT" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path) as f:
    src = f.read()

old = (
    '\tif (!calledFunc || calledFunc->isDefinition() ||\n'
    '\t\t\t!startsWith(calledFunc->getInitialName(), "llvm.ctpop.")) {\n'
    '\t\tFuncOptimizer::visit(stmt);\n'
    '\t\treturn;\n'
    '\t}\n'
    '\n'
    '\t// It is a call to llvm.ctpop.*. Remove it.\n'
)
new = (
    '\tconst auto& _iname = calledFunc->getInitialName();\n'
    '\tbool _strip = startsWith(_iname, "llvm.ctpop.")\n'
    '\t    || isStrippableLLVMIntrinsic(_iname);\n'
    '\tif (!calledFunc || calledFunc->isDefinition() || !_strip) {\n'
    '\t\tFuncOptimizer::visit(stmt);\n'
    '\t\treturn;\n'
    '\t}\n'
    '\t// Strip metadata / no-op intrinsic call.\n'
)
if old in src:
    src = src.replace(old, new, 1)
    with open(path, 'w') as f:
        f.write(src)
    print("    Patched llvm_intrinsics_optimizer.cpp")
else:
    print("    WARNING: llvm_intrinsics_optimizer guard not found verbatim")
PYEOF
fi

#─────────────────────────────────────────────────
# Patch 2 — if_structure_optimizer patterns 6 & 7
#─────────────────────────────────────────────────
echo "[2/4] if_structure_optimizer Patterns 6 & 7"

cp "$PATCHES/if_structure_ext/if_structure_optimizer_ext.cpp" \
   "$ROOT/src/llvmir2hll/optimizer/optimizers/if_structure_optimizer_ext.cpp"
cp "$PATCHES/if_structure_ext/if_structure_optimizer_ext.h" \
   "$ROOT/include/retdec/llvmir2hll/optimizer/optimizers/if_structure_optimizer_ext.h"

if ! grep -q "if_structure_optimizer_ext" "$HLL_CMAKE"; then
    sed -i 's|optimizer/optimizers/if_structure_optimizer.cpp|optimizer/optimizers/if_structure_optimizer.cpp\n\toptimizer/optimizers/if_structure_optimizer_ext.cpp|' "$HLL_CMAKE"
    echo "    Added to $HLL_CMAKE"
fi

IF_STRUCT="$ROOT/src/llvmir2hll/optimizer/optimizers/if_structure_optimizer.cpp"
if ! grep -q "tryOptimization6" "$IF_STRUCT"; then
    sed -i '1s|^|#include "retdec/llvmir2hll/optimizer/optimizers/if_structure_optimizer_ext.h"\n|' "$IF_STRUCT"
    python3 - "$IF_STRUCT" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path) as f:
    src = f.read()

old = '\ttryOptimization5(stmt);\n}\n'
new = '\ttryOptimization5(stmt);\n\ttryOptimization6(stmt);\n\ttryOptimization7(stmt);\n}\n'
if old in src:
    src = src.replace(old, new, 1)
    with open(path, 'w') as f:
        f.write(src)
    print("    Wired tryOptimization6 and tryOptimization7")
else:
    print("    WARNING: tryOptimization5 call not found verbatim")
PYEOF
fi

#─────────────────────────────────────────────────
# Patch 3 — pow2_sub_optimizer (simplify_arithm_expr)
#─────────────────────────────────────────────────
echo "[3/4] Pow2 Arithmetic Sub-Optimizer"

cp "$PATCHES/pow2_arithm_ext/pow2_sub_optimizer.cpp" \
   "$ROOT/src/llvmir2hll/optimizer/optimizers/simplify_arithm_expr/pow2_sub_optimizer.cpp"
cp "$PATCHES/pow2_arithm_ext/pow2_sub_optimizer.h" \
   "$ROOT/include/retdec/llvmir2hll/optimizer/optimizers/simplify_arithm_expr/pow2_sub_optimizer.h"

if ! grep -q "pow2_sub_optimizer" "$HLL_CMAKE"; then
    sed -i 's|optimizer/optimizers/simplify_arithm_expr/zero_sub_optimizer.cpp|optimizer/optimizers/simplify_arithm_expr/zero_sub_optimizer.cpp\n\toptimizer/optimizers/simplify_arithm_expr/pow2_sub_optimizer.cpp|' "$HLL_CMAKE"
    echo "    Added to $HLL_CMAKE"
fi
echo "    Pow2SubOptimizer auto-registers via REGISTER_AT_FACTORY macro"

#─────────────────────────────────────────────────
# Patch 4 — strength_reduction bin2llvmir pass
#─────────────────────────────────────────────────
echo "[4/4] Strength Reduction Pass (bin2llvmir)"

mkdir -p "$ROOT/src/bin2llvmir/optimizations/strength_reduction"
mkdir -p "$ROOT/include/retdec/bin2llvmir/optimizations/strength_reduction"

cp "$PATCHES/strength_reduction/strength_reduction.cpp" \
   "$ROOT/src/bin2llvmir/optimizations/strength_reduction/strength_reduction.cpp"
cp "$PATCHES/strength_reduction/strength_reduction.h" \
   "$ROOT/include/retdec/bin2llvmir/optimizations/strength_reduction/strength_reduction.h"

if ! grep -q "strength_reduction" "$BIN2_CMAKE"; then
    sed -i 's|optimizations/constants/constants.cpp|optimizations/constants/constants.cpp\n\toptimizations/strength_reduction/strength_reduction.cpp|' "$BIN2_CMAKE"
    echo "    Added to $BIN2_CMAKE"
fi


# Wire strength_reduction into the JSON pass pipeline
DCFG="$ROOT/src/retdec-decompiler/decompiler-config.json"
if [[ -f "$DCFG" ]] && ! grep -q "retdec-strength-reduction" "$DCFG"; then
    sed -i 's|"retdec-constants",|"retdec-constants",\n            "retdec-strength-reduction",|' "$DCFG"
    echo "    Wired retdec-strength-reduction into decompiler-config.json (after retdec-constants)"
fi

echo ""
echo "==> All patches applied."
