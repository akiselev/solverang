# Implementation Plan: Expression Graph Python API (Design F)

## Executive Summary

Build a Python API where operator overloading (`+`, `*`, `**`, etc.) constructs
Rust-side expression trees that are symbolically differentiated, lowered to
`ConstraintOp` opcodes, JIT-compiled via Cranelift, and solved entirely in Rust
with the GIL released. Control flow is handled via branchless `Select` nodes
(hardware cmov/csel), making piecewise functions differentiable and fast.

**Result**: Users write natural Python math, get Rust-native solve performance
with automatic Jacobians. No Python callbacks during solve.

```python
import solverang as sr

x, y = sr.variables("x y")
result = sr.solve(residuals=[x**2 + y**2 - 1, x - y], x0=[0.5, 0.5])
# Entire pipeline: differentiate → JIT compile → solve runs in Rust, GIL released
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Python                                                                  │
│   x, y = sr.variables("x y")                                           │
│   r = x**2 + y**2 - 1.0       ← operator overloads build Rust Expr    │
│   sr.solve(residuals=[r], x0=...)  ← triggers Rust pipeline           │
└────────────────┬────────────────────────────────────────────────────────┘
                 │ PyO3 boundary (expressions cross as Rust structs)
┌────────────────▼────────────────────────────────────────────────────────┐
│ Rust: crates/solverang/src/expr/  (new module)                         │
│                                                                         │
│   RuntimeExpr tree                                                      │
│       │                                                                 │
│       ├─► differentiate()  → Jacobian RuntimeExpr trees                │
│       ├─► simplify()       → algebraically reduced trees               │
│       ▼                                                                 │
│   emit() via OpcodeEmitter → Vec<ConstraintOp>                         │
│       │                                                                 │
│       ▼                                                                 │
│   JITCompiler (Cranelift) → native fn pointers                         │
│       │                                                                 │
│       ▼                                                                 │
│   ExprProblem implements Problem trait                                  │
│       │                                                                 │
│       ▼                                                                 │
│   AutoSolver/LMSolver/Solver  (GIL released)                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: RuntimeExpr in Core Crate

**Goal**: A runtime expression tree with symbolic differentiation, simplification,
direct evaluation, and opcode emission. This is the foundation everything else
builds on.

**Location**: `crates/solverang/src/expr/` (new module, behind `runtime-expr`
feature flag)

### 1.1 RuntimeExpr Enum

Port the macro crate's `Expr` to a runtime-constructable version. The macro
crate's `Expr` lives in a proc-macro crate and operates on `TokenStream`; we
need a version that operates on values at runtime.

```
File: crates/solverang/src/expr/mod.rs
```

```rust
pub mod expr;
pub mod differentiate;
pub mod simplify;
pub mod emit;
pub mod evaluate;
pub mod display;
pub mod problem;

pub use expr::RuntimeExpr;
pub use problem::ExprProblem;
```

```
File: crates/solverang/src/expr/expr.rs
```

```rust
use std::collections::BTreeSet;

#[derive(Clone, Debug)]
pub enum RuntimeExpr {
    /// Variable reference by index into the flat state vector.
    Var(u32),

    /// Literal constant.
    Const(f64),

    /// Negation: -e
    Neg(Box<RuntimeExpr>),

    /// Addition: a + b
    Add(Box<RuntimeExpr>, Box<RuntimeExpr>),

    /// Subtraction: a - b
    Sub(Box<RuntimeExpr>, Box<RuntimeExpr>),

    /// Multiplication: a * b
    Mul(Box<RuntimeExpr>, Box<RuntimeExpr>),

    /// Division: a / b
    Div(Box<RuntimeExpr>, Box<RuntimeExpr>),

    /// Power with constant exponent: base^exp
    /// Exponent must be a compile-time constant (not a variable).
    Pow(Box<RuntimeExpr>, f64),

    /// Square root: sqrt(e)
    Sqrt(Box<RuntimeExpr>),

    /// Sine: sin(e)
    Sin(Box<RuntimeExpr>),

    /// Cosine: cos(e)
    Cos(Box<RuntimeExpr>),

    /// Tangent: tan(e)
    Tan(Box<RuntimeExpr>),

    /// Two-argument arctangent: atan2(y, x)
    Atan2(Box<RuntimeExpr>, Box<RuntimeExpr>),

    /// Absolute value: |e|
    Abs(Box<RuntimeExpr>),

    /// Maximum: max(a, b)
    Max(Box<RuntimeExpr>, Box<RuntimeExpr>),

    /// Minimum: min(a, b)
    Min(Box<RuntimeExpr>, Box<RuntimeExpr>),

    // ─── Control flow (branchless) ───

    /// Floating-point comparison: produces a boolean-like value.
    /// Result is 1.0 if condition holds, 0.0 otherwise.
    Compare {
        a: Box<RuntimeExpr>,
        b: Box<RuntimeExpr>,
        cond: CmpCondition,
    },

    /// Branchless conditional select: if condition != 0 then true_val else false_val.
    /// Both branches are always evaluated; the result is selected via cmov.
    Select {
        condition: Box<RuntimeExpr>,
        on_true: Box<RuntimeExpr>,
        on_false: Box<RuntimeExpr>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmpCondition {
    Gt,  // >
    Ge,  // >=
    Lt,  // <
    Le,  // <=
    Eq,  // ==
    Ne,  // !=
}

impl RuntimeExpr {
    pub fn is_zero(&self) -> bool {
        matches!(self, RuntimeExpr::Const(v) if *v == 0.0)
    }

    pub fn is_one(&self) -> bool {
        matches!(self, RuntimeExpr::Const(v) if *v == 1.0)
    }

    /// Collect all variable indices referenced in this expression.
    pub fn variables(&self) -> BTreeSet<u32> {
        let mut vars = BTreeSet::new();
        self.collect_vars(&mut vars);
        vars
    }

    fn collect_vars(&self, vars: &mut BTreeSet<u32>) {
        match self {
            RuntimeExpr::Var(idx) => { vars.insert(*idx); }
            RuntimeExpr::Const(_) => {}
            RuntimeExpr::Neg(e) | RuntimeExpr::Sqrt(e) | RuntimeExpr::Sin(e)
            | RuntimeExpr::Cos(e) | RuntimeExpr::Tan(e) | RuntimeExpr::Abs(e) => {
                e.collect_vars(vars);
            }
            RuntimeExpr::Pow(base, _) => { base.collect_vars(vars); }
            RuntimeExpr::Add(a, b) | RuntimeExpr::Sub(a, b)
            | RuntimeExpr::Mul(a, b) | RuntimeExpr::Div(a, b)
            | RuntimeExpr::Atan2(a, b) | RuntimeExpr::Max(a, b)
            | RuntimeExpr::Min(a, b) => {
                a.collect_vars(vars);
                b.collect_vars(vars);
            }
            RuntimeExpr::Compare { a, b, .. } => {
                a.collect_vars(vars);
                b.collect_vars(vars);
            }
            RuntimeExpr::Select { condition, on_true, on_false } => {
                condition.collect_vars(vars);
                on_true.collect_vars(vars);
                on_false.collect_vars(vars);
            }
        }
    }
}
```

### 1.2 Symbolic Differentiation

Direct port from `crates/macros/src/expr.rs` `Expr::differentiate()`, adapted
for `RuntimeExpr`. Uses `u32` variable index instead of `VarRef`.

```
File: crates/solverang/src/expr/differentiate.rs
```

Key differences from macro crate:
- Uses `u32` var index instead of `VarRef.id`
- Adds differentiation rules for new nodes: `Max`, `Min`, `Compare`, `Select`

**Differentiation rules for control flow nodes**:

```rust
// d/dx max(a, b) = select(a >= b, da, db)
RuntimeExpr::Max(a, b) => {
    let da = a.differentiate(var_idx);
    let db = b.differentiate(var_idx);
    RuntimeExpr::Select {
        condition: Box::new(RuntimeExpr::Compare {
            a: a.clone(),
            b: b.clone(),
            cond: CmpCondition::Ge,
        }),
        on_true: Box::new(da),
        on_false: Box::new(db),
    }
}

// d/dx min(a, b) = select(a <= b, da, db)
RuntimeExpr::Min(a, b) => {
    let da = a.differentiate(var_idx);
    let db = b.differentiate(var_idx);
    RuntimeExpr::Select {
        condition: Box::new(RuntimeExpr::Compare {
            a: a.clone(),
            b: b.clone(),
            cond: CmpCondition::Le,
        }),
        on_true: Box::new(da),
        on_false: Box::new(db),
    }
}

// d/dx |e| = select(e >= 0, de, -de)  (subgradient at 0: use +de)
RuntimeExpr::Abs(e) => {
    let de = e.differentiate(var_idx);
    RuntimeExpr::Select {
        condition: Box::new(RuntimeExpr::Compare {
            a: e.clone(),
            b: Box::new(RuntimeExpr::Const(0.0)),
            cond: CmpCondition::Ge,
        }),
        on_true: Box::new(de.clone()),
        on_false: Box::new(RuntimeExpr::Neg(Box::new(de))),
    }
}

// d/dx select(c, f, g) = select(c, df, dg)
// The condition is treated as non-differentiable (it's a boolean).
RuntimeExpr::Select { condition, on_true, on_false } => {
    let dt = on_true.differentiate(var_idx);
    let df = on_false.differentiate(var_idx);
    RuntimeExpr::Select {
        condition: condition.clone(),
        on_true: Box::new(dt),
        on_false: Box::new(df),
    }
}

// Compare nodes are non-differentiable (step functions).
// d/dx (a > b) = 0 everywhere (except at the switching point, undefined).
RuntimeExpr::Compare { .. } => RuntimeExpr::Const(0.0),
```

### 1.3 Simplification

Port from macro crate's `Expr::simplify()`. Add simplification rules for new nodes.

```
File: crates/solverang/src/expr/simplify.rs
```

Additional rules:
- `Select(Const(1.0), t, f)` → `t`
- `Select(Const(0.0), t, f)` → `f`
- `Select(c, t, t)` → `t` (both branches identical)
- `Max(Const(a), Const(b))` → `Const(a.max(b))`
- `Min(Const(a), Const(b))` → `Const(a.min(b))`

### 1.4 Opcode Emission

Lower `RuntimeExpr` trees to `Vec<ConstraintOp>` via the existing `OpcodeEmitter`.

```
File: crates/solverang/src/expr/emit.rs
```

```rust
use crate::jit::{ConstraintOp, OpcodeEmitter, Reg};
use super::expr::{RuntimeExpr, CmpCondition};

impl RuntimeExpr {
    /// Emit opcodes for this expression, returning the register holding the result.
    pub fn emit(&self, emitter: &mut OpcodeEmitter) -> Reg {
        match self {
            RuntimeExpr::Var(idx) => emitter.load_var(*idx),
            RuntimeExpr::Const(v) => emitter.const_f64(*v),
            RuntimeExpr::Neg(e) => {
                let r = e.emit(emitter);
                emitter.neg(r)
            }
            RuntimeExpr::Add(a, b) => {
                let ra = a.emit(emitter);
                let rb = b.emit(emitter);
                emitter.add(ra, rb)
            }
            // ... Sub, Mul, Div same pattern ...
            RuntimeExpr::Pow(base, exp) => {
                let rb = base.emit(emitter);
                match *exp {
                    0.0 => emitter.const_f64(1.0),
                    1.0 => rb,
                    2.0 => emitter.square(rb),
                    0.5 => emitter.sqrt(rb),
                    -1.0 => {
                        let one = emitter.one();
                        emitter.div(one, rb)
                    }
                    -2.0 => {
                        let sq = emitter.square(rb);
                        let one = emitter.one();
                        emitter.div(one, sq)
                    }
                    n if n == (n as i32) as f64 && n > 0.0 && n <= 8.0 => {
                        // Small positive integer: expand as repeated multiplication
                        let mut result = rb;
                        for _ in 1..(n as i32) {
                            result = emitter.mul(result, rb);
                        }
                        result
                    }
                    _ => {
                        // General case: emit exp(exp * ln(base))
                        // Requires Exp and Ln opcodes (see Phase 4)
                        // For now, fall back to approximate or panic
                        todo!("general power requires Exp/Ln opcodes")
                    }
                }
            }
            RuntimeExpr::Max(a, b) => {
                let ra = a.emit(emitter);
                let rb = b.emit(emitter);
                emitter.max(ra, rb)
            }
            RuntimeExpr::Min(a, b) => {
                let ra = a.emit(emitter);
                let rb = b.emit(emitter);
                emitter.min(ra, rb)
            }
            RuntimeExpr::Compare { a, b, cond } => {
                let ra = a.emit(emitter);
                let rb = b.emit(emitter);
                emitter.fcmp(ra, rb, *cond)  // new emitter method
            }
            RuntimeExpr::Select { condition, on_true, on_false } => {
                let rc = condition.emit(emitter);
                let rt = on_true.emit(emitter);
                let rf = on_false.emit(emitter);
                emitter.select(rc, rt, rf)  // new emitter method
            }
            // ... Sqrt, Sin, Cos, Tan, Atan2, Abs same as existing patterns ...
        }
    }
}
```

### 1.5 Direct Evaluation (Interpreted Fallback)

For platforms where Cranelift is unavailable (not x86_64/aarch64) and for
debugging/testing.

```
File: crates/solverang/src/expr/evaluate.rs
```

```rust
impl RuntimeExpr {
    /// Evaluate this expression with the given variable values.
    /// This is the interpreted (non-JIT) fallback.
    pub fn evaluate(&self, vars: &[f64]) -> f64 {
        match self {
            RuntimeExpr::Var(idx) => vars[*idx as usize],
            RuntimeExpr::Const(v) => *v,
            RuntimeExpr::Neg(e) => -e.evaluate(vars),
            RuntimeExpr::Add(a, b) => a.evaluate(vars) + b.evaluate(vars),
            RuntimeExpr::Sub(a, b) => a.evaluate(vars) - b.evaluate(vars),
            RuntimeExpr::Mul(a, b) => a.evaluate(vars) * b.evaluate(vars),
            RuntimeExpr::Div(a, b) => a.evaluate(vars) / b.evaluate(vars),
            RuntimeExpr::Pow(base, exp) => base.evaluate(vars).powf(*exp),
            RuntimeExpr::Sqrt(e) => e.evaluate(vars).sqrt(),
            RuntimeExpr::Sin(e) => e.evaluate(vars).sin(),
            RuntimeExpr::Cos(e) => e.evaluate(vars).cos(),
            RuntimeExpr::Tan(e) => e.evaluate(vars).tan(),
            RuntimeExpr::Atan2(y, x) => y.evaluate(vars).atan2(x.evaluate(vars)),
            RuntimeExpr::Abs(e) => e.evaluate(vars).abs(),
            RuntimeExpr::Max(a, b) => a.evaluate(vars).max(b.evaluate(vars)),
            RuntimeExpr::Min(a, b) => a.evaluate(vars).min(b.evaluate(vars)),
            RuntimeExpr::Compare { a, b, cond } => {
                let va = a.evaluate(vars);
                let vb = b.evaluate(vars);
                let result = match cond {
                    CmpCondition::Gt => va > vb,
                    CmpCondition::Ge => va >= vb,
                    CmpCondition::Lt => va < vb,
                    CmpCondition::Le => va <= vb,
                    CmpCondition::Eq => va == vb,
                    CmpCondition::Ne => va != vb,
                };
                if result { 1.0 } else { 0.0 }
            }
            RuntimeExpr::Select { condition, on_true, on_false } => {
                if condition.evaluate(vars) != 0.0 {
                    on_true.evaluate(vars)
                } else {
                    on_false.evaluate(vars)
                }
            }
        }
    }
}
```

### 1.6 Display (Pretty-Printing)

```
File: crates/solverang/src/expr/display.rs
```

Human-readable display for debugging and Python `__repr__`:
- `Var(0)` → `"x0"` (or named if provided)
- `Add(Var(0), Const(1.0))` → `"x0 + 1"`
- `Pow(Var(0), 2.0)` → `"x0**2"`
- `Select(Compare(...), a, b)` → `"where(x0 > 0, x0**2, -x0)"`

### 1.7 ExprProblem (Problem Trait Implementation)

```
File: crates/solverang/src/expr/problem.rs
```

```rust
pub struct ExprProblem {
    name: String,
    num_vars: usize,
    residual_exprs: Vec<RuntimeExpr>,
    /// Sparse Jacobian: for each residual row, a list of (col, derivative_expr)
    jacobian_exprs: Vec<Vec<(u32, RuntimeExpr)>>,
    /// JIT-compiled evaluation (None if JIT unavailable or compilation failed)
    jit_fn: Option<JITFunction>,
}

impl ExprProblem {
    pub fn new(name: String, num_vars: usize, residuals: Vec<RuntimeExpr>) -> Self {
        // 1. For each residual, find which variables it references
        // 2. Differentiate w.r.t. each referenced variable
        // 3. Simplify the derivative expressions
        // 4. Filter out zero derivatives (sparse)
        // 5. Try to JIT compile
    }
}

impl Problem for ExprProblem {
    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        // Use JIT if available, else interpreted fallback
    }
    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        // Use JIT if available, else interpreted fallback
    }
    // ...
}
```

### 1.8 New Opcodes: FCmp and Select

Add to existing `ConstraintOp` enum:

```
File: crates/solverang/src/jit/opcodes.rs (modify)
```

```rust
// New variants in ConstraintOp:

/// Floating-point comparison. Result is 1.0 if condition holds, 0.0 otherwise.
/// Lowers to Cranelift fcmp instruction.
FCmp {
    dst: Reg,
    a: Reg,
    b: Reg,
    cond: CmpCondition,
},

/// Branchless conditional select: dst = condition != 0 ? true_val : false_val.
/// Lowers to Cranelift select (cmov/csel on hardware).
///
/// SAFETY NOTE: In the branchless opcode stream, both branches are computed
/// before the select. For domain-restricted operations (sqrt of negative,
/// division by zero), users must guard the branch inputs to avoid NaN/Inf.
/// The library provides safe_div() and safe_sqrt() helpers for common cases.
///
/// ALTERNATIVE: For cases where branch evaluation cost or domain safety is
/// critical, we can emit basic blocks with conditional jumps instead of
/// cmov. This is a compilation strategy choice -- the RuntimeExpr::Select
/// node can lower to either form. The branchless form is preferred for
/// solver workloads because:
///   1. No branch mispredictions (solver evaluates at many different x values)
///   2. Flat opcode stream (simpler JIT, simpler interpreted fallback)
///   3. Both branches are typically cheap arithmetic
/// If profiling shows that lazy evaluation is needed (e.g., one branch is
/// very expensive), we can add a LazySelect variant that uses basic blocks.
Select {
    dst: Reg,
    condition: Reg,
    true_val: Reg,
    false_val: Reg,
},
```

### 1.9 Cranelift Translation for FCmp and Select

```
File: crates/solverang/src/jit/cranelift.rs (modify)
```

```rust
// In translate_ops(), add:

ConstraintOp::FCmp { dst, a, b, cond } => {
    let va = registers[&a];
    let vb = registers[&b];
    let cc = match cond {
        CmpCondition::Gt => FloatCC::GreaterThan,
        CmpCondition::Ge => FloatCC::GreaterThanOrEqual,
        CmpCondition::Lt => FloatCC::LessThan,
        CmpCondition::Le => FloatCC::LessThanOrEqual,
        CmpCondition::Eq => FloatCC::Equal,
        CmpCondition::Ne => FloatCC::NotEqual,
    };
    let cmp_bool = builder.ins().fcmp(cc, va, vb);
    // Convert boolean to f64: 1.0 if true, 0.0 if false
    // (Needed because our register file is all f64)
    let one = builder.ins().f64const(1.0);
    let zero = builder.ins().f64const(0.0);
    let result = builder.ins().select(cmp_bool, one, zero);
    registers.insert(*dst, result);
}

ConstraintOp::Select { dst, condition, true_val, false_val } => {
    let vc = registers[&condition];
    let vt = registers[&true_val];
    let vf = registers[&false_val];
    // Compare condition to zero to get a boolean
    let zero = builder.ins().f64const(0.0);
    let is_nonzero = builder.ins().fcmp(FloatCC::NotEqual, vc, zero);
    // Select: returns true_val if condition != 0, else false_val
    // This lowers to cmov on x86-64, csel on AArch64 -- no branches
    let result = builder.ins().select(is_nonzero, vt, vf);
    registers.insert(*dst, result);
}
```

### 1.10 OpcodeEmitter Extensions

```
File: crates/solverang/src/jit/lower.rs (modify)
```

Add methods to `OpcodeEmitter`:

```rust
/// Emit a floating-point comparison.
pub fn fcmp(&mut self, a: Reg, b: Reg, cond: CmpCondition) -> Reg {
    let dst = self.alloc_reg();
    self.ops.push(ConstraintOp::FCmp { dst, a, b, cond });
    dst
}

/// Emit a branchless select.
pub fn select(&mut self, condition: Reg, true_val: Reg, false_val: Reg) -> Reg {
    let dst = self.alloc_reg();
    self.ops.push(ConstraintOp::Select { dst, condition, true_val, false_val });
    dst
}
```

### Phase 1 Tests

```
File: crates/solverang/src/expr/tests.rs
```

- Differentiation correctness: verify `d/dx (x^2) = 2x` by evaluating at multiple points
- Differentiation of all node types (chain rule, product rule, quotient rule, trig)
- Differentiation of control flow: `d/dx max(x, 0)` = `select(x >= 0, 1, 0)`
- Simplification: `0 + x → x`, `1 * x → x`, `0 * x → 0`, constant folding
- Opcode emission: verify emitted opcodes match expected sequence
- Interpreted evaluation: verify correctness against hand-computed values
- JIT evaluation: verify JIT matches interpreted evaluation
- `ExprProblem` end-to-end: define a problem, solve it, verify solution
- Compare `ExprProblem` Jacobian against `verify_jacobian()` (finite differences)

### Phase 1 Deliverables

| File | Est. Lines | Status |
|------|-----------|--------|
| `crates/solverang/src/expr/mod.rs` | 15 | New |
| `crates/solverang/src/expr/expr.rs` | 120 | New |
| `crates/solverang/src/expr/differentiate.rs` | 180 | Port from macro crate |
| `crates/solverang/src/expr/simplify.rs` | 140 | Port from macro crate |
| `crates/solverang/src/expr/emit.rs` | 100 | New |
| `crates/solverang/src/expr/evaluate.rs` | 70 | New |
| `crates/solverang/src/expr/display.rs` | 80 | New |
| `crates/solverang/src/expr/problem.rs` | 150 | New |
| `crates/solverang/src/expr/tests.rs` | 300 | New |
| `crates/solverang/src/jit/opcodes.rs` | +30 | Modify |
| `crates/solverang/src/jit/cranelift.rs` | +40 | Modify |
| `crates/solverang/src/jit/lower.rs` | +20 | Modify |
| `crates/solverang/src/lib.rs` | +10 | Modify |
| **Total** | **~1,255** | |

---

## Phase 2: PyO3 Bindings

**Goal**: Expose RuntimeExpr to Python via operator overloading. Build the
`solverang-python` crate with maturin.

### 2.1 Crate Setup

```
File: crates/solverang-python/Cargo.toml
```

```toml
[package]
name = "solverang-python"
version = "0.1.0"
edition = "2021"

[lib]
name = "_solverang"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.22", features = ["extension-module", "abi3-py39"] }
numpy = "0.22"
solverang = { path = "../solverang", features = [
    "geometry", "jit", "runtime-expr", "parallel", "sparse"
] }
```

```
File: crates/solverang-python/pyproject.toml
```

```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "solverang"
requires-python = ">=3.9"
dependencies = ["numpy>=1.20"]
classifiers = [
    "Programming Language :: Rust",
    "Programming Language :: Python :: Implementation :: CPython",
]

[tool.maturin]
features = ["pyo3/extension-module"]
module-name = "solverang._solverang"
python-source = "python"
```

### 2.2 PyExpr: The Expression Node

```
File: crates/solverang-python/src/expr.rs
```

```rust
use pyo3::prelude::*;
use solverang::expr::RuntimeExpr;

/// A symbolic expression node. Built via operator overloads.
/// Immutable (frozen) -- all operations return new PyExpr instances.
#[pyclass(frozen, name = "Expr")]
#[derive(Clone)]
pub struct PyExpr {
    pub inner: RuntimeExpr,
    /// Display name for variables (e.g., "x", "y")
    pub name: Option<String>,
}

/// Accept either a PyExpr or a plain float from Python.
/// Enables writing `x + 1.0` without explicit wrapping.
#[derive(FromPyObject)]
pub enum ExprOrFloat {
    Expr(PyExpr),
    Float(f64),
    Int(i64),
}

impl ExprOrFloat {
    pub fn into_expr(self) -> RuntimeExpr {
        match self {
            ExprOrFloat::Expr(e) => e.inner,
            ExprOrFloat::Float(v) => RuntimeExpr::Const(v),
            ExprOrFloat::Int(v) => RuntimeExpr::Const(v as f64),
        }
    }
}

#[pymethods]
impl PyExpr {
    // ─── Arithmetic operators ───

    fn __add__(&self, other: ExprOrFloat) -> Self { /* Add node */ }
    fn __radd__(&self, other: ExprOrFloat) -> Self { /* Add node, reversed */ }
    fn __sub__(&self, other: ExprOrFloat) -> Self { /* Sub node */ }
    fn __rsub__(&self, other: ExprOrFloat) -> Self { /* Sub node, reversed */ }
    fn __mul__(&self, other: ExprOrFloat) -> Self { /* Mul node */ }
    fn __rmul__(&self, other: ExprOrFloat) -> Self { /* Mul node, reversed */ }
    fn __truediv__(&self, other: ExprOrFloat) -> Self { /* Div node */ }
    fn __rtruediv__(&self, other: ExprOrFloat) -> Self { /* Div node, reversed */ }
    fn __neg__(&self) -> Self { /* Neg node */ }
    fn __abs__(&self) -> Self { /* Abs node */ }

    fn __pow__(&self, exp: ExprOrFloat, _modulo: Option<PyObject>) -> PyResult<Self> {
        match exp {
            ExprOrFloat::Float(v) => Ok(PyExpr {
                inner: RuntimeExpr::Pow(Box::new(self.inner.clone()), v),
                name: None,
            }),
            ExprOrFloat::Int(v) => Ok(PyExpr {
                inner: RuntimeExpr::Pow(Box::new(self.inner.clone()), v as f64),
                name: None,
            }),
            ExprOrFloat::Expr(e) => {
                // Variable exponent: x**y requires Exp and Ln
                // Defer to Phase 4 or raise informative error
                Err(PyValueError::new_err(
                    "variable exponents (x**y) not yet supported; \
                     use constant exponents like x**2 or x**0.5"
                ))
            }
        }
    }

    // ─── Comparison operators (return Expr, not bool) ───
    // Python's __gt__ etc. must return a type that Python can use.
    // We return PyExpr wrapping a Compare node.
    //
    // __eq__ and __ne__ are overloaded to raise TypeError, preventing
    // silent bugs where `x == y` would use Python's default identity
    // comparison (always False) instead of building a Compare node.
    // Users must use sr.eq(x, y) for equality comparisons in expressions
    // and sr.ne(x, y) for inequality. This follows the "errors should
    // never pass silently" principle from the Zen of Python.

    fn __eq__(&self, _other: ExprOrFloat) -> PyResult<bool> {
        Err(PyTypeError::new_err(
            "Cannot use == on Expr objects (it would break hashing). \
             Use sr.eq(a, b) to build an equality comparison expression, \
             or sr.ne(a, b) for inequality."
        ))
    }

    fn __ne__(&self, _other: ExprOrFloat) -> PyResult<bool> {
        Err(PyTypeError::new_err(
            "Cannot use != on Expr objects (it would break hashing). \
             Use sr.ne(a, b) to build an inequality comparison expression."
        ))
    }

    fn __gt__(&self, other: ExprOrFloat) -> Self {
        PyExpr {
            inner: RuntimeExpr::Compare {
                a: Box::new(self.inner.clone()),
                b: Box::new(other.into_expr()),
                cond: CmpCondition::Gt,
            },
            name: None,
        }
    }

    fn __ge__(&self, other: ExprOrFloat) -> Self { /* Compare Ge */ }
    fn __lt__(&self, other: ExprOrFloat) -> Self { /* Compare Lt */ }
    fn __le__(&self, other: ExprOrFloat) -> Self { /* Compare Le */ }

    // ─── Math methods ───

    fn sqrt(&self) -> Self { /* Sqrt node */ }
    fn sin(&self) -> Self { /* Sin node */ }
    fn cos(&self) -> Self { /* Cos node */ }
    fn tan(&self) -> Self { /* Tan node */ }

    // ─── Symbolic differentiation ───

    fn diff(&self, var: &PyExpr) -> PyResult<Self> {
        match &var.inner {
            RuntimeExpr::Var(idx) => Ok(PyExpr {
                inner: self.inner.differentiate(*idx).simplify(),
                name: None,
            }),
            _ => Err(PyValueError::new_err(
                "can only differentiate with respect to a Variable"
            )),
        }
    }

    // ─── Inspection ───

    #[getter]
    fn variables(&self) -> Vec<u32> {
        self.inner.variables().into_iter().collect()
    }

    /// Evaluate the expression with concrete variable values.
    /// Useful for debugging.
    fn eval(&self, values: Vec<f64>) -> f64 {
        self.inner.evaluate(&values)
    }

    fn __repr__(&self) -> String {
        // Use Display implementation from display.rs
        format!("{}", self.inner)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}
```

### 2.3 Module-Level Functions

```
File: crates/solverang-python/src/functions.rs
```

```rust
use std::sync::atomic::{AtomicU32, Ordering};

/// Global allocator for unique variable indices across all `variables()` calls.
/// This prevents index collisions when variables are created in separate calls:
///   x = sr.variables("x")   # Var(0)
///   y = sr.variables("y")   # Var(1), not Var(0)!
///
/// Can be reset with sr.reset_variables() for a fresh problem.
static NEXT_VAR_INDEX: AtomicU32 = AtomicU32::new(0);

/// Create symbolic variables with globally unique indices.
/// Usage: x, y = sr.variables("x y")
///        xs = sr.variables("x", count=10)
#[pyfunction]
#[pyo3(signature = (names, *, count=None))]
fn variables(names: &str, count: Option<usize>) -> Vec<PyExpr> {
    match count {
        Some(n) => (0..n).map(|i| {
            let idx = NEXT_VAR_INDEX.fetch_add(1, Ordering::Relaxed);
            PyExpr {
                inner: RuntimeExpr::Var(idx),
                name: Some(format!("{}_{}", names.trim(), i)),
            }
        }).collect(),
        None => names.split_whitespace().map(|name| {
            let idx = NEXT_VAR_INDEX.fetch_add(1, Ordering::Relaxed);
            PyExpr {
                inner: RuntimeExpr::Var(idx),
                name: Some(name.to_string()),
            }
        }).collect(),
    }
}

/// Reset the global variable index counter. Call before defining a new problem
/// to start variable indices from 0.
#[pyfunction]
fn reset_variables() {
    NEXT_VAR_INDEX.store(0, Ordering::Relaxed);
}

/// Module-level math functions that operate on expressions.
#[pyfunction]
fn sqrt(e: ExprOrFloat) -> PyExpr { /* Sqrt node */ }
#[pyfunction]
fn sin(e: ExprOrFloat) -> PyExpr { /* Sin node */ }
#[pyfunction]
fn cos(e: ExprOrFloat) -> PyExpr { /* Cos node */ }
#[pyfunction]
fn tan(e: ExprOrFloat) -> PyExpr { /* Tan node */ }
#[pyfunction]
fn atan2(y: ExprOrFloat, x: ExprOrFloat) -> PyExpr { /* Atan2 node */ }

/// Branchless conditional: where(condition, on_true, on_false)
/// Both branches are always evaluated; the result is selected.
///
/// Usage:
///   r = sr.where(x > 0, x**2, -x)
///   r = sr.where(x > y, x - y, y - x)
#[pyfunction]
#[pyo3(name = "where")]
fn where_(condition: &PyExpr, on_true: ExprOrFloat, on_false: ExprOrFloat) -> PyExpr {
    PyExpr {
        inner: RuntimeExpr::Select {
            condition: Box::new(condition.inner.clone()),
            on_true: Box::new(on_true.into_expr()),
            on_false: Box::new(on_false.into_expr()),
        },
        name: None,
    }
}

/// Create a residual from an equation: eq(lhs, rhs) → lhs - rhs
#[pyfunction]
fn eq(lhs: ExprOrFloat, rhs: ExprOrFloat) -> PyExpr {
    PyExpr {
        inner: RuntimeExpr::Sub(
            Box::new(lhs.into_expr()),
            Box::new(rhs.into_expr()),
        ),
        name: None,
    }
}

/// max(a, b) as an expression node (differentiable via subgradient)
#[pyfunction]
fn max(a: ExprOrFloat, b: ExprOrFloat) -> PyExpr { /* Max node */ }

/// min(a, b) as an expression node (differentiable via subgradient)
#[pyfunction]
fn min(a: ExprOrFloat, b: ExprOrFloat) -> PyExpr { /* Min node */ }

/// Smooth absolute value: sqrt(x^2 + epsilon)
/// Useful when the derivative at x=0 matters for solver convergence.
#[pyfunction]
#[pyo3(signature = (e, epsilon=1e-8))]
fn smooth_abs(e: ExprOrFloat, epsilon: f64) -> PyExpr {
    let inner = e.into_expr();
    PyExpr {
        inner: RuntimeExpr::Sqrt(Box::new(RuntimeExpr::Add(
            Box::new(RuntimeExpr::Pow(Box::new(inner), 2.0)),
            Box::new(RuntimeExpr::Const(epsilon)),
        ))),
        name: None,
    }
}

/// Safe division: a / b when b != 0, else fill (default 0.0).
/// Avoids NaN/Inf from division by zero in both-branches-evaluated Select nodes.
#[pyfunction]
#[pyo3(signature = (a, b, fill=0.0))]
fn safe_div(a: ExprOrFloat, b: ExprOrFloat, fill: f64) -> PyExpr {
    let a_expr = a.into_expr();
    let b_expr = b.into_expr();
    PyExpr {
        inner: RuntimeExpr::Select {
            condition: Box::new(RuntimeExpr::Compare {
                a: Box::new(b_expr.clone()),
                b: Box::new(RuntimeExpr::Const(0.0)),
                cond: CmpCondition::Ne,
            }),
            on_true: Box::new(RuntimeExpr::Div(
                Box::new(a_expr),
                Box::new(b_expr),
            )),
            on_false: Box::new(RuntimeExpr::Const(fill)),
        },
        name: None,
    }
}

/// Equality comparison as expression node: sr.eq(a, b) → Compare(a, b, Eq)
/// Returns an Expr (not bool). Use instead of == which raises TypeError.
#[pyfunction]
#[pyo3(name = "eq")]
fn expr_eq(a: ExprOrFloat, b: ExprOrFloat) -> PyExpr {
    PyExpr {
        inner: RuntimeExpr::Compare {
            a: Box::new(a.into_expr()),
            b: Box::new(b.into_expr()),
            cond: CmpCondition::Eq,
        },
        name: None,
    }
}

/// Inequality comparison as expression node: sr.ne(a, b) → Compare(a, b, Ne)
/// Returns an Expr (not bool). Use instead of != which raises TypeError.
#[pyfunction]
fn ne(a: ExprOrFloat, b: ExprOrFloat) -> PyExpr {
    PyExpr {
        inner: RuntimeExpr::Compare {
            a: Box::new(a.into_expr()),
            b: Box::new(b.into_expr()),
            cond: CmpCondition::Ne,
        },
        name: None,
    }
}

/// Clamp: max(lo, min(hi, x))
#[pyfunction]
fn clamp(e: ExprOrFloat, lo: ExprOrFloat, hi: ExprOrFloat) -> PyExpr {
    let inner = e.into_expr();
    PyExpr {
        inner: RuntimeExpr::Max(
            Box::new(lo.into_expr()),
            Box::new(RuntimeExpr::Min(
                Box::new(hi.into_expr()),
                Box::new(inner),
            )),
        ),
        name: None,
    }
}
```

### 2.4 Solve Function

```
File: crates/solverang-python/src/solve.rs
```

```rust
/// Accept x0 as either a Python list or a numpy array.
/// This matches user expectations from the examples (x0=[0.5, 0.5])
/// while also supporting numpy arrays for larger problems.
#[derive(FromPyObject)]
enum InitialPoint<'py> {
    Array(PyReadonlyArray1<'py, f64>),
    List(Vec<f64>),
}

#[pyfunction]
#[pyo3(signature = (*, residuals=None, equations=None, x0,
                    solver=None, tolerance=None, max_iterations=None))]
fn solve(
    py: Python<'_>,
    residuals: Option<Vec<PyExpr>>,
    equations: Option<Vec<PyExpr>>,
    x0: InitialPoint<'_>,
    solver: Option<&str>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<PySolveResult> {
    // Error if both or neither residuals/equations provided
    let expr_vec: Vec<PyExpr> = match (residuals, equations) {
        (Some(res), None) => res,
        (None, Some(eq)) => eq,
        (Some(_), Some(_)) => {
            return Err(PyValueError::new_err(
                "must provide exactly one of 'residuals' or 'equations', not both",
            ));
        }
        (None, None) => {
            return Err(PyValueError::new_err(
                "must provide 'residuals' or 'equations'",
            ));
        }
    };

    let exprs: Vec<RuntimeExpr> = expr_vec.into_iter()
        .map(|e| e.inner)
        .collect();

    // Convert x0 to Vec<f64> regardless of input type
    let x0_vec: Vec<f64> = match x0 {
        InitialPoint::Array(arr) => arr.as_slice()?.to_vec(),
        InitialPoint::List(v) => v,
    };
    let num_vars = x0_vec.len();

    // Validate: all variable indices must be < num_vars
    for (i, expr) in exprs.iter().enumerate() {
        for var_idx in expr.variables() {
            if var_idx as usize >= num_vars {
                return Err(PyValueError::new_err(format!(
                    "residual {} references variable index {}, but x0 has only {} elements",
                    i, var_idx, num_vars
                )));
            }
        }
    }

    // Build problem: auto-differentiate + JIT compile
    let problem = ExprProblem::new("python_expr".into(), num_vars, exprs);

    // Solve with GIL released
    let result = py.allow_threads(move || {
        match solver.unwrap_or("auto") {
            "auto" => AutoSolver::new().solve(&problem, &x0_vec),
            "nr" | "newton-raphson" => {
                let mut config = SolverConfig::default();
                if let Some(tol) = tolerance { config.tolerance = tol; }
                if let Some(max) = max_iterations { config.max_iterations = max; }
                Solver::new(config).solve(&problem, &x0_vec)
            }
            "lm" | "levenberg-marquardt" => {
                let mut config = LMConfig::default();
                if let Some(tol) = tolerance { config = config.with_tol(tol); }
                if let Some(max) = max_iterations {
                    // LMConfig.patience controls max function evaluations via:
                    //   max_fev = patience * (num_vars + 1)
                    // Translate the user-facing max_iterations into patience,
                    // ensuring at least 1 unit of patience.
                    let evals_per_unit = num_vars + 1;
                    config.patience = std::cmp::max(1, max / evals_per_unit);
                }
                LMSolver::new(config).solve(&problem, &x0_vec)
            }
            _ => /* error */
        }
    });

    Ok(PySolveResult::from(result))
}
```

### 2.5 SolveResult

```
File: crates/solverang-python/src/result.rs
```

```rust
#[pyclass(frozen, name = "SolveResult")]
pub struct PySolveResult {
    solution: Vec<f64>,
    converged: bool,
    iterations: usize,
    residual_norm: f64,
    error_message: Option<String>,
}

#[pymethods]
impl PySolveResult {
    #[getter]
    fn x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.solution)
    }

    #[getter]
    fn solution<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.x(py)
    }

    #[getter]
    fn converged(&self) -> bool { self.converged }

    #[getter]
    fn success(&self) -> bool { self.converged }

    #[getter]
    fn iterations(&self) -> usize { self.iterations }

    #[getter]
    fn residual_norm(&self) -> f64 { self.residual_norm }

    fn raise_on_failure(&self) -> PyResult<()> { /* raise SolverError if !converged */ }

    fn __bool__(&self) -> bool { self.converged }
    fn __repr__(&self) -> String { /* ... */ }
}
```

### 2.6 Module Entry Point

```
File: crates/solverang-python/src/lib.rs
```

```rust
use pyo3::prelude::*;

mod expr;
mod functions;
mod solve;
mod result;
mod geometry;
mod exceptions;

use expr::PyExpr;
use result::PySolveResult;
use functions::*;
use solve::solve;

#[pymodule]
fn _solverang(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyExpr>()?;
    m.add_class::<PySolveResult>()?;

    m.add_function(wrap_pyfunction!(variables, m)?)?;
    m.add_function(wrap_pyfunction!(reset_variables, m)?)?;
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add_function(wrap_pyfunction!(eq, m)?)?;
    m.add_function(wrap_pyfunction!(ne, m)?)?;
    m.add_function(wrap_pyfunction!(where_, m)?)?;

    // Math functions
    m.add_function(wrap_pyfunction!(sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(sin, m)?)?;
    m.add_function(wrap_pyfunction!(cos, m)?)?;
    m.add_function(wrap_pyfunction!(tan, m)?)?;
    m.add_function(wrap_pyfunction!(atan2, m)?)?;
    m.add_function(wrap_pyfunction!(max, m)?)?;
    m.add_function(wrap_pyfunction!(min, m)?)?;
    m.add_function(wrap_pyfunction!(smooth_abs, m)?)?;
    m.add_function(wrap_pyfunction!(safe_div, m)?)?;
    m.add_function(wrap_pyfunction!(clamp, m)?)?;

    // Exceptions
    m.add("SolverError", m.py().get_type::<exceptions::SolverError>())?;
    m.add("ConvergenceError", m.py().get_type::<exceptions::ConvergenceError>())?;
    m.add("DimensionError", m.py().get_type::<exceptions::DimensionError>())?;

    Ok(())
}
```

### 2.7 Python Package Files

```
File: crates/solverang-python/python/solverang/__init__.py
```

```python
"""Solverang: Fast nonlinear solver with expression-graph Python API."""

from ._solverang import (
    Expr,
    SolveResult,
    variables, reset_variables,
    solve,
    eq, ne,
    where,
    sqrt, sin, cos, tan, atan2,
    max, min,
    smooth_abs, safe_div, clamp,
    SolverError, ConvergenceError, DimensionError,
)

# Note: `where` is NOT a Python keyword (unlike `for`, `if`, `class`).
# It can be used directly as an identifier. We also provide `where_`
# as an alias for users who prefer the trailing-underscore convention.
where_ = where

__all__ = [
    "Expr", "SolveResult",
    "variables", "reset_variables", "solve",
    "eq", "ne", "where", "where_",
    "sqrt", "sin", "cos", "tan", "atan2",
    "max", "min", "smooth_abs", "safe_div", "clamp",
    "SolverError", "ConvergenceError", "DimensionError",
]
```

```
File: crates/solverang-python/python/solverang/py.typed
(empty file -- PEP 561 marker)
```

### Phase 2 Deliverables

| File | Est. Lines | Status |
|------|-----------|--------|
| `crates/solverang-python/Cargo.toml` | 25 | New |
| `crates/solverang-python/pyproject.toml` | 25 | New |
| `crates/solverang-python/src/lib.rs` | 50 | New |
| `crates/solverang-python/src/expr.rs` | 250 | New |
| `crates/solverang-python/src/functions.rs` | 150 | New |
| `crates/solverang-python/src/solve.rs` | 120 | New |
| `crates/solverang-python/src/result.rs` | 100 | New |
| `crates/solverang-python/src/exceptions.rs` | 30 | New |
| `crates/solverang-python/python/solverang/__init__.py` | 25 | New |
| `crates/solverang-python/python/solverang/_solverang.pyi` | 120 | New |
| `crates/solverang-python/python/solverang/py.typed` | 0 | New |
| **Total** | **~895** | |

---

## Phase 3: Geometry Integration

**Goal**: Expose `ConstraintSystem2D`/`3D` to Python, and allow expression-based
custom constraints to compose with built-in geometric constraints.

### 3.1 ConstraintSystem2D PyClass

```
File: crates/solverang-python/src/geometry.rs
```

Key design: store constraint specs as a Rust enum, build the real
`ConstraintSystem<2>` only at solve time. This avoids the builder-pattern
ownership problem.

```rust
#[pyclass(name = "ConstraintSystem2D")]
struct PyConstraintSystem2D {
    name: String,
    points: Vec<[f64; 2]>,
    fixed: Vec<bool>,
    builtin_constraints: Vec<ConstraintSpec>,
    /// Custom expression-based residuals (from add_residual)
    custom_residuals: Vec<RuntimeExpr>,
}

#[pymethods]
impl PyConstraintSystem2D {
    #[new]
    #[pyo3(signature = (name=None))]
    fn new(name: Option<String>) -> Self { /* ... */ }

    #[pyo3(signature = (x, y, *, fixed=false))]
    fn add_point(&mut self, x: f64, y: f64, fixed: bool) -> usize { /* ... */ }

    fn fix_point(&mut self, index: usize) { /* ... */ }

    // ─── Built-in constraints ───
    fn constrain_distance(&mut self, p1: usize, p2: usize, distance: f64) { /* ... */ }
    fn constrain_horizontal(&mut self, p1: usize, p2: usize) { /* ... */ }
    fn constrain_vertical(&mut self, p1: usize, p2: usize) { /* ... */ }
    fn constrain_angle(&mut self, p1: usize, p2: usize, degrees: f64) { /* ... */ }
    fn constrain_parallel(&mut self, ...) { /* ... */ }
    fn constrain_perpendicular(&mut self, ...) { /* ... */ }
    fn constrain_coincident(&mut self, p1: usize, p2: usize) { /* ... */ }
    fn constrain_midpoint(&mut self, mid: usize, start: usize, end: usize) { /* ... */ }
    fn constrain_point_on_line(&mut self, ...) { /* ... */ }
    fn constrain_point_on_circle(&mut self, ...) { /* ... */ }
    fn constrain_equal_length(&mut self, ...) { /* ... */ }

    // ─── Expression-based constraints ───

    /// Get symbolic coordinate expressions for a point.
    /// Returns (x_expr, y_expr) bound to this point's variable indices.
    fn coords(&self, point_index: usize) -> PyResult<(PyExpr, PyExpr)> {
        // Map point index to variable index in the flat array
        // (skipping fixed points)
        let var_base = self.free_var_index(point_index)?;
        Ok((
            PyExpr {
                inner: RuntimeExpr::Var(var_base),
                name: Some(format!("p{}.x", point_index)),
            },
            PyExpr {
                inner: RuntimeExpr::Var(var_base + 1),
                name: Some(format!("p{}.y", point_index)),
            },
        ))
    }

    /// Add a custom expression-based residual.
    fn add_residual(&mut self, expr: &PyExpr) {
        self.custom_residuals.push(expr.inner.clone());
    }

    // ─── Solve ───

    #[pyo3(signature = (*, solver=None, tolerance=None, max_iterations=None))]
    fn solve(&self, py: Python<'_>, ...) -> PyResult<PyGeometryResult> {
        // 1. Build ConstraintSystem<2> from specs
        // 2. Build combined Problem (built-in + custom expression residuals)
        // 3. Release GIL, solve
        // 4. Return result with point positions
    }

    // ─── Info ───

    #[getter]
    fn num_points(&self) -> usize { self.points.len() }
    #[getter]
    fn num_constraints(&self) -> usize {
        self.builtin_constraints.len() + self.custom_residuals.len()
    }
    #[getter]
    fn degrees_of_freedom(&self) -> isize { /* ... */ }
    fn __repr__(&self) -> String { /* ... */ }
}
```

### 3.2 Macro for 2D/3D Deduplication

Most methods are identical between 2D and 3D (distance, coincident, midpoint,
etc.). Use a Rust macro to generate both implementations:

```rust
macro_rules! impl_constraint_system {
    ($py_name:ident, $dim:literal) => {
        #[pymethods]
        impl $py_name {
            fn constrain_distance(&mut self, p1: usize, p2: usize, distance: f64) {
                self.builtin_constraints.push(
                    ConstraintSpec::Distance { p1, p2, target: distance }
                );
            }
            // ... shared methods
        }
    }
}

impl_constraint_system!(PyConstraintSystem2D, 2);
impl_constraint_system!(PyConstraintSystem3D, 3);
```

### Phase 3 Deliverables

| File | Est. Lines |
|------|-----------|
| `crates/solverang-python/src/geometry.rs` | 400 |
| Tests | 200 |
| **Total** | **~600** |

---

## Phase 4: Extended Capabilities

Implement after the core (Phases 1-3) is working and tested.

### 4.1 Exp and Ln Opcodes

Add `Exp` and `Ln` to `ConstraintOp` to support:
- `x**y` (variable exponents): `exp(y * ln(x))`
- Smooth max via LogSumExp: `ln(exp(a) + exp(b)) / alpha`
- Natural growth/decay models

**Cranelift translation**: Use Taylor series (like sin/cos) or call out to
libm. Cranelift doesn't have native exp/ln -- will need either:
- Polynomial approximation (Remez or minimax)
- Function call to a C math library

**Differentiation**:
- `d/dx exp(f) = exp(f) * df`
- `d/dx ln(f) = df / f`

### 4.2 Parameter Type (Mutable Constants)

```python
r = sr.Parameter("radius", value=1.0)
residuals = [x**2 + y**2 - r**2, ...]

# Solve, change parameter, solve again without recompilation
r.value = 2.0
```

Implementation: `Arc<AtomicU64>` storing f64 bits. The JIT-compiled code loads
the parameter value from a pointer at each evaluation, so changing the parameter
does not require recompilation.

Requires adding a `LoadParam { dst, param_ptr }` opcode that loads from an
external pointer rather than the variables array.

### 4.3 Callback Fallback Path

For problems that genuinely need control flow beyond `where()`:

```python
# Callback path (slower, can't release GIL, but fully flexible)
def my_residuals(x):
    if some_complex_condition(x):
        return complex_branch_a(x)
    else:
        return complex_branch_b(x)

result = sr.solve_callback(
    residuals=my_residuals,
    num_residuals=2,
    num_variables=3,
    x0=[1.0, 2.0, 3.0],
)
```

### 4.4 Batch Solve

```python
# Solve many instances in parallel (all GIL-free)
problems = [make_problem(p) for p in params]
results = sr.solve_batch(problems, x0s)
```

Uses rayon internally; the GIL is released for the entire batch.

### 4.5 Expression Caching / Structural Hashing

For repeated solves with the same expression structure, cache the JIT-compiled
function. Use structural hashing of the expression tree to detect identical
structures.

---

## Control Flow Design (Detail)

This section explains how control flow works end-to-end with concrete examples.

### The `where()` Function

```python
import solverang as sr

x, = sr.variables("x")

# Piecewise: f(x) = x^2 for x > 0, else -x
r = sr.where(x > 0, x**2, -x)
```

**Expression tree built**:

```
Select(
    condition: Compare(Var(0), Const(0.0), Gt),
    on_true:   Pow(Var(0), 2.0),
    on_false:  Neg(Var(0)),
)
```

**Differentiation** (d/dx):

```
Select(
    condition: Compare(Var(0), Const(0.0), Gt),    # same condition
    on_true:   Mul(Const(2.0), Var(0)),             # d/dx(x^2) = 2x
    on_false:  Const(-1.0),                          # d/dx(-x) = -1
)
```

**Opcode emission**:

```
; Residual
LoadVar   r0, 0          ; x
LoadConst r1, 0.0
FCmp      r2, r0, r1, Gt ; x > 0 ? 1.0 : 0.0
Mul       r3, r0, r0     ; x^2  (true branch, always computed)
Neg       r4, r0         ; -x   (false branch, always computed)
Select    r5, r2, r3, r4 ; pick one
StoreResidual 0, r5

; Jacobian dr/dx
LoadVar   r0, 0
LoadConst r1, 0.0
FCmp      r2, r0, r1, Gt
LoadConst r3, 2.0
Mul       r4, r3, r0     ; 2*x  (true branch derivative)
LoadConst r5, -1.0       ; -1   (false branch derivative)
Select    r6, r2, r4, r5 ; pick correct derivative
StoreJacobianIndexed 0, r6
```

**Cranelift native code** (x86-64):

```asm
; FCmp + Select → fcmp + cmov, no branches
ucomisd xmm0, xmm1      ; compare x to 0.0
cmova   xmm5, xmm3      ; select x^2 if x > 0
```

**Key property**: both branches are always evaluated (both `x^2` and `-x` are
computed), but only one result is kept. This means:

1. No branch mispredictions
2. The opcode stream stays flat (no basic blocks, no jumps)
3. Both branches must be numerically valid for all inputs

### Nested Control Flow

```python
# Clamp: max(0, min(1, x))
r = sr.where(x < 0, 0.0, sr.where(x > 1, 1.0, x))
```

This creates nested `Select` nodes. The derivative is also nested:

```python
dr/dx = where(x < 0, 0.0, where(x > 1, 0.0, 1.0))
```

### Both-Branches-Safe Requirement

Because both branches are always evaluated, they must not produce NaN/Inf
for any input. For example:

```python
# DANGEROUS: division by zero in false branch when x > 0
r = sr.where(x > 0, x, 1.0 / x)  # 1/x is computed even when x > 0

# SAFE: use safe_div which guards against zero denominators
r = sr.where(x > 0, x, sr.safe_div(1.0, x))

# SAFE: or guard the denominator manually
r = sr.where(x > 0, x, 1.0 / sr.where(sr.ne(x, 0), x, 1.0))

# SAFE: use sqrt with safe_distance for domain safety
r = sr.where(x > 0, sr.sqrt(x), 0.0)   # sqrt(neg) = NaN!
r = sr.where(x > 0, sr.sqrt(sr.max(x, 0.0)), 0.0)  # guarded
```

The library provides `sr.safe_div(a, b, fill=0.0)` which evaluates to
`a / b` when `b != 0` and `fill` otherwise, using a single `Select` node
internally. This makes guarded division ergonomic and less error-prone
than nested `sr.where()` calls.

This is the same constraint that PyTorch's `torch.where` has. Document it
clearly and provide the `smooth_abs`/`smooth_max`/`safe_div` alternatives
for cases where the exact boundary matters.

### Comparison to JAX/PyTorch

| Feature | solverang `where()` | JAX `lax.select` | PyTorch `torch.where` |
|---------|--------------------|-----------------|-----------------------|
| Both branches evaluated | Yes | Yes | Yes |
| Branchless in compiled code | Yes (cmov) | N/A (XLA) | N/A (eager) |
| Gradient routing | `select(c, df, dg)` | `select(c, df, dg)` | `where(c, grad, 0)` * |
| NaN-safe gradient | Yes | Yes | No (0 * NaN = NaN) |
| Hardware acceleration | Cranelift JIT | XLA/TPU | CUDA kernels |

\* PyTorch's gradient implementation multiplies unselected branch by zero,
which can produce NaN. Our approach selects the correct gradient directly,
avoiding this problem.

---

## Testing Strategy

### Unit Tests (Rust)

1. **RuntimeExpr differentiation**: verify all rules against hand-computed derivatives
2. **Simplification**: verify algebraic identities
3. **Evaluation**: verify against f64 arithmetic
4. **Opcode emission**: verify expected opcode sequences
5. **JIT vs interpreted**: verify identical results for all expression types
6. **ExprProblem Jacobian**: verify against `verify_jacobian()` finite differences
7. **Control flow differentiation**: verify `where(x > 0, x^2, -x)` derivative
8. **Edge cases**: NaN, Inf, division by zero, x^0, 0^x

### Integration Tests (Python)

1. **Simple equations**: `x^2 = 2` → `x ≈ ±√2`
2. **System of equations**: circle-line intersection
3. **Rosenbrock**: 100-variable Rosenbrock function
4. **Geometry**: triangle with distance constraints
5. **Mixed geometry + expression**: built-in + custom constraints
6. **Control flow**: piecewise functions, clamped values
7. **Error handling**: dimension mismatch, singular Jacobian, unsupported ops
8. **Performance**: benchmark against scipy.optimize for known problems

### Property-Based Tests

Use `proptest` or similar to generate random expression trees and verify:
- Differentiation + evaluation matches finite differences
- `simplify()` preserves evaluation results
- JIT matches interpreted for random inputs
- Jacobian sparsity pattern matches variable references

---

## File Layout Summary

```
crates/
  solverang/
    src/
      expr/                          # NEW module (Phase 1)
        mod.rs                       # 15 lines
        expr.rs                      # 120 lines
        differentiate.rs             # 180 lines
        simplify.rs                  # 140 lines
        emit.rs                      # 100 lines
        evaluate.rs                  # 70 lines
        display.rs                   # 80 lines
        problem.rs                   # 150 lines
        tests.rs                     # 300 lines
      jit/
        opcodes.rs                   # +30 lines (FCmp, Select)
        cranelift.rs                 # +40 lines (FCmp, Select translation)
        lower.rs                     # +20 lines (fcmp, select emitter methods)
      lib.rs                         # +10 lines (feature gate, re-export)
    Cargo.toml                       # +1 line (runtime-expr feature)

  solverang-python/                  # NEW crate (Phase 2+3)
    Cargo.toml                       # 25 lines
    pyproject.toml                   # 25 lines
    src/
      lib.rs                         # 50 lines
      expr.rs                        # 250 lines
      functions.rs                   # 150 lines
      solve.rs                       # 120 lines
      result.rs                      # 100 lines
      exceptions.rs                  # 30 lines
      geometry.rs                    # 400 lines (Phase 3)
    python/
      solverang/
        __init__.py                  # 25 lines
        _solverang.pyi               # 120 lines
        py.typed                     # 0 lines

Total new code: ~2,750 lines
Total modified: ~100 lines
```

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Expression tree bloat for large problems (1000+ vars) | Memory, construction time | Implement CSE (Phase 4.5); profile realistic workloads |
| Taylor approx for trig in Cranelift may lose precision | Incorrect solutions | Add `libm` call-out option; validate against interpreted eval |
| `__pow__` with variable exponent breaks user expectations | Confusing error | Clear error message; implement Exp/Ln in Phase 4.1 |
| Both-branches-evaluated can produce NaN | Silent wrong answers | Document clearly; provide `smooth_*` alternatives; add NaN checks |
| Maturin/PyO3 version churn | Build failures | Pin versions; test against multiple Python versions in CI |
| Cranelift only on x86_64/aarch64 | No JIT on other platforms | Interpreted fallback always works; JIT is an optimization |
| Complex expressions produce large Jacobians | Slow compilation | Add opcode budget/timeout; lazy JIT (compile on second solve) |
| Python `if x > 0` evaluates eagerly, bypasses graph | Subtle bugs | Document; consider adding a runtime check (warn if Expr used in bool context) |

---

## Implementation Order

```
Phase 1 (RuntimeExpr + opcodes)  ← START HERE
  │
  ├─ 1.1-1.6: RuntimeExpr type + methods
  ├─ 1.7: ExprProblem
  ├─ 1.8-1.10: FCmp/Select opcodes + Cranelift
  └─ Tests: Rust unit tests
  │
  ▼
Phase 2 (PyO3 bindings)
  │
  ├─ 2.1: Crate + maturin setup
  ├─ 2.2: PyExpr with operator overloads
  ├─ 2.3: Module-level functions (variables, solve, where, etc.)
  ├─ 2.4-2.5: solve() + SolveResult
  ├─ 2.6-2.7: Module entry + Python package
  └─ Tests: Python integration tests
  │
  ▼
Phase 3 (Geometry integration)
  │
  ├─ 3.1: ConstraintSystem2D
  ├─ 3.2: ConstraintSystem3D (macro-generated)
  └─ Tests: Mixed geometry + expression tests
  │
  ▼
Phase 4 (Extended capabilities, as needed)
  ├─ 4.1: Exp/Ln opcodes
  ├─ 4.2: Parameter type
  ├─ 4.3: Callback fallback
  ├─ 4.4: Batch solve
  └─ 4.5: Expression caching
```
