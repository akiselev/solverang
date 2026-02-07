# Level 1: Make It Work

**Goal**: JIT produces correct results and is reachable from the public API.

**Prerequisite reading**: `docs/notes/jit-integration-analysis.md`

---

## Overview

The JIT module compiles and passes tests but is architecturally disconnected:
`JITSolver::solve()` always falls back to interpreted evaluation, trig functions
produce garbage outside a narrow range, and only 6 of 18 constraint types can be
lowered. This level fixes all of that.

**Estimated effort**: ~1,200 new/modified LOC across 8 files.

---

## Step 1: Fix Trigonometric Functions

**Files**: `jit/cranelift.rs`

**Problem**: `approximate_sin`, `approximate_cos`, and `approximate_atan2` use
raw Taylor series with no range reduction. They produce garbage for |x| > ~3,
and constraint angles routinely span 0-2π (~6.28).

**Approach**: Register libm functions as external symbols via Cranelift's
`JITBuilder::symbol()` API, then emit `call` instructions instead of inline
polynomials.

### 1a. Register libm symbols during compiler creation

In `JITCompiler::new()`, after creating the `JITBuilder`:

```rust
let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

// Register libm math functions as callable symbols via extern "C" wrappers.
// IMPORTANT: Rust's f64::sin/cos/atan2 use Rust ABI, NOT C ABI. Cranelift
// emits calls using the platform C calling convention, so we must wrap every
// function in an extern "C" shim to guarantee ABI compatibility. Passing
// Rust-ABI function pointers directly can crash or produce wrong results on
// targets where the ABIs differ (this is not detected at compile time).
builder.symbol("libm_sin", libm_sin_wrapper as *const u8);
builder.symbol("libm_cos", libm_cos_wrapper as *const u8);
builder.symbol("libm_atan2", libm_atan2_wrapper as *const u8);
```

All three functions need `extern "C"` wrappers — not just `atan2`:

```rust
extern "C" fn libm_sin_wrapper(x: f64) -> f64 {
    x.sin()
}

extern "C" fn libm_cos_wrapper(x: f64) -> f64 {
    x.cos()
}

extern "C" fn libm_atan2_wrapper(y: f64, x: f64) -> f64 {
    y.atan2(x)
}
```

On Windows, also verify that these delegate to the correct libm implementation
(MSVC's `sin`/`cos` may have different precision characteristics). Consider
using the `libm` crate for cross-platform consistency if needed.

### 1b. Declare functions in the module

Add a helper struct to `JITCompiler` that caches `FuncId`s:

```rust
struct MathFunctions {
    sin: FuncId,
    cos: FuncId,
    atan2: FuncId,
}
```

During `compile()`, declare these once:

```rust
fn declare_math_functions(&mut self) -> Result<MathFunctions, JITError> {
    let mut sig1 = self.module.make_signature();
    sig1.params.push(AbiParam::new(types::F64));
    sig1.returns.push(AbiParam::new(types::F64));

    let sin = self.module.declare_function("libm_sin", Linkage::Import, &sig1)?;
    let cos = self.module.declare_function("libm_cos", Linkage::Import, &sig1)?;

    let mut sig2 = self.module.make_signature();
    sig2.params.push(AbiParam::new(types::F64));
    sig2.params.push(AbiParam::new(types::F64));
    sig2.returns.push(AbiParam::new(types::F64));

    let atan2 = self.module.declare_function("libm_atan2", Linkage::Import, &sig2)?;

    Ok(MathFunctions { sin, cos, atan2 })
}
```

### 1c. Replace approximations with calls in `translate_ops`

Change `translate_ops` to accept a `&MathFunctions` parameter (or the function
references after `declare_func_in_func`), then:

```rust
ConstraintOp::Sin { dst, src } => {
    let src_val = get_reg(&registers, *src);
    let local_sin = module.declare_func_in_func(math.sin, builder.func);
    let call = builder.ins().call(local_sin, &[src_val]);
    let result = builder.inst_results(call)[0];
    registers.insert(*dst, result);
}
```

### 1d. Delete dead approximation functions

Remove `approximate_sin`, `approximate_cos`, `approximate_atan`,
`approximate_atan2`. They are no longer needed.

### 1e. Tests

Add to `jit/cranelift.rs` tests:

```rust
#[test]
fn test_jit_sin_full_range() {
    // Compile a single-residual program: residual = sin(x)
    // Test at x = 0, π/6, π/4, π/2, π, 3π/2, 2π, -π, 7.5
    // Assert |jit_sin(x) - x.sin()| < 1e-12 for each
}

#[test]
fn test_jit_cos_full_range() { /* same pattern */ }

#[test]
fn test_jit_atan2_all_quadrants() {
    // Test (1,1), (-1,1), (-1,-1), (1,-1), (0,1), (1,0), (0,-1), (-1,0)
    // Assert |jit_atan2(y,x) - y.atan2(x)| < 1e-12
}
```

**Acceptance criteria**: All trig tests pass with < 1e-12 error across the full
input range. No Taylor series remain in the codebase.

---

## Step 2: Fix StoreJacobian No-Op

**Files**: `jit/cranelift.rs`, `jit/opcodes.rs`

**Problem**: `ConstraintOp::StoreJacobian { row, col, src }` reads the source
register but discards the value (`cranelift.rs:324-329`). This is dead code —
all actual Jacobian storage goes through `StoreJacobianIndexed`.

**Approach**: Remove the `StoreJacobian` variant entirely.

### 2a. Remove `StoreJacobian` from `ConstraintOp` enum

In `jit/opcodes.rs`, delete:

```rust
StoreJacobian {
    row: u32,
    col: u32,
    src: Reg,
},
```

Update `uses_register()` and `defines_register()` match arms.

### 2b. Remove from `translate_ops`

In `jit/cranelift.rs`, delete the `StoreJacobian` match arm.

### 2c. Remove from `OpcodeEmitter` if referenced

Check `jit/lower.rs` — the emitter's `store_jacobian()` method uses
`StoreJacobianIndexed`, not `StoreJacobian`. Confirm and remove any dead
references.

### 2d. Update validation

If `CompiledConstraints::validate()` references `StoreJacobian`, update it.

**Acceptance criteria**: `cargo check --features jit` succeeds. No `StoreJacobian`
references remain. All existing tests still pass.

---

## Step 3: Complete Constraint Lowering Coverage

**Files**: `jit/geometry_lowering.rs`

**Problem**: 6 of 18 constraint types have `Lowerable` implementations. The
remaining 12 cannot be JIT-compiled.

### Coverage plan

Constraints are grouped by implementation complexity. Within each group, the
math operations needed are listed.

#### Group A: Pure arithmetic (trivial — constant Jacobians)

These constraints use only add/sub/mul with constant Jacobian entries. Each
should take ~30 LOC for residual + jacobian lowering.

| Constraint | Residuals | Operations | Notes |
|---|---|---|---|
| `MidpointConstraint<2>` | 2 | sub, mul | `2*M[k] - A[k] - B[k]` |
| `MidpointConstraint<3>` | 3 | sub, mul | same, 3D |
| `SymmetricConstraint<2>` | 2 | add, sub, mul | `P1[k] + P2[k] - 2*C[k]` |
| `SymmetricConstraint<3>` | 3 | add, sub, mul | same, 3D |

#### Group B: Cross/dot products (straightforward)

These use products of differences. The Jacobian has non-constant entries but
no transcendentals.

| Constraint | Residuals | Operations | Notes |
|---|---|---|---|
| `ParallelConstraint<2>` | 1 | sub, mul | 2D cross product |
| `ParallelConstraint<3>` | 2 | sub, mul | 3D cross product (2 components) |
| `PerpendicularConstraint<2>` | 1 | sub, mul, add | dot product |
| `PerpendicularConstraint<3>` | 1 | sub, mul, add | dot product, 3 terms |
| `PointOnLineConstraint<2>` | 1 | sub, mul | cross product |
| `PointOnLineConstraint<3>` | 2 | sub, mul | cross product (2 components) |
| `CollinearConstraint<2>` | 2 | sub, mul | two cross products |
| `CollinearConstraint<3>` | 4 | sub, mul | two 3D cross products |
| `SymmetricAboutLineConstraint` | 2 | sub, mul, add | cross + dot product |

#### Group C: Distance-based (moderate — reuse distance pattern)

These follow the same pattern as `DistanceConstraint`: compute Euclidean
distance, compare to target. The template is already established.

| Constraint | Residuals | Operations | Notes |
|---|---|---|---|
| `PointOnCircleConstraint<2>` | 1 | sub, mul, add, sqrt, div | dist - radius |
| `PointOnCircleConstraint<3>` | 1 | same | sphere variant |
| `EqualLengthConstraint<2>` | 1 | sub, mul, add, sqrt, div | len1 - len2 |
| `EqualLengthConstraint<3>` | 1 | same | 3D |
| `CircleTangentConstraint` | 1 | sub, mul, add, sqrt, div | center dist - target |
| `PointOnCircleVariableRadiusConstraint<2>` | 1 | sub, mul, add, sqrt, div | two distances |
| `PointOnCircleVariableRadiusConstraint<3>` | 1 | same | 3D |

#### Group D: Requires abs (one constraint)

| Constraint | Residuals | Operations | Notes |
|---|---|---|---|
| `LineTangentConstraint` | 1 | sub, mul, add, sqrt, abs, div | perp distance |

### Implementation order

1. Group A first (simplest, validates the pattern)
2. Group B next (covers the cross/dot product idiom)
3. Group C (reuses the DistanceConstraint template)
4. Group D last (LineTangentConstraint has the most complex formula)

### Template for each impl

Each `Lowerable` impl follows the pattern in `geometry_lowering.rs`:

```rust
#[cfg(feature = "geometry")]
impl Lowerable for MidpointConstraint<2> {
    fn lower_residual(&self, emitter: &mut OpcodeEmitter, ctx: &LoweringContext) {
        let base = ctx.current_residual;
        for k in 0..2 {
            let m_k = emitter.load_var(ctx.var_index(self.midpoint, k));
            let a_k = emitter.load_var(ctx.var_index(self.point_a, k));
            let b_k = emitter.load_var(ctx.var_index(self.point_b, k));
            let two = emitter.const_f64(2.0);
            let two_m = emitter.mul(two, m_k);
            let sum_ab = emitter.add(a_k, b_k);
            let residual = emitter.sub(two_m, sum_ab);
            emitter.store_residual(base + k as u32, residual);
        }
    }

    fn lower_jacobian(&self, emitter: &mut OpcodeEmitter, ctx: &LoweringContext) {
        let base = ctx.current_residual;
        let two = emitter.const_f64(2.0);
        let neg_one = emitter.const_f64(-1.0);
        for k in 0..2 {
            emitter.set_residual_index(base + k as u32);
            emitter.store_jacobian_current(ctx.var_index(self.midpoint, k), two);
            emitter.store_jacobian_current(ctx.var_index(self.point_a, k), neg_one);
            emitter.store_jacobian_current(ctx.var_index(self.point_b, k), neg_one);
        }
    }

    fn residual_count(&self) -> usize { 2 }
    fn variable_indices(&self) -> Vec<usize> {
        vec![self.midpoint, self.point_a, self.point_b]
    }
}
```

### 3e. Equivalence tests

For **every** constraint type with a `Lowerable` impl, add a test:

```rust
#[test]
fn test_<constraint>_jit_matches_interpreted() {
    // 1. Create constraint
    // 2. Lower to CompiledConstraints
    // 3. JIT compile
    // 4. Evaluate at 5+ test points
    // 5. Compare JIT residuals to GeometricConstraint::residuals()
    // 6. Compare JIT Jacobian to GeometricConstraint::jacobian()
    // 7. Assert max error < 1e-12
}
```

**Acceptance criteria**: All 18 constraint types have `Lowerable` impls. All
equivalence tests pass with < 1e-12 error. `cargo test --features jit,geometry`
passes.

---

## Step 4: Bridge JIT to the Public API

**Files**: `jit/mod.rs`, `jit/lower.rs`, `solver/jit_solver.rs`,
`geometry/system.rs` (minimal touch)

**Problem**: `ConstraintSystem<D>` implements `Problem` but not `Lowerable`.
`JITSolver::solve()` receives `&dyn Problem` and has no access to the concrete
constraint types inside `ConstraintSystem<D>`.

**Approach**: Don't fight the type erasure. Instead, add an optional lowering
method to the `GeometricConstraint<D>` trait with a default no-op, and give
`ConstraintSystem<D>` a `compile()` method that iterates its constraints and
lowers those that support it.

### 4a. Extend `GeometricConstraint<D>` with optional lowering

In whichever file defines the `GeometricConstraint` trait:

```rust
pub trait GeometricConstraint<const D: usize>: Send + Sync {
    // ... existing methods ...

    /// Try to lower this constraint's residual computation to JIT opcodes.
    ///
    /// Returns `true` if lowering succeeded, `false` to fall back to interpreted.
    /// Default implementation returns `false`.
    #[cfg(feature = "jit")]
    fn try_lower_residual(
        &self,
        _emitter: &mut crate::jit::OpcodeEmitter,
        _ctx: &crate::jit::LoweringContext,
    ) -> bool {
        false
    }

    /// Try to lower this constraint's Jacobian computation to JIT opcodes.
    #[cfg(feature = "jit")]
    fn try_lower_jacobian(
        &self,
        _emitter: &mut crate::jit::OpcodeEmitter,
        _ctx: &crate::jit::LoweringContext,
    ) -> bool {
        false
    }
}
```

### 4b. Implement `try_lower_*` on concrete types

For each constraint type that has a `Lowerable` impl, forward to it:

```rust
#[cfg(feature = "jit")]
impl<const D: usize> DistanceConstraint<D> {
    // ... (only for D=2 and D=3 via specialization or separate impls)
}
```

Actually, since Rust doesn't support specialization on const generics cleanly,
the pragmatic approach is to implement `try_lower_*` directly on the concrete
types via a macro:

```rust
macro_rules! impl_jit_lowering_for_constraint {
    ($ty:ty, $dim:expr) => {
        #[cfg(feature = "jit")]
        impl GeometricConstraint<$dim> for $ty {
            // ... (override try_lower_residual and try_lower_jacobian)
        }
    };
}
```

Or more simply: since each concrete type already impls both `GeometricConstraint<D>`
and `Lowerable`, implement `try_lower_*` by delegating to the `Lowerable` methods:

```rust
#[cfg(feature = "jit")]
fn try_lower_residual(&self, emitter: &mut OpcodeEmitter, ctx: &LoweringContext) -> bool {
    self.lower_residual(emitter, ctx);
    true
}
```

### 4c. Add `compile()` to `ConstraintSystem<D>`

```rust
#[cfg(feature = "jit")]
impl<const D: usize> ConstraintSystem<D> {
    /// Attempt to JIT-compile this constraint system.
    ///
    /// Returns `Some(JITFunction)` if all constraints support lowering,
    /// `None` if any constraint does not.
    pub fn try_compile(&self) -> Option<crate::jit::JITFunction> {
        let n_vars = self.total_variable_count();
        let n_residuals = self.equation_count();
        let dimension = D;

        let mut residual_emitter = OpcodeEmitter::new();
        let mut jacobian_emitter = OpcodeEmitter::new();
        let mut ctx = LoweringContext::new(dimension, n_vars);

        // Lower all constraints
        for constraint in &self.constraints {
            if !constraint.try_lower_residual(&mut residual_emitter, &ctx) {
                return None; // Constraint doesn't support JIT
            }
            // ... advance residual index ...
        }

        // Reset context, lower Jacobians
        ctx.current_residual = 0;
        for constraint in &self.constraints {
            if !constraint.try_lower_jacobian(&mut jacobian_emitter, &ctx) {
                return None;
            }
            // ... advance residual index ...
        }

        // Build CompiledConstraints, compile via JITCompiler
        let mut cc = CompiledConstraints::new(n_vars, n_residuals);
        cc.residual_ops = residual_emitter.into_ops();
        cc.jacobian_ops = jacobian_emitter.ops().to_vec();
        cc.jacobian_pattern = jacobian_emitter.take_jacobian_entries();
        cc.jacobian_nnz = cc.jacobian_pattern.len();

        let mut compiler = JITCompiler::new().ok()?;
        compiler.compile(&cc).ok()
    }
}
```

### 4d. Wire `JITSolver::solve()` to actually use JIT

Replace the hard-coded fallback in `jit_solver.rs:94-97`:

```rust
pub fn solve<P: Problem>(&mut self, problem: &P, x0: &[f64]) -> SolveResult {
    // ... dimension validation (keep as-is) ...

    // For generic Problem types, we cannot lower automatically.
    // Use interpreted path.
    self.solve_interpreted(problem, x0)
}

/// Solve a constraint system, attempting JIT compilation.
///
/// If all constraints in the system support JIT lowering and the problem
/// size exceeds the JIT threshold, this compiles to native code. Otherwise
/// falls back to interpreted evaluation.
#[cfg(feature = "geometry")]
pub fn solve_constraint_system<const D: usize>(
    &mut self,
    system: &ConstraintSystem<D>,
    x0: &[f64],
) -> SolveResult {
    // Check if JIT should be used
    if self.will_use_jit(system) {
        if let Some(jit_fn) = system.try_compile() {
            return self.solve_with_jit(&jit_fn, x0);
        }
    }

    // Fallback to interpreted
    self.solve_interpreted(system, x0)
}
```

### 4e. Tests

```rust
#[test]
fn test_solve_constraint_system_with_jit() {
    // Build a ConstraintSystem with 2 points + distance constraint
    // Solve with JITSolver using solve_constraint_system()
    // Verify convergence and correct solution
}

#[test]
fn test_jit_fallback_when_unsupported_constraint() {
    // Build a ConstraintSystem with a constraint that returns false
    // from try_lower_residual
    // Verify solve_constraint_system() still converges (via fallback)
}

#[test]
fn test_jit_end_to_end_triangle() {
    // 3 points, 3 distance constraints forming a triangle
    // Solve via JIT, verify geometry is correct
}
```

**Acceptance criteria**: `JITSolver::solve_constraint_system()` uses JIT when
all constraints support lowering and the threshold is met. Falls back gracefully
otherwise. End-to-end test passes.

---

## Step 5: Integration Tests

**Files**: new file `tests/jit_tests.rs`

Create a comprehensive integration test suite that exercises the full JIT
pipeline on realistic constraint systems.

### Test cases

1. **Single distance constraint** (2 points, 1 constraint)
2. **Triangle** (3 points, 3 distance constraints)
3. **Rectangle** (4 points, 4 distance + 2 perpendicular + 1 horizontal)
4. **Over-constrained** (add a redundant constraint, verify solve still works)
5. **Large grid** (10x10 grid of points with distance/horizontal/vertical constraints)
6. **3D tetrahedron** (4 3D points, 6 distance constraints)
7. **Mixed constraint types** (distance + coincident + fixed + horizontal + angle)
8. **JIT vs interpreted equivalence** (solve the same problem both ways, compare
   solution vectors element-by-element)

**Acceptance criteria**: All integration tests pass. JIT and interpreted paths
produce solutions that agree to within 1e-10.

---

## Dependency Graph

```
Step 1 (trig) ────────────────────────┐
                                       │
Step 2 (StoreJacobian) ───────────────┤
                                       ├── Step 3 (constraint coverage)
                                       │        │
                                       │        ├── Step 4 (API bridge)
                                       │        │        │
                                       │        │        └── Step 5 (integration tests)
                                       │        │
                                       │        └── Step 5 (integration tests)
```

Steps 1 and 2 are independent and can be done in parallel.
Step 3 depends on Step 1 (trig functions needed for AngleConstraint).
Step 4 depends on Step 3 (needs all constraint types lowerable).
Step 5 depends on Step 4 (needs the API bridge).

---

## Files Changed Summary

| File | Change |
|------|--------|
| `jit/cranelift.rs` | Replace trig approximations with libm calls, remove `StoreJacobian` arm |
| `jit/opcodes.rs` | Remove `StoreJacobian` variant |
| `jit/lower.rs` | Minor: remove any `StoreJacobian` references |
| `jit/geometry_lowering.rs` | Add 12 new `Lowerable` impls + equivalence tests |
| `jit/mod.rs` | Re-export new types if needed |
| `geometry/constraints/*.rs` | Add `try_lower_*` impls (or via trait extension) |
| `geometry/system.rs` | Add `try_compile()` method |
| `solver/jit_solver.rs` | Add `solve_constraint_system()` method |
| `tests/jit_tests.rs` | New integration test file |

**Estimated total**: ~1,200 new/modified LOC.
