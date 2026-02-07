# Plan A: Problem Metadata Layer (Revised)

## Status: PROPOSAL (v2 — revised from v1)
## Priority: Foundation (blocks Plans 1-6)
## Estimated scope: Small-Medium (metadata only, no solving changes)

---

## Revision Notes (v2)

**What changed from v1**: Scoped `ProblemBase` to metadata and classification only.
It is NOT a solving interface — solvers never accept `&dyn ProblemBase`. Removed the
`ContinuousProblem` alias; the existing `Problem` trait stays as-is and remains the
primary interface for continuous solvers. The hierarchy diagram was misleading — this
is a metadata sidecar, not a supertrait that solvers dispatch on.

---

## 1. Motivation

To classify problems, recommend solvers, and compose hybrid systems (Plan 6), we need
a way to ask "what kind of variables and constraints does this problem have?" without
requiring the caller to know which specific trait (`Problem`, `DiscreteProblem`,
`OptimizationProblem`) the problem implements.

`ProblemBase` answers metadata questions. It does NOT evaluate constraints, compute
residuals, or produce Jacobians. It's introspection, not computation.

## 2. What ProblemBase Is and Is NOT

### Is:
- A way to query variable types (continuous, integer, boolean, enumerated)
- A way to query constraint structure (count, which variables each touches)
- Input to the classifier (Plan 4) and hybrid decomposer (Plan 6)
- Automatically derived for any `Problem` via blanket impl

### Is NOT:
- An argument to any `solve()` function
- A replacement for `Problem`, `DiscreteProblem`, or `OptimizationProblem`
- A unified solving interface (that was the v1 mistake)

## 3. Design

```
                ┌──────────────────┐
                │   ProblemBase    │  metadata / introspection only
                │  (never solved   │
                │   directly)      │
                └────────┬─────────┘
                         │ blanket impl
         ┌───────────────┼────────────────┐
         │               │                │
   ┌─────▼──────┐  ┌─────▼───────┐  ┌────▼──────────┐
   │  Problem    │  │ Discrete    │  │ Optimization  │
   │  (existing, │  │ Problem     │  │ Problem       │
   │  unchanged) │  │ (Plan 3)    │  │ (Plan 1)      │
   └─────────────┘  └─────────────┘  └───────────────┘
         │                │                │
         ▼                ▼                ▼
   NR/LM/Sparse/    CSPSolver        AugLagSolver
   Parallel/JIT      (typed)          (typed)
   (typed)
```

Each solving trait is independent. Solvers accept their specific trait, not `ProblemBase`.

### 3.1 `ProblemBase`

```rust
pub trait ProblemBase: Send + Sync {
    /// Problem name for reporting and debugging.
    fn name(&self) -> &str;

    /// Number of variables.
    fn variable_count(&self) -> usize;

    /// Number of constraints.
    fn constraint_count(&self) -> usize;

    /// Domain of variable i.
    fn variable_domain(&self, index: usize) -> VariableDomain;

    /// Which variables does constraint j touch?
    /// Returns variable indices. Used for decomposition and classification.
    fn constraint_scope(&self, index: usize) -> Vec<usize>;

    /// Whether this problem has an optimization objective.
    fn has_objective(&self) -> bool { false }
}
```

Note: `variable_domain()` by index instead of returning a full `Vec<VariableDescriptor>`.
Cheaper, no allocation for the common case where you're checking a few variables.

### 3.2 `VariableDomain`

```rust
#[derive(Clone, Debug, PartialEq)]
pub enum VariableDomain {
    Continuous { lower: f64, upper: f64 },
    Integer { lower: i64, upper: i64 },
    Boolean,
    Enumerated { values: Vec<i64> },
}
```

Unchanged from v1 — this is a good design.

### 3.3 Blanket Impl for `Problem`

```rust
impl<T: Problem> ProblemBase for T {
    fn name(&self) -> &str { Problem::name(self) }
    fn variable_count(&self) -> usize { Problem::variable_count(self) }
    fn constraint_count(&self) -> usize { self.residual_count() }

    fn variable_domain(&self, _index: usize) -> VariableDomain {
        VariableDomain::Continuous {
            lower: f64::NEG_INFINITY,
            upper: f64::INFINITY,
        }
    }

    fn constraint_scope(&self, constraint_index: usize) -> Vec<usize> {
        // Default: derive from Jacobian sparsity at x=0
        // This is the same pattern as DecomposableProblem::constraint_graph()
        let x0 = self.initial_point(1.0);
        self.jacobian(&x0).into_iter()
            .filter(|(row, _, _)| *row == constraint_index)
            .map(|(_, col, _)| col)
            .collect()
    }
}
```

### 3.4 Blanket impls for other problem traits

Each new problem trait (Plans 1, 3) also gets a blanket impl, so any
`OptimizationProblem` or `DiscreteProblem` is automatically a `ProblemBase`.

## 4. File Layout

```
crates/solverang/src/
├── problem.rs           # UNCHANGED
├── problem_base.rs      # NEW — ProblemBase, VariableDomain, blanket impl
```

That's it. Two types and one blanket impl. No new modules, no feature flags needed
for the base metadata types.

## 5. What This Enables

- **Plan 4 (Classifier)**: `ProblemClassifier::classify(&dyn ProblemBase)` — works
  on any problem type without knowing which specific trait it implements.
- **Plan 6 (Hybrid)**: The decomposer uses `constraint_scope()` to partition variables
  into continuous and discrete blocks, then routes to the right solver.
- **Plan 5 (DSL)**: `Model::build()` generates a `ProblemBase` impl alongside the
  specific problem trait, enabling automatic classification.

## 6. Acceptance Criteria

- [ ] `ProblemBase` trait defined in `problem_base.rs`
- [ ] Blanket `impl ProblemBase for T where T: Problem` compiles
- [ ] All existing tests pass without modification (zero changes to existing code)
- [ ] `ProblemClassifier` (Plan 4) can accept `&dyn ProblemBase`
