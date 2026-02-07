# Plan A: Expanding the Problem Trait into a Generic Hierarchy

## Status: PROPOSAL
## Priority: Foundation (blocks all other plans)
## Estimated scope: Large (core architectural change)

---

## 1. Motivation

The current `Problem` trait is tightly coupled to continuous nonlinear systems:

```rust
pub trait Problem: Send + Sync {
    fn residuals(&self, x: &[f64]) -> Vec<f64>;
    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)>;
    // ...
}
```

Every variable is `f64`, every constraint evaluates to a continuous residual, and every
solver assumes gradient information via Jacobians. This works for Newton-Raphson and
Levenberg-Marquardt but cannot express:

- **Integer variables** (scheduling, MIP)
- **Boolean variables** (SAT, configuration)
- **Finite-domain variables** (CSP, combinatorial)
- **Optimization objectives** (minimize/maximize, not just feasibility)
- **Logical constraints** (implications, disjunctions)

To evolve solverang into a generic constraint solver, the `Problem` trait must be expanded
**without breaking** the existing continuous solver ecosystem.

## 2. Design Principles

1. **Backward compatibility**: Existing `Problem` implementations must continue to work
   unchanged. The current trait becomes one specialization in a larger hierarchy.

2. **Additive, not destructive**: New abstractions are added alongside, not in place of,
   existing ones. Use trait hierarchy and blanket implementations.

3. **Feature-gated isolation**: New problem types live behind feature flags so the core
   crate remains lean for users who only need continuous solving.

4. **Concurrent-safe**: Other agents are modifying solvers, geometry, JIT, and macros.
   This plan must define **stable interface boundaries** so changes compose cleanly.

5. **Incremental adoption**: A problem can implement only the traits it needs. A pure
   continuous problem need not know about discrete variables.

## 3. Proposed Trait Hierarchy

```
                    ┌─────────────────┐
                    │  ProblemBase     │  (name, variable/constraint metadata)
                    └────────┬────────┘
                             │
            ┌────────────────┼──────────────────┐
            │                │                   │
   ┌────────▼────────┐ ┌────▼──────────┐ ┌──────▼──────────┐
   │ ContinuousProblem│ │ DiscreteProblem│ │  MixedProblem   │
   │ (current Problem)│ │ (CSP/SAT)     │ │  (MIP, hybrid)  │
   └────────┬────────┘ └───────────────┘ └─────────────────┘
            │
   ┌────────▼─────────┐
   │ OptimizationProblem│  (adds objective function)
   └──────────────────┘
```

### 3.1 `ProblemBase` — The Universal Root

```rust
/// Universal base for all solvable problems.
///
/// This trait captures metadata common to every problem type:
/// identity, variable descriptors, and constraint descriptors.
/// It does NOT define how to evaluate constraints — that's
/// the job of specialized sub-traits.
pub trait ProblemBase: Send + Sync {
    /// Problem name for reporting and debugging.
    fn name(&self) -> &str;

    /// Descriptors for each variable in the problem.
    fn variables(&self) -> &[VariableDescriptor];

    /// Number of constraints.
    fn constraint_count(&self) -> usize;

    /// Variable count convenience method.
    fn variable_count(&self) -> usize {
        self.variables().len()
    }
}
```

### 3.2 `VariableDescriptor` — Typed Variable Metadata

```rust
/// Describes a single variable's type, domain, and identity.
#[derive(Clone, Debug)]
pub struct VariableDescriptor {
    /// Human-readable name (e.g., "point_0_x", "machine_assignment").
    pub name: String,
    /// Domain this variable lives in.
    pub domain: VariableDomain,
    /// Index in the problem's variable array.
    pub index: usize,
}

/// The domain (type) of a variable.
#[derive(Clone, Debug, PartialEq)]
pub enum VariableDomain {
    /// Continuous real variable, optionally bounded.
    Continuous {
        lower: f64,  // f64::NEG_INFINITY for unbounded
        upper: f64,  // f64::INFINITY for unbounded
    },
    /// Integer variable with bounds.
    Integer { lower: i64, upper: i64 },
    /// Boolean variable (0 or 1).
    Boolean,
    /// Finite enumerated domain (e.g., {Red, Green, Blue}).
    Enumerated { values: Vec<i64> },
}

impl VariableDomain {
    pub fn is_continuous(&self) -> bool {
        matches!(self, VariableDomain::Continuous { .. })
    }

    pub fn is_discrete(&self) -> bool {
        !self.is_continuous()
    }

    pub fn is_bounded(&self) -> bool {
        match self {
            VariableDomain::Continuous { lower, upper } => {
                lower.is_finite() || upper.is_finite()
            }
            _ => true,
        }
    }
}
```

### 3.3 Preserving `Problem` as `ContinuousProblem`

```rust
/// A continuous nonlinear problem: F(x) = 0 where x in R^n.
///
/// This is the existing `Problem` trait, renamed for clarity in the
/// new hierarchy. A blanket impl or type alias ensures backward compat.
pub trait ContinuousProblem: ProblemBase {
    fn residual_count(&self) -> usize;
    fn residuals(&self, x: &[f64]) -> Vec<f64>;
    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)>;
    fn initial_point(&self, factor: f64) -> Vec<f64>;

    // ... (all existing Problem methods with defaults)
}
```

**Backward compatibility bridge**:

```rust
/// Blanket implementation: anything implementing the legacy Problem trait
/// automatically implements ProblemBase and ContinuousProblem.
impl<T: Problem> ProblemBase for T {
    fn name(&self) -> &str { Problem::name(self) }
    fn variables(&self) -> Vec<VariableDescriptor> {
        // Auto-generate: n unbounded continuous variables
        (0..self.variable_count())
            .map(|i| VariableDescriptor {
                name: format!("x_{}", i),
                domain: VariableDomain::Continuous {
                    lower: f64::NEG_INFINITY,
                    upper: f64::INFINITY,
                },
                index: i,
            })
            .collect()
    }
    fn constraint_count(&self) -> usize { self.residual_count() }
}
```

**CRITICAL**: The existing `Problem` trait remains **unchanged**. All current solver
code (Newton-Raphson, LM, Auto, Robust, Parallel, Sparse, JIT) continues to accept
`&dyn Problem` or `impl Problem`. The new hierarchy is layered on top.

## 4. File Layout

```
crates/solverang/src/
├── problem.rs                    # UNCHANGED — existing Problem trait
├── problem_base.rs               # NEW — ProblemBase, VariableDescriptor, VariableDomain
├── problem_ext.rs                # NEW — ContinuousProblem bridge, OptimizationProblem
├── discrete/                     # NEW (feature: "discrete")
│   ├── mod.rs
│   ├── problem.rs                # DiscreteProblem trait
│   └── variable.rs               # Discrete variable types
├── mixed/                        # NEW (feature: "mixed")
│   ├── mod.rs
│   └── problem.rs                # MixedProblem trait
```

## 5. Migration Strategy

### Phase 1: Add `ProblemBase` alongside `Problem` (non-breaking)
- Create `problem_base.rs` with `ProblemBase`, `VariableDescriptor`, `VariableDomain`
- Add blanket `impl ProblemBase for T where T: Problem`
- No existing code changes required
- Existing solvers still accept `&dyn Problem`

### Phase 2: New solvers accept `ProblemBase` (non-breaking)
- The new optimization solver (Plan 1) accepts `OptimizationProblem`
- The new CSP solver (Plan 3) accepts `DiscreteProblem`
- The new dispatcher (Plan 4) accepts `&dyn ProblemBase`
- All gated behind feature flags

### Phase 3: Optional deprecation (future, breaking)
- In a major version bump, optionally deprecate `Problem` in favor of
  `ContinuousProblem` which extends `ProblemBase`
- Provide a one-line migration: `use solverang::ContinuousProblem as Problem;`
- This is NOT required and can be deferred indefinitely

## 6. Interaction with Concurrent Work

| Active Work Area      | Impact of This Plan                                      |
|-----------------------|----------------------------------------------------------|
| Geometry module       | None. `ConstraintSystem<D>` implements `Problem`, which  |
|                       | gets blanket `ProblemBase` impl. No changes needed.      |
| JIT solver            | None. JITSolver accepts `Problem`. Still works.          |
| Auto-Jacobian macros  | None. `#[auto_jacobian]` generates `Problem` impls.      |
| Sparse solver         | None. SparseSolver accepts `Problem`.                    |
| Inequality transform  | Minor. `SlackVariableTransform` gets automatic           |
|                       | `ProblemBase` via its `Problem` impl.                    |
| Decomposition         | Future opportunity: `DecomposableProblem` could extend   |
|                       | `ProblemBase` instead of `Problem` to support discrete   |
|                       | problem decomposition.                                   |

## 7. Design Decisions & Rationale

**Q: Why not just add methods to `Problem`?**
A: Adding `variables() -> Vec<VariableDescriptor>` to `Problem` with a default impl
would technically work, but it conflates continuous and discrete concerns. A user
defining a SAT problem shouldn't need to implement `residuals()` and `jacobian()`.

**Q: Why a trait hierarchy instead of an enum-based `GenericProblem`?**
A: Trait hierarchy enables zero-cost dispatch for users who know their problem type
at compile time (most use cases). Enum-based would require runtime matching everywhere.
The trait approach also enables independent feature-gated compilation.

**Q: Why `VariableDomain` as an enum instead of a trait?**
A: Domains are a closed set of well-known types (continuous, integer, boolean, enumerated).
New domain types would be a rare, breaking change. An enum gives exhaustive matching,
`Clone`/`Debug` for free, and simple serialization.

**Q: Should `ProblemBase::variables()` return `&[VariableDescriptor]` or `Vec<VariableDescriptor>`?**
A: Return `Cow<[VariableDescriptor]>` or have a `variable_descriptors()` that returns
owned, with a `variable_domain(index)` accessor for cheap lookups. For the blanket impl
over `Problem`, we need to construct descriptors dynamically, so owned is necessary.
For hand-implemented `ProblemBase`, storing them is natural. Use `Cow` to support both.

## 8. Open Questions

1. **Value representation for mixed problems**: Should mixed-type variable vectors use
   `Vec<f64>` with convention (integers stored as f64, booleans as 0.0/1.0) or a
   typed `Value` enum? The f64 convention is simpler and avoids generics explosion
   but loses type safety.

2. **Constraint descriptors**: Should `ProblemBase` also expose `ConstraintDescriptor`
   (type, arity, variables involved)? This would help the classifier in Plan 4 but
   adds API surface.

3. **Lifetime of variable descriptors**: If problems are constructed dynamically,
   descriptors may change. Should `variables()` be infallible or return `Result`?

## 9. Acceptance Criteria

- [ ] `ProblemBase` trait defined with `VariableDescriptor` and `VariableDomain`
- [ ] Blanket `impl ProblemBase for T where T: Problem` compiles and passes tests
- [ ] All existing tests pass without modification
- [ ] At least one non-continuous problem type compiles against the new trait
- [ ] Documentation with examples for both legacy and new-style problem definitions
