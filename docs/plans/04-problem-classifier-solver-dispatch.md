# Plan 4: Problem Classifier and Solver Dispatch

## Status: PROPOSAL
## Priority: Medium (unifying layer — grows in value as more solvers exist)
## Depends on: Plan A (ProblemBase), and at least one of Plans 1-3 for non-trivial dispatch
## Feature flag: `dispatch` (or always-on if lightweight enough)

---

## 1. Motivation

Solverang already has an `AutoSolver` that chooses between Newton-Raphson and
Levenberg-Marquardt based on m vs n. As more solver paradigms are added (optimization,
MIP, CSP), the selection problem grows from "which continuous solver?" to "which
entire paradigm?":

```
User's Problem
     │
     ▼
┌─────────────────────────┐
│   Problem Classifier    │ ← Analyzes variable types, constraint structure
└────────────┬────────────┘
             │
     ┌───────┼───────┬──────────┬──────────────┐
     ▼       ▼       ▼          ▼              ▼
   NR/LM   Sparse  Optimization  B&B/MIP    CSP
  (square) (large)  (has obj)  (integers)  (discrete)
```

This plan generalizes `AutoSolver` into a **solver registry** that can classify any
`ProblemBase` and dispatch to the appropriate solver.

## 2. Design

### 2.1 Problem Classification

```rust
/// Classification of a problem's structure.
///
/// This captures the "what kind of problem is this?" question,
/// independent of which solver will handle it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProblemClass {
    /// Types of variables present.
    pub variable_types: VariableTypeSet,
    /// Whether the problem has an optimization objective.
    pub has_objective: bool,
    /// Structural characteristics.
    pub structure: ProblemStructure,
    /// Size category.
    pub size: ProblemSize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VariableTypeSet {
    pub has_continuous: bool,
    pub has_integer: bool,
    pub has_boolean: bool,
    pub has_enumerated: bool,
}

impl VariableTypeSet {
    pub fn is_pure_continuous(&self) -> bool {
        self.has_continuous && !self.has_integer && !self.has_boolean && !self.has_enumerated
    }

    pub fn is_pure_discrete(&self) -> bool {
        !self.has_continuous && (self.has_integer || self.has_boolean || self.has_enumerated)
    }

    pub fn is_mixed(&self) -> bool {
        self.has_continuous && (self.has_integer || self.has_boolean || self.has_enumerated)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProblemStructure {
    /// System shape: square (m=n), over-determined (m>n), under-determined (m<n).
    pub shape: SystemShape,
    /// Estimated Jacobian sparsity (0.0 = dense, 1.0 = empty).
    pub sparsity: SparsityLevel,
    /// Whether the problem decomposes into independent components.
    pub decomposable: bool,
    /// Number of independent components (if decomposable).
    pub component_count: Option<usize>,
    /// Whether all constraints are linear.
    pub is_linear: bool,
    /// Whether all constraints are convex (if known).
    pub is_convex: Option<bool>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SystemShape { Square, OverDetermined, UnderDetermined }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SparsityLevel { Dense, Moderate, Sparse, VerySparse }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProblemSize { Small, Medium, Large, VeryLarge }
```

### 2.2 Classifier

```rust
/// Analyzes a problem and produces a classification.
pub struct ProblemClassifier;

impl ProblemClassifier {
    /// Classify a problem from its ProblemBase metadata.
    ///
    /// This is a cheap operation that uses only the problem's metadata
    /// (variable descriptors, constraint count, etc.) — no function evaluations.
    pub fn classify(problem: &dyn ProblemBase) -> ProblemClass {
        let vars = problem.variables();
        let n = vars.len();
        let m = problem.constraint_count();

        let variable_types = VariableTypeSet {
            has_continuous: vars.iter().any(|v| v.domain.is_continuous()),
            has_integer: vars.iter().any(|v| matches!(v.domain, VariableDomain::Integer { .. })),
            has_boolean: vars.iter().any(|v| matches!(v.domain, VariableDomain::Boolean)),
            has_enumerated: vars.iter().any(|v| matches!(v.domain, VariableDomain::Enumerated { .. })),
        };

        let shape = match m.cmp(&n) {
            Ordering::Equal => SystemShape::Square,
            Ordering::Greater => SystemShape::OverDetermined,
            Ordering::Less => SystemShape::UnderDetermined,
        };

        let size = match n {
            0..=10 => ProblemSize::Small,
            11..=100 => ProblemSize::Medium,
            101..=10_000 => ProblemSize::Large,
            _ => ProblemSize::VeryLarge,
        };

        ProblemClass {
            variable_types,
            has_objective: false, // set by specific trait checks
            structure: ProblemStructure {
                shape,
                sparsity: SparsityLevel::Dense, // refined later
                decomposable: false,
                component_count: None,
                is_linear: false,
                is_convex: None,
            },
            size,
        }
    }

    /// Deeper classification that may evaluate the problem.
    ///
    /// This can check sparsity by sampling the Jacobian, test decomposability,
    /// etc. More expensive but more accurate.
    pub fn classify_deep(problem: &dyn Problem) -> ProblemClass {
        let mut class = Self::classify(problem);

        // Check sparsity from Jacobian sample
        let x0 = problem.initial_point(1.0);
        let jac = problem.jacobian(&x0);
        let density = jac.len() as f64 / (problem.residual_count() * problem.variable_count()) as f64;
        class.structure.sparsity = match density {
            d if d > 0.5 => SparsityLevel::Dense,
            d if d > 0.1 => SparsityLevel::Moderate,
            d if d > 0.01 => SparsityLevel::Sparse,
            _ => SparsityLevel::VerySparse,
        };

        class
    }
}
```

### 2.3 Solver Registry

```rust
/// A registry of solvers that can handle different problem classes.
///
/// The registry maps problem classifications to solver instances,
/// generalizing AutoSolver from "NR vs LM" to "any paradigm."
pub struct SolverRegistry {
    /// Registered solver entries, checked in priority order.
    entries: Vec<SolverEntry>,
    /// Fallback solver for unclassified problems.
    fallback: Box<dyn AnySolver>,
}

struct SolverEntry {
    /// Human-readable name for diagnostics.
    name: String,
    /// Predicate: can this solver handle this problem class?
    can_handle: Box<dyn Fn(&ProblemClass) -> bool>,
    /// Priority (higher = preferred when multiple solvers match).
    priority: i32,
    /// The solver itself.
    solver: Box<dyn AnySolver>,
}

/// Trait for solvers that can be registered in the dispatch system.
///
/// This is a unified interface across all solver paradigms.
pub trait AnySolver: Send + Sync {
    /// Solve the problem, returning a generic result.
    fn solve_any(&self, problem: &dyn ProblemBase, initial: &SolverInput) -> SolverOutput;

    /// Name of this solver for diagnostics.
    fn name(&self) -> &str;
}

/// Input to a generic solver (flexible enough for all paradigms).
pub enum SolverInput {
    /// Continuous initial point.
    ContinuousPoint(Vec<f64>),
    /// Discrete initial assignment.
    DiscreteAssignment(Vec<i64>),
    /// Mixed initial values.
    Mixed { continuous: Vec<f64>, discrete: Vec<i64> },
    /// No initial point needed (solver generates its own).
    None,
}

/// Output from a generic solver.
pub enum SolverOutput {
    /// Continuous solution (from NR, LM, optimization).
    Continuous(SolveResult),
    /// Optimization solution.
    Optimization(OptimizationResult),
    /// Integer solution (from B&B).
    MixedInteger(MIPResult),
    /// Discrete solution (from CSP).
    Discrete(CSPResult),
}
```

### 2.4 Default Registry Configuration

```rust
impl SolverRegistry {
    /// Create a registry with all available solvers pre-registered.
    ///
    /// This replaces AutoSolver as the "just solve it" entry point.
    pub fn default_registry() -> Self {
        let mut registry = Self::new(Box::new(RobustSolverAdapter));

        // CSP solver for pure discrete problems
        #[cfg(feature = "csp")]
        registry.register(SolverEntry {
            name: "CSP".into(),
            can_handle: Box::new(|c| c.variable_types.is_pure_discrete() && !c.has_objective),
            priority: 100,
            solver: Box::new(CSPSolverAdapter::default()),
        });

        // B&B for mixed-integer problems
        #[cfg(feature = "mixed-integer")]
        registry.register(SolverEntry {
            name: "MIP".into(),
            can_handle: Box::new(|c| c.variable_types.is_mixed() ||
                                     (c.variable_types.has_integer && c.has_objective)),
            priority: 90,
            solver: Box::new(MIPSolverAdapter::default()),
        });

        // Optimization solver for continuous + objective
        #[cfg(feature = "optimization")]
        registry.register(SolverEntry {
            name: "AugLag".into(),
            can_handle: Box::new(|c| c.variable_types.is_pure_continuous() && c.has_objective),
            priority: 80,
            solver: Box::new(AugLagAdapter::default()),
        });

        // Sparse solver for large sparse systems
        #[cfg(feature = "sparse")]
        registry.register(SolverEntry {
            name: "Sparse".into(),
            can_handle: Box::new(|c| {
                c.variable_types.is_pure_continuous() &&
                matches!(c.size, ProblemSize::Large | ProblemSize::VeryLarge) &&
                matches!(c.structure.sparsity, SparsityLevel::Sparse | SparsityLevel::VerySparse)
            }),
            priority: 70,
            solver: Box::new(SparseSolverAdapter::default()),
        });

        // Parallel solver for decomposable problems
        #[cfg(feature = "parallel")]
        registry.register(SolverEntry {
            name: "Parallel".into(),
            can_handle: Box::new(|c| c.structure.decomposable && c.structure.component_count.unwrap_or(1) > 1),
            priority: 60,
            solver: Box::new(ParallelSolverAdapter::default()),
        });

        // Newton-Raphson for small/medium square continuous systems
        registry.register(SolverEntry {
            name: "Newton-Raphson".into(),
            can_handle: Box::new(|c| {
                c.variable_types.is_pure_continuous() &&
                c.structure.shape == SystemShape::Square &&
                !c.has_objective
            }),
            priority: 50,
            solver: Box::new(NRAdapter::default()),
        });

        // Levenberg-Marquardt as general continuous fallback
        registry.register(SolverEntry {
            name: "Levenberg-Marquardt".into(),
            can_handle: Box::new(|c| c.variable_types.is_pure_continuous() && !c.has_objective),
            priority: 40,
            solver: Box::new(LMAdapter::default()),
        });

        registry
    }
}
```

### 2.5 Top-Level User API

```rust
/// The universal "just solve it" function.
///
/// Classifies the problem, selects the best available solver, and runs it.
/// This is the primary entry point for users who don't want to choose a solver.
pub fn solve(problem: &dyn ProblemBase) -> SolverOutput {
    let registry = SolverRegistry::default_registry();
    registry.solve(problem)
}

/// Solve with explicit initial values.
pub fn solve_with(problem: &dyn ProblemBase, initial: SolverInput) -> SolverOutput {
    let registry = SolverRegistry::default_registry();
    registry.solve_with(problem, initial)
}
```

## 3. Solver Recommendation / Explain Mode

Useful for users who want to understand why a solver was chosen:

```rust
/// Explains which solver would be selected and why.
pub struct SolverRecommendation {
    pub selected_solver: String,
    pub classification: ProblemClass,
    pub rationale: String,
    pub alternatives: Vec<AlternativeSolver>,
}

pub struct AlternativeSolver {
    pub name: String,
    pub why_not_selected: String,
}

impl SolverRegistry {
    pub fn recommend(&self, problem: &dyn ProblemBase) -> SolverRecommendation { ... }
}
```

## 4. File Layout

```
crates/solverang/src/
├── dispatch/                      # NEW (feature: "dispatch")
│   ├── mod.rs                     # Module exports
│   ├── classifier.rs              # ProblemClassifier, ProblemClass
│   ├── registry.rs                # SolverRegistry, SolverEntry
│   ├── any_solver.rs              # AnySolver trait, SolverInput/Output
│   ├── adapters.rs                # Adapters wrapping existing solvers
│   └── recommend.rs               # SolverRecommendation
```

## 5. Implementation Phases

### Phase 1: Classification + registry with existing solvers only
- `ProblemClass`, `ProblemClassifier`
- `SolverRegistry` with NR, LM, Sparse, Parallel entries
- `AnySolver` trait + adapters for existing solvers
- Replace `AutoSolver` decision logic with classifier-based dispatch
- Test: verify that dispatch selects the same solver as `AutoSolver` for all
  existing test problems

### Phase 2: Deep classification
- `classify_deep()` with sparsity analysis and decomposition check
- Integrate with `should_use_sparse()` from sparse_solver.rs
- Integrate with `decompose()` from decomposition.rs

### Phase 3: New solver integration
- As Plans 1, 2, 3 deliver solvers, register them in the default registry
- Each new solver adds its `can_handle` predicate and adapter

### Phase 4: Explain mode
- `SolverRecommendation` with human-readable rationale
- CLI integration: `solverang explain <problem>`

## 6. Relationship to Existing `AutoSolver`

`AutoSolver` currently lives at `solver/auto.rs` and makes a binary choice:

```rust
fn auto_select(&self, problem: &P) -> SolverChoice {
    if m == n { NewtonRaphson } else { LevenbergMarquardt }
}
```

The new dispatch system **subsumes** `AutoSolver`:

- `AutoSolver` remains available for users who only need continuous solving
- `SolverRegistry::default_registry()` includes the same logic but extends to
  all paradigms
- In a future major version, `AutoSolver` could delegate to `SolverRegistry`
  internally

## 7. Interaction with Concurrent Work

| Active Work Area      | Impact                                                    |
|-----------------------|-----------------------------------------------------------|
| AutoSolver            | Complementary. AutoSolver stays for continuous-only use.  |
|                       | Registry generalizes the concept.                         |
| RobustSolver          | Becomes a fallback strategy within the registry.          |
| Existing solvers      | Each gets an `AnySolver` adapter. No code changes needed. |
| New solvers (Plans 1-3)| Each registers in the default registry via adapter.      |
| Geometry module       | ConstraintSystem implements Problem, automatically gets   |
|                       | classified as continuous + square/over-determined.         |

## 8. Open Questions

1. **Static vs dynamic dispatch**: Should the registry use trait objects (`dyn AnySolver`)
   or an enum of known solvers? Trait objects are more extensible; enum is faster and
   enables exhaustive matching.

2. **Configuration forwarding**: When the registry selects a solver, how does the user
   configure that specific solver (e.g., NR tolerance, LM patience)? Options:
   - Registry holds pre-configured solver instances (current design)
   - Registry accepts a generic config map
   - User configures via the adapter

3. **Automatic problem transformation**: Should the classifier also suggest
   transformations? E.g., "this problem has inequalities; applying slack transform
   would make it solvable by NR." This blurs classifier/transformer/solver boundaries.

4. **Caching**: Should the registry cache classifications? Useful if the same problem
   is solved repeatedly with different initial points.

## 9. Acceptance Criteria

- [ ] `ProblemClassifier` correctly classifies existing Problem implementations
- [ ] `SolverRegistry` dispatches to NR for square, LM for non-square (matching AutoSolver)
- [ ] Sparse problems dispatched to SparseSolver when feature enabled
- [ ] Decomposable problems dispatched to ParallelSolver when feature enabled
- [ ] `AnySolver` adapter works for NR, LM, Sparse, Parallel solvers
- [ ] `SolverRecommendation` produces readable explanations
- [ ] Adding a new solver type requires only implementing `AnySolver` + registering
