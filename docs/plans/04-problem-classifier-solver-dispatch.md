# Plan 4: Problem Classifier and Solver Recommendation (Revised)

## Status: PROPOSAL (v2 — revised from v1)
## Priority: Medium (diagnostics + enables Plan 6 decomposition)
## Depends on: Plan A (ProblemBase)
## Feature flag: none needed (lightweight, always available)

---

## Revision Notes (v2)

**What changed from v1**: Removed `AnySolver` trait, `SolverRegistry`, `SolverInput`,
and `SolverOutput`. These attempted to create a unified `solve()` interface across
all paradigms, which requires type erasure (downcasting `&dyn ProblemBase` to the
actual trait) and returns an enum the caller must match on. This is worse than just
calling the right typed solver directly.

The classifier remains — it's genuinely useful for diagnostics, recommendations, and
as input to the hybrid decomposer (Plan 6). But it classifies problems; it doesn't
solve them.

---

## 1. What This Plan Does

1. **Classifies** any `ProblemBase` into a `ProblemClass` (variable types, structure,
   size, sparsity, decomposability)
2. **Recommends** which solver to use, with human-readable rationale
3. **Partitions** variables into continuous/discrete blocks — the key input to Plan 6's
   hybrid decomposer

## 2. What This Plan Does NOT Do

- No generic `solve()` function
- No `AnySolver` trait or solver registry
- No `SolverOutput` enum
- No runtime solver dispatch (the DSL in Plan 5 handles this at build time;
  Plan 6 handles this for hybrid problems via composition protocols)

## 3. Design

### 3.1 Problem Classification

```rust
#[derive(Clone, Debug, PartialEq)]
pub struct ProblemClass {
    pub variable_types: VariableTypeSet,
    pub has_objective: bool,
    pub structure: ProblemStructure,
    pub size: ProblemSize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VariableTypeSet {
    pub continuous_count: usize,
    pub integer_count: usize,
    pub boolean_count: usize,
    pub enumerated_count: usize,
}

impl VariableTypeSet {
    pub fn is_pure_continuous(&self) -> bool {
        self.continuous_count > 0
            && self.integer_count == 0
            && self.boolean_count == 0
            && self.enumerated_count == 0
    }

    pub fn is_pure_discrete(&self) -> bool {
        self.continuous_count == 0
            && (self.integer_count > 0 || self.boolean_count > 0 || self.enumerated_count > 0)
    }

    pub fn is_mixed(&self) -> bool {
        self.continuous_count > 0
            && (self.integer_count > 0 || self.boolean_count > 0 || self.enumerated_count > 0)
    }

    pub fn total(&self) -> usize {
        self.continuous_count + self.integer_count + self.boolean_count + self.enumerated_count
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProblemStructure {
    pub shape: SystemShape,
    pub sparsity: Option<f64>,         // 0.0 = dense, 1.0 = empty. None = unknown.
    pub component_count: Option<usize>,
    pub is_linear: Option<bool>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SystemShape { Square, OverDetermined, UnderDetermined }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProblemSize { Small, Medium, Large, VeryLarge }
```

### 3.2 Classifier

```rust
pub struct ProblemClassifier;

impl ProblemClassifier {
    /// Cheap classification from metadata only (no function evaluations).
    pub fn classify(problem: &dyn ProblemBase) -> ProblemClass { ... }
}
```

### 3.3 Variable Partition — Key Integration Point for Plan 6

This is the most important output of the classifier for solver composition:

```rust
/// Partition of a problem into continuous and discrete blocks.
///
/// This is the structural analysis that Plan 6's hybrid decomposer uses
/// to decide how to compose solvers.
#[derive(Clone, Debug)]
pub struct VariablePartition {
    /// Indices of continuous variables.
    pub continuous_vars: Vec<usize>,
    /// Indices of discrete variables.
    pub discrete_vars: Vec<usize>,
    /// Constraints touching only continuous variables.
    pub pure_continuous_constraints: Vec<usize>,
    /// Constraints touching only discrete variables.
    pub pure_discrete_constraints: Vec<usize>,
    /// Constraints touching BOTH continuous and discrete variables.
    /// These are the "coupling" constraints that make hybrid solving hard.
    /// If empty, the problem is separable and the two parts can be solved independently.
    pub coupling_constraints: Vec<usize>,
}

impl VariablePartition {
    /// True if continuous and discrete parts are completely independent.
    pub fn is_separable(&self) -> bool {
        self.coupling_constraints.is_empty()
    }
}

impl ProblemClassifier {
    /// Partition variables by type and identify coupling constraints.
    pub fn partition_variables(problem: &dyn ProblemBase) -> VariablePartition {
        let mut continuous = Vec::new();
        let mut discrete = Vec::new();

        for i in 0..problem.variable_count() {
            match problem.variable_domain(i) {
                VariableDomain::Continuous { .. } => continuous.push(i),
                _ => discrete.push(i),
            }
        }

        let mut pure_continuous = Vec::new();
        let mut pure_discrete = Vec::new();
        let mut coupling = Vec::new();

        for j in 0..problem.constraint_count() {
            let scope = problem.constraint_scope(j);
            let touches_continuous = scope.iter().any(|v| continuous.contains(v));
            let touches_discrete = scope.iter().any(|v| discrete.contains(v));

            match (touches_continuous, touches_discrete) {
                (true, true) => coupling.push(j),
                (true, false) => pure_continuous.push(j),
                (false, true) => pure_discrete.push(j),
                (false, false) => {}
            }
        }

        VariablePartition {
            continuous_vars: continuous,
            discrete_vars: discrete,
            pure_continuous_constraints: pure_continuous,
            pure_discrete_constraints: pure_discrete,
            coupling_constraints: coupling,
        }
    }
}
```

### 3.4 Solver Recommendation (diagnostics, not dispatch)

```rust
pub struct SolverRecommendation {
    pub solver_name: String,
    pub rationale: String,
    pub classification: ProblemClass,
    pub alternatives: Vec<(String, String)>,  // (name, why_not_primary)
}

impl ProblemClassifier {
    /// Produce a human-readable solver recommendation.
    /// This is for diagnostics and user guidance — it does not invoke a solver.
    pub fn recommend(problem: &dyn ProblemBase) -> SolverRecommendation { ... }
}
```

## 4. File Layout

```
crates/solverang/src/
├── classify.rs          # NEW — ProblemClassifier, ProblemClass, VariablePartition
```

One file. The classifier is lightweight enough to be always-on (no feature flag).

## 5. Integration Points

| Consumer              | What it uses                                             |
|-----------------------|----------------------------------------------------------|
| Plan 5 (DSL)         | `classify()` at `Model::build()` time to choose traits   |
| Plan 6 (Hybrid)      | `partition_variables()` to split a mixed problem          |
| User diagnostics     | `recommend()` for human-readable solver selection advice  |
| Existing AutoSolver  | Could optionally delegate to classifier, but not required |

## 6. Acceptance Criteria

- [ ] `ProblemClassifier::classify()` works on any `&dyn ProblemBase`
- [ ] Existing `Problem` impls classified correctly via blanket ProblemBase impl
- [ ] `partition_variables()` correctly separates continuous/discrete/coupling
- [ ] `recommend()` produces useful human-readable output
- [ ] Zero changes to existing solver code
