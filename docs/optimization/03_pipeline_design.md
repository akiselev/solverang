# Pipeline Design: Extending the Solve Pipeline for Optimization

## 1. Executive Summary

This document specifies how the Solverang `ConstraintSystem` and `SolvePipeline` extend to
accommodate optimization problems (minimizing an objective subject to constraints) alongside
the existing pure constraint-solving infrastructure. The design preserves full backward
compatibility: every existing `ConstraintSystem` usage continues to work without modification.

The key architectural decision is a **unified system with a separate entry point** rather
than a parallel `OptimizationSystem` type. The `ConstraintSystem` gains an optional
`Objective` and an `OptimizationConfig`, and the pipeline gains new phases that activate
only when an objective is present. This keeps the codebase cohesive, avoids trait duplication,
and allows the objective to be added incrementally to an existing constraint system.

---

## 2. Current Architecture Recap

### 2.1 Pipeline Flow

```
Decompose --> Analyze --> Reduce --> Solve --> PostProcess
```

- **Decompose** (`DefaultDecompose`): union-find over shared parameters produces
  `Vec<ClusterData>`. Each `ClusterData` holds `constraint_indices`, `param_ids`, `entity_ids`.

- **Analyze** (`DefaultAnalyze`): per-cluster DOF, redundancy, pattern detection.
  Produces `ClusterAnalysis` (advisory, does not mutate state).

- **Reduce** (`DefaultReduce`): eliminates single-free-param constraints analytically,
  merges coincident params. Produces `ReducedCluster`.

- **Solve** (`DefaultSolve`): tries closed-form patterns first, then LM for leftovers.
  Produces `ClusterSolution`.

- **PostProcess** (`DefaultPostProcess`): converts `ClusterSolution` into `ClusterResult`
  with diagnostics.

### 2.2 Key Types

| Type | Role |
|------|------|
| `ParamStore` | Central f64 storage with generational IDs, fixed/free tracking |
| `SolverMapping` | ParamId <-> solver column bidirectional map |
| `SystemConfig` | Holds `LMConfig` + `SolverConfig` |
| `SystemResult` | Status + per-cluster results + timing |
| `Problem` trait | `residuals(x)` + `jacobian(x)` for numerical solvers |

### 2.3 Algorithm Selection (Current)

`AutoSolver` dispatches based on equation count vs. variable count:
- Square (m == n): Newton-Raphson
- Over-determined (m > n): Levenberg-Marquardt
- Under-determined (m < n): Levenberg-Marquardt

---

## 3. Unified System Design

### 3.1 Why Not a Separate `OptimizationSystem`

A separate type would duplicate entity/constraint management, param allocation, change
tracking, and caching. Instead, we add optimization as an *overlay* on `ConstraintSystem`:

```rust
pub struct ConstraintSystem {
    // ... existing fields unchanged ...
    params: ParamStore,
    entities: Vec<Option<Box<dyn Entity>>>,
    constraints: Vec<Option<Box<dyn Constraint>>>,
    config: SystemConfig,
    pipeline: SolvePipeline,
    change_tracker: ChangeTracker,
    solution_cache: SolutionCache,

    // --- NEW: optimization overlay ---
    objective: Option<Box<dyn Objective>>,
    opt_config: OptimizationConfig,          // initialized to Default::default()
    multiplier_store: MultiplierStore,       // empty when no objective
    last_opt_result: Option<OptimizationResult>,
}
```

The existing `solve()` method is unchanged. A new `optimize()` method activates the
optimization pipeline.

### 3.2 API Surface

```rust
impl ConstraintSystem {
    // Existing (unchanged)
    pub fn solve(&mut self) -> SystemResult { ... }

    // New: optimization
    pub fn set_objective(&mut self, objective: Box<dyn Objective>) { ... }
    pub fn clear_objective(&mut self) { ... }
    pub fn set_opt_config(&mut self, config: OptimizationConfig) { ... }
    pub fn optimize(&mut self) -> OptimizationResult { ... }

    // New: multiplier queries (post-solve)
    pub fn multiplier(&self, constraint: ConstraintId) -> Option<&[f64]> { ... }
    pub fn multipliers(&self) -> &MultiplierStore { ... }
}
```

**Backward compatibility guarantee**: calling `solve()` on a system that has an objective
set ignores the objective and solves constraints only. This is deliberate -- `solve()` and
`optimize()` are distinct entry points with distinct semantics.

---

## 4. New Pipeline Phases

### 4.1 Extended Pipeline (Optimization Mode)

```
 Classify --> Decompose* --> Analyze* --> Reduce --> MultiplierInit --> Solve* --> PostProcess*
    |                                                    |                |            |
    v                                                    v                v            v
ProblemClass                                     MultiplierState   OptSolution   OptResult
```

Phases marked with `*` have modified behavior in optimization mode. The `Classify` and
`MultiplierInit` phases are new and only activate when an objective is present.

### 4.2 Phase 0: Classify (NEW)

```rust
/// Classification of the overall problem type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProblemClass {
    /// Pure constraint solving: F(x) = 0.
    /// No objective present. Use existing NR/LM pipeline.
    ConstraintSolving,

    /// Unconstrained optimization: min f(x).
    /// No constraints. Use BFGS or L-BFGS.
    UnconstrainedOptimization,

    /// Equality-constrained optimization: min f(x) s.t. c(x) = 0.
    /// Use SQP (small/medium) or augmented Lagrangian (large).
    EqualityConstrained {
        n_vars: usize,
        n_eq: usize,
    },

    /// Inequality-constrained optimization: min f(x) s.t. c_eq(x) = 0, c_ineq(x) >= 0.
    /// Use SQP with active-set or IPM.
    MixedConstrained {
        n_vars: usize,
        n_eq: usize,
        n_ineq: usize,
    },
}

/// Phase 0 trait: classify the problem.
pub trait Classify: Send + Sync {
    fn classify(
        &self,
        objective: Option<&dyn Objective>,
        constraints: &[Option<Box<dyn Constraint>>],
        store: &ParamStore,
    ) -> ProblemClass;
}
```

The classifier examines:
1. Whether an objective is present
2. Total equation count vs. free variable count
3. Whether any constraints are marked as inequality constraints (a new `Constraint` trait
   method `fn is_inequality(&self) -> bool`, defaulting to `false`)

> **REVIEW NOTE:** The `is_inequality()` method is not yet present in the `Constraint`
> trait in the codebase (`crates/solverang/src/pipeline/traits.rs`). Adding it as a
> defaulted method (returning `false`) is non-breaking for all existing implementations,
> but the name must be finalized before implementation begins. Also note: the `Classify`
> phase may be simplified by checking `self.objective.is_none()` directly in `optimize()`
> rather than maintaining a separate phase slot. Consider removing `Classify` and folding
> the check into `optimize()` to reduce pipeline complexity.

When `ProblemClass::ConstraintSolving` is returned, the pipeline falls through to the
existing code path with zero overhead.

### 4.3 Phase 1: Decompose (MODIFIED)

**For pure constraints**: unchanged.

**For optimization**: decomposition must account for the objective coupling variables.
The objective function f(x) may reference any subset of parameters. Two strategies:

#### Strategy A: Monolithic (default for optimization)

When an objective is present, all parameters referenced by the objective are forced into a
single "optimization cluster." Constraints that share parameters with this cluster are
absorbed into it. Remaining constraints that are fully decoupled from the objective form
independent constraint-only clusters and are solved with the existing pipeline.

```
                         +---------------------------+
                         |   Optimization Cluster    |
                         |  (objective + coupled     |
                         |   constraints)            |
                         +---------------------------+
                                    |
              +---------------------+---------------------+
              |                     |                     |
    +---------+---------+  +--------+--------+  +---------+---------+
    | Constraint-only   |  | Constraint-only |  | Constraint-only   |
    | Cluster A         |  | Cluster B       |  | Cluster C         |
    +---------+---------+  +--------+--------+  +---------+---------+
```

#### Strategy B: Partial Separability Detection (advanced)

For objectives with known structure `f(x) = sum_k f_k(x_k)` where each `f_k` depends on
a subset of variables, the decomposer can detect partial separability and split the
optimization into smaller coupled blocks. This requires the `Objective` trait to expose
sub-objective structure (see Section 7).

```rust
/// Extended decomposition for optimization problems.
pub trait DecomposeOpt: Decompose + Send + Sync {
    /// Decompose with awareness of the objective function's variable dependencies.
    fn decompose_with_objective(
        &self,
        objective: &dyn Objective,
        constraints: &[Option<Box<dyn Constraint>>],
        entities: &[Option<Box<dyn Entity>>],
        store: &ParamStore,
    ) -> OptDecomposition;
}

// > **REVIEW NOTE:** `DecomposeOpt` extends `Decompose`, but `SolvePipeline` stores
// > `decompose: Box<dyn Decompose>`. A `Box<dyn Decompose>` cannot be upcast to
// > `Box<dyn DecomposeOpt>` at runtime in Rust. The extended `SolvePipeline` must store
// > the optimization decomposer in a **separate field**:
// >
// >     decompose_opt: Option<Box<dyn DecomposeOpt>>,
// >
// > defaulting to `None`. When `optimize()` runs, it checks for `decompose_opt` first;
// > if absent, it uses a default `DefaultDecomposeOpt` that implements both traits. The
// > existing `decompose: Box<dyn Decompose>` field is unchanged for constraint-only solving.

/// Result of optimization-aware decomposition.
#[derive(Clone, Debug)]
pub struct OptDecomposition {
    /// The primary optimization cluster (objective + coupled constraints).
    pub opt_cluster: OptClusterData,
    /// Independent constraint-only clusters (solved with existing pipeline).
    pub constraint_clusters: Vec<ClusterData>,
}

/// An optimization cluster: objective + constraints that share its variables.
#[derive(Clone, Debug)]
pub struct OptClusterData {
    /// All param IDs in this cluster (union of objective and constraint params).
    pub param_ids: Vec<ParamId>,
    /// Constraint indices coupled to the objective.
    pub constraint_indices: Vec<usize>,
    /// Entity IDs referenced.
    pub entity_ids: Vec<EntityId>,
    /// Whether partial separability was detected.
    pub separable_blocks: Option<Vec<SeparableBlock>>,
}

impl OptClusterData {
    /// Convert to a `ClusterData` for use with the existing `Reduce` phase.
    ///
    /// This conversion is lossy: `separable_blocks` information is discarded.
    /// The returned `ClusterData` uses a fixed placeholder `ClusterId(0)` since
    /// optimization clusters are not tracked in the same ID space as constraint clusters.
    pub fn as_cluster_data(&self) -> ClusterData {
        ClusterData {
            id: crate::id::ClusterId(0),
            constraint_indices: self.constraint_indices.clone(),
            param_ids: self.param_ids.clone(),
            entity_ids: self.entity_ids.clone(),
        }
    }
}

/// A separable block within the optimization cluster.
#[derive(Clone, Debug)]
pub struct SeparableBlock {
    /// Sub-objective index (into the Objective's sub_objectives()).
    pub sub_objective_idx: usize,
    /// Parameters in this block.
    pub param_ids: Vec<ParamId>,
    /// Constraint indices local to this block.
    pub constraint_indices: Vec<usize>,
}
```

### 4.4 Phase 2: Analyze (MODIFIED)

In optimization mode, the analyzer additionally computes:

```rust
/// Extended analysis for optimization clusters.
#[derive(Clone, Debug, Default)]
pub struct OptClusterAnalysis {
    /// Standard cluster analysis (DOF, redundancy, patterns).
    pub base: ClusterAnalysis,
    /// Problem classification for this cluster.
    pub problem_class: ProblemClass,
    /// Estimated condition number of the constraint Jacobian (if computed).
    pub constraint_condition: Option<f64>,
    /// Whether the objective Hessian appears to be positive definite
    /// (sampled at the current point).
    pub objective_convexity: Option<ConvexityEstimate>,
    /// Recommended algorithm based on problem structure.
    pub recommended_algorithm: AlgorithmChoice,
}

#[derive(Clone, Debug)]
pub enum ConvexityEstimate {
    /// All eigenvalues of the sampled Hessian are positive.
    LikelyConvex,
    /// Mixed eigenvalues detected.
    LikelyNonconvex,
    /// Hessian not available or too expensive to compute.
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlgorithmChoice {
    /// Newton-Raphson / LM (pure constraint solving).
    ConstraintSolver,
    /// BFGS or L-BFGS (unconstrained).
    QuasiNewton,
    /// Sequential Quadratic Programming (equality/mixed constrained, small/medium).
    SQP,
    /// Interior Point Method (mixed constrained, large scale).
    IPM,
    /// Augmented Lagrangian (large-scale equality constrained).
    AugmentedLagrangian,
}
```

The `OptClusterAnalysis` is produced by an `AnalyzeOpt` trait (analogous to `Analyze` for
constraint clusters):

```rust
/// Phase 2 trait for optimization clusters.
pub trait AnalyzeOpt: Send + Sync {
    fn analyze_opt(
        &self,
        opt_cluster: &OptClusterData,
        objective: &dyn Objective,
        constraints: &[Option<Box<dyn Constraint>>],
        entities: &[Option<Box<dyn Entity>>],
        store: &ParamStore,
    ) -> OptClusterAnalysis;
}
```

This must be stored as a separate field `analyze_opt: Box<dyn AnalyzeOpt>` in the extended
`SolvePipeline` (see Section 11 note on pipeline extension).

**Algorithm selection logic** (implemented in `DefaultOptAnalyze`):

> **REVIEW NOTE (threshold):** The threshold `n + m_eq < 500` mixes variable count and
> constraint count. This is a heuristic and should not be interpreted as a derived bound.
> The internal `QPSolverChoice` documentation in Section 6.2 states Dense QP is for
> `n < 200`; those thresholds are independent. The SQP auto-selection threshold governs
> the outer algorithm; QP solver selection (Dense vs Sparse) happens inside SQP.
> Additionally, per `00_synthesis.md`, ALM is implemented first. Until SQP is available,
> `AugmentedLagrangian` should serve as the fallback for all equality-constrained problems
> regardless of size. Document this as a v1 constraint.

```
// NOTE: Thresholds below are heuristics. In v1, AugmentedLagrangian serves all
// equality-constrained cases until SQP is implemented. SQP and IPM are target-state
// algorithm choices, not v1 defaults.
select_algorithm(problem):
    if no objective:
        return ConstraintSolver

    n = variable_count          // number of free parameters
    m_eq = equality_constraint_count
    m_ineq = inequality_constraint_count

    if m_eq == 0 and m_ineq == 0:
        return QuasiNewton  // unconstrained

    if m_ineq == 0:
        // equality constrained only
        if n < 200:             // n (variables only), not n + m_eq
            return SQP
        else:
            return AugmentedLagrangian

    // mixed constrained
    if n < 200:
        return SQP
    else:
        return IPM
```

### 4.5 Phase 3: Reduce (UNCHANGED)

The reduce phase operates on constraints only and is unaware of the objective. It
eliminates trivial constraints before they reach the optimization solver, shrinking the
problem. Eliminated parameters have their values substituted into the objective
automatically (the objective reads from `ParamStore`, and eliminated values are written
there during reduction).

### 4.6 Phase 3.5: MultiplierInit (NEW)

After reduction and before solving, Lagrange multipliers are initialized for each active
equality constraint in the optimization cluster.

```rust
/// Phase 3.5 trait: initialize Lagrange multipliers.
pub trait InitMultipliers: Send + Sync {
    fn init_multipliers(
        &self,
        opt_cluster: &OptClusterData,
        reduced: &ReducedCluster,
        constraints: &[Option<Box<dyn Constraint>>],
        store: &ParamStore,
        multiplier_store: &mut MultiplierStore,
        config: &OptimizationConfig,
    ) -> MultiplierState;
}

/// Snapshot of multiplier values entering the solve phase.
#[derive(Clone, Debug)]
pub struct MultiplierState {
    /// Multiplier values indexed by (constraint_index, equation_row).
    pub values: Vec<f64>,
    /// Mapping: position in values vec -> (constraint_index, equation_row).
    pub mapping: Vec<(usize, usize)>,
}
```

**Initialization strategies** (configurable via `MultiplierInitStrategy`):

| Strategy | Description | When to use |
|----------|-------------|-------------|
| `Zero` | All multipliers start at 0.0 | Default, simple, works for convex problems |
| `LeastSquares` | Solve J^T * lambda = -grad_f for lambda | Better for SQP warm-start |
| `WarmStart` | Reuse from previous solve | Incremental re-optimization |
| `Penalty` | lambda = mu * c(x) (penalty parameter times residual) | Augmented Lagrangian |

### 4.7 Phase 4: Solve (MODIFIED)

The solve phase gains a new dispatch path for optimization clusters:

```rust
/// Extended solve trait for optimization.
pub trait SolveOptCluster: Send + Sync {
    fn solve_opt_cluster(
        &self,
        opt_cluster: &OptClusterData,
        reduced: &ReducedCluster,
        analysis: &OptClusterAnalysis,
        objective: &dyn Objective,
        constraints: &[Option<Box<dyn Constraint>>],
        store: &ParamStore,
        multiplier_state: &MultiplierState,
        warm_start: Option<&[f64]>,
        config: &OptimizationConfig,
    ) -> OptClusterSolution;
}

/// Solution of an optimization cluster.
#[derive(Clone, Debug)]
pub struct OptClusterSolution {
    /// Solve status.
    pub status: OptSolveStatus,
    /// Optimal parameter values.
    pub param_values: Vec<(ParamId, f64)>,
    /// Final Lagrange multipliers (one per constraint equation).
    pub multipliers: Vec<f64>,
    /// Multiplier mapping (same as MultiplierState::mapping).
    pub multiplier_mapping: Vec<(usize, usize)>,
    /// Optimal objective value.
    pub objective_value: f64,
    /// Constraint violation norm at solution.
    pub constraint_violation: f64,
    /// Number of outer iterations (SQP/IPM iterations).
    pub iterations: usize,
    /// Solver mapping used.
    pub mapping: Option<SolverMapping>,
    /// Raw solution vector.
    pub numerical_solution: Option<Vec<f64>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptSolveStatus {
    /// KKT conditions satisfied within tolerance.
    Optimal,
    /// Converged but KKT residual slightly above tolerance.
    FeasibleSuboptimal,
    /// Constraints satisfied but objective not minimized.
    FeasibleNotConverged,
    /// Solver did not converge.
    NotConverged,
    /// Problem detected as infeasible.
    Infeasible,
}
```

**Dispatch logic** (in `DefaultSolveOpt`):

```
match analysis.recommended_algorithm {
    ConstraintSolver => {
        // Delegate to existing DefaultSolve/NumericalOnlySolve
        // (should not happen in optimization path, but handle gracefully)
    }
    QuasiNewton => {
        // Use BFGSSolver or LBFGSSolver
        solve_unconstrained(objective, store, config.bfgs_config)
    }
    SQP => {
        // Use SQPSolver
        solve_sqp(objective, constraints, store, multipliers, config.sqp_config)
    }
    IPM => {
        // Use IPMSolver
        solve_ipm(objective, constraints, store, multipliers, config.ipm_config)
    }
    AugmentedLagrangian => {
        // Use AugLagSolver (outer loop updates multipliers, inner loop uses LM/BFGS)
        solve_auglag(objective, constraints, store, multipliers, config.auglag_config)
    }
}
```

### 4.8 Phase 5: PostProcess (MODIFIED)

Post-processing additionally extracts multiplier values and computes sensitivity
information.

```rust
/// Extended post-processing for optimization.
pub trait PostProcessOpt: Send + Sync {
    fn post_process_opt(
        &self,
        solution: &OptClusterSolution,
        analysis: &OptClusterAnalysis,
        opt_cluster: &OptClusterData,
    ) -> OptClusterResult;
}
```

---

## 5. ParamStore Extension for Multipliers

### 5.1 Design Decision: Separate MultiplierStore

Lagrange multipliers are **not** stored in `ParamStore`. Rationale:

1. **Multipliers are not geometric parameters.** They have no `EntityId` owner, no
   fixed/free semantics, and no meaning outside optimization.
2. **Lifetime differs.** Multipliers exist only during and after optimization. They are
   recomputed each solve and have no change-tracking or caching semantics.
3. **Mixing them would complicate `ParamStore`.** The generational ID, owner, and
   fixed/free machinery would need special-casing for multipliers.

### 5.2 MultiplierStore

<!-- Decision: MultiplierId uses semantic addressing { constraint_id: ConstraintId, equation_row: usize }. No generational index needed — multipliers are recomputed each solve. Both 01_mathematical_architecture.md and 00_synthesis.md now use this form. -->

```rust
/// Identifier for a Lagrange multiplier.
///
/// Addresses a multiplier by the constraint it belongs to and the equation row
/// within that constraint. No generational bookkeeping is needed because multipliers
/// are recomputed each optimization solve.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MultiplierId {
    /// Which constraint this multiplier is associated with.
    pub constraint_id: ConstraintId,
    /// Which equation row within that constraint (0-based).
    pub equation_row: usize,
}

/// Storage for Lagrange multipliers, separate from ParamStore.
#[derive(Clone, Debug, Default)]
pub struct MultiplierStore {
    /// Multiplier values indexed by MultiplierId.
    entries: HashMap<MultiplierId, f64>,
}

impl MultiplierStore {
    pub fn new() -> Self { Self::default() }

    /// Set the multiplier for a specific constraint equation.
    pub fn set(&mut self, id: MultiplierId, value: f64) {
        self.entries.insert(id, value);
    }

    /// Get the multiplier value. Returns 0.0 if not set.
    pub fn get(&self, id: MultiplierId) -> f64 {
        self.entries.get(&id).copied().unwrap_or(0.0)
    }

    /// Get all multipliers for a constraint as a slice-like view.
    pub fn for_constraint(&self, cid: ConstraintId, neq: usize) -> Vec<f64> {
        (0..neq).map(|row| {
            self.get(MultiplierId { constraint_id: cid, equation_row: row })
        }).collect()
    }

    /// Clear all multipliers.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Number of stored multipliers.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over all multiplier entries.
    pub fn iter(&self) -> impl Iterator<Item = (&MultiplierId, &f64)> {
        self.entries.iter()
    }

    /// Extract multiplier values in a flat vector matching a given ordering.
    pub fn extract_ordered(&self, ordering: &[(ConstraintId, usize)]) -> Vec<f64> {
        ordering.iter().map(|&(cid, row)| {
            self.get(MultiplierId { constraint_id: cid, equation_row: row })
        }).collect()
    }

    /// Write multiplier values back from a flat vector using the given ordering.
    pub fn write_ordered(&mut self, values: &[f64], ordering: &[(ConstraintId, usize)]) {
        for (i, &(cid, row)) in ordering.iter().enumerate() {
            if i < values.len() {
                self.set(MultiplierId { constraint_id: cid, equation_row: row }, values[i]);
            }
        }
    }
}
```

### 5.3 Multiplier-to-Solver Mapping

Analogous to `SolverMapping` for parameters, the optimization solver needs a mapping
between multiplier positions and the flat vector used internally:

```rust
/// Bidirectional mapping: MultiplierId <-> position in the multiplier vector.
#[derive(Clone, Debug)]
pub struct MultiplierMapping {
    pub id_to_pos: HashMap<MultiplierId, usize>,
    pub pos_to_id: Vec<MultiplierId>,
}
```

This is built once per optimization solve from the active constraint list.

---

## 6. Configuration Types

### 6.1 OptimizationConfig

```rust
/// Top-level optimization configuration.
#[derive(Clone, Debug)]
pub struct OptimizationConfig {
    /// Algorithm selection strategy.
    pub algorithm: AlgorithmSelection,
    /// Multiplier initialization strategy.
    pub multiplier_init: MultiplierInitStrategy,
    /// KKT tolerance for declaring optimality.
    pub kkt_tolerance: f64,
    /// Constraint violation tolerance.
    pub feasibility_tolerance: f64,
    /// Maximum outer iterations.
    pub max_iterations: usize,
    /// SQP-specific configuration.
    pub sqp: SQPConfig,
    /// IPM-specific configuration.
    pub ipm: IPMConfig,
    /// BFGS configuration (unconstrained).
    pub bfgs: BFGSConfig,
    /// Augmented Lagrangian configuration.
    pub auglag: AugLagConfig,
    /// Whether to compute sensitivity information post-solve.
    pub compute_sensitivity: bool,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            algorithm: AlgorithmSelection::Auto,
            multiplier_init: MultiplierInitStrategy::LeastSquares,
            kkt_tolerance: 1e-8,
            feasibility_tolerance: 1e-10,
            max_iterations: 200,
            sqp: SQPConfig::default(),
            ipm: IPMConfig::default(),
            bfgs: BFGSConfig::default(),
            auglag: AugLagConfig::default(),
            compute_sensitivity: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlgorithmSelection {
    /// Automatic selection based on problem structure.
    Auto,
    /// Force SQP.
    ForceSQP,
    /// Force IPM.
    ForceIPM,
    /// Force augmented Lagrangian.
    ForceAugLag,
    /// Force quasi-Newton (ignore constraints -- penalty method).
    ForceQuasiNewton,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MultiplierInitStrategy {
    /// All multipliers start at zero.
    Zero,
    /// Least-squares estimate from KKT system.
    LeastSquares,
    /// Reuse values from MultiplierStore (warm start).
    WarmStart,
    /// Penalty-based: lambda = mu * c(x).
    Penalty { mu: f64 },
}
```

### 6.2 SQPConfig

```rust
/// Configuration for the SQP solver.
#[derive(Clone, Debug)]
pub struct SQPConfig {
    /// Maximum SQP iterations.
    pub max_iterations: usize,
    /// Step tolerance for convergence.
    pub step_tolerance: f64,
    /// Merit function parameter (l1 penalty).
    pub merit_penalty: f64,
    /// Hessian approximation strategy.
    pub hessian: HessianStrategy,
    /// Line search parameters.
    pub line_search: LineSearchConfig,
    /// QP sub-problem solver.
    pub qp_solver: QPSolverChoice,
}

impl Default for SQPConfig {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            step_tolerance: 1e-10,
            merit_penalty: 10.0,
            hessian: HessianStrategy::BFGS,
            line_search: LineSearchConfig::default(),
            qp_solver: QPSolverChoice::Dense,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HessianStrategy {
    /// Exact Hessian of the Lagrangian (requires second derivatives).
    Exact,
    /// BFGS quasi-Newton approximation (no second derivatives needed).
    BFGS,
    /// SR1 quasi-Newton approximation (can capture indefiniteness).
    SR1,
    /// Gauss-Newton approximation (Hessian ~ J^T J, good near solution).
    GaussNewton,
}

#[derive(Clone, Debug)]
pub struct LineSearchConfig {
    pub armijo_c: f64,
    pub backtrack_factor: f64,
    pub max_iterations: usize,
}

impl Default for LineSearchConfig {
    fn default() -> Self {
        Self {
            armijo_c: 1e-4,
            backtrack_factor: 0.5,
            max_iterations: 20,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QPSolverChoice {
    /// Dense QP solver (for small problems, n < 200).
    Dense,
    /// Sparse QP solver (for larger problems).
    Sparse,
}
```

> **REVIEW NOTE:** The `Dense` QP solver is documented as suitable for `n < 200` (variable
> count). The `SQP` algorithm selection in Section 4.4 uses `n < 200` as the SQP threshold
> (corrected from the original `n + m_eq < 500`). These two thresholds are now consistent:
> when SQP is selected (n < 200), the default `Dense` QP solver is appropriate. For n >= 200,
> `AugmentedLagrangian` is selected instead, which does not require a QP sub-solver.

### 6.3 IPMConfig

```rust
/// Configuration for the Interior Point Method solver.
#[derive(Clone, Debug)]
pub struct IPMConfig {
    /// Maximum IPM iterations.
    pub max_iterations: usize,
    /// Barrier parameter reduction factor.
    pub barrier_reduction: f64,
    /// Initial barrier parameter.
    pub initial_barrier: f64,
    /// Minimum barrier parameter before switching to exact solve.
    pub min_barrier: f64,
    /// Mehrotra corrector steps.
    pub use_mehrotra: bool,
    /// Linear system solver for the KKT system.
    pub linear_solver: LinearSolverChoice,
}

impl Default for IPMConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            barrier_reduction: 0.2,
            initial_barrier: 1.0,
            min_barrier: 1e-12,
            use_mehrotra: true,
            linear_solver: LinearSolverChoice::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearSolverChoice {
    /// Automatic selection based on matrix structure.
    Auto,
    /// Dense LU/Cholesky factorization.
    Dense,
    /// Sparse LDL factorization.
    SparseLDL,
}
```

### 6.4 BFGSConfig and AugLagConfig

```rust
/// Configuration for BFGS/L-BFGS unconstrained optimization.
#[derive(Clone, Debug)]
pub struct BFGSConfig {
    pub max_iterations: usize,
    pub gradient_tolerance: f64,
    pub step_tolerance: f64,
    /// Memory depth for L-BFGS (0 = full BFGS).
    pub lbfgs_memory: usize,
    pub line_search: LineSearchConfig,
}

impl Default for BFGSConfig {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            gradient_tolerance: 1e-8,
            step_tolerance: 1e-10,
            lbfgs_memory: 20,
            line_search: LineSearchConfig::default(),
        }
    }
}

/// Configuration for augmented Lagrangian method.
#[derive(Clone, Debug)]
pub struct AugLagConfig {
    /// Maximum outer iterations (multiplier updates).
    pub max_outer_iterations: usize,
    /// Maximum inner iterations per outer iteration.
    pub max_inner_iterations: usize,
    /// Initial penalty parameter.
    pub initial_penalty: f64,
    /// Penalty increase factor.
    pub penalty_increase: f64,
    /// Maximum penalty parameter.
    pub max_penalty: f64,
    /// Inner solver tolerance schedule (tightens each outer iteration).
    pub inner_tol_start: f64,
    pub inner_tol_end: f64,
}

impl Default for AugLagConfig {
    fn default() -> Self {
        Self {
            max_outer_iterations: 50,
            max_inner_iterations: 200,
            initial_penalty: 10.0,
            penalty_increase: 10.0,
            max_penalty: 1e12,
            inner_tol_start: 1e-2,
            inner_tol_end: 1e-10,
        }
    }
}
```

---

## 7. Objective Trait

<!-- Decision: System-level Objective trait has fn id() -> ObjectiveId as required, fn gradient() returning Vec<(ParamId, f64)> (sparse, ParamStore-based). Hessian is an inline defaulted method returning None. Problem-level ObjectiveFunction (in OptimizationBuilder) uses array-based interface with no id(). Adapters convert between levels. -->

```rust
/// An objective function to minimize: f(x).
///
/// The objective reads parameter values from ParamStore, just like constraints.
/// It produces a scalar value and a gradient.
pub trait Objective: Send + Sync {
    /// Unique identifier for this objective (for registry and result lookup).
    fn id(&self) -> ObjectiveId;

    /// Human-readable name.
    fn name(&self) -> &str;

    /// Parameters this objective depends on.
    fn param_ids(&self) -> &[ParamId];

    /// Evaluate f(x) at the current parameter values.
    fn value(&self, store: &ParamStore) -> f64;

    /// Gradient: df/dp for each parameter. Returns (ParamId, partial_derivative).
    fn gradient(&self, store: &ParamStore) -> Vec<(ParamId, f64)>;

    /// Hessian of the objective (optional, for exact second-order methods).
    ///
    /// Returns sparse triplets: (ParamId_i, ParamId_j, d2f/dp_i dp_j).
    /// **Only the lower triangle (including diagonal) need be provided** -- the
    /// Hessian is symmetric, and entries above the diagonal are ignored by assemblers.
    ///
    /// Returns `None` if exact Hessians are not available; solvers fall back to
    /// quasi-Newton (L-BFGS) approximation.
    fn hessian_entries(&self, _store: &ParamStore) -> Option<Vec<(ParamId, ParamId, f64)>> {
        None
    }

    /// Sub-objectives for partial separability detection (optional).
    /// If f(x) = sum_k f_k(x_k), return the sub-objectives.
    fn sub_objectives(&self) -> Option<Vec<Box<dyn SubObjective>>> {
        None
    }
}

/// A sub-objective for partially separable functions.
pub trait SubObjective: Send + Sync {
    /// Parameters this sub-objective depends on.
    fn param_ids(&self) -> &[ParamId];
    /// Evaluate this sub-objective.
    fn value(&self, store: &ParamStore) -> f64;
    /// Gradient of this sub-objective.
    fn gradient(&self, store: &ParamStore) -> Vec<(ParamId, f64)>;
}
```

**Note on `ObjectiveHessian` supertrait pattern**: `00_synthesis.md` recommends defining
Hessian support as a separate supertrait `pub trait ObjectiveHessian: Objective` rather
than an inline defaulted method. This allows solvers to use Rust's type system to
statically guarantee Hessian availability, avoiding `Option` unwrapping. The inline
defaulted method above is acceptable for an initial implementation. For production SQP/IPM
solvers that require exact Hessians, the supertrait pattern from `00_synthesis.md` is
preferred.

**Note on `ConstraintHessian`**: SQP and IPM assemblers also need per-constraint Hessian
contributions to form the full Hessian of the Lagrangian. This is defined in
`01_mathematical_architecture.md` as an extension trait `ConstraintHessian: Constraint`
with method `fn residual_hessian(equation_row, store) -> Option<Vec<(ParamId, ParamId, f64)>>`.
See that document for the full definition.

---

## 8. Decomposition with Objectives

### 8.1 The Coupling Problem

An objective `f(x1, x2, ..., xn)` that depends on all variables defeats decomposition
into independent clusters. In constraint-only mode, two constraints on disjoint parameter
sets form independent clusters. With an objective, they become coupled through `f`.

### 8.2 Partial Separability Detection

When the objective implements `sub_objectives()`, the decomposer can detect block
structure:

```
Algorithm: OptDecompose

Input:  objective O with sub-objectives {f_k}, constraints {c_j}
Output: OptDecomposition

1. Build parameter dependency graph:
   - For each sub-objective f_k, record param_ids(f_k)
   - For each constraint c_j, record param_ids(c_j)

2. Union-find over parameters:
   - For each f_k: union all params in param_ids(f_k)
   - For each c_j: union all params in param_ids(c_j)

3. Group sub-objectives and constraints by parameter component.

4. Each component with at least one sub-objective becomes a SeparableBlock.
   Components with only constraints become constraint-only ClusterData.
```

### 8.3 Fallback: Monolithic Optimization

When `sub_objectives()` returns `None`, all objective parameters and all constraints
sharing those parameters form one large optimization cluster. This is the common case for
CAD problems where the objective is typically simple (e.g., "minimize total constraint
relaxation" or "minimize distance from reference configuration").

```
 Objective params: {p1, p2, p3, p4, p5}

 Constraint C1: {p1, p2}       --> absorbed into opt cluster
 Constraint C2: {p3, p6}       --> absorbed (shares p3 with objective)
 Constraint C3: {p7, p8}       --> independent constraint cluster
 Constraint C4: {p6, p9}       --> absorbed (shares p6 with C2, transitively)

 Result:
   Opt cluster:        {p1,p2,p3,p4,p5,p6,p9}, constraints {C1,C2,C4}
   Constraint cluster: {p7,p8}, constraints {C3}
```

---

## 9. Result Types

### 9.1 OptimizationResult

```rust
/// Result of solving an optimization problem.
#[derive(Clone, Debug)]
pub struct OptimizationResult {
    /// Overall status.
    pub status: OptimizationStatus,
    /// Optimal objective value.
    pub objective_value: f64,
    /// Constraint violation norm at solution.
    pub constraint_violation: f64,
    /// KKT residual norm at solution.
    pub kkt_residual: f64,
    /// Total outer iterations.
    pub iterations: usize,
    /// Wall-clock duration.
    pub duration: std::time::Duration,
    /// Per-cluster results for independent constraint clusters.
    pub constraint_cluster_results: Vec<ClusterResult>,
    /// Result for the optimization cluster.
    pub opt_cluster_result: Option<OptClusterResult>,
    /// Sensitivity information (if requested).
    pub sensitivity: Option<SensitivityInfo>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimizationStatus {
    /// Optimal solution found (KKT satisfied).
    Optimal,
    /// Feasible point found but objective not fully minimized.
    Feasible,
    /// Some clusters solved, optimization cluster did not converge.
    PartiallyOptimal,
    /// Problem appears infeasible.
    Infeasible,
    /// Solver did not converge.
    NotConverged,
}

/// Result for the optimization cluster.
#[derive(Clone, Debug)]
pub struct OptClusterResult {
    /// Solve status.
    pub status: OptSolveStatus,
    /// Optimal objective value.
    pub objective_value: f64,
    /// Constraint violation.
    pub constraint_violation: f64,
    /// KKT residual.
    pub kkt_residual: f64,
    /// Iterations.
    pub iterations: usize,
    /// Algorithm that was used.
    pub algorithm: AlgorithmChoice,
    /// Residual norm for the constraint portion.
    pub residual_norm: f64,
}
```

### 9.2 Sensitivity Information

```rust
/// Post-solve sensitivity analysis results.
#[derive(Clone, Debug)]
pub struct SensitivityInfo {
    /// Lagrange multipliers per constraint equation.
    /// multipliers[i] = (ConstraintId, equation_row, multiplier_value).
    pub multipliers: Vec<(ConstraintId, usize, f64)>,

    /// Sensitivity of the optimal objective to constraint perturbations.
    /// For equality constraint c_i(x) = 0, the sensitivity is:
    ///   d(f*)/d(b_i) = -lambda_i
    /// where b_i is the RHS perturbation of constraint i.
    pub objective_sensitivities: Vec<(ConstraintId, usize, f64)>,

    /// Reduced Hessian eigenvalues (if computed).
    /// Positive eigenvalues confirm a local minimum.
    pub reduced_hessian_eigenvalues: Option<Vec<f64>>,

    /// Active constraint set at the solution (for inequality constraints).
    pub active_constraints: Vec<ConstraintId>,
}
```

Multiplier interpretation for sensitivity analysis:

- `lambda_i > 0`: tightening constraint i increases the objective (binding constraint)
- `lambda_i = 0`: constraint i is not active at the solution (inequality constraint)
- `|lambda_i|` large: objective is highly sensitive to constraint i -- changing it would
  significantly change the optimal design

---

## 10. How Existing ConstraintSystem Continues to Work

### 10.1 Zero-Cost When No Objective

When no objective is set:

1. `MultiplierStore` is empty (zero heap allocation until first `set()`).
2. `opt_config` is `OptimizationConfig::default()` (never None).
3. `last_opt_result` is `None`.
4. `solve()` calls the existing pipeline exactly as before.
5. `optimize()` checks for no objective and returns early with an error/diagnostic.

### 10.2 Calling `solve()` with an Objective Set

`solve()` ignores the objective. It runs the standard constraint-solving pipeline. This
is by design: the user may want to first solve constraints to a feasible point, then call
`optimize()` to minimize the objective. This two-phase workflow is common in CAD.

### 10.3 SystemConfig vs OptimizationConfig

`SystemConfig` is not modified. It retains `lm_config` and `solver_config` for constraint
solving. `OptimizationConfig` is stored separately and used only by `optimize()`.

```rust
// Existing (unchanged)
pub struct SystemConfig {
    pub lm_config: LMConfig,
    pub solver_config: SolverConfig,
}

// New (separate)
pub struct OptimizationConfig { ... }
```

<!-- Decision: opt_config is stored as OptimizationConfig (not Option<OptimizationConfig>), initialized to Default::default(). This is consistent with how SystemConfig is handled in the existing codebase. -->

---

## 11. Pipeline Orchestration: `optimize()` Method

> **REVIEW NOTE (SolvePipeline extension):** The pseudocode below calls methods such as
> `self.pipeline.classify(...)`, `self.pipeline.decompose_opt(...)`,
> `self.pipeline.analyze_opt(...)`, `self.pipeline.init_multipliers(...)`,
> `self.pipeline.solve_opt_cluster(...)`, and `self.pipeline.post_process_opt(...)`.
> These do NOT exist on the current `SolvePipeline` struct in
> `crates/solverang/src/pipeline/mod.rs`. The extended struct must add new fields:
>
> ```rust
> pub struct SolvePipeline {
>     // --- existing fields (unchanged) ---
>     decompose: Box<dyn Decompose>,
>     analyze: Box<dyn Analyze>,
>     reduce: Box<dyn Reduce>,
>     solve: Box<dyn SolveCluster>,
>     post_process: Box<dyn PostProcess>,
>     cached_clusters: Vec<ClusterData>,
>     clusters_valid: bool,
>
>     // --- new optimization-only fields (None = use default) ---
>     classify: Option<Box<dyn Classify>>,
>     decompose_opt: Option<Box<dyn DecomposeOpt>>,
>     analyze_opt: Option<Box<dyn AnalyzeOpt>>,
>     init_multipliers: Option<Box<dyn InitMultipliers>>,
>     solve_opt: Option<Box<dyn SolveOptCluster>>,
>     post_process_opt: Option<Box<dyn PostProcessOpt>>,
> }
> ```
>
> `PipelineBuilder` must also gain setters for each of these six new phases. All six
> default to `None`; `optimize()` instantiates defaults lazily (e.g.,
> `DefaultClassify`, `DefaultDecomposeOpt`, `DefaultOptAnalyze`, `DefaultInitMultipliers`,
> `DefaultSolveOpt`, `DefaultPostProcessOpt`).

```rust
impl ConstraintSystem {
    pub fn optimize(&mut self) -> OptimizationResult {
        let start = std::time::Instant::now();
        let opt_config = self.opt_config.clone();

        let objective = match &self.objective {
            Some(obj) => obj.as_ref(),
            None => return OptimizationResult::no_objective(),
        };

        // Phase 0: Classify
        // NOTE: This phase can be simplified to `self.objective.is_none()` check above.
        // Keeping it here for extensibility if custom classifiers are needed.
        let problem_class = self.pipeline.classify(
            Some(objective), &self.constraints, &self.params
        );

        // Phase 1: Decompose with objective awareness
        let decomposition = self.pipeline.decompose_opt(
            objective, &self.constraints, &self.entities, &self.params
        );

        // Phase 1b: Solve independent constraint-only clusters with existing pipeline.
        // These clusters are completely decoupled from the objective and are solved
        // exactly as in the regular solve() path (Analyze -> Reduce -> Solve -> PostProcess).
        let mut constraint_results = Vec::new();
        for cluster in &decomposition.constraint_clusters {
            let analysis = self.pipeline.analyze(
                cluster, &self.constraints, &self.entities, &self.params
            );
            let reduced = self.pipeline.reduce(
                cluster, &self.constraints, &mut self.params
            );
            let warm = self.solution_cache.get(&cluster.id)
                .map(|c| c.solution.as_slice());
            let solution = self.pipeline.solve_cluster(
                &reduced, &analysis, &self.constraints, &self.params,
                warm, &self.config
            );
            for &(pid, val) in &solution.param_values { self.params.set(pid, val); }
            if let (Some(m), Some(v)) = (&solution.mapping, &solution.numerical_solution) {
                self.params.write_free_values(v, m);
            }
            for &(pid, _) in &reduced.eliminated_params { self.params.unfix(pid); }
            let result = self.pipeline.post_process(
                &solution, &analysis, cluster
            );
            constraint_results.push(result);
        }

        // Phase 2: Analyze the optimization cluster
        let opt_analysis = self.pipeline.analyze_opt(
            &decomposition.opt_cluster, objective, &self.constraints,
            &self.entities, &self.params
        );

        // Phase 3: Reduce the optimization cluster's constraints.
        // OptClusterData::as_cluster_data() converts to ClusterData (lossy: drops
        // separable_blocks). See Section 4.3 for the conversion definition.
        let opt_reduced = self.pipeline.reduce(
            &decomposition.opt_cluster.as_cluster_data(),
            &self.constraints, &mut self.params
        );

        // Phase 3.5: Initialize multipliers.
        // WarmStart reuses self.multiplier_store from the previous optimize() call.
        // For the first call (or after clear_objective()), the store is empty and
        // WarmStart falls back to Zero initialization.
        let multiplier_state = self.pipeline.init_multipliers(
            &decomposition.opt_cluster, &opt_reduced,
            &self.constraints, &self.params,
            &mut self.multiplier_store, &opt_config
        );

        // Phase 4: Solve the optimization cluster.
        // Warm start: use last_opt_result's raw solution if available.
        let warm_start: Option<Vec<f64>> = self.last_opt_result
            .as_ref()
            .and_then(|_r| {
                // Extract param values in solver-column order from the previous solution.
                // This is a best-effort warm start; the mapping may differ after
                // structural changes, in which case the solver restarts from the current
                // ParamStore values.
                None // TODO: implement warm-start extraction from last_opt_result
            });
        let opt_solution = self.pipeline.solve_opt_cluster(
            &decomposition.opt_cluster, &opt_reduced, &opt_analysis,
            objective, &self.constraints, &self.params,
            &multiplier_state, warm_start.as_deref(), &opt_config
        );

        // Write solution back to ParamStore
        for &(pid, val) in &opt_solution.param_values {
            self.params.set(pid, val);
        }

        // Write multipliers back to MultiplierStore.
        // SAFETY: constraint indices in multiplier_mapping come from the fresh
        // decomposition performed above, so they are guaranteed to reference live
        // (non-None) constraint slots.
        let multiplier_ordering: Vec<(ConstraintId, usize)> = opt_solution
            .multiplier_mapping
            .iter()
            .filter_map(|&(ci, row)| {
                // Use filter_map instead of unwrap() to guard against tombstoned slots.
                self.constraints.get(ci)?.as_ref().map(|c| (c.id(), row))
            })
            .collect();
        self.multiplier_store.write_ordered(
            &opt_solution.multipliers, &multiplier_ordering
        );

        // Phase 5: Post-process
        let opt_result = self.pipeline.post_process_opt(
            &opt_solution, &opt_analysis, &decomposition.opt_cluster
        );

        // Sensitivity analysis (if configured).
        // compute_sensitivity() assembles SensitivityInfo from the multiplier store,
        // the reduced cluster, and the objective gradient. Its full interface is TBD;
        // it computes d(f*)/d(b_i) = -lambda_i for each constraint perturbation b_i
        // and optionally the reduced Hessian eigenvalues.
        let sensitivity = if opt_config.compute_sensitivity {
            Some(compute_sensitivity(
                objective, &self.constraints, &self.params,
                &self.multiplier_store, &opt_reduced
            ))
        } else {
            None
        };

        // Un-fix eliminated params (mirror of what solve() does for ReducedCluster).
        for &(pid, _) in &opt_reduced.eliminated_params {
            self.params.unfix(pid);
        }

        // determine_status() maps OptClusterResult + Vec<ClusterResult> to
        // OptimizationStatus. Logic: if opt_result.status is Optimal and all
        // constraint_results converged -> Optimal; if opt is Optimal but some
        // constraint clusters failed -> PartiallyOptimal; etc.
        // compute_kkt_residual() computes ||grad_x L||_inf where
        //   L = f + lambda^T * c(x).
        // For inequality constraints, KKT also checks complementarity mu_j * h_j = 0.
        let result = OptimizationResult {
            status: determine_status(&opt_result, &constraint_results),
            objective_value: opt_solution.objective_value,
            constraint_violation: opt_solution.constraint_violation,
            kkt_residual: compute_kkt_residual(
                objective, &self.constraints, &self.params, &self.multiplier_store
            ),
            iterations: opt_solution.iterations,
            duration: start.elapsed(),
            constraint_cluster_results: constraint_results,
            opt_cluster_result: Some(opt_result),
            sensitivity,
        };

        self.last_opt_result = Some(result.clone());
        // Clear change tracker after optimize(), same as after solve().
        // NOTE: change_tracker is read for dirty-cluster detection in Phase 1b above.
        // It must be cleared AFTER Phase 1b, which this ordering guarantees.
        self.change_tracker.clear();
        result
    }
}
```

---

## 12. Algorithm Dispatch: Detailed Flow

### 12.1 Constraint Solving (Existing, Unchanged)

```
Problem arrives at SolveCluster::solve_cluster()
    |
    +--> Try closed-form patterns (from analysis)
    |       |
    |       +--> Success: remove handled constraints
    |       +--> Failure: fall through
    |
    +--> Build ReducedSubProblem for remaining constraints
    |
    +--> LMSolver::solve(sub_problem, x0)
    |
    +--> Return ClusterSolution
```

### 12.2 Optimization Solving (New)

```
OptCluster arrives at SolveOptCluster::solve_opt_cluster()
    |
    +--> Read analysis.recommended_algorithm
    |
    +--[QuasiNewton]--> BFGSSolver
    |                     - Build OptProblem wrapping Objective + ParamStore
    |                     - Run L-BFGS iterations
    |                     - Return OptClusterSolution (multipliers empty)
    |
    +--[SQP]--> SQPSolver
    |             - Build KKT system from Objective gradient + Constraint Jacobians
    |             - Initialize Hessian approximation (BFGS or exact)
    |             - Outer loop:
    |               1. Evaluate gradient, Jacobian, residuals
    |               2. Solve QP subproblem for step (dx, dlambda)
    |               3. Line search with l1 merit function
    |               4. Update x, lambda
    |               5. Check KKT conditions
    |             - Return OptClusterSolution (with multipliers)
    |
    +--[IPM]--> IPMSolver
    |             - Add slack variables for inequality constraints
    |             - Build augmented KKT system with barrier terms
    |             - Outer loop (barrier reduction):
    |               1. Solve augmented KKT for (dx, dlambda, ds)
    |               2. Step-size selection (fraction-to-boundary)
    |               3. Update all variables
    |               4. Reduce barrier parameter
    |             - Return OptClusterSolution (with multipliers)
    |
    +--[AugLag]--> AugLagSolver
                    - Outer loop (multiplier updates):
                      1. Form augmented Lagrangian: L_A = f + lambda^T c + (mu/2)||c||^2
                      2. Minimize L_A using LM/BFGS (inner solve)
                      3. Update multipliers: lambda += mu * c(x)
                      4. Possibly increase mu
                      5. Check feasibility + optimality
                    - Return OptClusterSolution (with multipliers)
```

### 12.3 Interaction with Existing Solvers

The existing `LMSolver`, `Solver` (NR), `AutoSolver`, and `RobustSolver` remain
unchanged. They continue to solve the `Problem` trait (residuals + Jacobian).

The new optimization solvers work with a different abstraction:

```rust
/// An optimization problem for optimization solvers.
/// Unlike Problem (which is residuals-based), this exposes objective + constraints.
pub trait OptProblem: Send + Sync {
    fn variable_count(&self) -> usize;
    fn objective_value(&self, x: &[f64]) -> f64;
    fn objective_gradient(&self, x: &[f64]) -> Vec<f64>;
    fn objective_hessian(&self, x: &[f64]) -> Option<Vec<(usize, usize, f64)>>;
    fn constraint_count(&self) -> usize;
    fn constraint_residuals(&self, x: &[f64]) -> Vec<f64>;
    fn constraint_jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)>;
}
```

A bridge struct `ReducedOptProblem` adapts `Objective` + `[Constraint]` + `ParamStore` +
`SolverMapping` into the `OptProblem` interface, handling the ID-to-index mapping.

---

## 13. PipelineBuilder Extension

```rust
pub struct PipelineBuilder {
    // Existing
    decompose: Option<Box<dyn Decompose>>,
    analyze: Option<Box<dyn Analyze>>,
    reduce: Option<Box<dyn Reduce>>,
    solve: Option<Box<dyn SolveCluster>>,
    post_process: Option<Box<dyn PostProcess>>,

    // New: optimization phases
    classify: Option<Box<dyn Classify>>,
    decompose_opt: Option<Box<dyn DecomposeOpt>>,
    analyze_opt: Option<Box<dyn AnalyzeOpt>>,
    init_multipliers: Option<Box<dyn InitMultipliers>>,
    solve_opt: Option<Box<dyn SolveOptCluster>>,
    post_process_opt: Option<Box<dyn PostProcessOpt>>,
}
```

All new fields default to `None` and use default implementations when not set. The
builder gains corresponding setter methods. Existing builder usage is unchanged.

---

## 14. Data Flow Diagram

### 14.1 Constraint-Only Mode (Existing)

```
ConstraintSystem::solve()
    |
    v
SolvePipeline::run()
    |
    +--[Decompose]--> Vec<ClusterData>
    |
    +--for each cluster:
    |    |
    |    +--[Analyze]--> ClusterAnalysis
    |    +--[Reduce]---> ReducedCluster
    |    +--[Solve]----> ClusterSolution
    |    +--[PostProc]-> ClusterResult
    |
    +--[Aggregate]---> SystemResult
```

### 14.2 Optimization Mode (New)

```
ConstraintSystem::optimize()
    |
    v
[Classify] ------> ProblemClass
    |
    v
[DecomposeOpt] --> OptDecomposition
    |                  |
    |                  +-- opt_cluster: OptClusterData
    |                  +-- constraint_clusters: Vec<ClusterData>
    |
    +--for each constraint_cluster (existing pipeline):
    |    |
    |    +--[Analyze]-----> ClusterAnalysis
    |    +--[Reduce]------> ReducedCluster
    |    +--[Solve]-------> ClusterSolution ------> write to ParamStore
    |    +--[PostProcess]-> ClusterResult
    |
    +--for opt_cluster:
    |    |
    |    +--[AnalyzeOpt]--------> OptClusterAnalysis (includes AlgorithmChoice)
    |    +--[Reduce]------------> ReducedCluster (constraint reduction only)
    |    +--[InitMultipliers]---> MultiplierState
    |    +--[SolveOptCluster]---> OptClusterSolution --> write to ParamStore
    |    |                                           --> write to MultiplierStore
    |    +--[PostProcessOpt]----> OptClusterResult
    |
    +--[Sensitivity]? ----------> SensitivityInfo
    |
    +--[Aggregate] -------------> OptimizationResult
```

---

## 15. Multiplier Initialization Strategies (Detailed)

### 15.1 Zero Initialization

```
lambda_0 = 0
```

Simple, always works. Suitable when the starting point is far from optimal and the
multiplier estimate would be inaccurate anyway.

### 15.2 Least-Squares Initialization

At the current point x_0, solve the overdetermined system for lambda:

```
J(x_0)^T * lambda = -grad_f(x_0)
```

where J is the constraint Jacobian. The least-squares solution is:

```
lambda_0 = -(J J^T)^{-1} J grad_f(x_0)
```

This gives the best linear estimate of multipliers and is the default for SQP. It
requires one Jacobian evaluation and one small linear solve.

### 15.3 Warm Start

Reuse the multiplier values from `MultiplierStore` (from a previous `optimize()` call).
Check that the constraint IDs still match; for any new/removed constraints, fall back to
zero or least-squares for those specific multipliers.

### 15.4 Penalty Initialization

For augmented Lagrangian, set:

```
lambda_0 = mu * c(x_0)
```

where mu is the initial penalty parameter and c(x_0) is the constraint residual vector.

---

## 16. Post-Solve Multiplier Exposure

After `optimize()` completes:

```rust
// Query individual multipliers
let lambda = system.multiplier(constraint_id);
// Returns Option<&[f64]> -- one value per equation row in the constraint

// Query all multipliers
let store = system.multipliers();
for (id, &value) in store.iter() {
    println!("Constraint {:?} row {}: lambda = {}",
             id.constraint_id, id.equation_row, value);
}

// Sensitivity: how much does optimal f change if constraint i is relaxed?
if let Some(ref sensitivity) = result.sensitivity {
    for &(cid, row, df_db) in &sensitivity.objective_sensitivities {
        // df_db = -lambda_i (sensitivity of optimal objective to RHS perturbation)
        println!("Relaxing constraint {:?} row {} changes f* by {:.4} per unit",
                 cid, row, df_db);
    }
}
```

---

## 17. Constraint Trait Extension for Inequality Support

To support inequality constraints (`c(x) >= 0`) in optimization, the `Constraint` trait
gains one default method:

```rust
pub trait Constraint: Send + Sync {
    // ... existing methods unchanged ...

    /// Whether this is an inequality constraint (c(x) >= 0).
    ///
    /// Default: false (equality constraint c(x) = 0).
    /// When true, the optimization solver treats residuals as inequality
    /// constraints. The pure constraint solver (solve()) ignores this flag
    /// and treats all constraints as equalities.
    fn is_inequality(&self) -> bool {
        false
    }
}
```

This is a backward-compatible addition (default returns `false`). The existing
`solve()` path ignores it entirely.

---

## 18. Implementation Phases

### Phase 1: Foundation (no new algorithms)
1. Add `Objective` trait
2. Add `MultiplierStore` and `MultiplierId`
3. Add `OptimizationConfig` and sub-configs
4. Add `OptimizationResult` and related types
5. Add `ProblemClass` and `Classify` trait
6. Extend `ConstraintSystem` with `objective`, `opt_config`, `multiplier_store` fields
7. Implement `optimize()` that delegates to existing LM solver (penalty method as
   baseline -- minimize `f(x) + mu/2 * ||c(x)||^2`)

### Phase 2: SQP
1. Implement `SQPSolver` with BFGS Hessian approximation
2. Implement `ReducedOptProblem` bridge
3. Implement `SolveOptCluster` trait and `DefaultSolveOpt`
4. Implement `MultiplierInit` (zero + least-squares strategies)
5. Implement `DecomposeOpt` (monolithic strategy)
6. Wire into `optimize()` pipeline

### Phase 3: Algorithm Selection and Analysis
1. Implement `OptClusterAnalysis` and `AnalyzeOpt`
2. Implement automatic algorithm selection
3. Implement `PostProcessOpt` with sensitivity computation
4. Implement `is_inequality()` on Constraint and inequality handling in SQP

### Phase 4: Advanced Solvers
1. Implement `IPMSolver`
2. Implement `AugLagSolver`
3. Implement partial separability detection in `DecomposeOpt`
4. Implement warm-start multiplier strategy

### Phase 5: Polish
1. Incremental re-optimization (reuse multipliers, decomposition caching)
2. `PipelineBuilder` extensions
3. Comprehensive testing with CAD optimization scenarios

---

## 19. Summary of New Types

| Type | Module | Purpose |
|------|--------|---------|
| `Objective` | `objective/mod.rs` | Trait: scalar objective function |
| `SubObjective` | `objective/mod.rs` | Trait: sub-function for partial separability |
| `MultiplierId` | `multiplier/store.rs` | ID for a Lagrange multiplier |
| `MultiplierStore` | `multiplier/store.rs` | Storage for multiplier values |
| `MultiplierMapping` | `multiplier/store.rs` | MultiplierId <-> solver position |
| `MultiplierState` | `pipeline/types.rs` | Snapshot entering solve phase |
| `ProblemClass` | `pipeline/classify.rs` | Enum: problem type classification |
| `Classify` | `pipeline/traits.rs` | Trait: phase 0 |
| `DecomposeOpt` | `pipeline/traits.rs` | Trait: optimization-aware decomposition |
| `OptDecomposition` | `pipeline/types.rs` | Decompose output for optimization |
| `OptClusterData` | `pipeline/types.rs` | An optimization cluster |
| `SeparableBlock` | `pipeline/types.rs` | A block in partial separability |
| `OptClusterAnalysis` | `pipeline/types.rs` | Analysis output for opt cluster |
| `ConvexityEstimate` | `pipeline/types.rs` | Enum: convexity classification |
| `AlgorithmChoice` | `pipeline/types.rs` | Enum: which algorithm to use |
| `InitMultipliers` | `pipeline/traits.rs` | Trait: phase 3.5 |
| `SolveOptCluster` | `pipeline/traits.rs` | Trait: phase 4 for optimization |
| `OptClusterSolution` | `pipeline/types.rs` | Solve output for opt cluster |
| `OptSolveStatus` | `pipeline/types.rs` | Enum: optimization solve status |
| `PostProcessOpt` | `pipeline/traits.rs` | Trait: phase 5 for optimization |
| `OptClusterResult` | `system.rs` | Result for an opt cluster |
| `OptimizationResult` | `system.rs` | Top-level optimization result |
| `OptimizationStatus` | `system.rs` | Enum: overall optimization status |
| `SensitivityInfo` | `system.rs` | Post-solve sensitivity data |
| `OptimizationConfig` | `system.rs` | Top-level opt config |
| `AlgorithmSelection` | `system.rs` | Enum: algorithm forcing |
| `MultiplierInitStrategy` | `system.rs` | Enum: how to initialize multipliers |
| `SQPConfig` | `solver/sqp_config.rs` | SQP solver configuration |
| `IPMConfig` | `solver/ipm_config.rs` | IPM solver configuration |
| `BFGSConfig` | `solver/bfgs_config.rs` | BFGS solver configuration |
| `AugLagConfig` | `solver/auglag_config.rs` | Aug. Lagrangian configuration |
| `HessianStrategy` | `solver/sqp_config.rs` | Enum: Hessian approximation |
| `LineSearchConfig` | `solver/config.rs` | Line search parameters |
| `QPSolverChoice` | `solver/sqp_config.rs` | Enum: QP sub-solver |
| `LinearSolverChoice` | `solver/ipm_config.rs` | Enum: KKT linear solver |
| `OptProblem` | `problem.rs` | Trait: optimization problem interface |
| `ReducedOptProblem` | `solve/reduced_opt.rs` | Bridge: Objective + Constraints -> OptProblem |
