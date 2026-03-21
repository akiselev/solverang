# Mathematical Architecture for Constrained Optimization in Solverang

## 1. Mathematical Foundation

### 1.1 Problem Statement

Solverang currently solves systems of the form:

    Find x such that F(x) = 0

The optimization extension generalizes this to:

    min  f(x)
    s.t. g_i(x) = 0,   i = 1..p    (equality constraints)
         h_j(x) <= 0,  j = 1..q    (inequality constraints)

where x in R^n, f: R^n -> R, g: R^n -> R^p, h: R^n -> R^q.

### 1.2 Why Optimization is a Strict Generalization of Root-Finding

Every root-finding problem F(x) = 0 is equivalent to the unconstrained optimization problem
min ||F(x)||^2. Conversely, every constrained optimization problem can be converted to a
structured root-finding problem via the KKT conditions. This bidirectional relationship is the
architectural foundation: the existing solver infrastructure handles the inner loop of
optimization algorithms, while a new outer layer manages the objective, multipliers, and
algorithmic strategy.

Note: the KKT conditions are *necessary* (not sufficient) conditions for optimality, and
they characterize the solution only when a constraint qualification holds (e.g., LICQ:
linear independence of active constraint gradients). In degenerate geometric configurations
— coincident points, collinear constraints, redundant constraints — constraint qualifications
may be violated and the KKT system may have no solution or infinitely many solutions. The
existing solver infrastructure already handles these cases (via DOF analysis and redundancy
detection); the optimization extension should integrate with those diagnostics rather than
assuming constraint qualifications always hold.

### 1.3 The Lagrangian

The Lagrangian function for the constrained problem is:

    L(x, lambda, mu) = f(x) + sum_i lambda_i * g_i(x) + sum_j mu_j * h_j(x)

where lambda in R^p are equality multipliers and mu in R^q are inequality multipliers.

The first-order KKT conditions are:

    grad_x L = grad f + sum_i lambda_i * grad g_i + sum_j mu_j * grad h_j = 0
    g_i(x) = 0                    for all i
    h_j(x) <= 0                   for all j
    mu_j >= 0                     for all j
    mu_j * h_j(x) = 0             for all j (complementarity)

### 1.4 Derivative Hierarchy

The fundamental escalation in derivative requirements:

| Level | Operation | Derivative Info | Matrix Size |
|-------|-----------|-----------------|-------------|
| 0 | Function evaluation | f(x), g(x), h(x) | Scalar / vectors |
| 1 | Gradient / Jacobian | grad f, Jg, Jh | n-vector / p x n, q x n |
| 2 | Hessian of Lagrangian | H_L = H_f + sum lambda_i H_{g_i} + sum mu_j H_{h_j} | n x n symmetric |

Root-finding requires Level 1 (Jacobians). Optimization requires Level 2 (Hessians) for
second-order methods, or can operate with Level 1 only for quasi-Newton approaches.

---

## 2. Trait Hierarchy

### 2.1 Design Principles

1. **Non-breaking extension**: All existing `Constraint` and `Problem` implementations
   continue to work without modification.
2. **Gradual opt-in**: Hessian information is optional; the system falls back to
   quasi-Newton approximations (BFGS/L-BFGS) when exact Hessians are unavailable.
3. **Composability**: The Lagrangian is assembled at runtime from independent per-function
   derivative contributions, exactly as the current Jacobian is assembled from
   per-constraint contributions.
4. **ParamId-centric**: All new traits speak in terms of `ParamId`, not column indices.
   The `SolverMapping` handles the translation, just as it does today.

### 2.2 New ID Types

```rust
/// Semantic address for a Lagrange multiplier (dual variable).
///
/// Addresses a multiplier by the constraint it belongs to and the equation row
/// within that constraint. No generational bookkeeping needed — multipliers
/// are recomputed each optimization solve.
///
/// Each equality constraint equation gets one multiplier (lambda_i).
/// Each inequality constraint equation gets one multiplier (mu_j).
/// Multipliers are managed by the MultiplierStore, not the ParamStore,
/// because they are dual variables with different semantics.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct MultiplierId {
    /// Which constraint this multiplier is associated with.
    pub constraint_id: ConstraintId,
    /// Which equation row within that constraint (0-based).
    pub equation_row: usize,
}

/// Generational index for an objective function in the system.
///
/// Most problems have a single objective, but multi-objective formulations
/// (scalarized via weighted sum) may register multiple objectives.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObjectiveId {
    pub(crate) index: u32,
    pub(crate) generation: Generation,
}
```

### 2.3 Core Optimization Traits

#### 2.3.1 Objective

```rust
/// A scalar objective function f(x) to be minimized.
///
/// This is the optimization analogue of `Constraint`. Where constraints produce
/// residual vectors that should be zero, an objective produces a scalar value
/// that should be minimized.
///
/// # Derivative Levels
///
/// - `value()` (required): Evaluate f(x). Always needed.
/// - `gradient()` (required): Compute grad f(x). Needed by all algorithms.
/// - `hessian_entries()` (optional): Compute H_f(x). Needed for exact
///   second-order methods (SQP, IPM). When absent, the solver uses L-BFGS
///   or Gauss-Newton approximations.
///
/// # ParamId Convention
///
/// Like `Constraint`, the objective returns gradient entries as
/// `(ParamId, value)` pairs. The `SolverMapping` translates to column
/// indices. This means objectives compose naturally with the existing
/// parameter infrastructure.
pub trait Objective: Send + Sync {
    /// Unique identifier for this objective.
    fn id(&self) -> ObjectiveId;

    /// Human-readable name for diagnostics.
    fn name(&self) -> &str;

    /// Which parameters this objective depends on (for graph building).
    fn param_ids(&self) -> &[ParamId];

    /// Evaluate the objective function: f(x).
    fn value(&self, store: &ParamStore) -> f64;

    /// Sparse gradient: (param_id, df/dp) pairs.
    ///
    /// Only non-zero entries need to be returned.
    fn gradient(&self, store: &ParamStore) -> Vec<(ParamId, f64)>;

    /// Sparse Hessian of f: (param_id_row, param_id_col, d^2f/dp_r dp_c) triplets.
    ///
    /// Returns only the lower triangle (including diagonal) since the Hessian
    /// is symmetric. Entries above the diagonal are ignored.
    ///
    /// Returns `None` if exact Hessians are not available, in which case the
    /// solver will use quasi-Newton (L-BFGS) approximations.
    fn hessian_entries(&self, store: &ParamStore) -> Option<Vec<(ParamId, ParamId, f64)>> {
        let _ = store;
        None
    }
}
```

#### 2.3.2 Extending Constraint for Optimization

The existing `Constraint` trait already provides exactly what is needed for equality
constraints in an optimization context: `residuals()` produces g(x) and `jacobian()`
produces the Jacobian of g. What is missing is the per-constraint Hessian, which is
needed to assemble the Hessian of the Lagrangian.

Rather than modifying the existing `Constraint` trait (which would break all existing
implementations), we define an extension trait:

```rust
/// Extension trait for constraints that can provide second-order derivative
/// information (Hessian of each residual function).
///
/// This is optional. Constraints that do not implement this trait will have
/// their Hessian contribution approximated via Gauss-Newton (J^T J) when
/// used in optimization, or via finite differences.
///
/// # Hessian Structure
///
/// For a constraint with k residuals r_0(x), ..., r_{k-1}(x), the Hessian
/// contribution to the Lagrangian for residual `equation_row` is:
///
///   lambda * nabla^2 r_{equation_row}(x)
///
/// where lambda is the multiplier for that equation row.
///
/// The `hessian_entries` method returns the Hessian of a single residual
/// (without the multiplier weight). The Lagrangian assembler applies the
/// multiplier weights at runtime.
pub trait ConstraintHessian: Constraint {
    /// Sparse Hessian of a single residual equation: (param_row, param_col, value).
    ///
    /// Returns d^2 r_{equation_row} / (dp_i dp_j) as lower-triangular triplets.
    ///
    /// `equation_row` is in 0..self.equation_count().
    ///
    /// Returns `None` for this equation if exact Hessian is unavailable.
    fn residual_hessian(
        &self,
        equation_row: usize,
        store: &ParamStore,
    ) -> Option<Vec<(ParamId, ParamId, f64)>>;

    /// Whether all residuals in this constraint have exact Hessians available.
    fn has_exact_hessians(&self) -> bool {
        true
    }
}
```

<!-- Decision: ConstraintHessian method is residual_hessian(equation_row: usize, store: &ParamStore) — index first, store second. This matches the LagrangianAssembler call pattern in section 3.2. -->


#### 2.3.3 Inequality Constraint (ParamStore-Based)

<!-- Decision: The system-level inequality trait is named InequalityFn (not SystemInequalityConstraint). This avoids collision with the existing array-based InequalityConstraint in constraints/inequality.rs. -->

The existing `InequalityConstraint` in `constraints/inequality.rs` uses raw `&[f64]`
variable vectors. For the optimization extension, we need a `ParamStore`-based version
that integrates with the generational ID system:

```rust
/// An inequality constraint h(x) <= 0 in the constraint system.
///
/// Convention: h(x) <= 0 is the standard form. This is the opposite sign
/// convention from the existing `InequalityConstraint` trait (which uses
/// g(x) >= 0). We adopt h(x) <= 0 because it matches the standard
/// optimization literature (Nocedal & Wright, Boyd & Vandenberghe) and
/// the Lagrangian sign convention L = f + lambda^T g + mu^T h.
///
/// # Relationship to Existing Trait
///
/// - Existing `InequalityConstraint`: `evaluate(x: &[f64]) -> f64` (g(x) >= 0)
/// - This trait: `values(store: &ParamStore) -> Vec<f64>` (h(x) <= 0)
///
/// To convert: h(x) = -g(x).
///
/// # Multiple Inequalities
///
/// A single `InequalityFn` can produce multiple scalar inequalities
/// (analogous to how a `Constraint` can produce multiple residuals).
pub trait InequalityFn: Send + Sync {
    /// Unique identifier.
    fn id(&self) -> ConstraintId;

    /// Human-readable name for diagnostics.
    fn name(&self) -> &str;

    /// Which entities this inequality references.
    fn entity_ids(&self) -> &[EntityId];

    /// Which parameters this inequality depends on.
    fn param_ids(&self) -> &[ParamId];

    /// Number of scalar inequalities.
    fn inequality_count(&self) -> usize;

    /// Evaluate h(x). Each element should be <= 0 when satisfied.
    fn values(&self, store: &ParamStore) -> Vec<f64>;

    /// Sparse Jacobian: (inequality_row, param_id, dh/dp) triplets.
    fn jacobian(&self, store: &ParamStore) -> Vec<(usize, ParamId, f64)>;

    /// Sparse Hessian of a single inequality: d^2 h_row / (dp_i dp_j).
    ///
    /// Returns lower-triangular triplets. Returns None if unavailable.
    fn hessian_entries(
        &self,
        inequality_row: usize,
        store: &ParamStore,
    ) -> Option<Vec<(ParamId, ParamId, f64)>> {
        let _ = (inequality_row, store);
        None
    }
}
```

#### 2.3.4 OptimizationProblem

<!-- Decision: Two-level design. System-level: Objective + InequalityFn (ParamStore-based, has id()). Problem-level: ObjectiveFunction + InequalityConstraint (array-based, no id()). OptimizationProblem below is the problem-level assembled struct. Adapters convert between levels. See the two-level table in 00_synthesis.md. -->

```rust
/// A complete constrained optimization problem (problem-level, array-indexed interface).
///
/// This trait is for use with standalone optimization solvers that operate on
/// raw `&[f64]` variable vectors. It is the optimization-layer analogue of the
/// `Problem` trait for root-finding.
///
/// For integration with `ConstraintSystem` and the entity/ParamId graph, implement
/// `Objective` and `InequalityFn` instead. This trait is useful for:
/// - Testing optimization solvers in isolation (Hock-Schittkowski benchmark problems)
/// - Wrapping external problem specifications
/// - Problems that pre-date the ParamStore infrastructure
///
/// # Assembly
///
/// The system holds:
/// - One or more `Objective`s (weighted sum if multiple)
/// - Zero or more `Constraint`s (equality: g(x) = 0)
/// - Zero or more `InequalityFn`s (inequality: h(x) <= 0)
///
/// The Lagrangian is:
///
///   L(x, lambda, mu) = f(x) + lambda^T g(x) + mu^T h(x)
///
/// where f(x) is the (possibly weighted) sum of all objectives.
///
/// # Relationship to ConstraintSystem
///
/// `OptimizationSystem` extends `ConstraintSystem` with objective and
/// inequality management. When no objective is present, the system
/// degenerates to feasibility (pure constraint satisfaction), which
/// the existing solver handles unchanged.
pub trait OptimizationProblem: Send + Sync {
    /// Number of primal variables (n).
    fn variable_count(&self) -> usize;

    /// Number of equality constraints (p).
    fn equality_count(&self) -> usize;

    /// Number of inequality constraints (q).
    fn inequality_count(&self) -> usize;

    /// Evaluate the objective function f(x).
    fn objective_value(&self, x: &[f64]) -> f64;

    /// Gradient of the objective: grad f(x) as dense n-vector.
    fn objective_gradient(&self, x: &[f64]) -> Vec<f64>;

    /// Evaluate equality constraints: g(x), p-vector (should be zero at solution).
    fn equality_values(&self, x: &[f64]) -> Vec<f64>;

    /// Jacobian of equalities: sparse (row, col, val) triplets for the p x n matrix.
    fn equality_jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)>;

    /// Evaluate inequality constraints: h(x), q-vector (should be <= 0 at solution).
    fn inequality_values(&self, x: &[f64]) -> Vec<f64>;

    /// Jacobian of inequalities: sparse (row, col, val) triplets for the q x n matrix.
    fn inequality_jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)>;

    /// Hessian of the Lagrangian: H_L(x, lambda, mu).
    ///
    /// Returns sparse lower-triangular (row, col, val) triplets for the n x n matrix:
    ///
    ///   H_L = H_f + sum_i lambda_i * H_{g_i} + sum_j mu_j * H_{h_j}
    ///
    /// If `None`, the solver uses quasi-Newton (L-BFGS or SR1) approximations.
    fn lagrangian_hessian(
        &self,
        x: &[f64],
        lambda: &[f64],
        mu: &[f64],
    ) -> Option<Vec<(usize, usize, f64)>> {
        let _ = (x, lambda, mu);
        None
    }

    /// Hessian-vector product: H_L(x, lambda, mu) * v.
    ///
    /// More efficient than forming the full Hessian when only products are needed
    /// (e.g., for CG-based inner solvers in SQP). Default implementation forms
    /// the sparse Hessian and multiplies. Override for matrix-free implementations.
    fn lagrangian_hessian_vec(
        &self,
        x: &[f64],
        lambda: &[f64],
        mu: &[f64],
        v: &[f64],
    ) -> Option<Vec<f64>> {
        let hessian = self.lagrangian_hessian(x, lambda, mu)?;
        let n = self.variable_count();
        let mut result = vec![0.0; n];
        for &(row, col, val) in &hessian {
            if row < n && col < n {
                result[row] += val * v[col];
                if row != col {
                    result[col] += val * v[row]; // symmetric
                }
            }
        }
        Some(result)
    }

    /// Initial point for the primal variables.
    fn initial_point(&self) -> Vec<f64>;

    /// Initial estimates for equality multipliers (optional).
    fn initial_lambda(&self) -> Option<Vec<f64>> {
        None
    }

    /// Initial estimates for inequality multipliers (optional).
    fn initial_mu(&self) -> Option<Vec<f64>> {
        None
    }
}
```

---

## 3. Lagrangian Assembly at Runtime

### 3.1 Assembly Architecture

The Lagrangian is never materialized as a single function object. Instead, the system
assembles its derivatives on demand by iterating over registered components and combining
their contributions. This mirrors how the current system assembles the global Jacobian from
per-constraint Jacobian entries.

```
Current (root-finding):
  For each Constraint c:
    entries += c.jacobian(store)      // (row, ParamId, val) triplets
  Global Jacobian = SolverMapping.assemble(entries)

Extended (optimization):
  Gradient:
    grad = objective.gradient(store)             // (ParamId, val) pairs
    For each Constraint c with multiplier lambda_c:
      For each (row, pid, val) in c.jacobian(store):
        grad[pid] += lambda_c[row] * val         // J_g^T * lambda
    For each InequalityFn h with multiplier mu_h:
      For each (row, pid, val) in h.jacobian(store):
        grad[pid] += mu_h[row] * val              // J_h^T * mu

  Hessian:
    H = objective.hessian_entries(store)          // (ParamId, ParamId, val)
    For each ConstraintHessian c with multiplier lambda_c:
      For each equation_row r:
        For each (pi, pj, val) in c.residual_hessian(r, store):
          H[pi][pj] += lambda_c[r] * val
    For each InequalityFn h with multiplier mu_h:
      For each inequality_row r:
        For each (pi, pj, val) in h.hessian_entries(r, store):
          H[pi][pj] += mu_h[r] * val
```

### 3.2 Concrete Assembler

```rust
/// Assembles the Lagrangian and its derivatives from individual components.
///
/// This is the runtime orchestrator that combines per-function derivative
/// information (from Objective, Constraint, InequalityFn) with multiplier
/// values to produce the Lagrangian's gradient and Hessian.
pub struct LagrangianAssembler<'a> {
    /// The objective function(s).
    objectives: Vec<&'a dyn Objective>,
    /// Equality constraints (existing Constraint trait objects).
    equalities: Vec<&'a dyn Constraint>,
    /// Extended constraints with Hessian info (subset of equalities).
    ///
    /// # Alignment invariant
    ///
    /// `equality_hessians.len()` MUST equal `equalities.len()`. Entry `[i]` is
    /// `Some` if `equalities[i]` implements `ConstraintHessian`, `None` otherwise.
    /// Violating this invariant causes `self.equality_hessians[idx]` to panic or
    /// index the wrong Hessian, producing a silently incorrect Lagrangian Hessian.
    /// The constructor must enforce this with a `debug_assert_eq!`.
    equality_hessians: Vec<Option<&'a dyn ConstraintHessian>>,
    /// Inequality constraints.
    inequalities: Vec<&'a dyn InequalityFn>,
    /// Solver mapping for ParamId -> column index.
    mapping: &'a SolverMapping,
}

impl<'a> LagrangianAssembler<'a> {
    /// Evaluate f(x).
    pub fn objective_value(&self, store: &ParamStore) -> f64 {
        self.objectives.iter().map(|o| o.value(store)).sum()
    }

    /// Assemble grad_x L = grad f + J_g^T lambda + J_h^T mu.
    ///
    /// Returns a dense n-vector in solver column order.
    pub fn gradient(
        &self,
        store: &ParamStore,
        lambda: &[f64],
        mu: &[f64],
    ) -> Vec<f64> {
        let n = self.mapping.len();
        let mut grad = vec![0.0; n];

        // Objective gradient
        for obj in &self.objectives {
            for (pid, val) in obj.gradient(store) {
                if let Some(&col) = self.mapping.param_to_col.get(&pid) {
                    grad[col] += val;
                }
            }
        }

        // Equality contribution: J_g^T * lambda
        let mut lambda_offset = 0;
        for eq in &self.equalities {
            for (row, pid, val) in eq.jacobian(store) {
                if let Some(&col) = self.mapping.param_to_col.get(&pid) {
                    let mult = lambda[lambda_offset + row];
                    grad[col] += mult * val;
                }
            }
            lambda_offset += eq.equation_count();
        }

        // Inequality contribution: J_h^T * mu
        let mut mu_offset = 0;
        for ineq in &self.inequalities {
            for (row, pid, val) in ineq.jacobian(store) {
                if let Some(&col) = self.mapping.param_to_col.get(&pid) {
                    let mult = mu[mu_offset + row];
                    grad[col] += mult * val;
                }
            }
            mu_offset += ineq.inequality_count();
        }

        grad
    }

    /// Assemble H_L = H_f + sum lambda_i H_{g_i} + sum mu_j H_{h_j}.
    ///
    /// Returns sparse lower-triangular COO triplets (col_i, col_j, value)
    /// in solver column order. Returns None if any component lacks exact Hessians,
    /// signaling that the solver should use a quasi-Newton approximation.
    ///
    /// When `allow_partial` is true, components without exact Hessians contribute
    /// zero (Gauss-Newton approximation for those terms). When false, any missing
    /// Hessian causes the entire assembly to return None.
    pub fn hessian(
        &self,
        store: &ParamStore,
        lambda: &[f64],
        mu: &[f64],
        allow_partial: bool,
    ) -> Option<Vec<(usize, usize, f64)>> {
        let mut triplets = Vec::new();

        // Objective Hessian
        for obj in &self.objectives {
            match obj.hessian_entries(store) {
                Some(entries) => {
                    for (pi, pj, val) in entries {
                        if let (Some(&ci), Some(&cj)) = (
                            self.mapping.param_to_col.get(&pi),
                            self.mapping.param_to_col.get(&pj),
                        ) {
                            let (r, c) = if ci >= cj { (ci, cj) } else { (cj, ci) };
                            triplets.push((r, c, val));
                        }
                    }
                }
                None if allow_partial => {} // skip, Gauss-Newton for this term
                None => return None,
            }
        }

        // Equality Hessians
        let mut lambda_offset = 0;
        for (idx, eq) in self.equalities.iter().enumerate() {
            let neq = eq.equation_count();
            if let Some(hess_provider) = self.equality_hessians[idx] {
                for row in 0..neq {
                    let mult = lambda[lambda_offset + row];
                    if mult.abs() < 1e-15 {
                        // Skip zero-weighted terms
                        continue;
                    }
                    match hess_provider.residual_hessian(row, store) {
                        Some(entries) => {
                            for (pi, pj, val) in entries {
                                if let (Some(&ci), Some(&cj)) = (
                                    self.mapping.param_to_col.get(&pi),
                                    self.mapping.param_to_col.get(&pj),
                                ) {
                                    let (r, c) = if ci >= cj {
                                        (ci, cj)
                                    } else {
                                        (cj, ci)
                                    };
                                    triplets.push((r, c, mult * val));
                                }
                            }
                        }
                        None if allow_partial => {}
                        None => return None,
                    }
                }
            } else if !allow_partial {
                return None;
            }
            lambda_offset += neq;
        }

        // Inequality Hessians
        let mut mu_offset = 0;
        for ineq in &self.inequalities {
            let nineq = ineq.inequality_count();
            for row in 0..nineq {
                let mult = mu[mu_offset + row];
                if mult.abs() < 1e-15 {
                    continue;
                }
                match ineq.hessian_entries(row, store) {
                    Some(entries) => {
                        for (pi, pj, val) in entries {
                            if let (Some(&ci), Some(&cj)) = (
                                self.mapping.param_to_col.get(&pi),
                                self.mapping.param_to_col.get(&pj),
                            ) {
                                let (r, c) = if ci >= cj {
                                    (ci, cj)
                                } else {
                                    (cj, ci)
                                };
                                triplets.push((r, c, mult * val));
                            }
                        }
                    }
                    None if allow_partial => {}
                    None => return None,
                }
            }
            mu_offset += nineq;
        }

        Some(triplets)
    }
}
```

### 3.3 Storage of Per-Function Hessians

Individual Hessians are never stored persistently. They are computed on demand during
each iteration and immediately combined with multiplier weights into the Lagrangian
Hessian. This mirrors the existing pattern where per-constraint Jacobians are computed
per iteration and immediately assembled into the global Jacobian.

The rationale: storing per-function Hessians would require O(p + q) sparse matrices
of size n x n each. For a system with hundreds of constraints, the memory cost would
be substantial and the data would be stale after each parameter update anyway.

The assembly sequence per iteration is:

```
1. Solver updates x (primal) and (lambda, mu) (dual)
2. Write x into ParamStore
3. For each component, call hessian_entries(store) -> immediate triplets
4. Weight each triplet by the corresponding multiplier
5. Accumulate into a single SparseJacobian (COO) or CsrMatrix (CSR)
6. Feed to the linear system solver
```

---

## 4. Hessian of the Lagrangian: Composition Mechanism

### 4.1 Mathematical Structure

The Hessian of the Lagrangian is:

    H_L(x, lambda, mu) = H_f(x) + sum_{i=1}^{p} lambda_i * H_{g_i}(x)
                                  + sum_{j=1}^{q} mu_j * H_{h_j}(x)

Each H_{g_i} is an n x n symmetric matrix (the Hessian of the i-th equality residual).
Each H_{h_j} is an n x n symmetric matrix (the Hessian of the j-th inequality).

Key structural observations:

1. **Sparsity**: Each H_{g_i} is typically very sparse. A constraint involving k
   parameters has a Hessian with at most k(k+1)/2 unique entries. For a distance
   constraint between two 2D points (k=4), this is at most 10 entries in an n x n
   matrix where n might be thousands.

2. **Superposition**: The final H_L is a weighted sum of sparse matrices. The
   sparsity pattern of H_L is the union of the individual patterns.

3. **Symmetry**: All constituent Hessians are symmetric (they are Hessians of scalar
   functions), so we only store the lower triangle.

### 4.2 Runtime Composition

```
Per-function Hessian (from macro or manual impl):
  H_{g_i} stored as: Vec<(ParamId, ParamId, f64)>  (lower tri, ParamId space)

Lagrangian Hessian assembly:
  H_L = SparseJacobian::new(n, n)        // in solver column space
  for each equality i:
    weight = lambda[i]
    for (pa, pb, val) in constraint_i.residual_hessian(row, store):
      col_a = mapping.param_to_col[pa]
      col_b = mapping.param_to_col[pb]
      H_L.add_entry(max(col_a, col_b), min(col_a, col_b), weight * val)
  // similarly for objective and inequality Hessians
  // convert to CSR for the linear solve
  H_L_csr = H_L.to_csr()  // duplicate entries at same position are summed
```

The `CsrMatrix::from_coo` already handles duplicate entry summation (see
`jacobian/sparse.rs` line 318-327), so overlapping Hessian contributions from
different constraints at the same (col_a, col_b) position are naturally summed.

### 4.3 Sparsity Pattern Caching

The sparsity pattern of H_L is determined by the constraint graph structure and does
not change between iterations (assuming no structural changes like adding/removing
constraints). We can cache the pattern and reuse it:

```rust
/// Cached sparsity structure for the Hessian of the Lagrangian.
///
/// Precomputed once when the constraint graph changes. Reused across
/// iterations for efficient value-only updates.
pub struct HessianPattern {
    /// CSR pattern (row_ptr, col_indices) without values.
    pub pattern: SparsityPattern,
    /// For each CSR entry, which (function_index, local_entry_index) contribute.
    /// Used for efficient scatter during assembly.
    pub contribution_map: Vec<Vec<(usize, usize)>>,
}
```

---

## 5. Per-Algorithm Derivative Requirements

### 5.1 Overview Table

| Algorithm | f(x) | grad f | g, h | Jg, Jh | H_L | H_L * v | Notes |
|-----------|-------|--------|------|---------|-----|---------|-------|
| **SQP** (Sequential Quadratic Programming) | Yes | Yes | Yes | Yes | Yes* | -- | *Can use BFGS approx |
| **IPM** (Interior Point Method) | Yes | Yes | Yes | Yes | Yes* | -- | *Can use quasi-Newton |
| **ALM** (Augmented Lagrangian Method) | Yes | Yes | Yes | Yes | No** | No | **Uses inner NR/LM solver |
| **L-BFGS-B** (Limited-memory BFGS with Bounds) | Yes | Yes | -- | -- | No | No | Bounds only, no general ineq |
| **Penalty Method** | Yes | Yes | Yes | Yes | No | No | Simple, slow convergence |
| **Projected Gradient** | Yes | Yes | -- | -- | No | No | Bounds only, first-order |

### 5.2 Detailed Algorithm Requirements

#### SQP (Sequential Quadratic Programming)

The workhorse for small-to-medium constrained optimization. At each iteration, solves
a Quadratic Programming (QP) subproblem:

    min   grad f^T d + 0.5 d^T H_L d
    s.t.  Jg * d + g = 0
          Jh * d + h <= 0

**Required derivatives per iteration:**
- f(x): 1 evaluation
- grad f(x): 1 evaluation (n-vector)
- g(x), h(x): 1 evaluation each (p-vector, q-vector)
- Jg(x), Jh(x): 1 evaluation each (p x n, q x n sparse)
- H_L(x, lambda, mu): 1 evaluation (n x n sparse symmetric)

**Hessian approximation fallback**: When exact H_L is unavailable, SQP uses
damped BFGS updates to build a quasi-Newton approximation. This is the B_k matrix
in standard SQP implementations (Nocedal & Wright, Ch. 18).

**How this maps to Solverang traits:**
- `Objective::value()` + `Objective::gradient()` -> f, grad f
- `Constraint::residuals()` + `Constraint::jacobian()` -> g, Jg
- `InequalityFn::values()` + `InequalityFn::jacobian()` -> h, Jh
- `LagrangianAssembler::hessian()` -> H_L
- Inner QP solved by existing Newton-Raphson (for active-set QP) or dedicated QP solver

#### IPM (Interior Point / Barrier Method)

Transforms inequalities into a barrier term:

    min  f(x) - tau * sum_j ln(-h_j(x))
    s.t. g(x) = 0

As tau -> 0, the solution approaches the constrained optimum.

**Required derivatives per iteration:**
- f(x): 1 evaluation
- grad f(x): 1 evaluation
- g(x): 1 evaluation
- Jg(x): 1 evaluation
- h(x): 1 evaluation (for barrier evaluation and gradient)
- Jh(x): 1 evaluation (for barrier gradient and Hessian contribution)
- H_L(x, lambda, mu): 1 evaluation (includes barrier Hessian)

**The barrier Hessian contribution** for each inequality j is:

    tau * (grad h_j * grad h_j^T) / h_j^2  -  tau * H_{h_j} / h_j

This is computed from first-order information (Jh) for the rank-1 part, plus the
second-order H_{h_j} if available. The rank-1 part dominates near the boundary,
so exact inequality Hessians are less critical for IPM than for SQP.

**KKT system per iteration** (symmetric indefinite, "augmented system" form):

    [ H_L + Sigma   Jg^T   Jh^T ] [ dx      ]   [ -grad_x L         ]
    [ Jg            0      0    ] [ dlambda  ] = [ -g                 ]
    [ Jh            0     -Theta] [ dmu_ineq ]   [ -h + tau ./ mu_ineq]

where:
- `tau` is the (scalar) barrier parameter being driven to zero
- `mu_ineq` is the current inequality multiplier vector (element-wise division `./ `)
- `Sigma` and `Theta` are diagonal matrices from the barrier terms

> **REVIEW NOTE (notation):** The symbol `mu` is used in this document for both (a)
> the barrier parameter (scalar, also called `tau`) and (b) the inequality multiplier
> vector. In the KKT system above, `mu_ineq` is the multiplier vector and `tau` is
> the barrier parameter. These are distinct quantities. Implementors must take care not
> to confuse them. Standard references (Nocedal & Wright §19.2) use `mu` for the
> barrier parameter and `lambda` for all multipliers, or use `mu` for multipliers and
> `epsilon`/`tau` for the barrier. Choose one convention before implementing and apply
> it consistently across all files in this doc series.

**How this maps to Solverang:**
The KKT system is a structured nonlinear system that could be solved by the existing
Newton-Raphson solver (for the linearized KKT step). The outer IPM loop manages the
barrier parameter tau.

#### ALM (Augmented Lagrangian Method)

Solves a sequence of unconstrained (or bound-constrained) subproblems:

    min  f(x) + lambda^T g(x) + (rho/2) ||g(x)||^2
         + penalty terms for inequality constraints

**Required derivatives per iteration:**
- Inner subproblem requires: f, grad f, g, Jg (and h, Jh if inequalities present)
- NO *explicit* `ConstraintHessian` trait implementations needed for the outer loop.
  However, the inner NR/LM solver's Jacobian *is* the Hessian of the augmented
  Lagrangian. When exact Hessians are unavailable, NR/LM approximates this via
  Gauss-Newton (J^T J) or finite differences — so second-order information is still
  needed, it is just approximated automatically rather than provided by the caller.
- The inner solver can be the existing LM solver (treating the augmented Lagrangian
  gradient as residuals in a least-squares sense) or NR for the gradient = 0 system

**Key advantage for Solverang:** ALM naturally reuses the existing NR/LM infrastructure.
The inner subproblem is "find x such that grad(augmented Lagrangian) = 0", which is
exactly F(x) = 0 with F = grad(augmented Lagrangian).

**How this maps to Solverang:**
```rust
/// Constructs a Problem from an augmented Lagrangian subproblem.
///
/// The residuals are the gradient of the augmented Lagrangian,
/// and the Jacobian is the Hessian of the augmented Lagrangian
/// (approximated via Gauss-Newton / finite differences if exact Hessians unavailable).
struct AugmentedLagrangianSubproblem { ... }
impl Problem for AugmentedLagrangianSubproblem { ... }
```

This is the recommended entry point for optimization in Solverang because it requires
the least new infrastructure: the existing solvers do the heavy lifting.

#### L-BFGS-B (Bounds Only)

For problems with only simple bound constraints (no general equality/inequality):

    min f(x)  s.t.  lower <= x <= upper

**Required derivatives:**
- f(x): 1 evaluation per iteration
- grad f(x): 1 evaluation per iteration
- NO Jacobian, NO Hessian (L-BFGS builds implicit curvature from gradient history)

This is the simplest optimization algorithm to add and useful for problems like
"minimize total wire length subject to component placement bounds."

---

## 6. Integration with Existing Architecture

### 6.1 How Existing Constraint Maps to Equality Constraints

The existing `Constraint` trait already provides exactly what optimization needs for
equality constraints:

| Constraint method | Optimization role | Optimization interpretation |
|---|---|---|
| `residuals(store)` | g(x) | Equality constraint values |
| `jacobian(store)` | Jg | Equality constraint Jacobian |
| `equation_count()` | p (per constraint) | Number of equality equations |
| `param_ids()` | Sparsity structure | Which variables appear |
| `is_soft()` | Soft -> penalty | Soft constraints become penalty terms |
| `weight()` | Scaling | Scales the constraint in the Lagrangian |

The only gap is second-order information, filled by the optional `ConstraintHessian`
extension trait.

### 6.2 Solver Pipeline Extension

The current pipeline is:

    Decompose -> Analyze -> Reduce -> Solve -> PostProcess

For optimization, the pipeline extends to:

    Classify -> Decompose -> Analyze -> Reduce -> MultiplierInit -> Solve -> PostProcess
        |                                               |               |
  [problem type]                               [init multipliers]  [dispatch]
  [feasibility only?] -> existing NR/LM
  [has objective?]    -> optimization solver
  [has inequalities?] -> select IPM/SQP/ALM

<!-- Decision: Pipeline order is Classify → Decompose → Analyze → MultiplierInit → Solve → PostProcess. Classify runs before Decompose so that objective-coupled variable grouping is known before decomposition. MultiplierInit is a distinct phase between Analyze and Solve. -->

The new `ClassifyProblem` phase examines each cluster to determine:
1. Does it have an objective? If not, it is a feasibility problem (use existing solver).
2. Does it have inequality constraints? If so, choose IPM, SQP, or ALM.
3. Are exact Hessians available? If so, prefer SQP. If not, prefer ALM or L-BFGS.

### 6.3 MultiplierStore

```rust
/// Storage for Lagrange multipliers (dual variables).
///
/// Analogous to ParamStore but for dual variables. Multipliers are indexed
/// by MultiplierId and have additional semantics:
/// - Equality multipliers (lambda): unconstrained sign
/// - Inequality multipliers (mu): must be >= 0
///
/// The store is separate from ParamStore because:
/// 1. Multipliers have different update rules (dual updates vs primal updates)
/// 2. Inequality multipliers have non-negativity constraints
/// 3. Multipliers are not "owned" by entities
/// 4. The existing solver infrastructure should not see multipliers as variables
pub struct MultiplierStore {
    /// Current multiplier values.
    values: Vec<f64>,
    /// Which constraint each multiplier belongs to (for diagnostics).
    constraint_ids: Vec<ConstraintId>,
    /// Whether this is an inequality multiplier (must be >= 0).
    is_inequality: Vec<bool>,
    /// Generation tracking for safe access.
    generations: Vec<Generation>,
    alive: Vec<bool>,
    free_list: Vec<u32>,
}

impl MultiplierStore {
    /// Allocate multipliers for an equality constraint (one per equation).
    pub fn alloc_equality(
        &mut self,
        constraint_id: ConstraintId,
        count: usize,
    ) -> Vec<MultiplierId> { ... }

    /// Allocate multipliers for an inequality constraint.
    pub fn alloc_inequality(
        &mut self,
        constraint_id: ConstraintId,
        count: usize,
    ) -> Vec<MultiplierId> { ... }

    /// Get multiplier value. For inequality multipliers, always >= 0.
    pub fn get(&self, id: MultiplierId) -> f64 { ... }

    /// Set multiplier value. Clamps inequality multipliers to >= 0.
    pub fn set(&mut self, id: MultiplierId, value: f64) { ... }

    /// Extract all equality multipliers in constraint order.
    ///
    /// # Ordering Contract
    ///
    /// Returns multipliers in the order that equality constraints were registered
    /// via `alloc_equality`. The `LagrangianAssembler` relies on this to index
    /// multipliers by `lambda_offset + row`. Violating this order silently applies
    /// wrong multipliers to constraints, producing an incorrect Lagrangian gradient.
    pub fn lambda_vector(&self) -> Vec<f64> { ... }

    /// Extract all inequality multipliers in constraint order.
    ///
    /// # Ordering Contract
    ///
    /// Returns multipliers in the order that inequality constraints were registered
    /// via `alloc_inequality`. Same ordering requirement as `lambda_vector`.
    pub fn mu_vector(&self) -> Vec<f64> { ... }
}
```

### 6.4 OptimizationSystem

```rust
/// Extension of ConstraintSystem with optimization capabilities.
///
/// Wraps a ConstraintSystem and adds:
/// - Objective function(s)
/// - Inequality constraints
/// - Multiplier management
/// - Optimization solver dispatch
///
/// When no objective is registered, solve() delegates to the underlying
/// ConstraintSystem (pure feasibility). When an objective is present,
/// it invokes the optimization solver pipeline.
pub struct OptimizationSystem {
    /// The underlying constraint system (manages entities, equalities, params).
    inner: ConstraintSystem,
    /// Registered objective functions.
    objectives: Vec<Option<Box<dyn Objective>>>,
    /// Registered inequality constraints.
    inequalities: Vec<Option<Box<dyn InequalityFn>>>,
    /// Lagrange multiplier storage.
    multipliers: MultiplierStore,
    /// Optimization algorithm configuration.
    opt_config: OptimizationConfig,
}

impl OptimizationSystem {
    /// Add an objective function to minimize.
    pub fn add_objective(&mut self, obj: Box<dyn Objective>) -> ObjectiveId { ... }

    /// Add an inequality constraint h(x) <= 0.
    pub fn add_inequality(&mut self, ineq: Box<dyn InequalityFn>) -> ConstraintId { ... }

    /// Solve the optimization problem.
    ///
    /// If no objective is registered, delegates to ConstraintSystem::solve().
    /// Otherwise, runs the optimization algorithm selected by OptimizationConfig.
    pub fn solve(&mut self) -> OptimizationResult { ... }
}
```

---

## 7. Macro Extension for Hessians

### 7.1 How `#[auto_jacobian]` Works Today

The `#[auto_jacobian]` macro reads `#[residual]`-annotated methods, symbolically
differentiates the expressions, and generates `jacobian_entries()` implementations.
This produces first-order derivatives automatically.

### 7.2 Extension to `#[auto_hessian]`

The same symbolic differentiation engine can be applied a second time to generate
Hessians. For each residual r(x), the macro already computes the symbolic expressions
for dr/dx_i. Differentiating these once more yields d^2r/dx_i dx_j.

```rust
/// Example of how the macro would generate Hessian code:
///
/// Given a constraint:
///   #[residual] fn r(&self, store: &ParamStore) -> f64 {
///       let x = store.get(self.px);
///       let y = store.get(self.py);
///       x * x + x * y - 1.0
///   }
///
/// The macro generates:
///   Jacobian: [(self.px, 2*x + y), (self.py, x)]
///   Hessian:  [(self.px, self.px, 2.0),
///              (self.px, self.py, 1.0)]  // lower tri only
///
/// In code:
///   fn residual_hessian(&self, row: usize, store: &ParamStore)
///       -> Option<Vec<(ParamId, ParamId, f64)>> {
///       let x = store.get(self.px);
///       let _ = store.get(self.py); // used for value but hessian is constant here
///       Some(vec![
///           (self.px, self.px, 2.0),
///           (self.px, self.py, 1.0),
///       ])
///   }
```

The `#[auto_hessian]` attribute would be opt-in (not all constraints need it) and
would automatically implement `ConstraintHessian` for the annotated struct.

Similarly, an `#[auto_objective]` macro could generate both gradient and Hessian for
objective functions from a single `#[value]`-annotated method.

---

## 8. Concrete Data Flow Example

Consider minimizing wire length between two points subject to a fixed-distance
constraint:

```
Objective: f(x) = ||p1 - p2||  (minimize distance, or some function of it)
Equality:  g(x) = ||p0 - p1|| - d = 0  (distance constraint)
Inequality: h(x) = x_min - p1_x <= 0  (p1.x >= x_min, a bound)
```

**Iteration k of SQP:**

1. Read current (x_k, lambda_k, mu_k)
2. Evaluate: f(x_k), g(x_k), h(x_k)
3. Compute gradients:
   - objective.gradient(store) -> [(p1_x, df/dp1x), (p1_y, df/dp1y), (p2_x, ...), (p2_y, ...)]
   - constraint.jacobian(store) -> [(0, p0_x, dg/dp0x), (0, p0_y, ...), (0, p1_x, ...), ...]
   - inequality.jacobian(store) -> [(0, p1_x, -1.0)]
4. Assemble H_L:
   - objective.hessian_entries(store) -> [(p1_x, p1_x, v1), (p1_x, p1_y, v2), ...]
   - constraint.residual_hessian(0, store) -> [(p0_x, p0_x, v3), ...], weighted by lambda_k[0]
   - inequality.hessian_entries(0, store) -> [] (linear, Hessian is zero)
5. Form QP subproblem:
   - Quadratic term: H_L (n x n)
   - Linear term: grad f (n-vector)
   - Equality constraint: Jg * d + g = 0
   - Inequality constraint: Jh * d + h <= 0
6. Solve QP -> step direction d_k
7. Line search -> step length alpha_k
8. Update: x_{k+1} = x_k + alpha_k * d_k
9. Update multipliers: lambda_{k+1}, mu_{k+1} from QP dual solution
10. Write x_{k+1} into ParamStore
11. Check convergence (KKT residual < tolerance)

---

## 9. Recommended Implementation Order

### Phase 1: ALM (Lowest Risk, Highest Reuse)

1. Implement `Objective` trait and `ObjectiveId`
2. Implement `AugmentedLagrangianSubproblem` as a `Problem` wrapper
3. Use existing NR/LM solvers for the inner subproblem
4. Implement outer ALM loop (multiplier updates, penalty parameter increase)
5. No Hessian infrastructure needed (inner solver uses Gauss-Newton)

**Why first:** This requires almost no changes to the existing solver. The inner
subproblem is just another `Problem` that the existing NR/LM solvers can handle.

### Phase 2: L-BFGS-B (Simple Bounds)

1. Implement L-BFGS-B solver (new solver module, does not need Jacobian infrastructure)
2. Add bounds to `ParamStore` (lower/upper per ParamId)
3. Useful standalone for unconstrained and bound-constrained optimization

### Phase 3: SQP (Full Second-Order)

1. Implement `ConstraintHessian` extension trait
2. Implement `InequalityFn` trait and `MultiplierStore`
3. Implement `LagrangianAssembler`
4. Implement SQP solver with BFGS fallback for missing Hessians
5. Extend `#[auto_jacobian]` macro to optionally generate Hessians

### Phase 4: IPM (Barrier Method)

1. Implement barrier function and log-barrier transforms
2. Implement IPM KKT system assembly
3. Reuse sparse linear algebra from existing `SparseSolver`

---

## 10. Summary of New Types

| Type | Purpose | Relationship to Existing |
|------|---------|------------------------|
| `ObjectiveId` | Generational index for objectives | Same pattern as `ConstraintId` |
| `MultiplierId` | Semantic address `{ constraint_id, equation_row }` for dual variables | No generational index needed |
| `Objective` trait | Scalar function to minimize | Analogous to `Constraint` |
| `ConstraintHessian` trait | Second-order info for constraints | Extends `Constraint` |
| `InequalityFn` trait | h(x) <= 0 constraints with ParamId | ParamStore-based `InequalityConstraint` |
| `OptimizationProblem` trait | Unified problem interface (dense) | Extends `Problem` |
| `LagrangianAssembler` | Runtime derivative combiner | Analogous to Jacobian assembly in pipeline |
| `MultiplierStore` | Dual variable storage | Analogous to `ParamStore` |
| `OptimizationSystem` | Top-level coordinator | Wraps `ConstraintSystem` |
| `HessianPattern` | Cached sparsity for H_L | Uses existing `SparsityPattern` |
| `OptimizationConfig` | Algorithm selection and tuning | Extends `SystemConfig` |
| `OptimizationResult` | Solution + multipliers + status | Extends `SystemResult` |

> **REVIEW NOTE (missing specifications):** Two items needed before implementation
> can begin:
>
> 1. **Convergence criteria**: `OptimizationResult` needs a defined notion of
>    convergence. Minimum required: primal feasibility tolerance (`||g(x)||`), dual
>    feasibility tolerance (`||grad_x L||`), and complementarity tolerance
>    (`max |mu_j * h_j(x)|`). Section 8 mentions "KKT residual < tolerance" but does
>    not define how these three quantities are combined into a single stopping test.
>    Without this, property tests cannot be written and `OptimizationResult::is_converged()`
>    cannot be implemented.
>
> 2. **`OptimizationResult` status variants**: The analogous `SystemStatus` in
>    `system.rs` has `Solved`, `PartiallySolved`, `DiagnosticFailure`. The optimization
>    equivalent needs at minimum: `Converged`, `MaxIterationsReached`, `Infeasible`,
>    `Diverged`. Define these before Phase 1 is complete.
