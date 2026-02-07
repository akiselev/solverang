# Solverang V3: Solver-First Architecture

## Strategy

Ship the constraint solver as a standalone, geometry-agnostic library. Build the
geometric kernel *underneath* it later. The solver is the product. Geometry is a
plugin.

This supersedes `rearchitecture-v2.md` in scope. V2's designs are still correct
but this document reframes them around the publish boundary: what's in the crate,
what's an extension point, and what's deferred to the geometry layer.

---

## The Key Architectural Principle

**The solver core never imports a geometry type.**

```
┌─────────────────────────────────────────────────────────────┐
│  YOUR APPLICATION (CAD, PCB, game physics, robotics, ...)   │
└──────────────────────────┬──────────────────────────────────┘
                           │ uses
┌──────────────────────────▼──────────────────────────────────┐
│  GEOMETRY PLUGIN (feature = "sketch2d", separate crate,     │
│  or user code)                                              │
│                                                             │
│  Implements: Entity, Constraint                             │
│  Uses internally: curvo, nalgebra, custom math              │
│  Provides: Point2D, Circle2D, Arc2D, NurbsCurve, ...       │
│  Provides: DistanceConstraint, TangentConstraint, ...       │
│                                                             │
│  THE SOLVER DOES NOT KNOW THIS LAYER EXISTS.                │
└──────────────────────────┬──────────────────────────────────┘
                           │ implements traits from
┌──────────────────────────▼──────────────────────────────────┐
│  SOLVERANG (published crate)                                │
│                                                             │
│  Exports: ParamStore, Entity, Constraint, ConstraintSystem  │
│  Exports: SolveResult, Diagnostics, ChangeTracker           │
│  Contains: graph decomposition, symbolic reduction,         │
│            incremental solving, diagnostics, NR, LM,        │
│            sparse, JIT, parallel                            │
│                                                             │
│  THIS IS THE PRODUCT.                                       │
└─────────────────────────────────────────────────────────────┘
```

The `Entity` and `Constraint` traits are the extension points. Any domain —
CAD sketches, PCB layout, mechanism simulation, molecular dynamics — implements
these traits with its own types. The solver operates on `ParamId`s and trait
objects. It never knows what a "circle" is.

When the geometric kernel comes later, it's just another consumer of these traits.
Nothing in the solver changes. The kernel provides rich entity types (NURBS curves,
B-Rep edges, assembly mates) that produce residuals and Jacobians through the
same `Constraint` interface.

---

## What Exists Today (Inventory)

### Production-quality (keep as-is)

| Component | File | Status |
|---|---|---|
| Newton-Raphson solver | `solver/newton_raphson.rs` | Ship |
| Levenberg-Marquardt solver | `solver/levenberg_marquardt.rs` | Ship |
| LM adapter (nalgebra bridge) | `solver/lm_adapter.rs` | Ship |
| Auto/Robust solver selection | `solver/auto.rs` | Ship |
| Sparse solver (faer) | `solver/sparse_solver.rs` | Ship |
| JIT solver (Cranelift) | `solver/jit_solver.rs` | Ship |
| Solver configs | `solver/config.rs`, `lm_config.rs` | Ship |
| SolveResult / SolveError | `solver/result.rs` | Extend |
| Union-find decomposition | `decomposition.rs` | Reuse (wrap) |
| SubProblem extraction | `decomposition.rs` | Reuse (wrap) |
| Parallel solver | `solver/parallel.rs` | Adapt |
| Jacobian utilities | `jacobian.rs` | Ship |
| NIST test problems | `test_problems/` | Ship |
| Inequality constraints | `constraints/inequality.rs` | Ship |
| Problem trait | `problem.rs` | **Keep as low-level API** |

### Replace / rewrite

| Component | Status | Reason |
|---|---|---|
| `GeometricConstraint<D>` trait | Move to geometry plugin | Points-only, dimension-generic |
| `ConstraintSystem<D>` | Replace with new `ConstraintSystem` | New param-based architecture |
| All 16 constraint impls | Move to geometry plugin | Rewrite with squared formulations |
| Point/Line/Circle/Vector types | Move to geometry plugin | Become Entity implementations |
| Builder | Replace | New API for new system |

---

## What's Missing (Gap Analysis)

### Tier 1: Core Infrastructure (must ship)

These are the bones of the new architecture. Without them, there's nothing to publish.

```
NEW CODE NEEDED                        EST. LOC    DEPENDS ON
─────────────────────────────────────────────────────────────
id.rs                                    80        nothing
  ParamId, EntityId, ConstraintId
  (generational indices)

param/store.rs                          300        id.rs
  ParamStore: alloc, free, get, set,
  fix, unfix, SolverMapping

entity/mod.rs                            40        id.rs
  Entity trait (id, params, name)

constraint/mod.rs                        60        id.rs, param/store
  Constraint trait (id, entity_ids,
  param_ids, residuals, jacobian)

graph/bipartite.rs                      200        id.rs
  ConstraintGraph: entity<->constraint
  bipartite graph, param->constraint index

graph/cluster.rs                        150        id.rs
  RigidCluster: entities, constraints,
  params, cached state, status

graph/decompose.rs                      200        bipartite, cluster
  Wrap existing union-find decomposition
  to work with new ID types

solve/sub_problem.rs                    250        param/store, constraint
  ReducedSubProblem: what solvers see.
  Bridges new types to existing Problem trait.

system.rs                               400        everything above
  ConstraintSystem: registries + graph +
  decompose + per-cluster solve + assemble
─────────────────────────────────────────────────────────────
TOTAL TIER 1                          ~1,700 LOC
```

### Tier 2: Differentiating Features (what makes it cutting-edge)

These are what separate solverang from "yet another NR/LM wrapper."

```
NEW CODE NEEDED                        EST. LOC    DEPENDS ON
─────────────────────────────────────────────────────────────
reduce/substitute.rs                    150        param/store, constraint
  Fixed param elimination: remove fixed
  params from variable set, simplify
  constraint evaluation

reduce/merge.rs                         200        param/store, constraint
  Coincident param merging: replace
  param B with param A everywhere,
  remove the coincident constraint

reduce/eliminate.rs                     150        constraint, graph
  Trivial constraint detection: when
  a constraint directly determines one
  variable, substitute and remove

graph/redundancy.rs                     250        cluster, solve
  Jacobian rank analysis via SVD/QR.
  Detect redundant and conflicting
  constraints. Report which constraints
  are involved.

graph/dof.rs                            150        cluster, redundancy
  Per-entity DOF: which params are
  free to move, in what directions
  (null space projection)

dataflow/tracker.rs                     200        graph, cluster
  ChangeTracker: dirty params, dirty
  clusters, structural changes.
  Incremental re-decomposition.

dataflow/cache.rs                       150        cluster, solve
  Per-cluster solution caching.
  Warm start from previous solution.
  Jacobian factorization cache.
─────────────────────────────────────────────────────────────
TOTAL TIER 2                          ~1,250 LOC
```

### Tier 3: Advanced Solving

```
NEW CODE NEEDED                        EST. LOC    DEPENDS ON
─────────────────────────────────────────────────────────────
solve/drag.rs                           200        dof, system
  Null-space projection for under-
  constrained drag. Project user intent
  onto constraint manifold.

solve/branch.rs                         150        solve, system
  Branch selection: when multiple
  solutions exist, pick the one closest
  to the previous configuration.

graph/pattern.rs                        300        cluster, constraint
  Solvable pattern detection. Match
  subgraphs to known closed-form
  templates (triangle, line-circle, etc.)

solve/closed_form.rs                    250        pattern
  Analytical solvers for matched
  patterns. Law of cosines, quadratic
  intersections, etc.
─────────────────────────────────────────────────────────────
TOTAL TIER 3                           ~900 LOC
```

### Tier 4: Batteries-Included Geometry (feature-gated)

Shipped as `feature = "sketch2d"` (and later `"sketch3d"`, `"assembly"`).
This is NOT the geometric kernel. It's a convenience layer of basic entity
and constraint types so users don't have to implement the traits themselves
for common 2D/3D sketching.

```
NEW CODE NEEDED                        EST. LOC    DEPENDS ON
─────────────────────────────────────────────────────────────
sketch2d/entities.rs                    400        entity trait
  Point2D, LineSegment2D, Circle2D,
  Arc2D, InfiniteLine2D

sketch2d/constraints.rs                 800        constraint trait
  Distance (Pt-Pt, Pt-Line),
  Coincident, Tangent (Line-Circle,
  Circle-Circle), Parallel, Perp,
  Angle, Horizontal, Vertical, Fixed,
  Midpoint, Symmetric, Equal, On-Entity
  ALL with squared formulations.

sketch2d/builder.rs                     300        system, entities
  Ergonomic builder API

sketch3d/entities.rs                    300        entity trait
  Point3D, LineSegment3D, Plane, Axis3D

sketch3d/constraints.rs                 400        constraint trait
  3D variants of 2D constraints plus
  Coplanar, Coaxial

assembly/entities.rs                    200        entity trait
  RigidBody (position + quaternion)

assembly/constraints.rs                 300        constraint trait
  Mate, Coaxial, Insert, Gear
─────────────────────────────────────────────────────────────
TOTAL TIER 4                          ~2,700 LOC
```

### Grand Total: ~6,550 new LOC + ~10K existing = ~16.5K LOC

---

## Core Traits (Public API)

### Entity

```rust
/// A solvable entity: a named group of parameters.
///
/// Entities represent geometric objects (points, circles, curves), physical
/// objects (rigid bodies, springs), or any other domain object with solvable
/// parameters. The solver treats all entities uniformly as parameter groups.
///
/// # Implementing for a geometric kernel
///
/// When building a geometry layer on top of solverang, your curve/surface
/// types implement this trait. The solver doesn't know or care what kind of
/// geometry the entity represents — it only needs the parameter IDs.
///
/// ```ignore
/// // Future: NURBS curve entity
/// struct NurbsCurve2D {
///     id: EntityId,
///     control_point_params: Vec<ParamId>,  // 2 per control point
///     weight_params: Vec<ParamId>,          // 1 per control point
///     // Knot vector is fixed (not a solver variable)
///     knots: Vec<f64>,
///     degree: usize,
/// }
///
/// impl Entity for NurbsCurve2D {
///     fn id(&self) -> EntityId { self.id }
///     fn params(&self) -> &[ParamId] { &self.all_params }
///     fn name(&self) -> &str { "NurbsCurve2D" }
/// }
/// ```
pub trait Entity: Send + Sync {
    fn id(&self) -> EntityId;
    fn params(&self) -> &[ParamId];
    fn name(&self) -> &str;
}
```

Note what's **not** on this trait:
- No `EntityKind` — that's for the geometry layer to define.
- No `evaluate()` — the solver doesn't evaluate geometry.
- No `dof()` — DOF is computed by the solver from the constraint graph.
- No dimension generic `<const D: usize>` — the solver is dimension-agnostic.

### Constraint

```rust
/// A constraint: a set of equations over parameters.
///
/// Constraints produce residuals (which should be zero when satisfied) and
/// Jacobians (partial derivatives of residuals w.r.t. parameters). The solver
/// uses these to iteratively find parameter values that satisfy all constraints.
///
/// # Implementing for a geometric kernel
///
/// When building geometry on top of solverang, each geometric constraint
/// (distance, tangent, parallel, etc.) implements this trait. The constraint
/// reads parameter values from the ParamStore and produces residuals.
///
/// ```ignore
/// // Future: point-on-NURBS-curve constraint
/// struct PointOnNurbsCurve {
///     id: ConstraintId,
///     point_entity: EntityId,
///     curve_entity: EntityId,
///     px: ParamId, py: ParamId,    // point params
///     t: ParamId,                   // curve parameter (implicit variable)
///     curve_params: Vec<ParamId>,   // control point coords
///     knots: Vec<f64>,              // fixed
///     degree: usize,
/// }
///
/// impl Constraint for PointOnNurbsCurve {
///     fn residuals(&self, store: &ParamStore) -> Vec<f64> {
///         let t = store.get(self.t);
///         let point = (store.get(self.px), store.get(self.py));
///         let curve_point = evaluate_nurbs(t, &self.curve_params, store, ...);
///         vec![point.0 - curve_point.0, point.1 - curve_point.1]
///     }
///     // Jacobian includes dC/dt (basis functions) and dC/d(control_points)
/// }
/// ```
pub trait Constraint: Send + Sync {
    fn id(&self) -> ConstraintId;
    fn name(&self) -> &str;

    /// Which entities this constraint binds.
    fn entity_ids(&self) -> &[EntityId];

    /// Which parameters this constraint depends on (for graph building).
    fn param_ids(&self) -> &[ParamId];

    /// Number of scalar equations this constraint produces.
    fn equation_count(&self) -> usize;

    /// Evaluate residuals. Each element should be zero when satisfied.
    fn residuals(&self, store: &ParamStore) -> Vec<f64>;

    /// Sparse Jacobian: (equation_row, param_id, partial_derivative).
    /// Only non-zero entries. ParamId → column mapping is done by the system.
    fn jacobian(&self, store: &ParamStore) -> Vec<(usize, ParamId, f64)>;

    /// Weight for soft constraints (default 1.0).
    fn weight(&self) -> f64 { 1.0 }

    /// Is this a soft constraint that can be relaxed?
    fn is_soft(&self) -> bool { false }
}
```

Note what's **not** on this trait:
- No `<const D: usize>` — constraints work in any dimension.
- No `points: &[Point<D>]` parameter — constraints read from `ParamStore`.
- No geometry types — the solver never sees `Point2D`, `Circle`, etc.
- Jacobian returns `ParamId`, not column indices — the system does the mapping.

### ParamStore

```rust
/// Central storage for all solvable parameter values.
///
/// Every solvable quantity in the system is a ParamId pointing into this store.
/// Entities own parameters. Constraints read parameters. The solver writes
/// parameters. The ParamStore is the single source of truth.
pub struct ParamStore { ... }

impl ParamStore {
    // Allocation
    pub fn alloc(&mut self, value: f64, owner: EntityId) -> ParamId;
    pub fn free(&mut self, id: ParamId);

    // Access
    pub fn get(&self, id: ParamId) -> f64;
    pub fn set(&mut self, id: ParamId, value: f64);

    // Fixed/free
    pub fn is_fixed(&self, id: ParamId) -> bool;
    pub fn fix(&mut self, id: ParamId);
    pub fn unfix(&mut self, id: ParamId);

    // Solver interface
    pub fn free_param_count(&self) -> usize;
    pub fn build_solver_mapping(&self) -> SolverMapping;
    pub fn extract_free_values(&self, mapping: &SolverMapping) -> Vec<f64>;
    pub fn write_free_values(&mut self, values: &[f64], mapping: &SolverMapping);
}

/// Bidirectional mapping: ParamId <-> column index in Jacobian.
/// Built once per solve (or once per decomposition change).
pub struct SolverMapping {
    pub param_to_col: HashMap<ParamId, usize>,
    pub col_to_param: Vec<ParamId>,
}
```

### ConstraintSystem (the coordinator)

```rust
/// The top-level constraint solver.
///
/// Manages entities, constraints, the constraint graph, decomposition,
/// and the solve pipeline. This is what users interact with.
pub struct ConstraintSystem {
    params: ParamStore,
    entities: Vec<Box<dyn Entity>>,
    constraints: Vec<Box<dyn Constraint>>,
    graph: ConstraintGraph,
    clusters: Vec<RigidCluster>,
    tracker: ChangeTracker,
    config: SolverConfig,
}

impl ConstraintSystem {
    pub fn new() -> Self;

    // Entity management
    pub fn add_entity(&mut self, entity: Box<dyn Entity>) -> EntityId;
    pub fn remove_entity(&mut self, id: EntityId);

    // Constraint management
    pub fn add_constraint(&mut self, constraint: Box<dyn Constraint>) -> ConstraintId;
    pub fn remove_constraint(&mut self, id: ConstraintId);

    // Parameter access (delegates to ParamStore)
    pub fn get_param(&self, id: ParamId) -> f64;
    pub fn set_param(&mut self, id: ParamId, value: f64);
    pub fn fix_param(&mut self, id: ParamId);
    pub fn unfix_param(&mut self, id: ParamId);

    // Solving
    pub fn solve(&mut self) -> SystemResult;
    pub fn solve_incremental(&mut self) -> SystemResult;

    // Diagnostics
    pub fn diagnostics(&self) -> &SystemDiagnostics;
    pub fn cluster_count(&self) -> usize;
    pub fn degrees_of_freedom(&self) -> i32;
    pub fn is_well_constrained(&self) -> bool;
    pub fn redundant_constraints(&self) -> Vec<ConstraintId>;
    pub fn conflicting_constraints(&self) -> Vec<Vec<ConstraintId>>;

    // Direct access to ParamStore (for geometry layer)
    pub fn params(&self) -> &ParamStore;
    pub fn params_mut(&mut self) -> &mut ParamStore;
}
```

---

## How the Geometry Kernel Slots In Later

The critical insight: **the solver's Entity and Constraint traits have no geometry
in them**. They are just "groups of parameters" and "equations over parameters."

When the geometry kernel arrives, it provides:

### 1. Rich entity types that implement `Entity`

```rust
// In the geometry crate (NOT in solverang)
pub struct NurbsCurve2D {
    id: EntityId,
    control_points: Vec<(ParamId, ParamId)>,  // (x, y) per control point
    weights: Vec<ParamId>,
    knots: Vec<f64>,  // NOT a ParamId — knots are structural, not solvable
    degree: usize,
    // Cache: all param IDs flattened for the trait method
    all_params: Vec<ParamId>,
}

impl Entity for NurbsCurve2D {
    fn id(&self) -> EntityId { self.id }
    fn params(&self) -> &[ParamId] { &self.all_params }
    fn name(&self) -> &str { "NurbsCurve2D" }
}

impl NurbsCurve2D {
    /// Evaluate curve position at parameter t. Uses curvo internally.
    pub fn evaluate(&self, t: f64, store: &ParamStore) -> (f64, f64) {
        // Build control points from ParamStore
        // Call curvo::NurbsCurve::point_at(t)
    }

    /// Evaluate curve tangent at parameter t.
    pub fn tangent(&self, t: f64, store: &ParamStore) -> (f64, f64) { ... }

    /// Basis function values at t (needed for Jacobians).
    pub fn basis_at(&self, t: f64) -> Vec<f64> { ... }
}
```

### 2. Constraints that use geometry internally

```rust
// In the geometry crate (NOT in solverang)
pub struct PointOnNurbsCurve {
    id: ConstraintId,
    point: EntityId,
    curve: EntityId,
    // Direct param references (cached at construction)
    px: ParamId, py: ParamId,
    t: ParamId,  // Curve parameter — an implicit solver variable
    // Reference to curve for evaluation
    curve_ref: Arc<NurbsCurve2D>,
    all_params: Vec<ParamId>,
}

impl Constraint for PointOnNurbsCurve {
    fn equation_count(&self) -> usize { 2 }  // C(t) - P = 0 in 2D

    fn residuals(&self, store: &ParamStore) -> Vec<f64> {
        let px = store.get(self.px);
        let py = store.get(self.py);
        let t = store.get(self.t);
        let (cx, cy) = self.curve_ref.evaluate(t, store);
        vec![cx - px, cy - py]
    }

    fn jacobian(&self, store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
        let t = store.get(self.t);
        let (tx, ty) = self.curve_ref.tangent(t, store);
        let basis = self.curve_ref.basis_at(t);

        let mut entries = vec![
            // dR/d(px), dR/d(py) — point params
            (0, self.px, -1.0),
            (1, self.py, -1.0),
            // dR/d(t) — curve parameter
            (0, self.t, tx),
            (1, self.t, ty),
        ];

        // dR/d(control_point_i) — basis function contributions
        for (i, &(cpx, cpy)) in self.curve_ref.control_points.iter().enumerate() {
            let b = basis[i];
            entries.push((0, cpx, b));   // dCx/d(cpx_i) = N_i(t)
            entries.push((1, cpy, b));   // dCy/d(cpy_i) = N_i(t)
        }

        entries
    }

    fn entity_ids(&self) -> &[EntityId] { &[self.point, self.curve] }
    fn param_ids(&self) -> &[ParamId] { &self.all_params }
    fn id(&self) -> ConstraintId { self.id }
    fn name(&self) -> &str { "PointOnNurbsCurve" }
}
```

### 3. The solver sees none of this

From the solver's perspective, `PointOnNurbsCurve` is just "a constraint with
2 equations, N parameters, and these Jacobian entries." It doesn't know about
NURBS, basis functions, or curvo. The decomposition, reduction, and solving
all work identically.

### What makes this possible

Three design decisions enable the geometry kernel to slot in:

**A. Jacobian returns `(row, ParamId, value)`, not `(row, col, value)`.** The
constraint doesn't need to know the column ordering. The solver's `SolverMapping`
handles it. This means a NURBS constraint with 40 control point params works
the same way as a distance constraint with 4 point params.

**B. Implicit parameters (like curve parameter t) are just ParamIds.** The solver
doesn't distinguish between "geometric" params (point coordinates) and "implicit"
params (curve parameters). They're all variables. The geometry layer creates the
t parameter, adds it to the ParamStore, and includes it in the constraint's
`param_ids()`. The solver optimizes it like any other variable.

**C. Entity is just "a group of ParamIds."** The solver doesn't need to
evaluate geometry. It doesn't call `curve.evaluate(t)`. Only the constraint
implementations call geometry methods, and they do it inside `residuals()` and
`jacobian()` — which are opaque to the solver.

---

## What Changes in Existing Code

### `Problem` trait: STAYS as the low-level API

```
BEFORE: Problem is the only public abstraction.
        Users implement it directly for custom problems.

AFTER:  Problem stays as the raw solver interface.
        ReducedSubProblem implements Problem internally.
        Advanced users can still implement Problem directly.
        ConstraintSystem is the high-level API built on top.
```

The `Problem` trait becomes solverang's "escape hatch" — if the Entity/Constraint
system doesn't fit your domain, you can still implement `Problem` directly and use
the solvers. NIST test problems continue to implement `Problem` directly. The
`ConstraintSystem` produces `ReducedSubProblem`s that implement `Problem`.

### Decomposition: WRAPPED

The existing `decomposition.rs` (union-find, `Component`, `SubProblem`, `DecomposableProblem`)
stays. The new `graph/decompose.rs` wraps it to work with the new ID types:

```rust
// graph/decompose.rs
pub fn decompose_clusters(graph: &ConstraintGraph, store: &ParamStore) -> Vec<RigidCluster> {
    // Build edge list from ConstraintGraph
    let edges = graph.to_constraint_variable_edges(store);
    // Delegate to existing decompose_from_edges
    let components = crate::decomposition::decompose_from_edges(
        graph.constraint_count(),
        store.free_param_count(),
        &edges,
    );
    // Convert Component -> RigidCluster (richer type with cached state)
    components.into_iter().map(|c| RigidCluster::from_component(c, graph)).collect()
}
```

### Parallel solver: ADAPTED

The existing `ParallelSolver` continues to work for `DecomposableProblem`. The
new `ConstraintSystem` uses it internally:

```rust
// system.rs (internal)
fn solve_cluster(&self, cluster: &RigidCluster) -> ClusterResult {
    let sub = self.build_reduced_sub_problem(cluster);
    // sub implements Problem — existing solvers work directly
    let x0 = cluster.warm_start_or(&sub.initial_point());
    let solver = LMSolver::new(self.config.lm_config.clone());
    solver.solve(&sub, &x0)
}
```

### Existing geometry module: MOVES to feature-gated plugin

```
BEFORE: solverang::geometry::{Point2D, ConstraintSystem<D>, ...}

AFTER:  solverang::sketch2d::{Point2D, LineSegment2D, Circle2D, ...}
        (feature = "sketch2d", enabled by default)
        These implement Entity and Constraint from the core.
```

The old `ConstraintSystem<D>` is replaced by the new geometry-agnostic
`ConstraintSystem`. The old Point/Line/Circle types become entity implementations
in the `sketch2d` module. The old constraint implementations get rewritten with
squared formulations and `ParamId`-based APIs.

---

## ReducedSubProblem: The Bridge

This is the crucial adapter that makes existing solvers work with the new system.

```rust
/// A reduced sub-problem for a single cluster, ready for numerical solving.
/// Implements the existing Problem trait so all existing solvers work unchanged.
pub(crate) struct ReducedSubProblem<'a> {
    store: &'a ParamStore,
    mapping: SolverMapping,      // Only free params in this cluster
    constraints: Vec<&'a dyn Constraint>,
    initial_values: Vec<f64>,
}

impl Problem for ReducedSubProblem<'_> {
    fn name(&self) -> &str { "cluster" }

    fn variable_count(&self) -> usize {
        self.mapping.col_to_param.len()
    }

    fn residual_count(&self) -> usize {
        self.constraints.iter().map(|c| c.equation_count()).sum()
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        // Write x into a temporary snapshot of the param store
        let mut snapshot = self.store.snapshot();
        for (col, &param) in self.mapping.col_to_param.iter().enumerate() {
            snapshot.set(param, x[col]);
        }

        // Evaluate all constraints
        let mut residuals = Vec::new();
        for constraint in &self.constraints {
            residuals.extend(constraint.residuals(&snapshot));
        }
        residuals
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        let mut snapshot = self.store.snapshot();
        for (col, &param) in self.mapping.col_to_param.iter().enumerate() {
            snapshot.set(param, x[col]);
        }

        let mut entries = Vec::new();
        let mut row_offset = 0;
        for constraint in &self.constraints {
            for (local_row, param_id, value) in constraint.jacobian(&snapshot) {
                // Map ParamId -> column index (skip if param is fixed/not in this cluster)
                if let Some(&col) = self.mapping.param_to_col.get(&param_id) {
                    entries.push((row_offset + local_row, col, value));
                }
            }
            row_offset += constraint.equation_count();
        }
        entries
    }

    fn initial_point(&self, _factor: f64) -> Vec<f64> {
        self.initial_values.clone()
    }
}
```

This means: **every existing solver (NR, LM, sparse, JIT, parallel) works
without modification.** They receive a `ReducedSubProblem` which implements the
`Problem` trait they already understand.

---

## SystemResult: Rich Diagnostics

```rust
/// Result of solving the full constraint system.
pub struct SystemResult {
    /// Overall status
    pub status: SystemStatus,
    /// Per-cluster results
    pub clusters: Vec<ClusterResult>,
    /// Total iterations across all clusters
    pub total_iterations: usize,
    /// Wall-clock time
    pub duration: std::time::Duration,
}

pub enum SystemStatus {
    /// All clusters converged.
    Solved,
    /// Some clusters converged, some didn't.
    PartiallySolved,
    /// Structural issues detected before solving.
    DiagnosticFailure(Vec<DiagnosticIssue>),
}

pub struct ClusterResult {
    pub cluster_id: ClusterId,
    pub status: ClusterStatus,
    pub iterations: usize,
    pub residual_norm: f64,
}

pub enum ClusterStatus {
    Converged,
    NotConverged,
    Redundant(Vec<ConstraintId>),
    Conflicting(Vec<ConstraintId>),
    Skipped,  // Not dirty, reused cached solution
}

pub enum DiagnosticIssue {
    RedundantConstraint {
        constraint: ConstraintId,
        implied_by: Vec<ConstraintId>,
    },
    ConflictingConstraints {
        constraints: Vec<ConstraintId>,
    },
    UnderConstrained {
        entity: EntityId,
        free_directions: usize,
    },
}
```

---

## Implementation Order

### Phase 1: Skeleton (1-2 weeks)

Build the type system. No solving yet — just the data structures.

```
id.rs, param/store.rs, entity/mod.rs, constraint/mod.rs
graph/bipartite.rs, graph/cluster.rs
```

Test: create entities, add constraints, build graph, verify connectivity.

### Phase 2: Bridge (1 week)

Connect new types to existing solvers.

```
solve/sub_problem.rs (ReducedSubProblem implements Problem)
graph/decompose.rs (wraps existing union-find)
system.rs (basic coordinator: add entities/constraints, solve)
```

Test: solve a hand-built system of two DistanceConstraint-like structs.
Verify existing NR and LM produce correct results through the new bridge.

### Phase 3: Sketch2D basics (1-2 weeks)

Port the most important geometry types with squared formulations.

```
sketch2d/entities.rs: Point2D, LineSegment2D, Circle2D
sketch2d/constraints.rs: Distance(Pt-Pt), Coincident, Fixed,
    Horizontal, Vertical, Parallel, Perpendicular, Tangent(Line-Circle)
sketch2d/builder.rs: ergonomic API
```

Test: solve all existing geometric test cases through new architecture.
Regression: triangle, rectangle, circle-tangent problems.

### Phase 4: Symbolic Reduction (1-2 weeks)

```
reduce/substitute.rs, reduce/merge.rs, reduce/eliminate.rs
```

Test: verify reduction produces smaller systems with same solutions.
Benchmark: reduced system solves faster than unreduced.

### Phase 5: Diagnostics (1 week)

```
graph/redundancy.rs, graph/dof.rs
```

Test: detect known-redundant and known-conflicting configurations.

### Phase 6: Incremental Solving (1-2 weeks)

```
dataflow/tracker.rs, dataflow/cache.rs
```

Test: change one param, verify only one cluster re-solves.
Benchmark: incremental vs full re-solve on 100+ entity system.

### Phase 7: Remaining Sketch2D (1 week)

```
Arc2D, Ellipse2D, InfiniteLine2D entities
Angle, Midpoint, Symmetric, EqualLength, Collinear, OnEntity constraints
```

### Phase 8: Sketch3D + Assembly (2 weeks)

```
sketch3d/: Point3D, LineSegment3D, Plane, Axis3D
assembly/: RigidBody (quaternion), Mate, Coaxial
```

### Phase 9: Advanced Solving (2 weeks)

```
solve/drag.rs, solve/branch.rs
graph/pattern.rs, solve/closed_form.rs
```

### Phase 10: Publish Prep (1 week)

API review, documentation, examples, changelog, version bump.

**Total: ~14-18 weeks for a publishable cutting-edge constraint solver.**

---

## Comparison: V1 (Today) vs V3 (This Plan)

| Aspect | V1 | V3 |
|---|---|---|
| Public API | `Problem` trait (implement it yourself) | `ConstraintSystem` + `Entity`/`Constraint` traits |
| Low-level escape hatch | N/A (it's the only API) | `Problem` trait still available |
| Entity model | Points only (`Point<D>`) | Any entity with `ParamId`s |
| Constraint binding | `&[Point<D>]` (point array) | `&ParamStore` (param values) |
| Jacobian format | `(row, col, value)` | `(row, ParamId, value)` (constraint) → `(row, col, value)` (system maps it) |
| Geometry awareness | Hard-coded in constraint trait | Zero geometry in solver core |
| NURBS/spline support | Impossible | Implement Entity + Constraint |
| Variable radii | Hack (`PointOnCircleVariableRadiusConstraint`) | Natural (`Circle2D.radius` is a `ParamId`) |
| Decomposition | Union-find (good) | Union-find + incremental tracking |
| Symbolic reduction | None | Substitute + merge + eliminate |
| Diagnostics | None | Redundancy, conflict, per-entity DOF |
| Incremental solving | Partial (ParallelSolver::solve_incremental) | Full: change tracking → dirty clusters only |
| Drag solving | None | Null-space projection |
| Residual formulations | `sqrt(d^2) - target` | `d^2 - target^2` (polynomial) |
| Extensibility | Implement `Problem` | Implement `Entity` + `Constraint` |

---

## The "Future Geometry Kernel" Extension Surface

When the kernel arrives, it needs exactly three things from the solver:

1. **`ParamStore::alloc()`** — to create ParamIds for control points, radii, etc.
2. **`impl Entity`** — to register NURBS curves, B-Rep edges, etc.
3. **`impl Constraint`** — to produce residuals/Jacobians for geometric relationships.

That's it. The solver exposes these three extension points and the kernel plugs in.

The solver doesn't need:
- A `Curve` trait
- A `Surface` trait
- A `BRep` module
- An `Intersection` algorithm
- A `Tessellation` routine

All of that lives in the kernel crate, which depends on solverang and curvo.

```
geometry-kernel (future crate)
├── depends on: solverang (constraint solver)
├── depends on: curvo (NURBS evaluation)
├── depends on: nalgebra (linear algebra)
├── provides: NurbsCurve, NurbsSurface, BRep, Boolean, Fillet, ...
└── implements: solverang::Entity, solverang::Constraint
```
