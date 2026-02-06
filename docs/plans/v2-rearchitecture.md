# Solverang v2: Ground-Up Rearchitecture Plan

## Executive Summary

This plan replaces the current point-only geometry layer with a **parameter-centric variational solver** that natively supports all 2D and 3D geometric primitives, integrates differential dataflow for incremental constraint graph maintenance, and builds decomposition/deduplication heuristics into the core from day one. No backward compatibility is maintained.

The solver core (Newton-Raphson, Levenberg-Marquardt, sparse, parallel) is **kept intact** — it is well-tested against MINPACK/NIST benchmarks and does not need rewriting. The rearchitecture targets the geometry layer above it: the data model, entity system, constraint trait, graph analysis, and solve pipeline.

---

## 1. Data Model: Flat Parameter Vector

### 1.1 Current Problem

The current `ConstraintSystem<D>` stores `Vec<Point<D>>` with `Vec<bool>` for fixed/free. All non-point values (radii, angles, weights) are compile-time constants baked into constraint structs. This makes it impossible for the solver to optimize a circle's radius or an arc's sweep angle.

### 1.2 New Design: Everything Is a Parameter

```rust
/// The central data store. All solver variables live in a single flat Vec<f64>.
/// Entity handles provide typed views into parameter ranges.
pub struct ParameterStore {
    /// Flat vector of all solver parameters. This is what gets passed to the solver.
    values: Vec<f64>,
    /// Per-parameter: is this value fixed (driven) or free (solved)?
    fixed: Vec<bool>,
    /// Per-parameter: human-readable label for diagnostics (e.g., "circle_3.radius")
    labels: Vec<String>,
    /// Next free index for allocation.
    next_idx: usize,
}
```

Every geometric quantity — point coordinates, radii, angles, control point positions, knot values — is a solver variable with an index into this flat vector. The solver sees only `&[f64]` and `Vec<(row, col, f64)>`. Entities are zero-cost typed handles:

```rust
/// A contiguous range of parameters belonging to one entity.
#[derive(Clone, Copy, Debug)]
pub struct ParamRange {
    pub start: usize,
    pub count: usize,
}

/// Typed entity handle — zero-cost wrapper providing named access.
#[derive(Clone, Copy, Debug)]
pub struct EntityHandle {
    pub id: EntityId,
    pub kind: EntityKind,
    pub params: ParamRange,
}
```

### 1.3 Variable Indexing

No more `point_idx * D + coord` gymnastics with PARAM_COL_BASE sentinels. The Jacobian column index **is** the parameter index. Period.

```rust
// Old: col = point_idx * D + coord (only points are variables)
// New: col = param_store_index (anything can be a variable)

// A circle's center.x is just params.start + 0
// A circle's center.y is just params.start + 1
// A circle's radius is just params.start + 2
```

### 1.4 Fixed vs. Free

Any parameter can be independently fixed or freed. Fixing a point means fixing its D coordinate parameters. Fixing a circle's center but freeing its radius is just `fixed[cx] = true; fixed[cy] = true; fixed[r] = false`. The system builds a `free_indices: Vec<usize>` map at solve time to compress the variable vector for the solver.

---

## 2. Entity Registry

### 2.1 EntityKind Enum

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EntityKind {
    // === 2D Primitives ===
    Point2D,          // params: [x, y]                           — 2 params
    Line2D,           // params: [x1, y1, x2, y2]                — 4 params (two endpoints)
    Circle2D,         // params: [cx, cy, r]                      — 3 params
    Arc2D,            // params: [cx, cy, r, start_angle, end_angle] — 5 params
    Ellipse2D,        // params: [cx, cy, rx, ry, rotation]       — 5 params
    EllipticArc2D,    // params: [cx, cy, rx, ry, rot, t0, t1]   — 7 params
    CubicBezier2D,    // params: [x0,y0, x1,y1, x2,y2, x3,y3]   — 8 params
    QuadBezier2D,     // params: [x0,y0, x1,y1, x2,y2]           — 6 params
    BSpline2D,        // params: control_points ++ knots           — variable
    // Composite / construction
    Polyline2D,       // params: [x0,y0, ..., xn,yn]             — 2*(n+1) params
    Offset2D,         // params: [distance]                       — 1 param (+ ref to base curve)

    // === 3D Primitives ===
    Point3D,          // params: [x, y, z]                        — 3 params
    Line3D,           // params: [x1,y1,z1, x2,y2,z2]            — 6 params
    Circle3D,         // params: [cx,cy,cz, nx,ny,nz, r]         — 7 params (center + normal + radius)
    Arc3D,            // params: [cx,cy,cz, nx,ny,nz, r, t0, t1] — 9 params
    Sphere,           // params: [cx,cy,cz, r]                    — 4 params
    Cylinder,         // params: [px,py,pz, dx,dy,dz, r]         — 7 params (axis point, dir, radius)
    Cone,             // params: [px,py,pz, dx,dy,dz, half_angle] — 7 params
    Torus,            // params: [cx,cy,cz, nx,ny,nz, R, r]      — 8 params
    Plane,            // params: [px,py,pz, nx,ny,nz]             — 6 params (point + normal)
    Ellipse3D,        // params: [cx,cy,cz, major_x,y,z, minor_x,y,z] — 9 params
    CubicBezier3D,    // params: [x0..z0, x1..z1, x2..z2, x3..z3] — 12 params
    BSpline3D,        // params: control_points ++ knots           — variable
    // NURBS is BSpline + weights as parameters
    NurbsCurve2D,     // params: control_points ++ weights ++ knots — variable
    NurbsCurve3D,     // params: control_points ++ weights ++ knots — variable
    NurbsSurface,     // params: control_net ++ weights ++ knots_u ++ knots_v — variable

    // === Auxiliary / Internal ===
    Scalar,           // params: [value]                          — 1 param (for auxiliary t, angle, etc.)
}
```

### 2.2 Entity Metadata

Each entity knows how to:
1. Report its parameter count
2. Evaluate a point at parameter t (for curves) or (u,v) (for surfaces)
3. Evaluate tangent/normal at t
4. Provide a bounding box estimate for spatial queries

```rust
pub trait EntityEvaluator {
    /// Number of parameters this entity occupies in the store.
    fn param_count(&self) -> usize;

    /// Evaluate position on curve at parameter t ∈ [0,1].
    /// Returns None for non-curve entities (points, planes).
    fn evaluate_at(&self, params: &[f64], t: f64) -> Option<[f64; 3]>;

    /// Evaluate tangent vector at parameter t.
    fn tangent_at(&self, params: &[f64], t: f64) -> Option<[f64; 3]>;

    /// Evaluate curvature scalar at parameter t.
    fn curvature_at(&self, params: &[f64], t: f64) -> Option<f64>;

    /// Axis-aligned bounding box: (min, max).
    fn bounds(&self, params: &[f64]) -> ([f64; 3], [f64; 3]);
}
```

### 2.3 Entity Allocation

```rust
impl ParameterStore {
    /// Allocate a new entity. Returns a typed handle.
    pub fn add_entity(&mut self, kind: EntityKind, initial_values: &[f64]) -> EntityHandle {
        let id = EntityId(self.next_entity_id);
        self.next_entity_id += 1;

        let start = self.next_idx;
        let count = initial_values.len();

        self.values.extend_from_slice(initial_values);
        self.fixed.extend(std::iter::repeat(false).take(count));
        self.labels.extend(
            (0..count).map(|i| format!("{}_{}.p{}", kind.name(), id.0, i))
        );
        self.next_idx += count;

        let handle = EntityHandle {
            id,
            kind,
            params: ParamRange { start, count },
        };
        self.entities.insert(id, handle);
        handle
    }

    /// Convenience: add a 2D point
    pub fn add_point_2d(&mut self, x: f64, y: f64) -> EntityHandle {
        self.add_entity(EntityKind::Point2D, &[x, y])
    }

    /// Convenience: add a 2D circle
    pub fn add_circle_2d(&mut self, cx: f64, cy: f64, r: f64) -> EntityHandle {
        self.add_entity(EntityKind::Circle2D, &[cx, cy, r])
    }
}
```

---

## 3. Constraint Trait: Rearchitected

### 3.1 New Trait

```rust
/// A geometric constraint operating on the flat parameter vector.
///
/// Constraints no longer receive `&[Point<D>]` — they receive `&[f64]`
/// (the full parameter store) and know which indices they care about.
pub trait Constraint: Send + Sync {
    /// Unique identifier for graph operations.
    fn id(&self) -> ConstraintId;

    /// Human-readable name.
    fn name(&self) -> &'static str;

    /// Number of scalar equations this constraint produces.
    fn equation_count(&self) -> usize;

    /// Which parameter indices this constraint reads.
    /// Used for graph construction and decomposition.
    fn dependencies(&self) -> &[usize];

    /// Evaluate residuals. `params` is the FULL parameter vector.
    /// Returns exactly `equation_count()` values.
    fn residuals(&self, params: &[f64]) -> Vec<f64>;

    /// Sparse Jacobian: (local_row, global_col, value).
    /// `local_row` ∈ 0..equation_count(), `global_col` is a parameter index.
    fn jacobian(&self, params: &[f64]) -> Vec<(usize, usize, f64)>;

    /// Whether this is a soft/preference constraint (minimized, not zeroed).
    fn is_soft(&self) -> bool { false }

    /// Priority weight for weighted least-squares.
    fn weight(&self) -> f64 { 1.0 }

    /// Estimated difficulty: helps solver selection heuristic.
    fn nonlinearity_hint(&self) -> Nonlinearity { Nonlinearity::Moderate }
}

#[derive(Clone, Copy, Debug)]
pub enum Nonlinearity {
    /// Linear constraint (e.g., horizontal, vertical, coincident).
    /// Can be solved in one NR step.
    Linear,
    /// Quadratic or mildly nonlinear (e.g., distance, point-on-circle).
    Moderate,
    /// Highly nonlinear (e.g., tangent, curvature continuity, angle).
    /// May need LM or careful initialization.
    High,
}
```

### 3.2 Key Differences from Current Design

| Aspect | Current | v2 |
|--------|---------|-----|
| Input | `&[Point<D>]` | `&[f64]` (full param vector) |
| Column indices | `point_idx * D + coord` | Direct parameter index |
| Non-point values | Compile-time constants | Solver variables |
| Dependency tracking | `variable_indices() -> Vec<usize>` (point indices) | `dependencies() -> &[usize]` (parameter indices) |
| Dimensionality | `GeometricConstraint<D>` (const generic) | `Constraint` (dimension-agnostic) |
| Solver hints | None | `nonlinearity_hint()` |

### 3.3 Dropping Const Generic D

The current `<const D: usize>` pattern forces every constraint to be parameterized, creates duplicate code paths for 2D/3D, and prevents mixing 2D and 3D entities in one system (which is needed for e.g., projecting 3D geometry onto a 2D sketch plane). In v2:

- Entities carry their own dimensionality via `EntityKind`.
- Constraints reference parameter indices directly — they don't care if those parameters form a 2D point or a 3D point.
- Type safety is maintained at the entity/builder level, not at the constraint trait level.

---

## 4. Full Constraint Catalog

### 4.1 Point-Point Constraints (existing, ported)

| Constraint | Equations | Description | Nonlinearity |
|-----------|-----------|-------------|-------------|
| `Distance` | 1 | \|p1-p2\| = d | Moderate |
| `Coincident` | D | p1 = p2 (per-coordinate) | Linear |
| `Fixed` | D | p = target | Linear |
| `Horizontal` | 1 | p1.y = p2.y | Linear |
| `Vertical` | 1 | p1.x = p2.x | Linear |
| `Midpoint` | D | m = (p1+p2)/2 | Linear |
| `Symmetric` | D | p1+p2 = 2*center | Linear |
| `SymmetricAboutLine` | D | reflection across axis | Moderate |

### 4.2 Line Constraints (existing, ported)

| Constraint | Equations | Description | Nonlinearity |
|-----------|-----------|-------------|-------------|
| `Parallel` | 1 (2D), 2 (3D) | cross product = 0 | Moderate |
| `Perpendicular` | 1 | dot product = 0 | Moderate |
| `Collinear` | 2 (2D), 4 (3D) | segments on same line | Moderate |
| `EqualLength` | 1 | \|L1\| = \|L2\| | Moderate |
| `Angle` | 1 | angle between lines = θ | High |
| `PointOnLine` | 1 (2D), 2 (3D) | point lies on line | Linear |

### 4.3 Circle/Arc Constraints (new)

| Constraint | Equations | Description | Nonlinearity |
|-----------|-----------|-------------|-------------|
| `PointOnCircle` | 1 | \|p-c\|² = r² (r is now a variable) | Moderate |
| `EqualRadius` | 1 | r1 = r2 | Linear |
| `ConcentricCircles` | D | c1 = c2 | Linear |
| `LineTangentCircle` | 1 | perpendicular dist from center to line = r | High |
| `CircleTangentCircle` | 1 | \|c1-c2\| = r1±r2 | High |
| `ArcEndpoint` | D | point at arc start/end matches arc evaluation | Moderate |
| `ArcSweep` | 1 | θ_end - θ_start = sweep_angle | Linear |
| `ArcCenter` | D | arc center = point | Linear |
| `PointOnArc` | 1 | \|p-c\|² = r² AND angle in range | High |

### 4.4 Ellipse Constraints (new)

| Constraint | Equations | Description | Nonlinearity |
|-----------|-----------|-------------|-------------|
| `PointOnEllipse` | 1 | ((x-cx)cos(r)+(y-cy)sin(r))²/a² + ... = 1 | High |
| `EllipseFociDistance` | 1 | sum of distances to foci = 2a | High |
| `EllipseEccentricity` | 1 | e = sqrt(1 - b²/a²) = target | Moderate |

### 4.5 Bezier/Spline Constraints (new)

| Constraint | Equations | Description | Nonlinearity |
|-----------|-----------|-------------|-------------|
| `PointOnBezier` | D | p = B(t), t is auxiliary variable | High |
| `BezierEndpoint` | D | P0 or P3 of bezier = target point | Linear |
| `G0Continuity` | D | end of curve_a = start of curve_b | Linear |
| `G1Continuity` | D | tangent vectors collinear at junction | High |
| `G2Continuity` | 1 | curvature equal at junction | High |
| `BezierTangentAtEnd` | D-1 | tangent direction at t=0 or t=1 = target | Moderate |
| `BSplinePointOn` | D | p = S(t), t auxiliary | High |
| `NurbsPointOn` | D | p = N(t) with rational weights | High |
| `NurbsWeightPositive` | 1 | w_i > 0 (inequality via slack) | Linear |

### 4.6 3D-Specific Constraints (new)

| Constraint | Equations | Description | Nonlinearity |
|-----------|-----------|-------------|-------------|
| `PointOnPlane` | 1 | n·(p-p0) = 0 | Linear |
| `PointOnSphere` | 1 | \|p-c\|² = r² | Moderate |
| `PointOnCylinder` | 1 | distance from axis = r | Moderate |
| `PointOnCone` | 1 | angle from axis at apex = half_angle | High |
| `PointOnTorus` | 1 | torus implicit equation | High |
| `PlaneParallel` | 2 | n1 × n2 = 0 | Moderate |
| `PlanePerpendicular` | 1 | n1 · n2 = 0 | Moderate |
| `PlaneDistance` | 1 | distance between parallel planes = d | Linear |
| `Coplanar` | 1 | point lies on plane | Linear |
| `LineOnPlane` | 2 | both endpoints on plane | Linear |
| `NormalToPlane` | 2 | line direction = plane normal | Moderate |
| `CylinderRadius` | 1 | cylinder.r = target | Linear |
| `ConcentricCylinders` | 4 | same axis | Moderate |

### 4.7 Cross-Dimension Constraints (new)

| Constraint | Equations | Description | Nonlinearity |
|-----------|-----------|-------------|-------------|
| `ProjectToPlane` | D | project 3D entity onto sketch plane | Linear |
| `LiftFromSketch` | 1 | 2D sketch point = 3D point projected | Linear |

### 4.8 Meta-Constraints (new)

| Constraint | Equations | Description | Nonlinearity |
|-----------|-----------|-------------|-------------|
| `FixedParam` | 1 | single parameter = constant | Linear |
| `EqualParam` | 1 | param_a = param_b | Linear |
| `ParamRange` | 0-2 | low ≤ param ≤ high (via slack variables) | Linear |
| `RatioParam` | 1 | param_a = k * param_b | Linear |

---

## 5. Constraint Graph with Differential Dataflow

### 5.1 Architecture

The constraint graph is the central data structure that the DD layer operates on. It is a bipartite graph between **entities** (which own parameters) and **constraints** (which reference parameters). The DD layer provides incremental maintenance of:

1. **Connected components** — which entities and constraints form independent subsystems
2. **DOF analysis** — per-component degrees of freedom
3. **Dirty tracking** — which components need re-solving after an edit
4. **Redundancy detection** — structurally redundant constraints

```
┌──────────────────────────────────────────────────────────────────┐
│                      User / Editor API                           │
│  add_entity(), remove_entity(), add_constraint(), move_point()   │
└────────────────────────────┬─────────────────────────────────────┘
                             │ edit operations as (entity, +1/-1)
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│              Differential Dataflow Layer                          │
│                                                                  │
│  Input Collections:                                              │
│    entities: Collection<(EntityId, EntityKind, ParamRange)>      │
│    constraints: Collection<(ConstraintId, Vec<ParamIdx>)>        │
│    param_fixed: Collection<(ParamIdx, bool)>                     │
│                                                                  │
│  Derived Collections (incrementally maintained):                 │
│    entity_constraint_edges: Collection<(EntityId, ConstraintId)> │
│    components: Collection<(EntityId, ComponentId)>               │
│    component_dof: Collection<(ComponentId, i32)>                 │
│    dirty_components: Collection<ComponentId>                     │
│    redundant_constraints: Collection<ConstraintId>               │
└────────────────────────────┬─────────────────────────────────────┘
                             │ components + dirty set
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                     Solve Pipeline                                │
│  For each dirty component:                                       │
│    1. Extract sub-problem                                        │
│    2. Select solver                                              │
│    3. Solve                                                      │
│    4. Write back solution                                        │
└──────────────────────────────────────────────────────────────────┘
```

### 5.2 Why DD and Not Just BFS

The current BFS approach recomputes all components from scratch on every edit. For small drawings (< 100 entities), this is fine. For large parametric models (1000+ entities, which is common in MCAD), the cost is O(E+V) per keystroke. DD reduces this to O(Δ) — proportional to what actually changed.

More importantly, DD composes: adding new derived analyses (DOF, redundancy, over-constraint detection, structural rank) is just adding new dataflow operators. With BFS, each analysis is a separate pass.

### 5.3 DD Implementation Strategy

Rather than using the `differential-dataflow` crate (which has a heavy runtime and complex API), we implement a **lightweight DD core** tailored to our graph operations:

```rust
/// Lightweight incremental graph engine.
/// Tracks a bipartite (entity, constraint) graph with change propagation.
pub struct IncrementalGraph {
    // --- Input state ---
    entities: HashMap<EntityId, EntityMeta>,
    constraints: HashMap<ConstraintId, ConstraintMeta>,
    /// entity → set of constraints referencing it
    entity_to_constraints: HashMap<EntityId, HashSet<ConstraintId>>,
    /// constraint → set of entities it references
    constraint_to_entities: HashMap<ConstraintId, HashSet<EntityId>>,

    // --- Derived state (incrementally maintained) ---
    /// entity → component ID (via incremental union-find)
    components: IncrementalUnionFind,
    /// component → DOF count
    component_dof: HashMap<ComponentId, i32>,
    /// components that need re-solving
    dirty: HashSet<ComponentId>,

    /// Monotonic version counter for change tracking.
    version: u64,
}
```

The key insight: our graph is sparse and changes are local (add/remove one entity or constraint at a time). We don't need the full generality of DD's lattice-based timestamps. We need:

1. **Incremental union-find** with split support (for component detection)
2. **Delta propagation** (when a constraint is added/removed, only affected components are re-analyzed)
3. **Dirty marking** (which components need re-solving)

### 5.4 Incremental Union-Find with Rollback

Standard union-find doesn't support splits (removing an edge can split a component). We use **link-cut trees** or, more practically, **epoch-based rebuild**: maintain a version counter, and when a removal happens, rebuild only the affected component(s).

```rust
pub struct IncrementalUnionFind {
    /// Standard union-find for fast queries
    uf: UnionFind,
    /// Edge list for rebuild-on-removal
    edges: Vec<(EntityId, EntityId)>,
    /// Version at which each component was last validated
    component_versions: HashMap<ComponentId, u64>,
    /// Current version
    version: u64,
}

impl IncrementalUnionFind {
    /// Add an edge (from shared constraint). O(α(n)) amortized.
    pub fn add_edge(&mut self, a: EntityId, b: EntityId) {
        self.edges.push((a, b));
        self.uf.union(a, b);
        self.version += 1;
    }

    /// Remove an edge. Rebuilds affected component if it might split.
    /// Worst case O(component_size), but only for the affected component.
    pub fn remove_edge(&mut self, a: EntityId, b: EntityId) {
        self.edges.retain(|&(x, y)| !(x == a && y == b) && !(x == b && y == a));
        // Rebuild only the component that contained a and b
        let old_root = self.uf.find(a);
        self.rebuild_component(old_root);
        self.version += 1;
    }
}
```

### 5.5 Change Propagation

```rust
impl IncrementalGraph {
    pub fn add_constraint(&mut self, id: ConstraintId, meta: ConstraintMeta) {
        let entities = meta.entity_deps.clone();
        self.constraints.insert(id, meta);

        for &eid in &entities {
            self.entity_to_constraints.entry(eid).or_default().insert(id);
            self.constraint_to_entities.entry(id).or_default().insert(eid);
        }

        // Union all entities referenced by this constraint
        if entities.len() >= 2 {
            for window in entities.windows(2) {
                self.components.add_edge(window[0], window[1]);
            }
        }

        // Mark affected component dirty
        if let Some(&eid) = entities.first() {
            let comp = self.components.find(eid);
            self.dirty.insert(comp);
            self.update_component_dof(comp);
        }

        self.version += 1;
    }

    pub fn remove_constraint(&mut self, id: ConstraintId) {
        if let Some(meta) = self.constraints.remove(&id) {
            let entities = meta.entity_deps;

            // Record which component(s) are affected BEFORE removal
            let affected_component = entities.first()
                .map(|&eid| self.components.find(eid));

            // Remove edges
            for &eid in &entities {
                self.entity_to_constraints.entry(eid).or_default().remove(&id);
            }
            self.constraint_to_entities.remove(&id);

            // Remove from union-find (may cause component split)
            for window in entities.windows(2) {
                self.components.remove_edge(window[0], window[1]);
            }

            // Mark new component(s) dirty
            for &eid in &entities {
                let comp = self.components.find(eid);
                self.dirty.insert(comp);
                self.update_component_dof(comp);
            }

            self.version += 1;
        }
    }

    pub fn move_parameter(&mut self, param_idx: usize, new_value: f64) {
        // Parameter change doesn't alter graph topology.
        // Just mark the owning entity's component as dirty.
        if let Some(entity_id) = self.param_to_entity(param_idx) {
            let comp = self.components.find(entity_id);
            self.dirty.insert(comp);
        }
    }

    /// Returns components that need re-solving, clearing the dirty set.
    pub fn take_dirty(&mut self) -> Vec<ComponentId> {
        self.dirty.drain().collect()
    }
}
```

---

## 6. Decomposition & Structural Analysis

### 6.1 Three Levels of Decomposition

The solve pipeline applies decomposition at three levels, from coarsest to finest:

```
Level 1: Connected Components (DD-maintained)
  Split independent subsystems. A sketch with two disconnected shapes
  becomes two independent solve problems.

Level 2: Dulmage-Mendelsohn (DM) Decomposition
  Within a connected component, find the structural block-triangular form.
  Identifies: well-constrained core, under-constrained parts, over-constrained parts.
  Allows solving blocks in topological order.

Level 3: Rigid Cluster Decomposition (DR-planner)
  Within a DM block, identify rigid sub-assemblies (triangles, rectangles)
  that can be solved independently and then assembled.
  This is the D-Cubed / Bettig-Hoffmann approach.
```

### 6.2 Dulmage-Mendelsohn Implementation

DM decomposition operates on the constraint-variable bipartite graph:

```rust
pub struct DMDecomposition {
    /// Under-determined part: more variables than constraints.
    /// These variables have degrees of freedom.
    pub under: DMBlock,
    /// Well-determined part: square blocks in topological order.
    /// Each block can be solved independently once its predecessors are solved.
    pub well: Vec<DMBlock>,
    /// Over-determined part: more constraints than variables.
    /// Indicates redundancy or inconsistency.
    pub over: DMBlock,
}

pub struct DMBlock {
    pub constraint_indices: Vec<usize>,
    pub variable_indices: Vec<usize>,
}
```

Algorithm:
1. Find maximum matching in bipartite graph (Hopcroft-Karp, O(E√V))
2. Run BFS/DFS on unmatched nodes to identify DM partition
3. Within the well-determined part, find strongly connected components (Tarjan's) to get fine blocks

### 6.3 Rigid Cluster Decomposition (Future Phase)

This is the DR-planner algorithm from Hoffmann et al.:
- Input: a well-constrained 2D geometric constraint system
- Output: a tree of rigid clusters with shared entities at joints
- Each leaf cluster is a small well-constrained subsystem (typically 3-6 entities)
- Solve leaf → propagate through tree → assemble at root

This is complex to implement correctly and is deferred to a later phase, but the data model supports it from day one by making entities and constraints first-class graph nodes.

---

## 7. Redundancy & Conflict Detection

### 7.1 Structural Redundancy

A constraint is **structurally redundant** if removing it doesn't change the structural rank of the Jacobian. The DD layer detects this via DM decomposition: constraints in the over-determined part are redundant.

```rust
impl IncrementalGraph {
    /// Constraints that are structurally redundant.
    /// These make the system over-constrained and may cause solver failure.
    pub fn redundant_constraints(&self, component: ComponentId) -> Vec<ConstraintId> {
        let dm = self.dm_decompose(component);
        dm.over.constraint_indices.iter()
            .filter_map(|&idx| self.constraint_at_index(idx))
            .collect()
    }
}
```

### 7.2 Numerical Redundancy

Two constraints might be structurally independent but numerically equivalent (e.g., distance(A,B)=5 and distance(B,A)=5). Detect by checking if the Jacobian rows are linearly dependent:

```rust
/// Check if constraint produces a Jacobian row that is a linear combination
/// of other constraint Jacobian rows in the same component.
pub fn detect_numerical_redundancy(
    constraints: &[&dyn Constraint],
    params: &[f64],
    tolerance: f64,
) -> Vec<ConstraintId> {
    // Build Jacobian matrix, compute rank via SVD or QR with column pivoting.
    // Constraints corresponding to zero/near-zero singular values are redundant.
    // ...
}
```

### 7.3 Conflict Detection

An over-constrained system where constraints are incompatible produces a conflict. After DM decomposition identifies the over-determined part, we attempt to solve it and check residual norms:

- If over-determined block solves to near-zero residuals → constraints are compatible (redundant but consistent)
- If residuals remain large → **conflict**: report the offending constraints to the user

### 7.4 Duplicate Constraint Deduplication

Before adding a constraint to the graph, check for exact duplicates:

```rust
impl IncrementalGraph {
    pub fn add_constraint_deduped(&mut self, constraint: Box<dyn Constraint>) -> AddResult {
        let deps = constraint.dependencies().to_vec();
        let name = constraint.name();

        // Check for exact duplicate (same type, same dependencies)
        for existing in self.constraints_on_entities(&deps) {
            if existing.name() == name && existing.dependencies() == deps.as_slice() {
                return AddResult::Duplicate(existing.id());
            }
        }

        // Check for semantic duplicate (e.g., distance(A,B) and distance(B,A))
        if let Some(canonical) = self.canonicalize_deps(name, &deps) {
            for existing in self.constraints_on_entities(&canonical) {
                if existing.name() == name {
                    return AddResult::Duplicate(existing.id());
                }
            }
        }

        let id = self.add_constraint(constraint.id(), constraint.into_meta());
        AddResult::Added(id)
    }
}
```

---

## 8. Solver Pipeline

### 8.1 Pipeline Overview

```
User edits parameter(s)
    │
    ▼
DD: mark affected component(s) dirty
    │
    ▼
For each dirty component:
    │
    ├─ 1. Extract sub-problem (variables + constraints for this component)
    │
    ├─ 2. DM decomposition → block-triangular form
    │     ├─ Under-constrained blocks → least-squares (minimize movement)
    │     ├─ Well-constrained blocks → solve in topological order
    │     └─ Over-constrained blocks → detect redundancy/conflict
    │
    ├─ 3. For each well-constrained block:
    │     ├─ Classify: linear-only? moderate? high nonlinearity?
    │     ├─ Select solver:
    │     │   ├─ All linear → single NR step (exact)
    │     │   ├─ Small (≤20 vars) + moderate → NR with LM fallback
    │     │   ├─ Small + high nonlinearity → LM from start
    │     │   ├─ Large (>100 vars) + sparse → SparseSolver
    │     │   └─ Default → AutoSolver cascade
    │     └─ Solve with warm start from current parameter values
    │
    ├─ 4. Write solution back to ParameterStore
    │
    └─ 5. Clear dirty flag for component
```

### 8.2 Solver Selection Heuristic

```rust
pub fn select_solver(block: &SubProblem, params: &[f64]) -> SolverChoice {
    let n_vars = block.variable_count();
    let n_eqs = block.equation_count();
    let max_nonlinearity = block.max_nonlinearity_hint();
    let sparsity = block.jacobian_sparsity_ratio(params);

    match (n_vars, max_nonlinearity, sparsity) {
        // Trivial: 1-2 variables, just use NR
        (0..=2, _, _) => SolverChoice::NewtonRaphson,

        // All linear constraints: one NR step suffices
        (_, Nonlinearity::Linear, _) => SolverChoice::NewtonRaphson,

        // Large and sparse: use sparse solver
        (n, _, s) if n > 100 && s < 0.1 => SolverChoice::Sparse,

        // Moderate nonlinearity, good initial guess likely: NR with LM fallback
        (_, Nonlinearity::Moderate, _) => SolverChoice::Robust, // NR → LM cascade

        // High nonlinearity: start with LM (more robust)
        (_, Nonlinearity::High, _) => SolverChoice::LevenbergMarquardt,
    }
}
```

### 8.3 Warm Starting

The key performance optimization: after the first solve, subsequent solves (triggered by small user edits) start from the previous solution. This is critical for interactive performance.

```rust
impl SolvePipeline {
    pub fn solve_dirty(&mut self) {
        let dirty = self.graph.take_dirty();

        for comp_id in dirty {
            let sub = self.extract_sub_problem(comp_id);

            // Warm start: use current parameter values as initial guess.
            // These are the previous solution (or user-placed initial values).
            let x0: Vec<f64> = sub.variable_indices.iter()
                .map(|&idx| self.params.values[idx])
                .collect();

            let solver = select_solver(&sub, &self.params.values);
            let result = solver.solve(&sub.as_problem(&self.params), &x0);

            if let SolveResult::Converged { solution, .. } = result {
                // Write back
                for (local_idx, &global_idx) in sub.variable_indices.iter().enumerate() {
                    self.params.values[global_idx] = solution[local_idx];
                }
            }
        }
    }
}
```

### 8.4 Under-Constrained Handling

When DOF > 0, the system has infinite solutions. We solve a **minimum-displacement** problem: find the solution closest to the current state.

```rust
// For under-constrained blocks, add regularization:
// minimize ||x - x_current||² subject to F(x) = 0
//
// This is equivalent to solving the augmented system:
//   [J; λI] * dx = [-F; -λ(x - x_current)]
//
// where λ is a small regularization weight.
```

This gives the "minimum surprise" behavior expected by interactive editor users — unconstrained DOFs don't randomly jump.

---

## 9. Builder API (v2)

### 9.1 Fluent Builder

```rust
let system = ConstraintSystem::builder()
    // Add entities (returns handles)
    .point_2d(0.0, 0.0, |p| p.fixed())              // p0, fixed
    .point_2d(10.0, 0.0, |p| p.name("right"))        // p1
    .circle_2d(5.0, 5.0, 3.0, |c| c.fix_center())   // circle with fixed center
    .line_2d(0.0, 0.0, 10.0, 0.0, |_| {})            // line from (0,0) to (10,0)
    .cubic_bezier_2d(
        [0.0, 0.0], [2.0, 4.0], [8.0, 4.0], [10.0, 0.0],
        |_| {}
    )
    // Add constraints (reference entities by index)
    .constrain(|e| e.distance(0, 1, 10.0))
    .constrain(|e| e.point_on_circle(1, 2))       // p1 on circle
    .constrain(|e| e.tangent_line_circle(3, 2))    // line tangent to circle
    .constrain(|e| e.g1_continuity(3, 4))          // line-bezier G1 junction
    .build();
```

### 9.2 Handle-Based API (Alternative)

For programmatic construction where you need to reference entities:

```rust
let mut sys = ConstraintSystem::new();

let p0 = sys.add_point_2d(0.0, 0.0);
let p1 = sys.add_point_2d(10.0, 0.0);
let circle = sys.add_circle_2d(5.0, 5.0, 3.0);

sys.fix_entity(p0);
sys.add_constraint(Distance::new(p0, p1, 10.0));
sys.add_constraint(PointOnCircle::new(p1, circle));

// Access specific parameters
sys.fix_param(circle.param(2));  // Fix radius only
let radius_idx = circle.params.start + 2;
```

---

## 10. JIT Integration

The existing Cranelift-based JIT compiler (`src/jit/`) compiles constraint residuals and Jacobians to native code. In v2, JIT operates on the flat parameter vector:

```rust
pub trait JitLowerable {
    /// Emit instructions to compute residuals given parameter registers.
    fn lower_residuals(&self, ctx: &mut LoweringContext, param_regs: &[Reg]) -> Vec<Reg>;

    /// Emit instructions to compute Jacobian entries.
    fn lower_jacobian(&self, ctx: &mut LoweringContext, param_regs: &[Reg])
        -> Vec<(usize, usize, Reg)>;
}
```

The JIT lowering is updated to use direct parameter indices instead of the current `point_idx * D + coord` scheme.

---

## 11. Module Structure

```
crates/solverang/src/
├── lib.rs                  # Crate root, feature gates, re-exports
├── problem.rs              # Problem trait (KEEP AS-IS)
│
├── solver/                 # Numerical solvers (KEEP AS-IS)
│   ├── newton_raphson.rs
│   ├── levenberg_marquardt.rs
│   ├── auto.rs
│   ├── sparse_solver.rs
│   ├── parallel.rs
│   └── jit_solver.rs
│
├── geometry/               # REWRITE: v2 geometry layer
│   ├── mod.rs
│   ├── params.rs           # ParameterStore, ParamRange, EntityHandle
│   ├── entity.rs           # EntityKind enum, EntityEvaluator trait
│   ├── entities/           # Per-entity-type evaluators
│   │   ├── point.rs
│   │   ├── line.rs
│   │   ├── circle.rs
│   │   ├── arc.rs
│   │   ├── ellipse.rs
│   │   ├── bezier.rs
│   │   ├── bspline.rs
│   │   ├── nurbs.rs
│   │   ├── plane.rs
│   │   ├── sphere.rs
│   │   ├── cylinder.rs
│   │   ├── cone.rs
│   │   └── torus.rs
│   ├── constraint.rs       # Constraint trait (new)
│   ├── constraints/        # Per-constraint-type implementations
│   │   ├── distance.rs
│   │   ├── coincident.rs
│   │   ├── fixed.rs
│   │   ├── horizontal.rs
│   │   ├── vertical.rs
│   │   ├── midpoint.rs
│   │   ├── symmetric.rs
│   │   ├── parallel.rs
│   │   ├── perpendicular.rs
│   │   ├── collinear.rs
│   │   ├── equal_length.rs
│   │   ├── angle.rs
│   │   ├── point_on_line.rs
│   │   ├── point_on_circle.rs
│   │   ├── point_on_ellipse.rs
│   │   ├── tangent.rs
│   │   ├── equal_radius.rs
│   │   ├── concentric.rs
│   │   ├── arc_constraints.rs
│   │   ├── bezier_continuity.rs
│   │   ├── point_on_curve.rs
│   │   ├── plane_constraints.rs
│   │   ├── surface_constraints.rs
│   │   └── meta_constraints.rs   # FixedParam, EqualParam, ParamRange
│   ├── system.rs           # ConstraintSystem (rewritten)
│   └── builder.rs          # Builder API (rewritten)
│
├── graph/                  # NEW: constraint graph + DD
│   ├── mod.rs
│   ├── incremental.rs      # IncrementalGraph (DD-lite)
│   ├── union_find.rs       # IncrementalUnionFind with rollback
│   ├── dm.rs               # Dulmage-Mendelsohn decomposition
│   ├── redundancy.rs       # Redundancy / conflict detection
│   └── diagnostics.rs      # DOF analysis, constraint status reporting
│
├── pipeline/               # NEW: solve pipeline orchestration
│   ├── mod.rs
│   ├── extract.rs          # Sub-problem extraction from components
│   ├── select.rs           # Solver selection heuristic
│   ├── warm_start.rs       # Warm start / minimum-displacement
│   └── solve.rs            # Top-level solve_dirty() orchestration
│
├── decomposition.rs        # KEEP: union-find (used as building block)
├── constraints/            # KEEP: inequality constraints, slack transforms
├── jacobian/               # KEEP: verification, sparse utilities
├── test_problems/          # KEEP: MINPACK + NIST benchmarks
└── jit/                    # UPDATE: new lowering for param-based constraints
```

---

## 12. Implementation Phases

### Phase 0: Foundation (Week 1-2)
**Goal**: New data model compiles, existing solvers still work.

1. Create `geometry/params.rs` with `ParameterStore`, `EntityHandle`, `ParamRange`
2. Create `geometry/entity.rs` with `EntityKind` enum
3. Create `geometry/constraint.rs` with new `Constraint` trait
4. Create `geometry/system.rs` v2: `ConstraintSystem` wrapping `ParameterStore` + `Vec<Box<dyn Constraint>>`
5. Implement `Problem` trait for new `ConstraintSystem`
6. Write tests: allocate entities, add constraints, verify `Problem` output matches expected

**Deliverable**: New `ConstraintSystem` that wraps the flat parameter model and bridges to the existing `Problem` trait. Can solve basic problems with NR/LM.

### Phase 1: Port Existing Constraints (Week 2-3)
**Goal**: All 16 existing constraint types reimplemented on new trait.

1. Port each constraint from `GeometricConstraint<D>` → `Constraint`:
   - Remove const generic D
   - Change from `&[Point<D>]` to `&[f64]` (parameter vector)
   - Change column indices from `point_idx * D + coord` to direct param indices
   - Store entity handles (not point indices)
2. Port `ConstraintSystemBuilder` to new API
3. Ensure all existing geometry tests pass (adapted to new API)
4. Verify Jacobians with `verify_jacobian` for every ported constraint

**Deliverable**: Feature parity with current system, all tests passing.

### Phase 2: New Primitives & Constraints (Week 3-5)
**Goal**: Circle, arc, ellipse, bezier as first-class entities with variable parameters.

1. Implement entity evaluators: `Circle2D`, `Arc2D`, `Ellipse2D`, `CubicBezier2D`
2. Implement new constraints: `PointOnCircle` (variable radius), `EqualRadius`, `ConcentricCircles`, `LineTangentCircle`, `CircleTangentCircle`
3. Implement arc constraints: `ArcEndpoint`, `ArcSweep`, `PointOnArc`
4. Implement ellipse constraints: `PointOnEllipse`
5. Implement Bezier constraints: `PointOnBezier` (with auxiliary t), `BezierEndpoint`, `G0Continuity`, `G1Continuity`, `G2Continuity`
6. Implement meta-constraints: `FixedParam`, `EqualParam`, `ParamRange`
7. Full Jacobian verification for every new constraint
8. Integration tests: solve a constrained drawing with mixed entity types

**Deliverable**: All 2D primitive types and their constraints. Drawings with circles, arcs, beziers, tangencies, continuity.

### Phase 3: Constraint Graph & DD (Week 5-7)
**Goal**: Incremental graph analysis replaces batch recomputation.

1. Implement `IncrementalUnionFind` with epoch-based rollback for splits
2. Implement `IncrementalGraph` with add/remove entity/constraint
3. Implement dirty tracking and change propagation
4. Implement DM decomposition on the constraint-variable bipartite graph
5. Implement redundancy detection (structural + numerical)
6. Implement conflict detection with diagnostic messages
7. Benchmarks: compare DD incremental vs BFS batch for 100, 1000, 10000 entities

**Deliverable**: `IncrementalGraph` that maintains components, DOF, and dirty sets incrementally.

### Phase 4: Solve Pipeline (Week 7-9)
**Goal**: Full solve pipeline with heuristic solver selection.

1. Implement sub-problem extraction from DD components
2. Implement solver selection heuristic based on size/nonlinearity/sparsity
3. Implement warm starting (minimum-displacement for under-constrained)
4. Implement `solve_dirty()` top-level orchestration
5. Integration tests: interactive editing simulation (add/move/delete in sequence)
6. Performance benchmarks: latency per edit for various drawing sizes

**Deliverable**: Complete solve pipeline that an editor can call on every user action.

### Phase 5: 3D Primitives & Cross-Dimension (Week 9-11)
**Goal**: Full 3D support.

1. Implement 3D entity evaluators: `Point3D`, `Line3D`, `Circle3D`, `Arc3D`, `Sphere`, `Cylinder`, `Cone`, `Torus`, `Plane`
2. Implement 3D constraints: `PointOnPlane`, `PointOnSphere`, `PointOnCylinder`, `PlaneParallel`, `PlanePerpendicular`, etc.
3. Implement cross-dimension: `ProjectToPlane`, `LiftFromSketch`
4. Tests: 3D assembly constraints, sketch-on-face workflows

**Deliverable**: Full 3D primitive and constraint support.

### Phase 6: Advanced Curves & JIT Update (Week 11-13)
**Goal**: B-splines, NURBS, and JIT for new constraints.

1. Implement `BSpline2D`, `BSpline3D` entity evaluators with De Boor's algorithm
2. Implement `NurbsCurve2D`, `NurbsCurve3D` with rational evaluation
3. Implement spline constraints: `BSplinePointOn`, `NurbsPointOn`, `NurbsWeightPositive`
4. Update JIT lowering for new constraint trait (flat param vector)
5. JIT-compile hot-path constraints (distance, point-on-circle, tangent)

**Deliverable**: Full curve support including NURBS. JIT acceleration for common constraints.

---

## 13. Testing Strategy

### 13.1 Per-Constraint Verification

Every constraint gets:
1. **Residual test**: known-satisfied configuration → residuals ≈ 0
2. **Jacobian verification**: `verify_jacobian()` against finite differences
3. **Solve test**: from perturbed initial guess → solver finds known solution
4. **Boundary tests**: degenerate cases (zero-length lines, coincident points, zero radius)

### 13.2 Integration Test Suite

Realistic 2D drawings solved end-to-end:

| Test | Description | Entities | Constraints |
|------|-------------|----------|-------------|
| Triangle | 3 points, 3 distances, 1 fixed | 3 | 4 |
| Rectangle | 4 points, horiz/vert/equal-length | 4 | 8 |
| Circle-tangent-lines | Circle + 2 tangent lines | 1 circle, 4 points | 6 |
| Bezier chain | 3 cubic beziers with G2 continuity | 3 beziers | 8 |
| Gear sketch | 20 arcs with equal radius + angular spacing | 20 arcs, 1 circle | 60 |
| Complex drawing | 50+ mixed entities | 50+ | 100+ |

### 13.3 DD Correctness

1. **Equivalence**: DD incremental component detection matches BFS batch for same inputs
2. **Add/remove symmetry**: add then remove returns to original state
3. **Stress**: random add/remove sequences with verification after each step

### 13.4 Performance Benchmarks

| Benchmark | Target |
|-----------|--------|
| solve_dirty() with 1 changed param, 100 entities | < 1ms |
| solve_dirty() with 1 changed param, 1000 entities | < 10ms |
| Full solve of 100-entity drawing from scratch | < 50ms |
| Full solve of 1000-entity drawing from scratch | < 500ms |
| DD component update after add_constraint | < 100μs |

---

## 14. Migration & Cleanup

Since this is a solo project with no backward-compatibility requirements:

1. **Delete** `geometry/point.rs` `Point<D>` struct (replaced by `EntityKind::Point2D/3D`)
2. **Delete** `geometry/vector.rs`, `geometry/line.rs`, `geometry/circle.rs` (helper structs replaced by entities)
3. **Delete** all files in `geometry/constraints/` (reimplemented on new trait)
4. **Delete** `geometry/system.rs` and `geometry/builder.rs` (rewritten)
5. **Keep** `decomposition.rs` (used as building block for `IncrementalUnionFind`)
6. **Keep** all solver code as-is
7. **Keep** `problem.rs` as-is (the bridge between geometry and solvers)
8. **Keep** `test_problems/` as-is (solver validation)
9. **Keep** `jacobian/` as-is (verification utilities)
10. **Update** `jit/geometry_lowering.rs` for new constraint trait

---

## 15. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| DD adds complexity without measurable perf gain for small drawings | Wasted effort | Implement DD behind a feature flag; keep BFS as fallback. Benchmark early. |
| DM decomposition is complex to implement correctly | Bugs in structural analysis | Start with Hopcroft-Karp from a well-known reference impl. Test against known DM results. |
| Auxiliary parameter t for point-on-curve creates many extra variables | Solver slowdown | Use substitution where possible (e.g., parameterize distance along curve instead of explicit t). Profile. |
| G2 continuity constraints have complex Jacobians | Incorrect derivatives | Use `verify_jacobian()` religiously. Consider auto-diff as a fallback. |
| Removing const generic D makes type errors runtime errors | Dimension mismatches | Validate entity dimensions at constraint creation time. Good error messages. |
| Flat parameter vector makes debugging harder | Hard to interpret solver state | `labels: Vec<String>` on every parameter. Debug formatter that groups by entity. |

---

## 16. Non-Goals (Explicitly Out of Scope)

- **B-Rep topology** (half-edges, faces, shells) — separate crate, not part of solver
- **Boolean operations** (union, intersection, subtraction) — separate crate
- **STEP I/O** — separate crate
- **Feature tree / parametric history** — editor-level concern, not solver
- **NurbsSurface** constraints — surfaces deferred to post-v2; curves only in v2 scope
- **Undo/redo** — editor-level concern, though `ParameterStore` snapshots enable it

---

## Appendix A: Comparison with Prior Approaches

| Aspect | Phase 1 Plan (incremental) | v2 Plan (this document) |
|--------|---------------------------|-------------------------|
| Data model | Points + PARAM_COL_BASE sentinel | Flat parameter vector |
| Breaking changes | Trait signature change only | Full rewrite of geometry layer |
| Entity types | Added as wrappers around points + params | First-class with typed handles |
| Graph analysis | Existing union-find only | DD + DM + redundancy detection |
| Solver selection | Manual | Automatic heuristic per component |
| 3D support | Same const-generic approach | Dimension-agnostic constraints |
| Constraint count | ~25 | ~50+ |
| Estimated effort | ~3,400 LOC | ~8,000-12,000 LOC |

## Appendix B: Academic References

1. **Bettig & Hoffmann (2011)** — "Geometric constraint solving in parametric CAD" — DR-planner decomposition
2. **Jermann et al. (2006)** — "Decomposition of geometric constraint systems" — DM + witness configurations
3. **Hoffmann & Vermeer (1995)** — "A spatial constraint solver" — Cluster merging
4. **Ait-Aoudia & Jegou (2015)** — "Reduction methods for geometric constraint solving" — Graph reduction
5. **Joan-Arinyo et al. (2003)** — "Revisiting decomposition analysis of geometric constraint graphs" — Tree decomposition
6. **Thierry et al. (2011)** — "Extensions of the witness method to characterize under/over/well-constrained geometric CSPs" — Structural analysis
7. **Schreck et al. (2006)** — "Using the Dulmage-Mendelsohn decomposition for solving geometric constraints" — DM for geometric CSP
