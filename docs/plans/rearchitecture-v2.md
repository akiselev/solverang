# Solverang V2: Ground-Up Constraint Solver Rearchitecture

## Context

Solo developer. No backward compatibility. Full rewrite of the constraint solver core.

The current `Problem` trait reduces everything to `F(x) = 0` over a flat `&[f64]` vector.
Entities are points only. Constraints reference points by index. There is no constraint graph,
no symbolic reduction, no incremental solving, no entity richness. The numerical core (NR, LM,
sparse, JIT) is production-quality and stays. Everything above it gets rebuilt.

This plan covers **Layer 4 only** (the constraint solver). Layers 1-3 (geometry, topology,
modeling) and Layers 5-6 (history, I/O) from `plan-mcad.md` are separate efforts that consume
this layer.

---

## Design Principles

1. **Parameters are the atoms.** Every solvable quantity is a `ParamId`. Entities own parameters.
   Constraints bind parameters. The solver sees only parameters.
2. **The constraint graph is the spine.** All intelligence (decomposition, redundancy detection,
   incremental updates, diagnostics) derives from the bipartite entity-constraint graph.
3. **Differential dataflow for incrementality.** When a user drags a point, only affected clusters
   re-solve. When a constraint is added, only the affected subgraph re-decomposes.
4. **Squared formulations by default.** Distance constraints use `d^2 - target^2 = 0`. No square
   roots in residuals. Polynomial Jacobians everywhere possible.
5. **Decompose before solving.** The solver never sees the full system. It sees small subproblems
   produced by the graph analysis layer.
6. **Diagnostics are not optional.** Redundancy, conflict, and DOF analysis are built into the
   graph layer, not bolted on after.

---

## Architecture Overview

```
User Action (drag point, add constraint, change dimension)
    |
    v
+------------------------------------------------------------------+
|  SYSTEM: ConstraintSystem                                        |
|  Central coordinator. Owns all registries and the dataflow state.|
+------------------------------------------------------------------+
    |
    v
+------------------------------------------------------------------+
|  LAYER A: REGISTRIES                                             |
|  ParamStore  |  EntityRegistry  |  ConstraintRegistry            |
|  (values)       (geometry)         (relationships)               |
+------------------------------------------------------------------+
    |
    v
+------------------------------------------------------------------+
|  LAYER B: CONSTRAINT GRAPH + DIFFERENTIAL DATAFLOW               |
|  Bipartite graph: entities <-> constraints                       |
|  Incremental connected components (DD or union-find)             |
|  Dirty cluster tracking                                          |
|  Change propagation                                              |
+------------------------------------------------------------------+
    |
    v
+------------------------------------------------------------------+
|  LAYER C: PER-CLUSTER ANALYSIS (for each dirty cluster)          |
|                                                                  |
|  C1. Diagnostics: redundancy, conflict, DOF                     |
|  C2. Symbolic Reduction: substitute fixed, merge coincident,     |
|      eliminate trivial                                           |
|  C3. Pattern Matching: detect triangles, line-circle, known      |
|      closed-form subgraphs                                      |
|  C4. Reduced SubProblem assembly                                 |
+------------------------------------------------------------------+
    |
    v
+------------------------------------------------------------------+
|  LAYER D: NUMERICAL SOLVING                                      |
|  Closed-form solver (for matched patterns)                       |
|  LM with trust region (primary numerical solver)                 |
|  Newton-Raphson (square well-conditioned systems)                |
|  Warm start from cached previous solution                        |
|  Branch selection (closest to previous configuration)            |
+------------------------------------------------------------------+
    |
    v
+------------------------------------------------------------------+
|  LAYER E: RESULT ASSEMBLY                                        |
|  Write solutions back to ParamStore                              |
|  Update cluster caches (solution, Jacobian factorization)        |
|  Emit diagnostics (status per constraint, DOF per entity)        |
|  Notify downstream (display, history, feature tree)              |
+------------------------------------------------------------------+
```

---

## Module Structure

```
crates/solverang/src/
|-- lib.rs                          # Public API re-exports
|-- id.rs                           # ParamId, EntityId, ConstraintId, ClusterId
|
|-- param/
|   |-- mod.rs
|   |-- store.rs                    # ParamStore: central value storage
|   +-- expr.rs                     # Algebraic expressions over params (future: driven dims)
|
|-- entity/
|   |-- mod.rs                      # Entity trait
|   |-- registry.rs                 # EntityRegistry: owns all entities
|   |-- kind.rs                     # EntityKind enum (for pattern matching)
|   |-- point.rs                    # Point2D (x,y), Point3D (x,y,z)
|   |-- line.rs                     # Line2D (p1,p2), Line3D (p1,p2), Ray, InfLine
|   |-- circle.rs                   # Circle2D (cx,cy,r), Arc2D (cx,cy,r,start,sweep)
|   |-- ellipse.rs                  # Ellipse2D, EllipticalArc2D
|   |-- spline.rs                   # BSpline2D/3D (control point params)
|   |-- plane.rs                    # Plane (origin + normal, 6 params)
|   |-- axis.rs                     # Axis3D (origin + direction)
|   |-- surface.rs                  # Sphere, Cylinder, Cone, Torus
|   +-- rigid_body.rs              # RigidBody3D (position + quaternion, 7 params)
|
|-- constraint/
|   |-- mod.rs                      # Constraint trait
|   |-- registry.rs                 # ConstraintRegistry
|   |-- kind.rs                     # ConstraintKind enum
|   |-- coincident.rs               # Point-Point, Point-Line, Point-Circle, Point-Plane, ...
|   |-- distance.rs                 # Point-Point, Point-Line, Point-Plane, Line-Line, ...
|   |-- angle.rs                    # Line-Line, Line-Horizontal, Plane-Plane, ...
|   |-- parallel.rs                 # Line-Line, Line-Plane, Plane-Plane
|   |-- perpendicular.rs            # Line-Line, Line-Plane, Plane-Plane
|   |-- tangent.rs                  # Line-Circle, Circle-Circle, Line-Ellipse, ...
|   |-- concentric.rs               # Circle-Circle, Arc-Arc, Sphere-Sphere, ...
|   |-- equal.rs                    # Length-Length, Radius-Radius, Angle-Angle
|   |-- symmetric.rs                # About point, about line, about plane
|   |-- midpoint.rs                 # Point at midpoint of segment
|   |-- collinear.rs                # Points/lines collinear
|   |-- on_entity.rs                # Point-on-line, point-on-circle, point-on-spline, ...
|   |-- horizontal.rs               # Line horizontal (2D)
|   |-- vertical.rs                 # Line vertical (2D)
|   |-- fixed.rs                    # Fix entity in place
|   |-- coplanar.rs                 # Entities on same plane (3D)
|   |-- coaxial.rs                  # Axes aligned (3D)
|   |-- mate.rs                     # Assembly: face-to-face
|   |-- gear.rs                     # Assembly: coupled rotation
|   |-- smooth.rs                   # G1/G2 continuity at curve junctions
|   |-- inequality.rs               # g(x) >= 0 via slack variables
|   +-- dimension.rs                # Driving/driven dimension constraints
|
|-- graph/
|   |-- mod.rs
|   |-- bipartite.rs                # Entity-constraint bipartite graph
|   |-- decompose.rs                # Cluster decomposition (union-find + DD)
|   |-- cluster.rs                  # RigidCluster: entities + constraints + cached state
|   |-- pattern.rs                  # Solvable pattern detection (triangles, etc.)
|   |-- redundancy.rs               # Jacobian rank analysis, conflict detection
|   |-- ordering.rs                 # Solve order within/across clusters
|   +-- dof.rs                      # DOF analysis per entity and per cluster
|
|-- reduce/
|   |-- mod.rs
|   |-- substitute.rs               # Fixed parameter elimination
|   |-- merge.rs                    # Coincident entity merging
|   |-- eliminate.rs                # Trivial constraint removal
|   +-- simplify.rs                 # Algebraic simplification of residuals
|
|-- solve/
|   |-- mod.rs
|   |-- sub_problem.rs              # ReducedSubProblem: what the solver actually sees
|   |-- newton.rs                   # Newton-Raphson (reuse existing, minor mods)
|   |-- lm.rs                       # Levenberg-Marquardt (reuse existing)
|   |-- closed_form.rs              # Analytical solutions for matched patterns
|   |-- branch.rs                   # Solution branch selection
|   |-- drag.rs                     # Drag solving: null-space projection for underconstrained
|   +-- result.rs                   # SolveResult, diagnostics, per-constraint status
|
|-- dataflow/
|   |-- mod.rs
|   |-- tracker.rs                  # ChangeTracker: dirty params, dirty clusters
|   |-- propagate.rs                # Propagate param changes through graph to clusters
|   |-- incremental.rs              # Incremental graph algorithms (component updates)
|   +-- cache.rs                    # Per-cluster solution/Jacobian cache
|
+-- system.rs                       # ConstraintSystem: top-level coordinator
```

---

## Core Types

### Identifiers

```rust
// id.rs
// All IDs are lightweight Copy types. Generational to detect use-after-delete.

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct ParamId { index: u32, generation: u32 }

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct EntityId { index: u32, generation: u32 }

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct ConstraintId { index: u32, generation: u32 }

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct ClusterId(u32);
```

### ParamStore

```rust
// param/store.rs
// Single source of truth for all parameter values.
// Every solvable quantity in the system is a ParamId pointing here.

pub struct ParamStore {
    values: Vec<f64>,            // Dense storage indexed by ParamId.index
    fixed: Vec<bool>,            // Whether parameter is locked
    generations: Vec<u32>,       // Generation counter per slot
    free_slots: Vec<u32>,        // Recycled slots
    names: Vec<Option<String>>,  // Debug names ("circle_3.radius")
    owners: Vec<EntityId>,       // Which entity owns this param
}

impl ParamStore {
    pub fn alloc(&mut self, initial: f64, owner: EntityId) -> ParamId;
    pub fn free(&mut self, id: ParamId);
    pub fn get(&self, id: ParamId) -> f64;
    pub fn set(&mut self, id: ParamId, value: f64);
    pub fn is_fixed(&self, id: ParamId) -> bool;
    pub fn fix(&mut self, id: ParamId);
    pub fn unfix(&mut self, id: ParamId);

    // Bulk operations for solver interface
    pub fn free_param_ids(&self) -> Vec<ParamId>;
    pub fn free_param_count(&self) -> usize;
    pub fn extract_free_values(&self) -> Vec<f64>;
    pub fn write_free_values(&mut self, values: &[f64]);

    // Mapping: flat solver index <-> ParamId
    pub fn build_solver_mapping(&self) -> SolverMapping;
}

pub struct SolverMapping {
    param_to_col: HashMap<ParamId, usize>,  // ParamId -> column in Jacobian
    col_to_param: Vec<ParamId>,             // column -> ParamId
}
```

### Entity Trait

```rust
// entity/mod.rs
// An entity is a geometric object with parameters.
// Constraints operate on entity parameters, not on raw coordinates.

pub trait Entity: Send + Sync {
    fn id(&self) -> EntityId;
    fn kind(&self) -> EntityKind;
    fn params(&self) -> &[ParamId];  // All parameters this entity owns
    fn param_count(&self) -> usize { self.params().len() }
    fn name(&self) -> &str;
}
```

### Entity Types (Exhaustive)

```rust
// entity/kind.rs
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum EntityKind {
    // 2D Sketch Entities
    Point2D,          // (x, y) = 2 params
    LineSegment2D,    // (x1, y1, x2, y2) = 4 params (two endpoints)
    Circle2D,         // (cx, cy, r) = 3 params
    Arc2D,            // (cx, cy, r, start_angle, sweep_angle) = 5 params
    Ellipse2D,        // (cx, cy, rx, ry, rotation) = 5 params
    EllipticalArc2D,  // (cx, cy, rx, ry, rotation, start, sweep) = 7 params
    BSpline2D,        // (control_points...) = 2*n params
    InfiniteLine2D,   // (px, py, angle) = 3 params
    Ray2D,            // (px, py, angle) = 3 params

    // 3D Entities
    Point3D,          // (x, y, z) = 3 params
    LineSegment3D,    // (x1,y1,z1, x2,y2,z2) = 6 params
    Circle3D,         // (cx,cy,cz, nx,ny,nz, r) = 7 params (center+normal+radius)
    Arc3D,            // (cx,cy,cz, nx,ny,nz, r, start, sweep) = 9 params
    BSpline3D,        // (control_points...) = 3*n params
    InfiniteLine3D,   // (px,py,pz, dx,dy,dz) = 6 params
    Plane,            // (px,py,pz, nx,ny,nz) = 6 params (point on plane + normal)
    Axis3D,           // (px,py,pz, dx,dy,dz) = 6 params (point + direction)
    Sphere,           // (cx,cy,cz, r) = 4 params
    Cylinder,         // (px,py,pz, dx,dy,dz, r) = 7 params
    Cone,             // (apex_x,y,z, axis_x,y,z, half_angle) = 7 params
    Torus,            // (cx,cy,cz, nx,ny,nz, major_r, minor_r) = 8 params

    // Assembly
    RigidBody,        // (tx,ty,tz, qw,qx,qy,qz) = 7 params (position + quaternion)
}
```

Note: `InfiniteLine2D` and `Plane` use minimal parameterizations. `LineSegment2D` uses
endpoint parameterization because that's how users think about sketch lines (and because
most constraints bind to endpoints). The solver handles both: constraints that need "the
line as a geometric object" extract direction/length from the endpoint params internally.

### Concrete Entity Examples

```rust
// entity/circle.rs
pub struct Circle2D {
    id: EntityId,
    cx: ParamId,
    cy: ParamId,
    r: ParamId,
}

impl Entity for Circle2D {
    fn id(&self) -> EntityId { self.id }
    fn kind(&self) -> EntityKind { EntityKind::Circle2D }
    fn params(&self) -> &[ParamId] { &[self.cx, self.cy, self.r] }  // not a slice field
    fn name(&self) -> &str { "Circle2D" }
}

impl Circle2D {
    pub fn center_x(&self) -> ParamId { self.cx }
    pub fn center_y(&self) -> ParamId { self.cy }
    pub fn radius(&self) -> ParamId { self.r }
}

// entity/point.rs
pub struct Point2D {
    id: EntityId,
    x: ParamId,
    y: ParamId,
}

impl Entity for Point2D {
    fn id(&self) -> EntityId { self.id }
    fn kind(&self) -> EntityKind { EntityKind::Point2D }
    fn params(&self) -> &[ParamId] { &[self.x, self.y] }
    fn name(&self) -> &str { "Point2D" }
}
```

### Constraint Trait

```rust
// constraint/mod.rs
// Key change from V1: constraints reference ParamIds, not point indices.
// The Jacobian returns (equation_row, ParamId, value) not (row, col, value).
// Column mapping happens at the system level.

pub trait Constraint: Send + Sync {
    fn id(&self) -> ConstraintId;
    fn kind(&self) -> ConstraintKind;
    fn name(&self) -> &str;

    // Which entities this constraint references (for graph building)
    fn entity_ids(&self) -> &[EntityId];

    // Which parameters this constraint depends on (subset of entity params)
    fn param_ids(&self) -> Vec<ParamId>;

    // Number of scalar equations
    fn equation_count(&self) -> usize;

    // Residuals: should be zero when satisfied.
    // Reads parameter values from the store.
    fn residuals(&self, store: &ParamStore) -> Vec<f64>;

    // Jacobian: (equation_row, param_id, partial_derivative)
    // Only non-zero entries. The system maps ParamId -> column index.
    fn jacobian(&self, store: &ParamStore) -> Vec<(usize, ParamId, f64)>;

    // Weight for soft constraints / weighted least squares
    fn weight(&self) -> f64 { 1.0 }

    // Priority for conflict resolution (higher = harder to relax)
    fn priority(&self) -> u32 { 100 }

    // Is this a soft constraint that can be relaxed?
    fn is_soft(&self) -> bool { false }

    // Quick check: is this constraint satisfied within tolerance?
    fn is_satisfied(&self, store: &ParamStore, tol: f64) -> bool {
        self.residuals(store).iter().all(|r| r.abs() < tol)
    }
}
```

### Constraint Examples (Squared Formulations)

```rust
// constraint/distance.rs
// SQUARED distance formulation: (dx^2 + dy^2) - target^2 = 0
// Advantages: no sqrt, no singularity at zero distance, polynomial Jacobian.

/// Distance between two Point2D entities.
pub struct DistancePoint2D {
    id: ConstraintId,
    entity_a: EntityId,
    entity_b: EntityId,
    // Cached param IDs (looked up at construction, not per-evaluation)
    ax: ParamId, ay: ParamId,
    bx: ParamId, by: ParamId,
    target: f64,
}

impl Constraint for DistancePoint2D {
    fn equation_count(&self) -> usize { 1 }

    fn residuals(&self, s: &ParamStore) -> Vec<f64> {
        let dx = s.get(self.bx) - s.get(self.ax);
        let dy = s.get(self.by) - s.get(self.ay);
        // Squared formulation: no sqrt
        vec![dx * dx + dy * dy - self.target * self.target]
    }

    fn jacobian(&self, s: &ParamStore) -> Vec<(usize, ParamId, f64)> {
        let dx = s.get(self.bx) - s.get(self.ax);
        let dy = s.get(self.by) - s.get(self.ay);
        // d(dx^2+dy^2)/d(ax) = -2*dx, etc. No division by distance.
        vec![
            (0, self.ax, -2.0 * dx),
            (0, self.ay, -2.0 * dy),
            (0, self.bx,  2.0 * dx),
            (0, self.by,  2.0 * dy),
        ]
    }

    fn entity_ids(&self) -> &[EntityId] { &[self.entity_a, self.entity_b] }
    fn param_ids(&self) -> Vec<ParamId> { vec![self.ax, self.ay, self.bx, self.by] }
    fn kind(&self) -> ConstraintKind { ConstraintKind::Distance }
    fn id(&self) -> ConstraintId { self.id }
    fn name(&self) -> &str { "Distance(Point2D, Point2D)" }
}

/// Distance from Point2D to Line2D (perpendicular distance, squared).
pub struct DistancePointLine2D {
    id: ConstraintId,
    point_entity: EntityId,
    line_entity: EntityId,
    px: ParamId, py: ParamId,          // point params
    lx1: ParamId, ly1: ParamId,        // line start params
    lx2: ParamId, ly2: ParamId,        // line end params
    target: f64,
}

impl Constraint for DistancePointLine2D {
    fn equation_count(&self) -> usize { 1 }

    fn residuals(&self, s: &ParamStore) -> Vec<f64> {
        // Perpendicular distance squared = (cross product)^2 / (line length)^2
        let px = s.get(self.px); let py = s.get(self.py);
        let x1 = s.get(self.lx1); let y1 = s.get(self.ly1);
        let x2 = s.get(self.lx2); let y2 = s.get(self.ly2);
        let dx = x2 - x1;
        let dy = y2 - y1;
        let cross = (px - x1) * dy - (py - y1) * dx;
        let len_sq = dx * dx + dy * dy;
        // cross^2 / len_sq - target^2 = 0
        // Multiply through: cross^2 - target^2 * len_sq = 0
        vec![cross * cross - self.target * self.target * len_sq]
    }

    // Jacobian omitted for brevity - chain rule on the above
    fn jacobian(&self, s: &ParamStore) -> Vec<(usize, ParamId, f64)> { todo!() }
    fn entity_ids(&self) -> &[EntityId] { &[self.point_entity, self.line_entity] }
    fn param_ids(&self) -> Vec<ParamId> {
        vec![self.px, self.py, self.lx1, self.ly1, self.lx2, self.ly2]
    }
    fn kind(&self) -> ConstraintKind { ConstraintKind::Distance }
    fn id(&self) -> ConstraintId { self.id }
    fn name(&self) -> &str { "Distance(Point2D, Line2D)" }
}

// constraint/tangent.rs
/// Tangent: Line tangent to Circle (2D).
/// Formulation: (perpendicular distance from center to line)^2 = radius^2
/// This is: cross^2 / line_len^2 = r^2, or cross^2 - r^2 * line_len^2 = 0
/// All polynomial, no sqrt, no abs().
pub struct TangentLineCircle2D {
    id: ConstraintId,
    line_entity: EntityId,
    circle_entity: EntityId,
    lx1: ParamId, ly1: ParamId,
    lx2: ParamId, ly2: ParamId,
    cx: ParamId, cy: ParamId, r: ParamId,
}

impl Constraint for TangentLineCircle2D {
    fn equation_count(&self) -> usize { 1 }

    fn residuals(&self, s: &ParamStore) -> Vec<f64> {
        let x1 = s.get(self.lx1); let y1 = s.get(self.ly1);
        let x2 = s.get(self.lx2); let y2 = s.get(self.ly2);
        let cx = s.get(self.cx); let cy = s.get(self.cy);
        let r = s.get(self.r);
        let dx = x2 - x1;
        let dy = y2 - y1;
        let cross = (cx - x1) * dy - (cy - y1) * dx;
        let len_sq = dx * dx + dy * dy;
        vec![cross * cross - r * r * len_sq]
    }

    fn jacobian(&self, s: &ParamStore) -> Vec<(usize, ParamId, f64)> { todo!() }
    fn entity_ids(&self) -> &[EntityId] { &[self.line_entity, self.circle_entity] }
    fn param_ids(&self) -> Vec<ParamId> {
        vec![self.lx1, self.ly1, self.lx2, self.ly2, self.cx, self.cy, self.r]
    }
    fn kind(&self) -> ConstraintKind { ConstraintKind::Tangent }
    fn id(&self) -> ConstraintId { self.id }
    fn name(&self) -> &str { "Tangent(Line2D, Circle2D)" }
}
```

### Complete Constraint Taxonomy

Every entry in this table is a concrete struct implementing `Constraint`.
Entity-pair dispatch happens at constraint creation time (the `ConstraintRegistry`
factory method picks the right struct based on `EntityKind` pairs).

```
CONSTRAINT TYPE          | ENTITY PAIRS                              | EQS | FORMULATION
-------------------------|-------------------------------------------|-----|---------------------------
Distance                 | Pt-Pt, Pt-Line, Pt-Circle, Pt-Plane,     | 1   | Squared distance
                         | Line-Line, Line-Plane, Plane-Plane        |     |
Coincident               | Pt-Pt, Pt-Line, Pt-Circle, Pt-Arc,       | 1-3 | Coordinate difference
                         | Pt-Ellipse, Pt-Spline, Pt-Plane          |     | or projection residual
Angle                    | Line-Line, Line-Horiz, Line-Vert,        | 1   | sin/cos formulation
                         | Plane-Plane, Line-Plane                   |     | (avoids atan2 wrapping)
Parallel                 | Line-Line, Line-Plane, Plane-Plane        | 1-2 | Cross product = 0
Perpendicular            | Line-Line, Line-Plane, Plane-Plane        | 1   | Dot product = 0
Tangent                  | Line-Circle, Line-Arc, Line-Ellipse,     | 1   | cross^2 - r^2*len^2 = 0
                         | Circle-Circle, Circle-Arc,                |     | or (dist - r1-r2)^2 = 0
                         | Circle-Ellipse, Line-Spline               |     |
Concentric               | Circle-Circle, Circle-Arc, Arc-Arc,      | 2   | center_x/y difference
                         | Sphere-Sphere, Cylinder-Cylinder          |     |
Coradial                 | Circle-Circle, Arc-Arc                    | 3   | center + radius match
Equal                    | Length-Length, Radius-Radius, Angle-Angle | 1   | param_a - param_b = 0
Symmetric (about point)  | Pt-Pt-Pt, Entity-Entity-Pt               | 2-3 | midpoint equations
Symmetric (about line)   | Pt-Pt + Line, Entity-Entity + Line       | 2-3 | reflection equations
Symmetric (about plane)  | Entity-Entity + Plane (3D)                | 3   | reflection equations
Midpoint                 | Pt on Line                                | 2   | p = (start+end)/2
Collinear                | 3+ Points, or Line-Line                   | 1-2 | Cross product = 0
On-Entity                | Pt-Line, Pt-Circle, Pt-Arc, Pt-Ellipse,  | 1-2 | Distance to entity = 0
                         | Pt-Spline, Pt-Plane, Pt-Sphere,           |     |
                         | Pt-Cylinder                                |     |
Horizontal               | Line2D, or Pt-Pt                          | 1   | dy = 0
Vertical                 | Line2D, or Pt-Pt                          | 1   | dx = 0
Fixed                    | Any entity                                | N   | params = constants
Smooth (G1)              | Curve-Curve at junction                   | 2   | Tangent vectors equal
Curvature (G2)           | Curve-Curve at junction                   | 3   | Tangent + curvature
Coplanar                 | Pt-Plane, Line-Plane, multi-point         | 1   | Signed distance = 0
Coaxial                  | Axis-Axis, Cyl-Cyl                        | 2-4 | Direction parallel +
                         |                                           |     | point on axis
Mate (flush)             | Face-Face (via Plane entities)             | 1-3 | Distance + parallel
Insert                   | Cylinder into hole (Mate + Coaxial)       | 4-5 | Combined
Gear ratio               | Arc/Circle rotation coupling               | 1   | theta1 * r1 = theta2 * r2
Dimension (driving)      | Any measurable quantity                    | 1   | param = expression
Dimension (driven)       | Any measurable quantity                    | 0   | Read-only display
Inequality               | Any g(x) >= 0                             | 1   | g(x) - s^2 = 0 (slack)
```

---

## Constraint Graph and Differential Dataflow

### The Bipartite Graph

```rust
// graph/bipartite.rs
pub struct ConstraintGraph {
    // Forward edges: entity -> constraints that reference it
    entity_to_constraints: HashMap<EntityId, SmallVec<[ConstraintId; 4]>>,
    // Reverse edges: constraint -> entities it references
    constraint_to_entities: HashMap<ConstraintId, SmallVec<[EntityId; 4]>>,
    // Derived: param -> constraints that depend on it (for change propagation)
    param_to_constraints: HashMap<ParamId, SmallVec<[ConstraintId; 4]>>,
    // Cluster assignments
    entity_cluster: HashMap<EntityId, ClusterId>,
    constraint_cluster: HashMap<ConstraintId, ClusterId>,
}
```

### Cluster Decomposition

The graph decomposes into **rigid clusters**: connected components of the bipartite graph.
Within each cluster, all entities and constraints are interdependent and must be solved together.
Across clusters, solutions are independent and can be parallelized.

```rust
// graph/decompose.rs
// Two strategies, selected by system size:
//
// Small systems (< 500 entities): Union-find.
//   O(n * alpha(n)) ~ O(n). Simple, fast, no overhead.
//
// Large systems or incremental updates: Differential dataflow.
//   Connected components as a DD computation. When an edge is added/removed,
//   DD incrementally updates the component assignment without re-scanning
//   the entire graph. This is the key innovation for interactive performance.

pub fn decompose_union_find(graph: &ConstraintGraph) -> Vec<RigidCluster> { ... }

pub fn decompose_incremental(
    graph: &ConstraintGraph,
    change: &GraphChange,
    previous: &[RigidCluster],
) -> Vec<RigidCluster> { ... }

pub enum GraphChange {
    AddConstraint(ConstraintId),
    RemoveConstraint(ConstraintId),
    AddEntity(EntityId),
    RemoveEntity(EntityId),
    // No change to graph structure, only param values changed
    ParamChanged(Vec<ParamId>),
}
```

### Rigid Cluster

```rust
// graph/cluster.rs
pub struct RigidCluster {
    pub id: ClusterId,
    pub entities: Vec<EntityId>,
    pub constraints: Vec<ConstraintId>,
    pub params: Vec<ParamId>,          // Union of all entity params in cluster

    // Cached solve state (invalidated when dirty)
    pub last_solution: Option<Vec<f64>>,
    pub last_residual_norm: Option<f64>,
    pub last_jacobian: Option<CachedJacobian>,

    // Diagnostics
    pub dof: i32,                       // variables - equations
    pub status: ClusterStatus,
}

pub enum ClusterStatus {
    WellConstrained,       // DOF = 0, Jacobian full rank
    UnderConstrained(i32), // DOF > 0, with DOF count
    OverConstrained(i32),  // DOF < 0, with excess count
    Redundant(Vec<ConstraintId>),  // DOF = 0 but Jacobian rank-deficient
    Conflicting(Vec<ConstraintId>), // Inconsistent constraints
    Unsolved,              // Not yet solved
}

pub struct CachedJacobian {
    pub factorization: Option<LUFactorization>,  // or QR, etc.
    pub param_order: Vec<ParamId>,  // Column ordering
    pub dirty: bool,
}
```

### Differential Dataflow Integration

The DD layer tracks three levels of change:

1. **Structural changes** (add/remove entity or constraint): Triggers incremental
   re-decomposition. May merge or split clusters.
2. **Parameter value changes** (user drags a point, changes a dimension): Does NOT
   change graph structure. Only marks affected clusters as dirty.
3. **Fixed/unfixed changes** (user locks/unlocks an entity): Changes the variable
   set within a cluster. Invalidates cached Jacobian and forces re-analysis.

```rust
// dataflow/tracker.rs
pub struct ChangeTracker {
    // Structural changes pending
    pending_structural: Vec<GraphChange>,
    // Parameter changes pending
    dirty_params: HashSet<ParamId>,
    // Clusters needing re-solve
    dirty_clusters: HashSet<ClusterId>,
    // Clusters needing re-analysis (structural change)
    invalidated_clusters: HashSet<ClusterId>,
}

impl ChangeTracker {
    // Called when user changes a parameter value
    pub fn notify_param_change(&mut self, param: ParamId, graph: &ConstraintGraph) {
        self.dirty_params.insert(param);
        // Trace param -> constraints -> clusters
        if let Some(constraints) = graph.param_to_constraints.get(&param) {
            for &cid in constraints {
                if let Some(&cluster) = graph.constraint_cluster.get(&cid) {
                    self.dirty_clusters.insert(cluster);
                }
            }
        }
    }

    // Called when graph structure changes
    pub fn notify_structural_change(&mut self, change: GraphChange) {
        self.pending_structural.push(change);
    }

    // Consume all pending changes
    pub fn drain(&mut self) -> PendingChanges {
        PendingChanges {
            structural: std::mem::take(&mut self.pending_structural),
            dirty_params: std::mem::take(&mut self.dirty_params),
            dirty_clusters: std::mem::take(&mut self.dirty_clusters),
            invalidated_clusters: std::mem::take(&mut self.invalidated_clusters),
        }
    }
}
```

### Full Incremental Solve Pipeline

```rust
// system.rs (pseudocode)
impl ConstraintSystem {
    pub fn solve_incremental(&mut self) -> SystemResult {
        let changes = self.tracker.drain();

        // Step 1: If structural changes, re-decompose
        if !changes.structural.is_empty() {
            for change in &changes.structural {
                self.graph.apply(change);
            }
            self.clusters = decompose_incremental(
                &self.graph, &changes, &self.clusters
            );
            // All affected clusters are now in dirty_clusters
        }

        // Step 2: Solve each dirty cluster
        let results: Vec<ClusterResult> = self.dirty_clusters
            .par_iter()  // rayon parallel
            .map(|&cluster_id| {
                let cluster = &self.clusters[cluster_id];
                self.solve_cluster(cluster)
            })
            .collect();

        // Step 3: Write results back
        for result in results {
            self.apply_cluster_result(result);
        }

        self.assemble_diagnostics()
    }

    fn solve_cluster(&self, cluster: &RigidCluster) -> ClusterResult {
        // C1. Diagnostics
        let diag = self.analyze_cluster(cluster);
        if diag.is_conflicting() {
            return ClusterResult::Conflict(diag);
        }

        // C2. Symbolic reduction
        let reduced = self.reduce_cluster(cluster);

        // C3. Pattern matching
        if let Some(closed_form) = self.match_pattern(&reduced) {
            return closed_form;
        }

        // C4. Build SubProblem for numerical solver
        let sub = reduced.to_sub_problem(&self.store);

        // C5. Warm start from previous solution
        let x0 = cluster.last_solution.clone()
            .unwrap_or_else(|| sub.initial_point());

        // C6. Numerical solve
        let result = self.lm_solver.solve(&sub, &x0);

        // C7. Branch selection
        self.select_nearest_branch(result, &x0)
    }
}
```

---

## Symbolic Reduction

Before a cluster goes to the numerical solver, we simplify it:

### Substitution (reduce/substitute.rs)

Fixed parameters are not solver variables. But rather than the V1 approach of
skipping them in the flat vector, we substitute their known values directly
into constraint equations, producing a smaller system.

```
Before: Distance(p1, p2) where p1 is fixed at (0,0)
  Variables: [p1.x, p1.y, p2.x, p2.y] (4 vars, but p1 is fixed)
  Equation: (p2.x - p1.x)^2 + (p2.y - p1.y)^2 - d^2 = 0

After substitution:
  Variables: [p2.x, p2.y] (2 vars)
  Equation: p2.x^2 + p2.y^2 - d^2 = 0
  (Simpler Jacobian: [2*p2.x, 2*p2.y])
```

### Merging (reduce/merge.rs)

Coincident constraints merge two parameters into one. Instead of having
`p1.x, p2.x` as separate variables with `p1.x - p2.x = 0`, we replace all
occurrences of `p2.x` with `p1.x` and remove the constraint.

```
Before: Coincident(P1, P2), Distance(P2, P3)
  Variables: [p1.x, p1.y, p2.x, p2.y, p3.x, p3.y] (6 vars)
  Equations: [p2.x-p1.x=0, p2.y-p1.y=0, dist(P2,P3)=d] (3 eqs)

After merge (replace p2 -> p1):
  Variables: [p1.x, p1.y, p3.x, p3.y] (4 vars)
  Equations: [dist(P1,P3)=d] (1 eq)
  (Removed 2 variables AND 2 equations, net DOF unchanged)
```

### Trivial Elimination (reduce/eliminate.rs)

Some constraints directly determine a variable:
- `Horizontal(P1, P2)`: `p2.y = p1.y`. Substitute and remove.
- `Vertical(P1, P2)`: `p2.x = p1.x`. Substitute and remove.
- `Fixed(P, (a,b))`: `p.x = a, p.y = b`. Substitute and remove.

These are detected by inspecting constraint type and checking if one side
has a single free variable that appears in no other constraint.

---

## Pattern Matching for Closed-Form Solutions

### Known Solvable Patterns (graph/pattern.rs)

Instead of running Newton iterations, certain small subgraphs have analytical solutions:

**Pattern: Two-Point Distance (2D)**
```
  P1(fixed) ---[distance=d]--- P2(free, 2 vars)
  1 equation, 2 variables -> 1 DOF (circle of solutions)
  With drag hint: project drag target onto circle of radius d centered at P1
```

**Pattern: Triangle (2D)**
```
  P1(fixed) ---[d1]--- P2(free) ---[d2]--- P3(free) ---[d3]--- P1
  3 equations, 4 variables (P2.x, P2.y, P3.x, P3.y) -> 1 DOF
  After fixing one more variable (e.g., P2 horizontal from P1):
  Closed-form via law of cosines.
```

**Pattern: Line-Circle Intersection**
```
  Point on Line + Point on Circle
  Closed-form: solve quadratic.
```

**Pattern: Circle-Circle Intersection**
```
  Two circle equations -> 2 intersections or none.
  Closed-form: standard algebraic solution.
```

Pattern matching traverses the reduced cluster graph, looking for subgraphs that
match known templates. Matched subgraphs are solved analytically. Unmatched
portions go to the numerical solver.

---

## Diagnostics (graph/redundancy.rs, graph/dof.rs)

### Redundancy Detection

After decomposition and reduction, for each cluster:

1. Compute Jacobian J at current point.
2. Compute rank(J) via SVD or QR with column pivoting.
3. If rank(J) < equation_count: constraints are **redundant**.
4. Identify which constraints are linearly dependent by examining
   the null space of J^T.
5. Report: "Constraint C7 is implied by constraints C3, C4, C5."

### Conflict Detection

1. Run solver on the cluster.
2. If solver converges with residual_norm < tolerance: no conflict.
3. If solver does NOT converge:
   a. Attempt to solve subsets of constraints (progressive relaxation).
   b. The minimal conflicting subset is the smallest set where removal
      of any one constraint allows convergence.
   c. Report: "Constraints C2, C5, C8 are mutually conflicting."

### DOF Analysis (Per-Entity)

For underconstrained clusters (DOF > 0):

1. Compute the null space of J (SVD).
2. For each entity, check which of its parameters appear in the null vectors.
3. Report: "Point P3 can move in direction (0.6, 0.8)" or
   "Circle C1's radius is unconstrained."

---

## Drag Solving (solve/drag.rs)

When DOF > 0 and the user drags an entity:

1. Compute the null space N of J at the current solution.
2. Express the user's desired movement as a vector in parameter space.
3. Project the desired movement onto the null space: delta = N * N^T * desired.
4. Apply: x_new = x_current + delta.
5. Re-solve to satisfy constraints (the projection may violate them slightly).

This gives the "minimum perturbation" behavior: the solution moves in the direction
the user wants while staying as close as possible to the constraint manifold.

For well-constrained systems (DOF = 0), drag solving uses the solver's initial
point mechanism: set the dragged entity's parameters to the target position, then
solve. The warm start from the previous solution ensures convergence to the
nearest valid configuration.

---

## ReducedSubProblem (solve/sub_problem.rs)

This is what the numerical solver actually sees. It is the V2 equivalent of the
V1 `Problem` trait, but it's an internal struct, not a public trait.

```rust
pub struct ReducedSubProblem<'a> {
    store: &'a ParamStore,
    // After reduction: only free, non-substituted parameters
    free_params: Vec<ParamId>,
    // After reduction: only non-trivial constraints
    constraints: Vec<&'a dyn Constraint>,
    // Mapping: free_param index -> ParamId
    param_mapping: SolverMapping,
    // Substitution table: ParamId -> known value (for fixed/merged params)
    substitutions: HashMap<ParamId, f64>,
}

// Implements the internal Problem interface for the solver
impl ReducedSubProblem<'_> {
    pub fn variable_count(&self) -> usize { self.free_params.len() }
    pub fn residual_count(&self) -> usize {
        self.constraints.iter().map(|c| c.equation_count()).sum()
    }

    pub fn residuals(&self, x: &[f64]) -> Vec<f64> {
        // Write x into a temporary param store snapshot
        // Evaluate all constraints
    }

    pub fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        // Evaluate constraint Jacobians (which return ParamId)
        // Map ParamId -> column index via param_mapping
        // Filter out substituted params (they're constants, Jacobian = 0)
    }
}
```

The existing Newton-Raphson and Levenberg-Marquardt solvers operate on this.
Their code barely changes - they still see `residuals(&[f64]) -> Vec<f64>` and
`jacobian(&[f64]) -> Vec<(usize, usize, f64)>`. The intelligence is above them.

---

## Angle Constraint Formulation

V1 used `dy*cos(theta) - dx*sin(theta) = 0` which has the theta vs theta+pi ambiguity.

V2 uses the two-equation sin/cos formulation for constraints between two lines:

```rust
// constraint/angle.rs
// Angle between two lines: formulated as cross/dot of direction vectors.
//
// Let d1 = (dx1, dy1), d2 = (dx2, dy2) be direction vectors.
// cross = dx1*dy2 - dy1*dx2 = |d1|*|d2|*sin(angle)
// dot   = dx1*dx2 + dy1*dy2 = |d1|*|d2|*cos(angle)
//
// Residual: cross*cos(target) - dot*sin(target) = 0
//
// This is a single equation that correctly identifies the angle
// (not angle + pi) because it uses both sin and cos components.
// Still avoids atan2. Polynomial in the point coordinates.
```

For "angle from horizontal" (line angle constraint), the formulation is simpler:
```
// d = (dx, dy), target angle theta
// Residual: dy*cos(theta) - dx*sin(theta) = 0
// BUT we add a second soft residual to break the pi ambiguity:
//   dx*cos(theta) + dy*sin(theta) > 0 (as inequality or heavy penalty)
// This prefers the direction matching theta over theta+pi.
```

---

## What Gets Reused from V1

| V1 Component | V2 Status |
|---|---|
| Newton-Raphson solver (`newton_raphson.rs`) | **Reuse.** Minor interface change: takes `ReducedSubProblem` instead of `&dyn Problem`. |
| Levenberg-Marquardt solver (`levenberg_marquardt.rs`) | **Reuse.** Same minor interface change. |
| LM adapter (`lm_adapter.rs`) | **Reuse.** Wraps `ReducedSubProblem` for the `levenberg-marquardt` crate. |
| Sparse solver (`sparse_solver.rs`) | **Reuse.** |
| JIT solver (`jit_solver.rs`) | **Reuse** for future performance optimization of hot clusters. |
| Solver configs (`config.rs`, `lm_config.rs`) | **Reuse.** |
| SolveResult / SolveError (`result.rs`) | **Extend.** Add per-constraint status, diagnostics. |
| Union-find decomposition (`decomposition.rs`) | **Reuse** as small-system fast path. |
| Parallel solver (`parallel.rs`) | **Replace** with new cluster-based parallel solver (simpler, uses new types). |
| `Problem` trait (`problem.rs`) | **Delete** as public API. Internal `ReducedSubProblem` replaces it. |
| `GeometricConstraint<D>` trait | **Delete.** Replaced by `Constraint` trait with `ParamId`. |
| `ConstraintSystem<D>` | **Delete.** Replaced by new `ConstraintSystem` (not generic over D). |
| All 16 constraint implementations | **Rewrite** with squared formulations and ParamId-based API. |
| Point/Line/Circle/Vector types | **Rewrite** as entities with ParamId-based storage. |

---

## Implementation Order

### Phase 1: Foundation (IDs, ParamStore, Entity/Constraint traits)

Build `id.rs`, `param/store.rs`, `entity/mod.rs`, `constraint/mod.rs`.
Implement `Point2D`, `LineSegment2D`, `Circle2D` entities.
Implement `DistancePoint2D`, `CoincidentPoint2D`, `FixedPoint2D` constraints.
Test: create entities, add constraints, evaluate residuals/Jacobians.

### Phase 2: Constraint Graph + Decomposition

Build `graph/bipartite.rs`, `graph/decompose.rs`, `graph/cluster.rs`.
Port existing union-find. Test: build graph, decompose, verify clusters.

### Phase 3: ReducedSubProblem + Solver Integration

Build `solve/sub_problem.rs`. Wire to existing NR and LM solvers.
Build `system.rs` as the coordinator.
Test: end-to-end solve of triangle, rectangle, circle-tangent problems.
Verify all existing geometric test cases pass with new architecture.

### Phase 4: Symbolic Reduction

Build `reduce/substitute.rs`, `reduce/merge.rs`, `reduce/eliminate.rs`.
Test: verify reduction produces smaller systems, same solutions.

### Phase 5: Diagnostics

Build `graph/redundancy.rs`, `graph/dof.rs`.
Test: detect redundant constraints, conflicting constraints, per-entity DOF.

### Phase 6: Remaining 2D Entities + Constraints

Implement `Arc2D`, `Ellipse2D`, `BSpline2D`, `InfiniteLine2D`.
Implement all 2D constraint variants (tangent-arc, point-on-spline, etc.).
Test: comprehensive 2D sketch solving.

### Phase 7: Differential Dataflow + Incremental Solving

Build `dataflow/tracker.rs`, `dataflow/propagate.rs`, `dataflow/cache.rs`.
Implement incremental decomposition.
Test: change one parameter, verify only affected cluster re-solves.
Benchmark: incremental vs full re-solve on 100+ entity systems.

### Phase 8: 3D Entities + Assembly Constraints

Implement `Point3D`, `LineSegment3D`, `Plane`, `Axis3D`, `Sphere`, `Cylinder`, etc.
Implement `RigidBody` with quaternion parameterization.
Implement assembly constraints: `Mate`, `Coaxial`, `Insert`.
Test: basic assembly positioning.

### Phase 9: Advanced Solving

Build `solve/drag.rs`, `solve/branch.rs`, `solve/closed_form.rs`.
Build `graph/pattern.rs` for closed-form pattern matching.
Test: drag solving, branch selection, analytical solution of triangles.

### Phase 10: Performance + Polish

Integrate JIT solver for hot clusters.
Tune decomposition thresholds (union-find vs DD).
Benchmark suite: 10/100/1000/10000 entity systems.
API stabilization and documentation.

---

## Comparison: V1 vs V2

| Aspect | V1 (Current) | V2 (This Plan) |
|---|---|---|
| Entity model | Points only | 20+ entity types with params |
| Constraint binding | Point indices | ParamId (entity-aware) |
| Variable tracking | Flat `&[f64]`, manual index math | `ParamStore` with `SolverMapping` |
| Distance formulation | `sqrt(d^2) - target` (singular at 0) | `d^2 - target^2` (polynomial) |
| Tangent formulation | `abs(cross)/len - r` (non-smooth) | `cross^2 - r^2*len^2` (polynomial) |
| Graph decomposition | Union-find (good) | Union-find + incremental DD |
| Symbolic reduction | None | Substitution + merge + elimination |
| Redundancy detection | None | Jacobian rank analysis |
| Conflict detection | None | Progressive relaxation |
| DOF analysis | Global count only | Per-entity, per-cluster, with directions |
| Drag solving | None | Null-space projection |
| Branch selection | Whatever NR converges to | Closest to previous configuration |
| Incremental solving | None | DD change tracking, per-cluster |
| Pattern matching | None | Closed-form for known subgraphs |
| Circle radius | Constant, baked into constraint | Variable `ParamId`, solvable |
| Public API | `Problem` trait (user implements) | `ConstraintSystem` (user adds entities) |
| Solver interface | Public `Problem` trait | Internal `ReducedSubProblem` |

---

## Open Questions

1. **DD crate vs custom?** The `differential-dataflow` crate (timely dataflow) is heavy
   machinery. For the constraint graph (hundreds of nodes, not billions), a custom
   incremental connected-components algorithm may be simpler and faster. Decision:
   start with custom incremental union-find, measure, upgrade to DD crate only if needed.

2. **Quaternion parameterization for rigid bodies?** Quaternions have 4 components but
   3 DOF (unit constraint). Options: (a) use 4 params + unit constraint, (b) use
   Rodrigues vector (3 params, singularity at pi), (c) use rotation matrix entries
   (9 params, 6 constraints). Decision: quaternion + unit constraint is standard.

3. **BSpline control point params?** A degree-3 BSpline with 10 control points has
   20 parameters in 2D. This makes entities with variable param counts. The `Entity`
   trait returns `&[ParamId]` which handles this, but storage is heap-allocated.
   Decision: acceptable. BSplines are rare enough that the allocation doesn't matter.

4. **Constraint factory dispatch?** When the user says "add tangent constraint between
   entity A and entity B", the system must pick the right concrete struct
   (`TangentLineCircle2D`, `TangentCircleCircle2D`, etc.) based on entity kinds.
   Decision: `ConstraintRegistry` has a `create_tangent(EntityId, EntityId)` method
   that dispatches on `(entity_a.kind(), entity_b.kind())`.
