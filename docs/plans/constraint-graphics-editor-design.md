# Constraint-Based 2D Vector Graphics Editor: Architecture & Design

## Table of Contents

1. [Academic State of the Art](#1-academic-state-of-the-art)
2. [How Solverang Maps to the Literature](#2-how-solverang-maps-to-the-literature)
3. [Translating Vector Operations to Constraints](#3-translating-vector-operations-to-constraints)
4. [Editor Architecture](#4-editor-architecture)
5. [UX Design](#5-ux-design)
6. [Test Suite Design](#6-test-suite-design)
7. [Key References](#7-key-references)

---

## 1. Academic State of the Art

### 1.1 Historical Arc

The field of constraint-based graphics begins with **Sutherland's Sketchpad (1963)**, which demonstrated that users could specify *design intent* (relationships like parallelism, equal lengths, perpendicularity) rather than absolute coordinates, and a constraint solver would maintain those relationships during editing. Every system since follows this paradigm.

Three major eras:

| Era | Focus | Key Systems |
|-----|-------|-------------|
| 1963-1990 | Foundations | Sketchpad, Juno, ThingLab |
| 1990-2005 | Graph-based decomposition | Owen, Hoffmann/Fudos, Kramer DOF analysis |
| 2005-2020 | Declarative/variational maturation | Penrose, Sketch-n-Sketch, Cassowary in Apple AutoLayout |
| 2020-present | ML augmentation + differentiable approaches | SketchGraphs, SketchGen, AIDL, Minkowski Penalties |

### 1.2 Solving Strategy Taxonomy

| Strategy | Method | Strengths | Weaknesses |
|----------|--------|-----------|------------|
| **Graph-based decomposition** | Translate constraints to a bipartite graph, decompose into triangular or tree subproblems (Owen 1991, Hoffmann/Fudos 1997, Sitharam's Frontier Algorithm) | Fast for 2D; dominant in CAD; polynomial-time decomposition | Hard to extend to 3D; can be misled by geometric theorems |
| **Numerical (Newton-Raphson / LM)** | Model all constraints as F(x)=0, solve iteratively | General; handles all constraint types | Requires good initial values; local convergence only |
| **Optimization-based** | Minimize sum of constraint violations as objective | Handles infeasible initialization; graceful degradation | Local minima; can be slower |
| **Symbolic** | Wu-Ritt characteristic sets, Grobner bases | Exact solutions | Computationally expensive (double-exponential worst case) |
| **Constraint propagation** | Incremental simplex (Cassowary) | Fast for linear systems; incremental | Limited to linear equalities/inequalities |
| **Hybrid** | Graph decomposition + numerical per-cluster | Best of both worlds | Implementation complexity |

**The state of the art for 2D CAD constraint solving is the hybrid approach**: graph-based decomposition to partition the problem into small clusters, then numerical solving (Newton-Raphson or Levenberg-Marquardt) for each irreducible cluster. This is the architecture used by DCM/D-Cubed (Siemens), LGS (LEDAS), SolveSpace, and virtually all commercial parametric CAD systems.

### 1.3 Degrees of Freedom Analysis

Following Kramer (1991), every geometric entity has a DOF count and every constraint removes DOF:

| Entity | DOF in 2D |
|--------|-----------|
| Point | 2 (x, y) |
| Line segment | 4 (x1, y1, x2, y2) |
| Circle (variable radius) | 3 (cx, cy, r) |
| Arc | 5 (cx, cy, r, start_angle, end_angle) |
| Cubic Bezier segment | 8 (4 control points x 2 coords) |
| Quadratic Bezier segment | 6 (3 control points x 2 coords) |

| Constraint | DOF removed |
|------------|-------------|
| Distance (point-point) | 1 |
| Coincident | 2 |
| Fixed position | 2 |
| Horizontal / Vertical | 1 |
| Angle | 1 |
| Parallel | 1 |
| Perpendicular | 1 |
| Tangent (line-circle) | 1 |
| Tangent (circle-circle) | 1 |
| Point on circle | 1 |
| Equal length | 1 |
| Midpoint | 2 |
| Symmetric | 2 |
| G1 continuity (tangent at junction) | 2 |
| G2 continuity (curvature match) | 3 |

A system with `n` total DOF and `m` total constraint equations is:
- **Well-constrained** when m = n (unique solution, up to discrete ambiguity)
- **Under-constrained** when m < n (remaining DOF are free for dragging)
- **Over-constrained** when m > n (least-squares best fit via LM)

**Laman's theorem** (for bar-joint systems in 2D): a graph G=(V,E) with |V|=n, |E|=m is generically rigid iff m = 2n-3 and every subgraph on n' vertices has at most 2n'-3 edges.

### 1.4 Key Open Problems

1. **Design intent preservation** — Autodesk's 2025 work uses RL alignment (DPO, GRPO) with a constraint solver in the feedback loop to train models that generate constraints matching designer intent. Achieves 93% fully-constrained sketches vs. 34% with naive fine-tuning.

2. **Solution multiplicity** — A well-constrained system may have exponentially many valid realizations. Sitharam's Equation and Solution Manager (ESM) offers incremental visual walk-through to avoid combinatorial explosion.

3. **Under/over-constrained systems** — Real user systems almost always contain mixed under- and over-constrained parts. Correct detection before numerical solving is essential (Zou et al., 2022 survey).

4. **Curves beyond lines and circles** — Constraint solving for free-form curves (Bezier, B-spline, NURBS) is substantially harder and less explored. Control points provide natural parameterization, but geometric continuity constraints (G1, G2) introduce coupling between adjacent curve segments.

5. **Inequality constraints** — Needed for proper design intent (forcing geometric objects to specific sides), but most solvers handle them poorly.

### 1.5 Relevant Systems

| System | Architecture | Key Innovation |
|--------|-------------|----------------|
| **SolveSpace** | Modified Newton's method on symbolic equations | Open-source parametric 2D/3D CAD; standalone solver library |
| **Penrose** (CMU, SIGGRAPH 2020) | Constrained numerical optimization from declarative specs | Translates mathematical notation to diagrams via constraint optimization |
| **Sketch-n-Sketch** (UIST 2019) | Trace-based program synthesis for SVG | Bidirectional: direct manipulation infers program updates |
| **Cassowary** (Badros et al., 2001) | Incremental dual simplex for linear constraints | Basis of Apple AutoLayout, Android ConstraintLayout, GTK4 |
| **Minkowski Penalties** (SIGGRAPH 2024) | Optimization using signed distance of Minkowski difference | Useful gradients from infeasible states; handles open curves |
| **diffvg** (Li et al., SIGGRAPH Asia 2020) | Differentiable vector graphics rasterization | Bridges vector/raster for optimization under constraints |

---

## 2. How Solverang Maps to the Literature

### 2.1 Current Architecture

Solverang already implements the two critical components of the hybrid approach:

1. **Graph-based decomposition** — The `decomposition` module uses union-find to detect connected components in the constraint-variable bipartite graph, and the `ParallelSolver` solves independent components concurrently.

2. **Numerical solving** — Multiple solver backends (Newton-Raphson, Levenberg-Marquardt, Sparse, JIT-compiled) handle each irreducible cluster.

The existing `sketch2d` module provides the entity/constraint vocabulary:

```
Entities:  Point2D, LineSegment2D, Circle2D, Arc2D, InfiniteLine2D
Constraints: DistancePtPt, DistancePtLine, Coincident, Parallel,
             Perpendicular, Angle, Horizontal, Vertical, Fixed,
             Midpoint, Symmetric, EqualLength, PointOnCircle,
             TangentLineCircle, TangentCircleCircle
```

The pipeline architecture (`pipeline/`) implements the five-phase approach:
```
Decompose → Analyze → Reduce → Solve → PostProcess
```

### 2.2 Gaps Relative to a Full Graphics Editor

| Capability | Status | What's Needed |
|-----------|--------|---------------|
| Points, lines, circles, arcs | Implemented | — |
| Bezier curves (quadratic, cubic) | **Missing** | New entities + constraints |
| Splines (composite Bezier/B-spline) | **Missing** | Spline entity + continuity constraints |
| Ellipses | **Missing** | Ellipse entity (5 DOF: cx, cy, rx, ry, rotation) |
| G0/G1/G2 continuity constraints | **Missing** | Junction constraints between curve segments |
| Point-on-curve constraints | **Partial** (PointOnCircle exists) | Generalize to PointOnBezier, PointOnArc |
| Tangent-at-point constraints | **Missing** | Constrain tangent direction at specific curve parameter |
| DOF analysis and reporting | **Partial** (equation/param counting) | Full structural analysis per Laman's theorem |
| Dragging / incremental re-solve | **Partial** (warm-start in ParallelSolver) | Full differential manipulation (Gleicher) |
| SVG/PDF export | **Missing** | Rendering layer |

---

## 3. Translating Vector Operations to Constraints

This section defines how standard 2D vector graphics primitives map to solverang's entity/constraint model.

### 3.1 Cubic Bezier Curve

A cubic Bezier segment is defined by four control points P0, P1, P2, P3:

```
B(t) = (1-t)^3 * P0 + 3(1-t)^2*t * P1 + 3(1-t)*t^2 * P2 + t^3 * P3
```

**Entity**: `CubicBezier2D` with 8 parameters: `[x0, y0, x1, y1, x2, y2, x3, y3]`

**Tangent at endpoints**:
- At t=0: tangent direction is `P1 - P0`
- At t=1: tangent direction is `P3 - P2`

**Constraints on a cubic Bezier**:

| Constraint | Residual | Equations |
|-----------|----------|-----------|
| Fix endpoint P0 | `[x0 - tx, y0 - ty]` | 2 |
| Fix endpoint P3 | `[x3 - tx, y3 - ty]` | 2 |
| Endpoint coincident with point | `[x0 - px, y0 - py]` | 2 |
| Tangent direction at P0 | `(y1-y0)*cos(a) - (x1-x0)*sin(a)` | 1 |
| Tangent direction at P3 | `(y3-y2)*cos(a) - (x3-x2)*sin(a)` | 1 |
| Tangent horizontal at P0 | `y1 - y0` | 1 |
| Tangent vertical at P3 | `x3 - x2` | 1 |
| Point on curve at parameter t | `B(t) - P` (substituted directly) | 2 |
| Curve length (approximate) | Gauss-Legendre quadrature of `\|B'(t)\|` | 1 |

### 3.2 Quadratic Bezier Curve

**Entity**: `QuadBezier2D` with 6 parameters: `[x0, y0, x1, y1, x2, y2]`

```
B(t) = (1-t)^2 * P0 + 2(1-t)*t * P1 + t^2 * P2
```

Same constraint patterns as cubic, but simpler. Tangent at t=0 is `P1 - P0`, at t=1 is `P2 - P1`.

### 3.3 Circular Arc

Already implemented as `Arc2D` with parameters `[cx, cy, r, start_angle, end_angle]`.

**Derived constraints**:

| Constraint | Residual | Equations |
|-----------|----------|-----------|
| Arc start point at P | `[cx + r*cos(a0) - px, cy + r*sin(a0) - py]` | 2 |
| Arc end point at P | `[cx + r*cos(a1) - px, cy + r*sin(a1) - py]` | 2 |
| Arc tangent at start | Direction is `(-sin(a0), cos(a0))` — perpendicular to radius | 1 |
| Arc sweep angle | `a1 - a0 - target_sweep` | 1 |
| Arc tangent to line at junction | Match tangent direction of arc endpoint with line direction | 1 |

### 3.4 Ellipse

**Entity**: `Ellipse2D` with 5 parameters: `[cx, cy, rx, ry, theta]`

Parametric form:
```
E(t) = [cx + rx*cos(t)*cos(theta) - ry*sin(t)*sin(theta),
         cy + rx*cos(t)*sin(theta) + ry*sin(t)*cos(theta)]
```

**Constraints**: Point-on-ellipse, tangent-to-ellipse, focus distance, eccentricity.

### 3.5 Composite Splines (G0/G1/G2 Continuity)

A spline is a sequence of curve segments joined end-to-end. The key constraints are continuity conditions at junction points:

**G0 (positional continuity)**: Endpoints coincide.
```
Residual: [x3_i - x0_{i+1}, y3_i - y0_{i+1}]   (2 equations)
```

**G1 (tangent continuity)**: Tangent directions match at junction.
For two cubic Bezier segments i and i+1 meeting at a junction:
```
Residual: (x3_i - x2_i) * (y1_{i+1} - y0_{i+1}) - (y3_i - y2_i) * (x1_{i+1} - x0_{i+1})
```
This is the cross product of the two tangent vectors = 0 (parallel condition). Combined with G0, this gives 3 equations total per junction.

**G2 (curvature continuity)**: Curvature magnitudes match at junction.
For cubic Beziers, curvature at t=0 is:
```
kappa = |P0P1 x P0P2| / |P0P1|^3    (cross product over cube of distance)
```
Matching curvature at the junction adds 1 more equation (4 total per G2 junction).

**Implementation strategy**: Each continuity level is a separate constraint type that references the parameters of two adjacent segments. The builder automatically generates these when constructing a spline.

### 3.6 Tangent Constraints for Curves

| Scenario | Formulation |
|----------|-------------|
| **Line tangent to circle** | Already implemented: `cross^2/len_sq - r^2 = 0` |
| **Line tangent to Bezier at endpoint** | Bezier tangent at t=0 is `P1-P0`; constrain parallel to line direction: cross product = 0 |
| **Circle tangent to Bezier** | At the tangent point (parameter t*), distance from center to B(t*) = r AND B'(t*) is perpendicular to (B(t*) - center). This requires introducing t* as an auxiliary variable. |
| **Bezier tangent to Bezier** | At junction: G1 continuity constraint (see above) |
| **Arc tangent to line at junction** | Arc tangent direction at endpoint is perpendicular to radius; constrain parallel to line direction |

### 3.7 Mapping Common Vector Editor Operations

| Editor Operation | Constraint Translation |
|-----------------|----------------------|
| **Draw rectangle** | 4 points, 4 line segments, 4 perpendicular constraints, optionally equal-length for square |
| **Draw rounded rectangle** | 4 line segments + 4 arcs, with G1 continuity at each line-arc junction, equal radii |
| **Draw regular polygon** | N points, N equal-length constraints, N equal-angle constraints |
| **Draw circle** | Circle2D entity, optionally fixed center/radius |
| **Draw ellipse** | Ellipse2D entity, or approximate with 4 cubic Bezier segments |
| **Draw Bezier path** | Sequence of CubicBezier2D with G0/G1/G2 at junctions |
| **Align objects** | Horizontal/Vertical constraints on corresponding points |
| **Distribute evenly** | Equal-distance constraints between consecutive objects |
| **Mirror** | Symmetric constraints about an axis |
| **Snap to grid** | Fixed constraints at grid-aligned coordinates |
| **Fillet (rounding a corner)** | Replace corner with arc; add tangent constraints to adjacent lines |
| **Chamfer** | Replace corner with line segment; add distance constraints from corner |
| **Offset path** | Parallel constraints + distance constraints for each segment |
| **Boolean operations** | Compute intersections, then re-constrain the resulting geometry |

---

## 4. Editor Architecture

### 4.1 Layer Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 5: UI / Rendering                                     │
│  Canvas rendering (SVG/WebGL), tool palette, property panel  │
│  Direct manipulation (drag handles, snapping, selection)     │
├─────────────────────────────────────────────────────────────┤
│  Layer 4: Editor Document Model                              │
│  Scene graph, undo/redo, selection state, tool state machine │
│  Serialization (save/load), SVG/PDF export                   │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: Constraint Manager                                 │
│  DOF analysis, constraint inference, auto-constrain          │
│  Drag-solve loop, incremental re-solve, warm-start           │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: Sketch2D Facade (existing)                         │
│  Sketch2DBuilder, entity/constraint vocabulary               │
│  Entity ↔ ParamId mapping, constraint wiring                 │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: Solver Core (existing)                             │
│  Pipeline: Decompose → Analyze → Reduce → Solve → PostProcess│
│  NR / LM / Sparse / Parallel / JIT backends                 │
│  ParamStore, generational IDs, Constraint trait              │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Component Details

#### Layer 1: Solver Core (Exists)

No changes needed. The existing solver infrastructure (Newton-Raphson, Levenberg-Marquardt, sparse solver, parallel decomposition, JIT compilation) provides the numerical backbone.

#### Layer 2: Sketch2D Facade (Exists, Needs Extension)

**Extensions needed:**

```rust
// New entities
pub struct CubicBezier2D {
    id: EntityId,
    // Control points: P0(x0,y0), P1(x1,y1), P2(x2,y2), P3(x3,y3)
    x0: ParamId, y0: ParamId,
    x1: ParamId, y1: ParamId,
    x2: ParamId, y2: ParamId,
    x3: ParamId, y3: ParamId,
    params: [ParamId; 8],
}

pub struct QuadBezier2D {
    id: EntityId,
    x0: ParamId, y0: ParamId,
    x1: ParamId, y1: ParamId,
    x2: ParamId, y2: ParamId,
    params: [ParamId; 6],
}

pub struct Ellipse2D {
    id: EntityId,
    cx: ParamId, cy: ParamId,
    rx: ParamId, ry: ParamId,
    rotation: ParamId,
    params: [ParamId; 5],
}
```

**New constraints:**

```rust
// G1 continuity between two cubic Bezier segments
pub struct G1Continuity {
    // Segment A end tangent parallel to Segment B start tangent
    // Residual: (x3a-x2a)*(y1b-y0b) - (y3a-y2a)*(x1b-x0b) = 0
    // Plus G0: x3a-x0b=0, y3a-y0b=0
}

// G2 continuity (curvature match)
pub struct G2Continuity {
    // G1 + curvature magnitude equality at junction
}

// Point on cubic Bezier at fixed parameter t
pub struct PointOnCubicBezier {
    // Residual: B(t) - P = [(1-t)^3*x0 + 3(1-t)^2*t*x1 + ... - px,
    //                        (1-t)^3*y0 + ... - py]
}

// Arc endpoint coincidence (arc start/end point = given point)
pub struct ArcEndpoint {
    // Residual: [cx + r*cos(angle) - px, cy + r*sin(angle) - py]
}

// Tangent direction at arc endpoint
pub struct ArcTangentDirection {
    // At start: tangent is (-sin(a0), cos(a0))
    // Constrain parallel to given direction
}
```

#### Layer 3: Constraint Manager (New)

This is the core new component that bridges the editor and the solver.

```rust
pub struct ConstraintManager {
    builder: Sketch2DBuilder,

    // DOF tracking
    total_dof: usize,           // Sum of entity DOFs
    constrained_dof: usize,     // Sum of constraint equations
    remaining_dof: usize,       // total - constrained (free DOF)

    // Solve state
    last_solution: Option<Vec<f64>>,  // For warm-starting
    dirty: bool,                       // Constraints changed since last solve

    // Constraint inference
    snap_tolerance: f64,        // Distance threshold for auto-snap
    angle_snap_degrees: f64,    // Angle threshold for auto-horizontal/vertical
}

impl ConstraintManager {
    /// Solve the current constraint system, warm-starting from previous solution.
    pub fn solve(&mut self) -> SolveResult { ... }

    /// Called during drag: update the dragged point's "target" and re-solve.
    /// Uses the previous solution as initial guess for fast convergence.
    pub fn drag_solve(&mut self, entity: EntityId, new_pos: (f64, f64)) -> SolveResult { ... }

    /// Analyze DOF status of the entire system.
    pub fn analyze_dof(&self) -> DofAnalysis { ... }

    /// Suggest constraints based on current geometry (auto-constrain).
    pub fn suggest_constraints(&self) -> Vec<SuggestedConstraint> { ... }

    /// Check if adding a constraint would over-constrain the system.
    pub fn would_overconstrain(&self, constraint: &dyn Constraint) -> bool { ... }
}
```

**Drag-solve loop** (following Gleicher's differential manipulation):

```
1. User starts dragging entity E
2. Temporarily add soft constraint: E.position = mouse_position (high weight)
3. Solve (warm-start from current state) → converges in 1-3 iterations
4. Update canvas with new positions
5. On mouse move: update target position, re-solve
6. On mouse release: remove soft constraint, final solve
```

This gives real-time interactive feedback because:
- The warm-start means the solver starts very close to the solution
- The perturbation is small (one frame's worth of mouse movement)
- Newton-Raphson converges quadratically near the solution

#### Layer 4: Editor Document Model (New)

```rust
pub struct EditorDocument {
    constraint_manager: ConstraintManager,

    // Scene graph
    objects: Vec<GraphicObject>,    // Visual objects (shapes, paths, groups)

    // Undo/redo
    history: UndoStack<EditorAction>,

    // Selection
    selection: HashSet<ObjectId>,

    // Tool state
    active_tool: Box<dyn Tool>,
}

pub enum GraphicObject {
    Path(PathObject),           // Composite Bezier path
    Rectangle(RectangleObject), // Constrained rectangle
    Circle(CircleObject),       // Constrained circle
    Ellipse(EllipseObject),
    Group(GroupObject),
    // Each wraps entity IDs + constraint IDs from the solver
}

pub trait Tool {
    fn on_mouse_down(&mut self, doc: &mut EditorDocument, pos: (f64, f64));
    fn on_mouse_move(&mut self, doc: &mut EditorDocument, pos: (f64, f64));
    fn on_mouse_up(&mut self, doc: &mut EditorDocument, pos: (f64, f64));
    fn on_key(&mut self, doc: &mut EditorDocument, key: Key);
}
```

#### Layer 5: UI / Rendering (New, Platform-Dependent)

Two viable rendering strategies:

1. **SVG-based** (web): Each solved entity emits SVG path elements. The constraint manager's solution updates SVG attributes directly.

2. **Immediate-mode** (native): Render to a GPU canvas (wgpu/skia) each frame using the solved parameter values.

### 4.3 Data Flow

```
User Action (draw, drag, add constraint)
    │
    ▼
Editor Document Model
    │ translate to entities + constraints
    ▼
Constraint Manager
    │ warm-start, DOF check
    ▼
Sketch2D Builder → ConstraintSystem
    │
    ▼
Pipeline: Decompose → Analyze → Reduce → Solve → PostProcess
    │                                       │
    │                    ┌──────────────────┘
    │                    ▼
    │              NR / LM / Sparse / JIT
    │                    │
    ▼                    ▼
ParamStore (updated with solution)
    │
    ▼
Read solved positions → Update scene graph → Render
```

### 4.4 Incremental Solving Strategy

For interactive editing, full re-solve on every frame is too expensive for large sketches. The pipeline's decomposition phase enables incremental solving:

1. **Component isolation**: Only re-solve components containing changed entities.
2. **Warm-start**: Use previous solution as initial guess (1-3 NR iterations typically suffice for small perturbations).
3. **Dirty tracking**: The `dataflow` module's cache tracks which parameters changed and which clusters need re-solving.
4. **JIT amortization**: For large, frequently re-solved systems, JIT-compile the residual/Jacobian functions once and reuse across frames.

---

## 5. UX Design

### 5.1 Constraint Visibility

Following SolveSpace and Onshape conventions:

- **Well-constrained entities**: Rendered in **green** (or default color)
- **Under-constrained entities**: Rendered in **blue** with visible DOF indicators (arrows showing free directions)
- **Over-constrained entities**: Rendered in **red** with conflicting constraints highlighted
- **Fixed entities**: Rendered with anchor icon

Constraints themselves are shown as annotations:
- Distance: dimension line with value
- Angle: arc with degree label
- Horizontal/Vertical: small H/V icon
- Coincident: concentric circle marker
- Tangent: T marker at tangent point
- Parallel: parallel lines icon (‖)
- Perpendicular: right-angle marker (⊥)

### 5.2 Tool Modes

| Tool | Behavior |
|------|----------|
| **Select/Move** | Click to select; drag to move (triggers drag-solve loop maintaining all constraints) |
| **Point** | Click to place point; auto-snap to existing geometry |
| **Line** | Click two endpoints; auto-creates two Point2D + LineSegment2D |
| **Rectangle** | Click-drag for bounding box; creates 4 points + 4 line segments + 4 perpendicular + 2 horizontal + 2 vertical constraints |
| **Circle** | Click center, drag radius; creates Circle2D |
| **Arc** | Click center, start angle, end angle; creates Arc2D |
| **Bezier Pen** | Click for anchor points; drag for control points; creates CubicBezier2D chain with G1 continuity |
| **Constrain** | Sub-tools for each constraint type; click two entities to constrain |
| **Dimension** | Click entity or pair → enter numerical value → creates distance/angle/radius constraint |

### 5.3 Auto-Constraint Inference

When the user draws or moves geometry, automatically suggest constraints based on proximity:

```rust
fn suggest_constraints(manager: &ConstraintManager, tolerance: f64) -> Vec<Suggestion> {
    let mut suggestions = vec![];

    // 1. Near-coincident points (distance < tolerance)
    for (p1, p2) in all_point_pairs() {
        if distance(p1, p2) < tolerance {
            suggestions.push(Suggestion::Coincident(p1, p2));
        }
    }

    // 2. Near-horizontal/vertical lines (angle within snap_degrees of 0/90)
    for line in all_lines() {
        let angle = line.angle();
        if (angle % 90.0).abs() < snap_degrees {
            suggestions.push(Suggestion::Horizontal(line) /* or Vertical */);
        }
    }

    // 3. Near-tangent line-circle pairs
    for (line, circle) in all_line_circle_pairs() {
        if (signed_distance(line, circle.center) - circle.radius).abs() < tolerance {
            suggestions.push(Suggestion::Tangent(line, circle));
        }
    }

    // 4. Near-equal-length segments
    for (l1, l2) in all_line_pairs() {
        if (l1.length() - l2.length()).abs() < tolerance {
            suggestions.push(Suggestion::EqualLength(l1, l2));
        }
    }

    suggestions
}
```

Suggestions are displayed as dashed constraint indicators; the user confirms or dismisses.

### 5.4 Constraint Conflict Resolution

When a user adds a constraint that would over-constrain the system:

1. **Detect**: `would_overconstrain()` checks if adding the constraint makes m > n.
2. **Warn**: Show the conflicting constraint set (highlight in red).
3. **Options**:
   - Add anyway (system becomes over-constrained; LM solver finds least-squares best fit)
   - Replace an existing constraint
   - Cancel

### 5.5 Keyboard Shortcuts

| Key | Action |
|-----|--------|
| H | Add horizontal constraint to selection |
| V | Add vertical constraint to selection |
| D | Add distance/dimension constraint |
| T | Add tangent constraint |
| P | Add parallel constraint |
| E | Add equal-length constraint |
| F | Fix selected point |
| Delete | Remove selected constraint |
| Tab | Cycle through constraint suggestions |

---

## 6. Test Suite Design

The test suite validates the constraint solver against known geometric configurations. Each test constructs a geometric figure from initial (possibly perturbed) values, applies constraints, solves, and verifies the result matches the expected geometry.

### 6.1 Test Categories

| Category | Tests | Purpose |
|----------|-------|---------|
| **Primitive shapes** | Triangle, square, regular polygon, circle | Basic constraint satisfaction |
| **Tangency** | Line-circle, circle-circle, arc-line | Tangent constraint correctness |
| **Continuity** | G0/G1/G2 Bezier junctions | Spline continuity constraints |
| **Composite shapes** | Rounded rectangle, filleted corner, star | Multi-constraint interaction |
| **Symmetry** | Symmetric pairs, mirrored shapes | Symmetric constraint correctness |
| **Pathological** | Over-constrained, under-constrained, near-singular | Robustness |
| **Scale** | 100+ constraint systems | Performance and decomposition |

### 6.2 Test Implementation

See `tests/graphics2d_test_suite.rs` for the full implementation.

Each test follows this pattern:

```rust
#[test]
fn test_SHAPE_NAME() {
    // 1. Create builder
    let mut b = Sketch2DBuilder::new();

    // 2. Add entities with approximate initial positions
    let p0 = b.add_point(x0_approx, y0_approx);
    // ...

    // 3. Add constraints
    b.constrain_distance(p0, p1, expected_distance);
    // ...

    // 4. Build and solve
    let system = b.build();
    let solution = solve_system(&system);

    // 5. Verify geometric properties
    assert_converged(&solution);
    assert_distance(solution, p0, p1, expected_distance, 1e-8);
    // ...
}
```

---

## 7. Key References

### Foundational
- Sutherland, I.E. "Sketchpad: A Man-Machine Graphical Communication System" (1963). The origin of constraint-based graphics.
- Kramer, G.A. "Using Degrees of Freedom Analysis to Solve Geometric Constraint Systems" (ACM Solid Modeling 1991). DOF-based symbolic reasoning.
- Owen, J.C. "Algebraic Solution for Geometry from Dimensional Constraints" (1991). Tri-connected graph decomposition.

### Graph Decomposition
- Fudos, I. & Hoffmann, C.M. "A Graph-Constructive Approach to Solving Systems of Geometric Constraints" (ACM TOG 1997). Reduction vs. decomposition analysis.
- Joan-Arinyo, R. et al. "Revisiting Decomposition Analysis of Geometric Constraint Graphs" (ACM 2003). Proves reduction and decomposition are equivalent.
- Sitharam, M. et al. "The Modified Frontier Algorithm" (arXiv:1507.01158). Optimal DR-planning in polynomial time.

### Surveys
- Hoffmann, C.M. & Joan-Arinyo, R. "A Brief on Constraint Solving" (CAD Journal 2005).
- Zou, H. et al. "A Review on Geometric Constraint Solving" (arXiv:2202.13795, 2022). Most comprehensive recent review.
- Bettig, B. & Hoffmann, C.M. "Geometric Constraint Solving in Parametric Computer-Aided Design" (ASME 2011).

### Constraint-Based Drawing
- Gleicher, M. & Witkin, A. "Drawing with Constraints" (Visual Computer 1994). Differential manipulation.
- Gleicher, M. "A Differential Approach to Graphical Interaction" (PhD thesis). Uniform Jacobian-based editing.

### Declarative/Optimization
- Ye, K. et al. "Penrose: From Mathematical Notation to Beautiful Diagrams" (SIGGRAPH 2020).
- Minarcik, J. et al. "Minkowski Penalties" (SIGGRAPH 2024). Shape arrangement via Minkowski signed distance.
- Li, T.-M. et al. "Differentiable Vector Graphics Rasterization" (SIGGRAPH Asia 2020).

### UI Layout
- Badros, G.J., Borning, A., & Stuckey, P.J. "The Cassowary Linear Arithmetic Constraint Solving Algorithm" (ACM TOCHI 2001).

### Programmatic/Bidirectional
- Chugh, R. et al. "Sketch-n-Sketch: Output-Directed Programming for SVG" (UIST 2019).

### ML-Augmented
- Seff, A. et al. "SketchGraphs" (2020). Dataset of 15M CAD sketches as constraint graphs.
- Para, W.R. et al. "SketchGen" (NeurIPS 2021). Autoregressive CAD sketch synthesis.
- Casey, J. et al. "Aligning Constraint Generation with Design Intent in Parametric CAD" (Autodesk 2025). RL alignment with solver-in-the-loop.
- Jones, B.D. et al. "AIDL: A Solver-Aided Hierarchical Language for LLM-Driven CAD Design" (CGF 2025).
