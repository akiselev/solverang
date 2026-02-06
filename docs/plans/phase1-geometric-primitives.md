# Phase 1: Geometric Primitives — Detailed Implementation Plan

## The Problem

`ConstraintSystem<D>` stores only `Vec<Point<D>>`. Every geometric entity is decomposed
into point indices. Non-point parameters like radius and angle are **constants** baked into
constraint structs (e.g., `PointOnCircleConstraint.radius: f64`,
`LineTangentConstraint.radius: f64`). The solver cannot vary these parameters.

This means:
- You can't ask "find the radius that makes this circle tangent to these two lines"
- You can't have "equal radius" between two circles (no radius variable to equate)
- You can't represent an arc whose angular extent is a solver unknown
- You can't express G1/G2 continuity at curve junctions naturally

## Design Decision: Points + Scalar Parameters

We extend the system with **scalar parameters** alongside points. This is the minimal
change that unlocks all curve primitives while preserving backward compatibility.

### Why not a flat parameter vector (SolveSpace style)?

A fully flat `Vec<f64>` where points and scalars are intermixed is more general but
requires rewriting the entire entity model, the builder, the Problem impl, and every
constraint. Points-plus-scalars gets us 95% of the benefit with 20% of the churn.

### Why not encode scalars as extra "virtual points"?

Encoding a radius as a `Point<2>` with a meaningless y-coordinate is hacky, wastes
solver effort on the phantom coordinate, and produces confusing DOF counts.

### Variable Vector Layout

```
[free_point_0.x, free_point_0.y, ..., free_point_N.x, free_point_N.y, free_param_0, free_param_1, ...]
 ├─────────── free_point_count * D ───────────────────┤├──── free_param_count ────┤
```

Point variables occupy columns `0..free_point_count*D` (unchanged from today).
Param variables occupy columns `free_point_count*D..free_point_count*D+free_param_count`.

### Jacobian Column Convention

Constraints already output `(row, col, value)` triplets where `col = point_idx * D + coord`.
We add a sentinel offset for param columns:

```rust
pub const PARAM_COL_BASE: usize = 1 << 20; // 1,048,576

pub fn param_col(param_idx: usize) -> usize {
    PARAM_COL_BASE + param_idx
}
```

Existing point-referencing constraints: columns 0..total_points*D (unchanged).
New param-referencing constraints: columns PARAM_COL_BASE+param_idx.
The system's `compute_jacobian()` detects `col >= PARAM_COL_BASE` and maps to the
param section of the variable vector.

This avoids changing the Jacobian entry type `(usize, usize, f64)` and keeps the
`Problem` trait completely untouched.

---

## Step-by-Step Implementation

### Step 1: Add Scalar Parameters to ConstraintSystem

**Files changed:** `src/geometry/system.rs`

Add fields to `ConstraintSystem<D>`:

```rust
pub struct ConstraintSystem<const D: usize> {
    points: Vec<Point<D>>,
    fixed: Vec<bool>,
    constraints: Vec<Box<dyn GeometricConstraint<D>>>,
    name: String,
    // NEW:
    params: Vec<f64>,         // scalar parameter values
    param_fixed: Vec<bool>,   // whether each param is fixed
    param_names: Vec<String>, // optional debug names ("radius", "start_angle", etc.)
}
```

Add methods:

```rust
/// Add a free scalar parameter. Returns its param index.
pub fn add_param(&mut self, value: f64) -> usize { ... }

/// Add a free scalar parameter with a debug name.
pub fn add_param_named(&mut self, value: f64, name: impl Into<String>) -> usize { ... }

/// Add a fixed scalar parameter.
pub fn add_param_fixed(&mut self, value: f64) -> usize { ... }

/// Fix/free a param.
pub fn fix_param(&mut self, index: usize) { ... }
pub fn free_param(&mut self, index: usize) { ... }

/// Get/set param values.
pub fn get_param(&self, index: usize) -> Option<f64> { ... }
pub fn set_param(&mut self, index: usize, value: f64) { ... }
pub fn params(&self) -> &[f64] { ... }

/// Count helpers.
pub fn param_count(&self) -> usize { ... }
pub fn free_param_count(&self) -> usize { ... }
```

Update existing methods:

```rust
pub fn total_variable_count(&self) -> usize {
    self.free_point_count() * D + self.free_param_count()
}

pub fn degrees_of_freedom(&self) -> i32 {
    self.total_variable_count() as i32 - self.equation_count() as i32
}

pub fn current_values(&self) -> Vec<f64> {
    let mut values = Vec::with_capacity(self.total_variable_count());
    // Free point coords (existing logic)
    for (i, point) in self.points.iter().enumerate() {
        if !self.fixed[i] {
            for k in 0..D { values.push(point.get(k)); }
        }
    }
    // Free params (NEW)
    for (i, &val) in self.params.iter().enumerate() {
        if !self.param_fixed[i] { values.push(val); }
    }
    values
}

pub fn set_values(&mut self, values: &[f64]) {
    let mut idx = 0;
    // Set free point coords (existing logic)
    for (i, point) in self.points.iter_mut().enumerate() {
        if !self.fixed[i] {
            for k in 0..D {
                point.set(k, values[idx]);
                idx += 1;
            }
        }
    }
    // Set free params (NEW)
    for (i, val) in self.params.iter_mut().enumerate() {
        if !self.param_fixed[i] {
            *val = values[idx];
            idx += 1;
        }
    }
}
```

Update `compute_jacobian()` to handle param columns:

```rust
pub fn compute_jacobian(&self) -> Vec<(usize, usize, f64)> {
    let free_point_map = self.build_free_point_map();
    let free_param_map = self.build_free_param_map();
    let param_base_col = self.free_point_count() * D;

    let mut all_entries = Vec::new();
    let mut row_offset = 0;

    for constraint in &self.constraints {
        let entries = constraint.jacobian(&self.points, &self.params);

        for (local_row, col, val) in entries {
            if col >= PARAM_COL_BASE {
                // Param variable
                let param_idx = col - PARAM_COL_BASE;
                if let Some(Some(free_idx)) = free_param_map.get(param_idx) {
                    let new_col = param_base_col + free_idx;
                    all_entries.push((row_offset + local_row, new_col, val));
                }
            } else {
                // Point variable (existing logic)
                let point_idx = col / D;
                let coord = col % D;
                if let Some(Some(free_idx)) = free_point_map.get(point_idx) {
                    let new_col = free_idx * D + coord;
                    all_entries.push((row_offset + local_row, new_col, val));
                }
            }
        }

        row_offset += constraint.equation_count();
    }
    all_entries
}
```

Update the `Problem` trait impl similarly (the `residuals` and `jacobian` methods that
take `&[f64]` need to unpack both points and params before calling constraints).

**Tests:** All existing tests must continue to pass with zero params. Add new tests
that create a system with params and verify variable counts, DOF, values round-trip.

---

### Step 2: Update GeometricConstraint Trait

**Files changed:** `src/geometry/constraints/mod.rs` + all 16 constraint files

Change the trait signature:

```rust
pub trait GeometricConstraint<const D: usize>: Send + Sync {
    fn equation_count(&self) -> usize;
    fn residuals(&self, points: &[Point<D>], params: &[f64]) -> Vec<f64>;
    fn jacobian(&self, points: &[Point<D>], params: &[f64]) -> Vec<(usize, usize, f64)>;
    fn variable_indices(&self) -> Vec<usize>;
    /// Param indices this constraint depends on. Default: none.
    fn param_indices(&self) -> Vec<usize> { vec![] }
    fn weight(&self) -> f64 { 1.0 }
    fn is_soft(&self) -> bool { false }
    fn name(&self) -> &'static str;
}
```

Add the PARAM_COL_BASE constant and `param_col()` helper to `mod.rs`.

Update all 16 existing constraint files. This is a **mechanical change** — just add
`_params: &[f64]` to each `residuals` and `jacobian` signature. The body is unchanged.
Example for DistanceConstraint:

```rust
// Before:
fn residuals(&self, points: &[Point<D>]) -> Vec<f64> { ... }
// After:
fn residuals(&self, points: &[Point<D>], _params: &[f64]) -> Vec<f64> { ... }
```

Files to update (all in `src/geometry/constraints/`):
- `angle.rs` (AngleConstraint)
- `coincident.rs` (CoincidentConstraint)
- `collinear.rs` (CollinearConstraint)
- `distance.rs` (DistanceConstraint)
- `equal_length.rs` (EqualLengthConstraint)
- `fixed.rs` (FixedConstraint)
- `horizontal.rs` (HorizontalConstraint)
- `midpoint.rs` (MidpointConstraint)
- `parallel.rs` (ParallelConstraint)
- `perpendicular.rs` (PerpendicularConstraint)
- `point_on_circle.rs` (PointOnCircleConstraint, PointOnCircleVariableRadiusConstraint)
- `point_on_line.rs` (PointOnLineConstraint)
- `symmetric.rs` (SymmetricConstraint, SymmetricAboutLineConstraint)
- `tangent.rs` (LineTangentConstraint, CircleTangentConstraint)
- `vertical.rs` (VerticalConstraint)

**Also update:** `src/jit/geometry_lowering.rs` (the Lowerable trait and its impls
for DistanceConstraint). The JIT lowering may need a parallel change or can be deferred.

**Tests:** All existing constraint tests must pass. The test helpers just add `&[]`
as the params argument.

---

### Step 3: Entity Descriptors

**New file:** `src/geometry/entities.rs`

Entities are lightweight descriptors that record which points and params belong to
a geometric primitive. They don't own data — they hold indices into the system.

```rust
/// Handle to a circle in the constraint system.
#[derive(Clone, Copy, Debug)]
pub struct CircleEntity {
    /// Point index of the center.
    pub center: usize,
    /// Param index of the radius.
    pub radius: usize,
}

/// Handle to a circular arc in the constraint system.
#[derive(Clone, Copy, Debug)]
pub struct ArcEntity {
    pub center: usize,       // point index
    pub radius: usize,       // param index
    pub start_angle: usize,  // param index
    pub end_angle: usize,    // param index
}

impl ArcEntity {
    /// Compute the start point of the arc given current system state.
    pub fn start_point<const D: usize>(&self, points: &[Point<D>], params: &[f64]) -> Point2D {
        let cx = points[self.center].get(0);
        let cy = points[self.center].get(1);
        let r = params[self.radius];
        let a = params[self.start_angle];
        Point2D::new(cx + r * a.cos(), cy + r * a.sin())
    }

    /// Compute the end point of the arc.
    pub fn end_point<const D: usize>(&self, points: &[Point<D>], params: &[f64]) -> Point2D {
        let cx = points[self.center].get(0);
        let cy = points[self.center].get(1);
        let r = params[self.radius];
        let a = params[self.end_angle];
        Point2D::new(cx + r * a.cos(), cy + r * a.sin())
    }

    /// Tangent direction at start point (perpendicular to radius, CCW).
    pub fn start_tangent<const D: usize>(&self, params: &[f64]) -> Vector2D {
        let a = params[self.start_angle];
        Vector2D::new(-a.sin(), a.cos())
    }

    /// Tangent direction at end point.
    pub fn end_tangent<const D: usize>(&self, params: &[f64]) -> Vector2D {
        let a = params[self.end_angle];
        Vector2D::new(-a.sin(), a.cos())
    }
}

/// Handle to an ellipse in the constraint system.
#[derive(Clone, Copy, Debug)]
pub struct EllipseEntity {
    pub center: usize,      // point index
    pub semi_major: usize,  // param index
    pub semi_minor: usize,  // param index
    pub rotation: usize,    // param index (angle of major axis from x-axis)
}

/// Handle to a cubic Bezier curve.
/// All control points are regular points — no params needed.
#[derive(Clone, Copy, Debug)]
pub struct CubicBezierEntity {
    pub p0: usize, // point index — start
    pub p1: usize, // point index — control 1
    pub p2: usize, // point index — control 2
    pub p3: usize, // point index — end
}

impl CubicBezierEntity {
    /// Evaluate the Bezier curve at parameter t ∈ [0,1].
    pub fn evaluate(&self, points: &[Point<2>], t: f64) -> Point2D {
        let p0 = points[self.p0];
        let p1 = points[self.p1];
        let p2 = points[self.p2];
        let p3 = points[self.p3];
        let u = 1.0 - t;
        let uu = u * u;
        let tt = t * t;
        Point2D::new(
            uu*u*p0.x() + 3.0*uu*t*p1.x() + 3.0*u*tt*p2.x() + tt*t*p3.x(),
            uu*u*p0.y() + 3.0*uu*t*p1.y() + 3.0*u*tt*p2.y() + tt*t*p3.y(),
        )
    }

    /// First derivative (tangent) at parameter t.
    pub fn derivative(&self, points: &[Point<2>], t: f64) -> Vector2D {
        let p0 = points[self.p0];
        let p1 = points[self.p1];
        let p2 = points[self.p2];
        let p3 = points[self.p3];
        let u = 1.0 - t;
        Vector2D::new(
            3.0*(u*u*(p1.x()-p0.x()) + 2.0*u*t*(p2.x()-p1.x()) + t*t*(p3.x()-p2.x())),
            3.0*(u*u*(p1.y()-p0.y()) + 2.0*u*t*(p2.y()-p1.y()) + t*t*(p3.y()-p2.y())),
        )
    }

    /// Second derivative at parameter t.
    pub fn second_derivative(&self, points: &[Point<2>], t: f64) -> Vector2D {
        let p0 = points[self.p0];
        let p1 = points[self.p1];
        let p2 = points[self.p2];
        let p3 = points[self.p3];
        let u = 1.0 - t;
        Vector2D::new(
            6.0*(u*(p2.x()-2.0*p1.x()+p0.x()) + t*(p3.x()-2.0*p2.x()+p1.x())),
            6.0*(u*(p2.y()-2.0*p1.y()+p0.y()) + t*(p3.y()-2.0*p2.y()+p1.y())),
        )
    }

    /// Curvature at parameter t.
    pub fn curvature(&self, points: &[Point<2>], t: f64) -> f64 {
        let d1 = self.derivative(points, t);
        let d2 = self.second_derivative(points, t);
        let cross = d1.coords[0] * d2.coords[1] - d1.coords[1] * d2.coords[0];
        let speed = d1.norm().max(MIN_EPSILON);
        cross / (speed * speed * speed)
    }
}

/// Handle to a quadratic Bezier curve.
#[derive(Clone, Copy, Debug)]
pub struct QuadraticBezierEntity {
    pub p0: usize, // point index
    pub p1: usize, // point index (control)
    pub p2: usize, // point index
}

// --- 3D-specific entities ---

/// Handle to a circle in 3D space.
/// A 3D circle requires a plane (specified by normal direction).
#[derive(Clone, Copy, Debug)]
pub struct Circle3DEntity {
    pub center: usize,          // point index
    pub radius: usize,          // param index
    pub normal_theta: usize,    // param index — polar angle of normal
    pub normal_phi: usize,      // param index — azimuthal angle of normal
}

/// Handle to a 3D arc.
#[derive(Clone, Copy, Debug)]
pub struct Arc3DEntity {
    pub center: usize,
    pub radius: usize,
    pub normal_theta: usize,
    pub normal_phi: usize,
    pub start_angle: usize,
    pub end_angle: usize,
}

/// Handle to a plane in 3D.
/// Defined by a point on the plane and a normal direction.
#[derive(Clone, Copy, Debug)]
pub struct PlaneEntity {
    pub point: usize,           // point index — a point on the plane
    pub normal_theta: usize,    // param index
    pub normal_phi: usize,      // param index
}
```

**Note on 3D normal representation:** Using spherical angles (theta, phi) for the normal
avoids the unit-vector constraint but has a singularity at the poles. This is acceptable
for Phase 1 — most 3D arcs don't have normals pointing exactly along the z-axis. A future
phase can switch to quaternion or Rodrigues parameterization if needed.

**Register entities in mod.rs**, export from `src/geometry/mod.rs`.

---

### Step 4: New Constraint Types for Circles and Arcs

**New files in `src/geometry/constraints/`:**

#### 4a. `concentric.rs` — Concentric constraint (2 circles share center)

Trivially reduces to CoincidentConstraint on the center points. But having a named
type improves diagnostics. Implementation: delegates to CoincidentConstraint internally.

```
Equations: D (same as CoincidentConstraint)
Points: center1, center2
Params: none
```

#### 4b. `equal_radius.rs` — Equal radius constraint

Two circles/arcs have the same radius.

```
Equation: params[r1] - params[r2] = 0
Count: 1
Points: none
Params: r1, r2
Jacobian: d/d(r1) = +1.0 at param_col(r1), d/d(r2) = -1.0 at param_col(r2)
```

#### 4c. `fixed_param.rs` — Fix a scalar parameter to a value

```
Equation: params[idx] - target = 0
Count: 1
Points: none
Params: idx
```

#### 4d. `point_on_arc.rs` — Point lies on arc (on circle AND within angular range)

Two constraints:
1. `|P - C| - radius = 0` (point on circle — existing equation)
2. Angle of (P-C) is within [start_angle, end_angle]

The angular range constraint is tricky with standard residuals because angles wrap.
Practical approach: For the "on arc" constraint, decompose into:
- Point on circle (distance)
- `atan2(py-cy, px-cx)` is between start and end angle

But atan2 has discontinuities. Better approach used by CAD solvers:
parameterize the point as `P = C + r*(cos(θ), sin(θ))` where θ is an **auxiliary
solver variable** constrained to `start ≤ θ ≤ end`. This introduces θ as a new param.

```
Params: radius, start_angle, end_angle, theta (auxiliary)
Equations:
  px - (cx + params[radius] * cos(params[theta])) = 0
  py - (cy + params[radius] * sin(params[theta])) = 0
  (theta constrained to [start, end] via inequality or soft constraint)
Count: 2 (+ optional inequality)
```

This is the cleanest formulation. The auxiliary theta param naturally handles the
angular range without atan2 discontinuities.

#### 4e. `arc_endpoint.rs` — Arc endpoint coincides with a point

The start/end point of an arc is a derived quantity:
`endpoint = center + radius * (cos(angle), sin(angle))`

```
Equations:
  px - (cx + params[r] * cos(params[angle])) = 0
  py - (cy + params[r] * sin(params[angle])) = 0
Count: 2
Points: point (the target), center
Params: radius, angle (start_angle or end_angle)

Jacobian (6 nonzero entries):
  d/d(px) = 1, d/d(py) = 1
  d/d(cx) = -1, d/d(cy) = -1
  d/d(r) = -cos(angle)  at param_col(radius) (row 0)
  d/d(r) = -sin(angle)  at param_col(radius) (row 1)
  d/d(angle) = r*sin(angle)   at param_col(angle) (row 0)
  d/d(angle) = -r*cos(angle)  at param_col(angle) (row 1)
```

#### 4f. `arc_tangent_line.rs` — G1 continuity between arc and line at junction

At the junction point, the line direction must match the arc tangent direction.
Arc tangent at angle θ is `(-sin(θ), cos(θ))`. Line direction is `(p2-p1)/|p2-p1|`.

```
Equation: (p2-p1) × (-sin(θ), cos(θ)) = 0
  i.e. (dx)*cos(θ) + (dy)*sin(θ) = 0   (where dx=p2x-p1x, dy=p2y-p1y)
  This says the line direction is perpendicular to the radius at θ.
Count: 1
Points: line_start, line_end
Params: angle (the arc angle at the junction point)
```

Note: this is just the tangent direction constraint. You also need `arc_endpoint`
to ensure the junction point actually lies on the arc. Typically used together.

#### 4g. `arc_tangent_arc.rs` — G1 continuity between two arcs at junction

At the junction, both arcs share a tangent direction. Since the arc tangent at angle θ
is perpendicular to the radius, G1 continuity means the two radii at the junction are
collinear with opposite signs (the tangent directions match, not just the tangent lines).

```
Equations:
  Tangent directions equal: same as collinear radii at junction point.
  Concretely: the line from center1 to center2 passes through the junction point.
  center1 + r1*(cos(θ1), sin(θ1)) = center2 + r2*(cos(θ2), sin(θ2))  [from endpoints]
  (cos(θ1), sin(θ1)) is parallel to (cos(θ2), sin(θ2))  [tangent match → radii parallel]
  So: θ1 - θ2 = 0 (same direction) or θ1 - θ2 = π (opposite, for external tangent)

Count: 1 (angle alignment)
```

Combined with arc_endpoint constraints on both arcs, this gives full G1 continuity.

#### 4h. `point_on_circle_variable_radius.rs` — Already exists, update to use params

The existing `PointOnCircleConstraint` has a constant radius. We need a version where
the radius is a solver param. Create `PointOnCircleParamRadius`:

```
Equation: |P - C| - params[radius] = 0
Count: 1
Points: point, center
Params: radius

Jacobian: same as PointOnCircleConstraint for point/center derivatives, plus
  d/d(radius) = -1.0 at param_col(radius)
```

#### 4i. `tangent_line_circle_param.rs` — Line tangent to circle with variable radius

Like the existing `LineTangentConstraint` but radius is a param.

```
Equation: perp_distance(center, line) - params[radius] = 0
Jacobian: same as LineTangentConstraint for point derivatives, plus
  d/d(radius) = -1.0 at param_col(radius)
```

---

### Step 5: Bezier Curve Constraints (2D)

**New files in `src/geometry/constraints/`:**

#### 5a. `bezier_g0.rs` — G0 positional continuity

Endpoint of curve A = startpoint of curve B. This is just `CoincidentConstraint(a.p3, b.p0)`.
No new code needed — document the pattern.

#### 5b. `bezier_g1.rs` — G1 tangent continuity at junction

The last handle of curve A, the junction point, and the first handle of curve B
must be collinear. Equivalently: `(P3-P2) × (Q1-Q0) = 0` where P3=Q0 (junction).

```
Equation: (p3x-p2x)*(q1y-q0y) - (p3y-p2y)*(q1x-q0x) = 0
  where p2=A.p2, p3=A.p3=B.p0=q0, q1=B.p1

Count: 1
Points: a_p2, junction (a_p3 = b_p0), b_p1
Params: none

Jacobian:
  Let ux = jx - p2x, uy = jy - p2y  (last handle vector of A)
  Let vx = q1x - jx, vy = q1y - jy  (first handle vector of B)
  f = ux*vy - uy*vx

  d/d(p2x) = -vy, d/d(p2y) = vx
  d/d(jx)  = vy - vy + ... = (vy - vy) [careful chain rule]
  Actually:
    df/djx = d(ux)/djx * vy + ux * d(vy)/djx - d(uy)/djx * vx - uy * d(vx)/djx
           = 1*vy + ux*(-1) - 0*vx - uy*(-1)
           ... let me compute properly:
    ux = jx - p2x, uy = jy - p2y
    vx = q1x - jx, vy = q1y - jy
    f = ux*vy - uy*vx
    df/djx = vy + ux*0 - 0*vx - uy*(-1) = vy + uy   WRONG

    Let me redo:
    f = (jx-p2x)(q1y-jy) - (jy-p2y)(q1x-jx)
    df/djx = (q1y-jy) + (jx-p2x)*(-1)... no wait:
    df/djx = 1*(q1y-jy) + (jx-p2x)*0 ... hmm

    Actually with ux = jx-p2x, vy = q1y-jy, uy = jy-p2y, vx = q1x-jx:
    df/djx = (dux/djx)*vy + ux*(dvy/djx) - (duy/djx)*vx - uy*(dvx/djx)
           = 1*vy + ux*(-1) - 0*vx - uy*(-1)
           = vy - ux + uy
           Wait that doesn't look right either. Let me be really careful:

    dux/djx = 1, duy/djx = 0
    dvx/djx = -1, dvy/djx = -1

    df/djx = 1*(q1y-jy) + (jx-p2x)*(-1) - 0*(q1x-jx) - (jy-p2y)*(-1)
           = vy - ux + uy

    df/djy = 0*(q1y-jy) + (jx-p2x)*(0) - 1*(q1x-jx) - (jy-p2y)*(0)
           ... wait, dvy/djy = -1, dvx/djy = 0:
    df/djy = 0*vy + ux*(-1) ... hmm, vy doesn't depend on jy?

    vy = q1y - jy, so dvy/djy = -1. vx = q1x - jx, so dvx/djy = 0.
    uy = jy - p2y, so duy/djy = 1. ux = jx - p2x, so dux/djy = 0.

    df/djy = 0*vy + ux*(-1) - 1*vx - uy*0
           = -ux - vx

    Full Jacobian (6 points × 2 coords = up to 6 entries, but 3 points × 2 = 6):
      df/dp2x = -vy,  df/dp2y = vx
      df/djx = vy - ux + uy,  df/djy = -ux - vx
      ... Actually I'll let the implementor derive this carefully with tests.
```

The implementation should use numerical Jacobian verification (the existing
`verify_jacobian` utility) to validate the analytical Jacobian.

#### 5c. `bezier_c1.rs` — C1 parametric continuity

Stronger than G1: handle vectors are equal, not just collinear.
`P3 - P2 = Q1 - Q0` (where P3 = Q0 = junction).

```
Equations:
  (jx - p2x) - (q1x - jx) = 0  →  2*jx - p2x - q1x = 0
  (jy - p2y) - (q1y - jy) = 0  →  2*jy - p2y - q1y = 0
Count: 2
Points: a_p2, junction, b_p1
Params: none
Jacobian: constant (same structure as MidpointConstraint)
  Row 0: d/dp2x = -1, d/djx = 2, d/dq1x = -1
  Row 1: d/dp2y = -1, d/djy = 2, d/dq1y = -1
```

This is actually identical to the MidpointConstraint (junction is midpoint of p2 and q1).

#### 5d. `bezier_g2.rs` — G2 curvature continuity at junction

Curvature of curve A at t=1 equals curvature of curve B at t=0.

Curvature at t=0 of a cubic Bezier:
```
κ(0) = 2/3 * |P0P1 × P0P2| / |P0P1|³
     = 2/3 * |(p1-p0) × (p2-p0)| / |p1-p0|³
```

For curve A at t=1:
```
κ_A(1) = 2/3 * |P3P2 × P3P1| / |P3P2|³
```

For curve B at t=0 (where B.P0 = A.P3):
```
κ_B(0) = 2/3 * |Q0Q1 × Q0Q2| / |Q0Q1|³
```

G2 constraint: κ_A(1) = κ_B(0)

```
Equation: κ_A(1) - κ_B(0) = 0
Count: 1
Points: a_p1, a_p2, junction (a_p3 = b_p0), b_p1, b_p2
Params: none

The Jacobian involves derivatives of the curvature formula w.r.t. control points.
This is complex but tractable. Use numerical Jacobian verification.
```

Alternatively, G2 can be formulated as a ratio condition on handle lengths plus the
G1 collinearity constraint, which avoids the curvature formula entirely. See:
Sederberg, "Computer Aided Geometric Design" for the standard formulation.

#### 5e. `point_on_bezier.rs` — Point constrained to lie on Bezier curve

Introduces an auxiliary parameter `t ∈ [0,1]` as a solver param.

```
Equations:
  px - B(t).x = 0
  py - B(t).y = 0
  where B(t) = (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3

Count: 2
Points: target_point, bezier.p0, bezier.p1, bezier.p2, bezier.p3
Params: t_param (auxiliary)

Jacobian:
  d/d(px) = 1, d/d(py) = 1 (for target point)
  d/d(p0x) = -(1-t)³, etc. (for each control point)
  d/d(t) = -dB/dt (for the t parameter) at param_col(t_param)
```

The t parameter should be initialized to the closest value (project the target point
onto the curve to find initial t).

#### 5f. `bezier_tangent_line.rs` — Bezier tangent direction at endpoint matches line

At t=0: tangent direction is (P1-P0). At t=1: tangent is (P3-P2).
This is structurally identical to ensuring collinearity of the handle with a line.

```
For t=0: (P1-P0) × (line_end - line_start) = 0
For t=1: (P3-P2) × (line_end - line_start) = 0
```

Reduces to a ParallelConstraint on the handle and the line direction.

---

### Step 6: Ellipse Constraints (2D)

**New files:**

#### 6a. `point_on_ellipse.rs` — Point lies on ellipse

An ellipse with center C, semi-axes a,b and rotation φ:
```
Point P on ellipse ↔ ((dx*cos(φ) + dy*sin(φ))² / a²) + ((−dx*sin(φ) + dy*cos(φ))² / b²) = 1
where dx = px - cx, dy = py - cy
```

```
Equation: (u²/a² + v²/b²) - 1 = 0
  where u = dx*cos(φ) + dy*sin(φ), v = -dx*sin(φ) + dy*cos(φ)
Count: 1
Points: point, center
Params: semi_major, semi_minor, rotation
```

Jacobian w.r.t. all 7 variables (px, py, cx, cy, a, b, φ). Complex but tractable.
Use numerical Jacobian verification.

---

### Step 7: 3D-Specific Primitives and Constraints

**3D differs from 2D in these ways:**

| Primitive | 2D | 3D Difference |
|---|---|---|
| Circle | center + radius | + normal direction (2 params: θ, φ) |
| Arc | center + radius + start_angle + end_angle | + normal direction |
| Ellipse | center + semi_axes + rotation | + normal direction + tilt |
| Bezier | 4×Point<2> | 4×Point<3> (works automatically via const generics) |
| Plane | N/A | point + normal direction |

**New 3D constraints:**

#### 7a. `coplanar.rs` — Two planes are coplanar / Point on plane

```
Equation: (P - P0) · normal = 0
  where P0 is a point on the plane, normal is derived from (θ, φ) params
Count: 1
Points: point, plane_point
Params: normal_theta, normal_phi
```

#### 7b. `point_on_circle_3d.rs` — Point on 3D circle

A point lies on a 3D circle if:
1. |P - C| = radius  (on sphere of that radius)
2. (P - C) · normal = 0  (on the plane through C with given normal)

```
Equations: 2
Points: point, center
Params: radius, normal_theta, normal_phi
```

#### 7c. `arc_endpoint_3d.rs`

3D arc endpoint: `C + r * (cos(θ)*u + sin(θ)*v)` where u,v are the local
coordinate axes of the arc's plane (derived from normal direction).

The u,v computation from normal (theta, phi):
```
normal = (sin(φ)*cos(θ), sin(φ)*sin(θ), cos(φ))
u = any vector perpendicular to normal (e.g., cross(normal, z_axis), normalized)
v = cross(normal, u)
```

**This is the most complex single constraint in the system** due to the chain of
trigonometric functions. Numerical Jacobian verification is essential.

#### 7d. `parallel_planes.rs`, `perpendicular_planes.rs`

Normal vectors are parallel / perpendicular.

```
Parallel: normal1 × normal2 = 0  (2 equations in 3D)
Perpendicular: normal1 · normal2 = 0  (1 equation)
```

These operate purely on param variables (the theta/phi of each normal).

---

### Step 8: Builder API Extensions

**File changed:** `src/geometry/builder.rs`

Add entity construction methods to `ConstraintSystemBuilder<2>`:

```rust
impl ConstraintSystemBuilder<2> {
    /// Add a circle entity. Returns a CircleEntity handle.
    pub fn circle(mut self, center: Point2D, radius: f64) -> (Self, CircleEntity) {
        let center_idx = self.system.add_point(center);
        let radius_idx = self.system.add_param(radius);
        (self, CircleEntity { center: center_idx, radius: radius_idx })
    }

    /// Add a circle with fixed radius.
    pub fn circle_fixed_radius(mut self, center: Point2D, radius: f64) -> (Self, CircleEntity) {
        let center_idx = self.system.add_point(center);
        let radius_idx = self.system.add_param_fixed(radius);
        (self, CircleEntity { center: center_idx, radius: radius_idx })
    }

    /// Add an arc entity.
    pub fn arc(mut self, center: Point2D, radius: f64,
               start_angle: f64, end_angle: f64) -> (Self, ArcEntity) {
        let center_idx = self.system.add_point(center);
        let radius_idx = self.system.add_param(radius);
        let start_idx = self.system.add_param(start_angle);
        let end_idx = self.system.add_param(end_angle);
        (self, ArcEntity {
            center: center_idx, radius: radius_idx,
            start_angle: start_idx, end_angle: end_idx,
        })
    }

    /// Add a cubic Bezier curve.
    pub fn cubic_bezier(mut self, p0: Point2D, p1: Point2D,
                        p2: Point2D, p3: Point2D) -> (Self, CubicBezierEntity) {
        let i0 = self.system.add_point(p0);
        let i1 = self.system.add_point(p1);
        let i2 = self.system.add_point(p2);
        let i3 = self.system.add_point(p3);
        (self, CubicBezierEntity { p0: i0, p1: i1, p2: i2, p3: i3 })
    }

    // Constraint helpers that operate on entity handles:

    /// Constrain a point to lie on a circle (variable radius).
    pub fn point_on_circle_entity(mut self, point: usize, circle: &CircleEntity) -> Self {
        self.system.add_constraint(Box::new(
            PointOnCircleParamRadius::new(point, circle.center, circle.radius)
        ));
        self
    }

    /// Constrain two circles to have equal radius.
    pub fn equal_radius(mut self, c1: &CircleEntity, c2: &CircleEntity) -> Self {
        self.system.add_constraint(Box::new(
            EqualRadiusConstraint::new(c1.radius, c2.radius)
        ));
        self
    }

    /// G1 continuity between two Bezier curves at their junction.
    pub fn bezier_g1(mut self, a: &CubicBezierEntity, b: &CubicBezierEntity) -> Self {
        // G0: endpoints coincide
        self.system.add_constraint(Box::new(
            CoincidentConstraint::<2>::new(a.p3, b.p0)
        ));
        // G1: handles collinear
        self.system.add_constraint(Box::new(
            BezierG1Constraint::new(a.p2, a.p3, b.p1) // junction = a.p3 = b.p0
        ));
        self
    }
}
```

**Design note:** The builder returns `(Self, EntityHandle)` tuples. This is slightly
awkward with fluent chaining. An alternative is to store entities in the system and
return their index. Either way, the entity handle is needed by subsequent constraint
calls.

A cleaner API might use a closure or side-channel:

```rust
let mut circle1 = None;
let system = ConstraintSystemBuilder::<2>::new()
    .circle(Point2D::new(0.0, 0.0), 5.0, |e| circle1 = Some(e))
    .point_on_circle_entity(0, &circle1.unwrap())
    .build();
```

Or store entities in the builder:

```rust
let mut builder = ConstraintSystemBuilder::<2>::new();
let circle1 = builder.add_circle(Point2D::new(0.0, 0.0), 5.0);
let circle2 = builder.add_circle(Point2D::new(10.0, 0.0), 3.0);
builder.equal_radius(&circle1, &circle2);
let system = builder.build();
```

This non-fluent style is probably better for entity-heavy construction.

---

### Step 9: Update JIT Geometry Lowering

**File changed:** `src/jit/geometry_lowering.rs`

The `Lowerable` trait needs the same `params` extension. For Phase 1, we can:
- Update the trait signature to accept params
- Only implement lowering for existing point-only constraints
- New param-aware constraints fall back to interpreted evaluation

Mark this as a follow-up optimization task.

---

### Step 10: Tests

#### Unit tests (per constraint):
Each new constraint file gets tests following the existing pattern:
- Residual is zero when constraint is satisfied
- Residual is nonzero when violated, with expected value
- Jacobian has correct size and finite values
- Jacobian matches numerical finite-difference Jacobian (use `verify_jacobian`)
- Edge cases: coincident points, zero radius, etc.

#### Integration tests (new file `tests/geometry_primitives.rs`):

```
test_circle_tangent_to_two_lines
  - Fixed radius circle, two fixed lines → solve for center position
  - Verify: perpendicular distance from center to each line = radius

test_circle_through_three_points
  - Three fixed points → solve for center and radius
  - Verify: all three points on circle, center matches analytical formula

test_equal_radius_circles
  - Two circles with equal_radius constraint → solve
  - Verify: both radii identical after solving

test_rounded_rectangle
  - 4 line segments + 4 fillet arcs with G1 at all junctions
  - Verify: all tangent junctions smooth, all fillet radii equal

test_bezier_g1_chain
  - 3 cubic Beziers chained with G1 continuity
  - Verify: handle collinearity at each junction

test_bezier_g2_chain
  - 2 cubic Beziers with G2 continuity
  - Verify: curvature matches at junction

test_inscribed_circle
  - Circle tangent to 3 sides of a triangle
  - Verify: radius matches incircle formula r = area/s

test_apollonius_problem
  - Circle tangent to 3 given circles
  - Verify: tangent distances correct

test_arc_endpoint_solve
  - Arc with fixed center, find radius and angles to pass through two points
  - Verify: endpoint positions match target points

test_line_arc_line_fillet
  - Two lines with arc fillet: G1 at both junctions
  - Verify: smooth transition, fillet radius matches target
```

#### Property tests (using proptest):
```
proptest! {
    fn circle_dof_is_3(cx, cy, r) {
        // Circle with no constraints has 3 DOF (center x, y + radius)
        let system = ...; // add circle entity, no constraints
        prop_assert_eq!(system.degrees_of_freedom(), 3);
    }

    fn arc_dof_is_5(cx, cy, r, a0, a1) {
        // Arc with no constraints has 5 DOF
        let system = ...; // add arc entity
        prop_assert_eq!(system.degrees_of_freedom(), 5);
    }

    fn bezier_dof_is_8(p0, p1, p2, p3) {
        // Cubic Bezier has 8 DOF (4 points × 2 coords)
        ...
        prop_assert_eq!(system.degrees_of_freedom(), 8);
    }

    fn fixed_radius_reduces_dof_by_1(cx, cy, r) {
        // Fixing radius reduces DOF from 3 to 2
        ...
    }

    fn jacobian_matches_numerical(constraint, points, params) {
        // For any constraint at any configuration, analytical Jacobian
        // matches finite-difference Jacobian within tolerance
        ...
    }
}
```

---

## Implementation Order

```
Step 1 (system.rs)  ──→ Step 2 (trait + 16 files)  ──→ Step 3 (entities.rs)
                                    │
                         ┌──────────┼──────────┐
                         ▼          ▼          ▼
                     Step 4      Step 5      Step 6
                    (circle/     (bezier    (ellipse
                     arc)        curves)    constraints)
                         │          │          │
                         └──────────┼──────────┘
                                    ▼
                              Step 7 (3D)
                                    │
                              Step 8 (builder)
                                    │
                              Step 9 (JIT, optional)
                                    │
                              Step 10 (tests throughout)
```

Steps 4, 5, 6 can proceed in parallel after Steps 1-3 are complete.

**Estimated scope per step:**
| Step | Files Changed | Files Created | LOC (estimate) |
|------|--------------|---------------|----------------|
| 1 | 1 (system.rs) | 0 | ~150 |
| 2 | 16 + mod.rs | 0 | ~100 (mechanical) |
| 3 | 0 | 1 (entities.rs) | ~300 |
| 4 | 0 | 6-8 constraint files | ~800 |
| 5 | 0 | 4-5 constraint files | ~600 |
| 6 | 0 | 1-2 constraint files | ~200 |
| 7 | 0 | 4-5 constraint files | ~500 |
| 8 | 1 (builder.rs) | 0 | ~200 |
| 9 | 1 (geometry_lowering.rs) | 0 | ~50 |
| 10 | test sections in each | 1 integration test file | ~500 |
| **Total** | **~20** | **~18** | **~3,400** |

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| PARAM_COL_BASE sentinel collides with actual point columns | 1<<20 supports 524K points. Assert in add_point() that point_count * D < PARAM_COL_BASE |
| 3D normal singularity at poles (spherical angles) | Document limitation. Only matters for normals exactly along z-axis. Future: quaternion parameterization |
| G2 curvature Jacobian complex to derive analytically | Use numerical Jacobian verification. Can fall back to numerical Jacobian if analytical is wrong |
| Builder API ergonomics with entity handles | Start with non-fluent `&mut self` style for entity creation. Fluent chaining for constraints |
| Backward compatibility of trait change | Mechanical — just add `_params: &[f64]` to all existing impls. No logic changes |
| Performance of param_col remapping in hot Jacobian assembly | Branch prediction will be perfect — existing constraints never produce PARAM_COL_BASE columns. No overhead for pure-point systems |
