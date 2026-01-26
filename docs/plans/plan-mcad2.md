# MCAD Geometric Kernel Implementation Plan

## Executive Summary

**Goal**: Build a SolidWorks-class B-rep (Boundary Representation) + NURBS (Non-Uniform Rational B-Spline) geometric kernel from scratch, fully integrated with Cadatomic's existing architecture.

**Why Not Truck?** We are NOT using the `truck` crate because:
1. **Differential Dataflow Integration**: Our architecture provides incremental feature regeneration - only recompute affected features when parameters change. This requires deep integration with our DD-based constraint solver.
2. **Event Sourcing**: Every geometric operation is an event, enabling time-travel debugging, undo/redo, and CRDT-based collaboration.
3. **Tight Constraint Solver Coupling**: Sketch profiles come directly from our constraint solver, not as external geometry.
4. **Modern Rust Architecture**: No C++ legacy, leveraging Rust's ownership model for memory safety in complex topology operations.

**Scope**: This plan covers the full vertical stack from NURBS primitives to STEP I/O, designed for AI agent implementation with TDD (Test-Driven Development).

---

## Architectural Advantages Over Existing Kernels

### 1. Incremental Feature Regeneration via Differential Dataflow

Traditional CAD kernels regenerate the entire feature tree on any parameter change. Our DD-based approach:

```
Traditional:                     Cadatomic DD-based:

Parameter Change                 Parameter Change
       │                                │
       ▼                                ▼
┌─────────────────┐              ┌─────────────────┐
│ Regenerate ALL  │              │ DD detects      │
│ features from   │              │ affected        │
│ the beginning   │              │ features only   │
└─────────────────┘              └─────────────────┘
       │                                │
       ▼                                ▼
 O(n) full rebuild              O(k) where k << n
```

**Implementation**: Feature dependencies are DD collections. When a sketch dimension changes, DD traces which extrusions/cuts/fillets depend on that sketch and only regenerates those.

### 2. Event-Sourced Geometry Operations

Every B-rep operation (extrude, cut, fillet, boolean) is an immutable event:

```rust
pub enum GeometryEvent {
    SketchCreated { sketch_id: SketchId, plane: PlaneRef },
    ProfileExtruded { feature_id: FeatureId, sketch_id: SketchId, depth: Length },
    BooleanPerformed { result_id: SolidId, op: BoolOp, operands: Vec<SolidId> },
    FilletApplied { feature_id: FeatureId, edges: Vec<EdgeRef>, radius: Length },
}
```

Benefits:
- **Time-travel debugging**: Replay to any point in feature history
- **Collaborative editing**: CRDT merge of concurrent geometric edits
- **Undo/redo**: Native support via event reversal
- **Audit trail**: Every change is tracked with HLC timestamps

### 3. Constraint Solver Integration

Sketch profiles flow directly from our `solverang` constraint system:

```
┌─────────────────────────────────────────────────────────────┐
│                    solverang                             │
│                                                              │
│  Constraints: [Horizontal, Distance=50, Tangent, ...]       │
│                          │                                   │
│                          ▼                                   │
│  ┌───────────────────────────────────────┐                  │
│  │ Newton-Raphson / LM Solver            │                  │
│  │ (DD-based component detection)        │                  │
│  └───────────────────────────────────────┘                  │
│                          │                                   │
└──────────────────────────┼──────────────────────────────────┘
                           │
                           ▼ Solved 2D Points
┌─────────────────────────────────────────────────────────────┐
│                   solverang_geometry                            │
│                                                              │
│  Profile Extraction: Closed loops → NURBS curves            │
│                          │                                   │
└──────────────────────────┼──────────────────────────────────┘
                           │
                           ▼ NURBS Profiles
┌─────────────────────────────────────────────────────────────┐
│                   solverang_features                            │
│                                                              │
│  Extrude/Revolve/Sweep → B-rep Solid                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4. Memory Safety in Topology Operations

B-rep topology is notoriously error-prone in C++ kernels. Rust's ownership model prevents:
- Dangling face/edge/vertex references after boolean operations
- Double-free of topology elements during shell reconstruction
- Data races in parallel feature regeneration

---

## 6-Layer Architecture

### Layer 1: Geometric Primitives (`solverang_geometry`)

**Purpose**: NURBS curves and surfaces, fundamental geometric types.

#### Key Types

```rust
// Fundamental primitives
pub struct Point3 { pub x: f64, pub y: f64, pub z: f64 }
pub struct Vector3 { pub x: f64, pub y: f64, pub z: f64 }
pub struct Plane { pub origin: Point3, pub normal: Vector3, pub u_axis: Vector3 }
pub struct Transform { pub matrix: [[f64; 4]; 4] }  // Affine transform

// NURBS curve
pub struct NurbsCurve {
    pub degree: usize,
    pub control_points: Vec<Point3>,
    pub weights: Vec<f64>,           // Rational weights (1.0 for non-rational)
    pub knot_vector: Vec<f64>,       // Non-decreasing knots
}

// NURBS surface
pub struct NurbsSurface {
    pub degree_u: usize,
    pub degree_v: usize,
    pub control_points: Vec<Vec<Point3>>,  // Grid of control points
    pub weights: Vec<Vec<f64>>,
    pub knot_vector_u: Vec<f64>,
    pub knot_vector_v: Vec<f64>,
}
```

#### Core Operations

| Operation                  | Complexity   | Notes                                                 |
| -------------------------- | ------------ | ----------------------------------------------------- |
| Curve point evaluation     | O(p)         | p = degree, using de Boor's algorithm                 |
| Surface point evaluation   | O(p_u * p_v) | Tensor product evaluation                             |
| Curve derivative           | O(p)         | First/second derivatives needed for tangent/curvature |
| Surface normal             | O(p_u * p_v) | Cross product of partial derivatives                  |
| Curve/curve intersection   | O(n log n)   | Bezier clipping or subdivision                        |
| Curve/surface intersection | O(n * m)     | The hard problem - iteration required                 |

#### Crate Evaluation: `curvo`

The `curvo` crate provides NURBS functionality. Evaluation criteria:

| Criteria                   | curvo   | Custom Implementation            |
| -------------------------- | ------- | -------------------------------- |
| Basic evaluation           | Good    | Equal effort                     |
| Derivatives                | Good    | Equal effort                     |
| Intersection               | Limited | Custom needed for SSI robustness |
| Event sourcing integration | No      | Full control                     |
| Performance                | Unknown | Optimizable                      |

**Recommendation**: Start with `curvo` for basic NURBS, implement custom surface-surface intersection (SSI) since that's the critical path for boolean operations.

---

### Layer 2: B-rep Topology (`solverang_topology`)

**Purpose**: Represent solid models as boundary faces with topological connectivity.

#### Data Structure: Incidence-Based B-rep

We use an incidence-based structure (not half-edge/DCEL) because:
1. More natural for non-manifold geometry (intermediate boolean states)
2. Simpler serialization for event sourcing
3. Well-suited for parallel traversal

```rust
// Core topology types with arena-based storage
pub type VertexId = Id<Vertex>;
pub type EdgeId = Id<Edge>;
pub type LoopId = Id<Loop>;
pub type FaceId = Id<Face>;
pub type ShellId = Id<Shell>;
pub type SolidId = Id<Solid>;

pub struct Vertex {
    pub point: Point3,
    pub edges: Vec<EdgeId>,  // Incident edges
}

pub struct Edge {
    pub curve: CurveGeometry,        // Geometric curve (NURBS, line, arc)
    pub vertices: (VertexId, VertexId),  // Start and end vertices
    pub faces: Vec<FaceId>,          // Incident faces (2 for manifold)
}

pub struct Loop {
    pub edges: Vec<(EdgeId, Orientation)>,  // Ordered edges forming closed loop
    pub face: FaceId,                        // Parent face
}

pub struct Face {
    pub surface: SurfaceGeometry,    // Geometric surface (NURBS, plane, cylinder)
    pub outer_loop: LoopId,          // Outer boundary
    pub inner_loops: Vec<LoopId>,    // Holes
    pub shell: ShellId,              // Parent shell
}

pub struct Shell {
    pub faces: Vec<FaceId>,
    pub solid: Option<SolidId>,      // None for open shells
}

pub struct Solid {
    pub outer_shell: ShellId,
    pub inner_shells: Vec<ShellId>,  // Voids
}

pub enum Orientation { Forward, Reversed }
```

#### Euler Operators

Euler operators maintain topological validity through atomic operations:

| Operator | Effect                   | DOF Change    |
| -------- | ------------------------ | ------------- |
| `mvfs`   | Make Vertex, Face, Shell | +1V, +1F, +1S |
| `mev`    | Make Edge, Vertex        | +1E, +1V      |
| `mef`    | Make Edge, Face          | +1E, +1F      |
| `kef`    | Kill Edge, Face          | -1E, -1F      |
| `kemr`   | Kill Edge, Make Ring     | -1E, +1L      |
| `mekr`   | Make Edge, Kill Ring     | +1E, -1L      |

Euler-Poincare formula for validation:
```
V - E + F - (L - F) - 2(S - G) = 0

Where:
  V = vertices, E = edges, F = faces
  L = loops, S = shells, G = genus (handles)
```

#### Topology Validation

```rust
impl Solid {
    pub fn validate(&self, store: &TopologyStore) -> Result<(), TopologyError> {
        // 1. Euler-Poincare formula
        self.check_euler_poincare(store)?;

        // 2. Every edge has exactly 2 incident faces (manifold)
        self.check_manifold_edges(store)?;

        // 3. Every loop is closed
        self.check_closed_loops(store)?;

        // 4. Face normals consistent (outward)
        self.check_orientation(store)?;

        // 5. No self-intersection
        self.check_no_self_intersection(store)?;

        Ok(())
    }
}
```

---

### Layer 3: Boolean Operations (`solverang_boolean`)

**Purpose**: Union, subtract, intersect operations on B-rep solids.

This is the most challenging layer due to:
1. Surface-surface intersection (SSI) robustness
2. Face classification complexity
3. Topology reconstruction

#### Boolean Algorithm Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Boolean Pipeline                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Surface-Surface Intersection (SSI)                      │
│     ├─ Find all face-face intersection curves               │
│     ├─ Split faces along intersection curves                │
│     └─ Handle tangent cases, edge-face intersections        │
│                                                              │
│  2. Face Classification                                      │
│     ├─ Classify each face of A relative to B (in/out/on)    │
│     ├─ Classify each face of B relative to A (in/out/on)    │
│     └─ Handle "on" faces (coplanar regions)                 │
│                                                              │
│  3. Face Selection by Boolean Type                          │
│     │                                                        │
│     ├─ Union: A_out + B_out + on_faces (same normal)        │
│     ├─ Subtract: A_out + B_in (flipped) + on_faces (opp)    │
│     └─ Intersect: A_in + B_in + on_faces (same normal)      │
│                                                              │
│  4. Topology Reconstruction                                  │
│     ├─ Build new shells from selected faces                 │
│     ├─ Sew edges along intersection curves                  │
│     └─ Validate resulting solid                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### Surface-Surface Intersection (The Hard Problem)

SSI strategies by surface type:

| Surface A | Surface B | Method                                  |
| --------- | --------- | --------------------------------------- |
| Plane     | Plane     | Analytical (line or coincident)         |
| Plane     | Cylinder  | Analytical (line, ellipse, or parabola) |
| Plane     | NURBS     | Marching + subdivision                  |
| Cylinder  | Cylinder  | Analytical or numerical                 |
| NURBS     | NURBS     | Marching squares + refinement           |

**Robustness Strategy**:
1. Use interval arithmetic for critical decisions
2. Maintain consistent tolerance throughout
3. Handle tangent/grazing intersections specially
4. Verify intersection curves lie on both surfaces

```rust
pub trait SurfaceSurfaceIntersection {
    fn intersect(
        &self,
        surface_a: &SurfaceGeometry,
        surface_b: &SurfaceGeometry,
        tolerance: f64,
    ) -> Result<Vec<IntersectionCurve>, SSIError>;
}

pub enum IntersectionCurve {
    ParametricCurve {
        curve_3d: NurbsCurve,
        uv_on_a: NurbsCurve,  // Curve in A's parameter space
        uv_on_b: NurbsCurve,  // Curve in B's parameter space
    },
    TangentPoint(Point3),
    Coincident(FaceRegion),
}
```

#### Face Classification

```rust
pub enum FaceClassification {
    Inside,   // Completely inside the other solid
    Outside,  // Completely outside the other solid
    On,       // Coplanar with a face of the other solid
    Split,    // Face is split by intersection (needs subdivision)
}

impl Face {
    pub fn classify(&self, other_solid: &Solid, store: &TopologyStore) -> FaceClassification {
        // Sample multiple points on face
        // Ray cast to determine inside/outside
        // Handle boundary cases carefully
    }
}
```

---

### Layer 4: Feature Modeling (`solverang_features`)

**Purpose**: Parametric features that generate/modify B-rep geometry.

#### Feature Types

```rust
pub enum Feature {
    // Sketch-based features
    Extrude(ExtrudeFeature),
    Revolve(RevolveFeature),
    Sweep(SweepFeature),
    Loft(LoftFeature),

    // Direct modeling
    Fillet(FilletFeature),
    Chamfer(ChamferFeature),
    Shell(ShellFeature),

    // Pattern features
    LinearPattern(LinearPatternFeature),
    CircularPattern(CircularPatternFeature),

    // Boolean operations
    BooleanCut(BooleanFeature),
    BooleanAdd(BooleanFeature),
}

pub struct ExtrudeFeature {
    pub id: FeatureId,
    pub sketch_id: SketchId,
    pub profile_loops: Vec<LoopId>,      // From constraint solver
    pub direction: ExtrudeDirection,
    pub depth: FeatureParameter,          // Can be expression/formula
    pub draft_angle: Option<Angle>,
    pub end_condition: EndCondition,
    pub operation: BooleanOperation,      // Add, cut, new body
}

pub enum EndCondition {
    Blind(Length),
    UpToSurface(FaceRef),
    ThroughAll,
    UpToVertex(VertexRef),
    MidPlane(Length),
}

pub enum FeatureParameter {
    Value(f64),
    Expression(String),                   // "width * 2 + 10"
    LinkedDimension(ConstraintId),        // From sketch
}
```

#### Feature Tree with DD Integration

```rust
// DD collections for feature dependencies
pub struct FeatureCollections {
    // Feature definitions: (FeatureId, FeatureData)
    features: Collection<(FeatureId, Feature), isize>,

    // Dependencies: (FeatureId, DependsOnFeatureId)
    dependencies: Collection<(FeatureId, FeatureId), isize>,

    // Computed B-rep: (FeatureId, BrepResult)
    brep_results: Collection<(FeatureId, Solid), isize>,

    // Validation status: (FeatureId, ValidationResult)
    validation: Collection<(FeatureId, ValidationResult), isize>,
}

impl FeatureCollections {
    /// When a feature changes, DD determines which downstream features
    /// need regeneration
    pub fn on_feature_changed(&mut self, feature_id: FeatureId) {
        // DD automatically propagates through dependency graph
        // Only affected features are re-computed
    }
}
```

#### Profile Extraction from Constraint Solver

```rust
pub struct ProfileExtractor {
    /// Extract closed loop profiles from solved sketch
    pub fn extract_profiles(
        sketch: &SolvedSketch,
    ) -> Result<Vec<Profile>, ProfileError> {
        // 1. Get all 2D curves from sketch
        let curves = sketch.get_curves();

        // 2. Find closed loops via graph traversal
        let loops = find_closed_loops(&curves)?;

        // 3. Determine loop hierarchy (outer vs inner/holes)
        let hierarchy = compute_loop_hierarchy(&loops)?;

        // 4. Convert 2D curves to NURBS
        let nurbs_profiles = loops.iter()
            .map(|loop_| convert_to_nurbs(loop_))
            .collect();

        Ok(nurbs_profiles)
    }
}

pub struct Profile {
    pub outer_boundary: NurbsCurve,
    pub holes: Vec<NurbsCurve>,
    pub plane: Plane,
}
```

---

### Layer 5: Geometric Queries (`solverang_query`)

**Purpose**: Spatial indexing and geometric queries for user interaction and algorithms.

#### Spatial Index: Bounding Volume Hierarchy (BVH)

```rust
pub struct BVH<T> {
    nodes: Vec<BVHNode<T>>,
    root: NodeId,
}

pub enum BVHNode<T> {
    Leaf {
        bounds: AABB,
        items: Vec<T>,
    },
    Internal {
        bounds: AABB,
        left: NodeId,
        right: NodeId,
    },
}

impl BVH<FaceId> {
    /// Build BVH from faces
    pub fn build(faces: &[FaceId], store: &TopologyStore) -> Self { ... }

    /// Ray cast query
    pub fn ray_cast(&self, ray: &Ray) -> Option<(FaceId, Point3, f64)> { ... }

    /// Frustum query (for selection)
    pub fn frustum_query(&self, frustum: &Frustum) -> Vec<FaceId> { ... }

    /// Nearest point on solid
    pub fn closest_point(&self, point: Point3) -> (FaceId, Point3, f64) { ... }
}
```

#### Query Operations

| Query              | Use Case                           | Implementation                     |
| ------------------ | ---------------------------------- | ---------------------------------- |
| Point-in-solid     | Determine if point is inside solid | Ray cast + odd/even crossing       |
| Closest point      | Snap to geometry                   | BVH + Newton iteration on surfaces |
| Ray cast           | Picking in 3D view                 | BVH + ray-surface intersection     |
| Interference check | Assembly validation                | BVH + SAT or GJK                   |
| Minimum distance   | Collision detection                | BVH + distance field               |
| Mass properties    | Engineering calculations           | Numeric integration over faces     |

```rust
pub trait GeometricQueries {
    fn point_in_solid(&self, point: Point3) -> bool;
    fn closest_point(&self, point: Point3) -> ClosestPointResult;
    fn ray_intersection(&self, ray: &Ray) -> Option<RayHit>;
    fn minimum_distance(&self, other: &Solid) -> f64;
    fn mass_properties(&self, density: f64) -> MassProperties;
    fn bounding_box(&self) -> AABB;
}

pub struct MassProperties {
    pub volume: f64,
    pub mass: f64,
    pub centroid: Point3,
    pub inertia_tensor: [[f64; 3]; 3],
}
```

---

### Layer 6: Serialization & I/O (`solverang_step`)

**Purpose**: STEP file import/export for interoperability.

#### STEP AP203/AP214 Support

STEP (ISO 10303) is the industry standard for CAD data exchange.

```rust
pub struct StepExporter {
    // Export B-rep to STEP AP203 (Configuration Controlled 3D Design)
    pub fn export_ap203(&self, solid: &Solid, store: &TopologyStore) -> StepFile { ... }

    // Export with product structure for assemblies
    pub fn export_ap214(&self, assembly: &Assembly, store: &TopologyStore) -> StepFile { ... }
}

pub struct StepImporter {
    // Import STEP with geometry healing
    pub fn import(&self, step_file: &StepFile) -> Result<ImportedModel, ImportError> { ... }

    // Healing operations
    fn heal_gaps(&self, brep: &mut Solid, tolerance: f64) { ... }
    fn stitch_faces(&self, faces: &[Face]) -> Shell { ... }
}
```

#### Event Sourcing for Geometry

```rust
// All geometric operations are events
#[derive(Clone, Serialize, Deserialize)]
pub enum GeometryEvent {
    // Primitive creation
    PointCreated { id: PointId, coords: Point3 },
    CurveCreated { id: CurveId, curve: NurbsCurve },
    SurfaceCreated { id: SurfaceId, surface: NurbsSurface },

    // Topology operations
    VertexCreated { id: VertexId, point_id: PointId },
    EdgeCreated { id: EdgeId, curve_id: CurveId, start: VertexId, end: VertexId },
    FaceCreated { id: FaceId, surface_id: SurfaceId, loops: Vec<LoopId> },
    SolidCreated { id: SolidId, shell: ShellId },

    // Feature operations
    FeatureCreated { id: FeatureId, feature: Feature },
    FeatureModified { id: FeatureId, changes: FeatureChanges },
    FeatureDeleted { id: FeatureId },

    // Boolean operations
    BooleanComputed { result_id: SolidId, operation: BoolOp, operands: Vec<SolidId> },
}

// Replay for time-travel debugging
impl GeometryStore {
    pub fn replay_to(&mut self, timestamp: HlcTimestamp) {
        // Reset state and replay events up to timestamp
    }
}
```

---

## Test Suites for TDD with AI Agents

Test suites are critical for AI agent development. Each test provides:
- **Clear PASS/FAIL criteria** (no subjective evaluation)
- **Numeric tolerance** where applicable
- **Performance threshold** for benchmarks
- **Reference output** for comparison

### NIST Test Suites (Authoritative Benchmarks)

#### 1. NIST Statistical Reference Data (StRD)

**Source**: https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

**Purpose**: Validate numerical solver accuracy with certified solutions.

| Dataset  | Variables | Residual Sum of Squares (certified) | Use For             |
| -------- | --------- | ----------------------------------- | ------------------- |
| Misra1a  | 2         | 1.2455138894E-01                    | Constraint solver   |
| Chwirut2 | 3         | 5.1304802941E+02                    | NURBS fitting       |
| Hahn1    | 7         | 1.5324382854E+00                    | Complex systems     |
| MGH17    | 5         | 5.4648946975E-05                    | High precision      |
| Lanczos3 | 6         | 1.6117193594E-08                    | Numerical stability |

**Test Structure**:
```rust
#[test]
fn nist_strd_misra1a() {
    let problem = nist::load_problem("Misra1a");
    let solution = solver.solve(&problem);

    // Certified residual sum of squares
    let expected_rss = 1.2455138894e-01;
    let actual_rss = solution.residual_sum_of_squares();

    // NIST requires 6 significant digits
    assert_relative_eq!(actual_rss, expected_rss, epsilon = 1e-6 * expected_rss);
}
```

#### 2. NIST STEP Conformance Testing

**Source**: ISO 10303 conformance test suites

**Purpose**: Validate STEP import/export correctness.

| Test Category         | Description               | Pass Criteria                  |
| --------------------- | ------------------------- | ------------------------------ |
| AP203 syntax          | File structure            | Parser accepts all valid files |
| Geometry roundtrip    | Export → import → compare | Volume difference < 0.1%       |
| Topology preservation | Edge/face counts          | Exact match                    |
| Product structure     | Assembly hierarchy        | Correct parent-child           |
| PMI data              | GD&T annotations          | Semantic preservation          |

**Test Structure**:
```rust
#[test]
fn step_roundtrip_volume_preservation() {
    let original = create_test_solid();
    let original_volume = original.volume();

    // Export to STEP
    let step_data = exporter.export_ap203(&original);

    // Import back
    let imported = importer.import(&step_data)?;
    let imported_volume = imported.volume();

    // Volume must match within 0.1%
    let tolerance = 0.001 * original_volume;
    assert!((imported_volume - original_volume).abs() < tolerance);
}
```

#### 3. NIST CAD Model Library

**Source**: https://www.nist.gov/ctl/psl/geometric-dimensioning-tolerancing-gdt

**Purpose**: Test boolean operations on real machined parts.

Models include:
- Simple prismatic parts (blocks with holes)
- Complex organic shapes
- Assemblies with interference fits
- Parts with GD&T annotations

### Open Test Suites

#### 1. OpenCASCADE Test Harness

**Source**: https://dev.opencascade.org/doc/overview/html/occt_dev_guides__tests.html

| Category  | Test Count | Description                   |
| --------- | ---------- | ----------------------------- |
| `boolean` | 500+       | Union, cut, common operations |
| `fillet`  | 200+       | Edge rounding                 |
| `offset`  | 100+       | Shell offset                  |
| `heal`    | 150+       | Geometry repair               |

**Usage**: Download OCCT test data, convert to our format, verify same results.

#### 2. ABC Dataset (Princeton)

**Source**: https://deep-geometry.github.io/abc-dataset/

**Purpose**: 1 million CAD models for robustness testing.

| Subset  | Models | Complexity     |
| ------- | ------ | -------------- |
| Simple  | 100K   | < 100 faces    |
| Medium  | 500K   | 100-1000 faces |
| Complex | 400K   | > 1000 faces   |

**Test Structure**:
```rust
#[test]
fn abc_dataset_import_1000() {
    let models = abc::load_subset("simple", 1000);
    let mut passed = 0;
    let mut failed = 0;

    for model in models {
        match importer.import(&model) {
            Ok(solid) => {
                if solid.validate().is_ok() {
                    passed += 1;
                } else {
                    failed += 1;
                }
            }
            Err(_) => failed += 1,
        }
    }

    // Require 95% pass rate
    assert!(passed as f64 / 1000.0 > 0.95);
}
```

#### 3. GrabCAD Community Models

**Source**: https://grabcad.com/library

**Purpose**: Real-world complexity, user-generated models.

Selected test cases:
- Automotive: Engine block, transmission housing
- Aerospace: Turbine blade, wing section
- Consumer: Phone case, game controller
- Industrial: Valve body, gearbox

### Performance Benchmarks (AI Agent Targets)

These are the performance targets that AI agents should verify during implementation:

| Operation                 | Target   | Measurement Method    | Notes                  |
| ------------------------- | -------- | --------------------- | ---------------------- |
| NURBS curve evaluation    | < 1 μs   | `criterion` benchmark | Single point, degree 3 |
| NURBS surface evaluation  | < 10 μs  | `criterion` benchmark | Single point, bicubic  |
| B-rep topology validation | < 10 ms  | `criterion` benchmark | 10,000 elements        |
| Boolean union (simple)    | < 50 ms  | `criterion` benchmark | 100 faces each         |
| Boolean union (complex)   | < 500 ms | `criterion` benchmark | 5,000 faces each       |
| Sketch solve (simple)     | < 5 ms   | `criterion` benchmark | 10 constraints         |
| Sketch solve (complex)    | < 50 ms  | `criterion` benchmark | 50 constraints         |
| Feature regeneration      | < 200 ms | End-to-end            | 5-feature chain        |
| STEP export               | < 1 s    | `criterion` benchmark | 10,000 faces           |
| STEP import               | < 2 s    | `criterion` benchmark | 10,000 faces           |
| Ray cast query            | < 1 ms   | `criterion` benchmark | 10,000 faces, BVH      |
| Closest point query       | < 5 ms   | `criterion` benchmark | 10,000 faces           |

**Benchmark Infrastructure**:
```rust
use criterion::{criterion_group, criterion_main, Criterion};

fn nurbs_curve_eval_benchmark(c: &mut Criterion) {
    let curve = create_test_curve(3, 10);  // degree 3, 10 control points

    c.bench_function("nurbs_curve_eval", |b| {
        b.iter(|| {
            curve.evaluate(0.5)  // Evaluate at parameter 0.5
        })
    });
}

fn boolean_union_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("boolean_union");

    for face_count in [100, 500, 1000, 5000] {
        let solid_a = create_test_solid(face_count);
        let solid_b = create_test_solid(face_count);

        group.bench_with_input(
            BenchmarkId::new("faces", face_count),
            &(solid_a, solid_b),
            |b, (a, b)| {
                b.iter(|| boolean_union(a, b))
            }
        );
    }

    group.finish();
}
```

### Test Infrastructure Layout

```
tests/
├── nist_strd/                  # NIST numerical benchmarks
│   ├── mod.rs
│   ├── misra1a.rs
│   ├── chwirut2.rs
│   └── datasets/               # Downloaded NIST data files
│       ├── Misra1a.dat
│       └── ...
│
├── step_conformance/           # STEP import/export validation
│   ├── mod.rs
│   ├── roundtrip_tests.rs
│   ├── ap203_tests.rs
│   ├── ap214_tests.rs
│   └── test_files/
│       ├── simple_block.step
│       ├── complex_assembly.step
│       └── ...
│
├── boolean_suite/              # Boolean operation edge cases
│   ├── mod.rs
│   ├── union_tests.rs
│   ├── subtract_tests.rs
│   ├── intersect_tests.rs
│   ├── tangent_cases.rs        # Known difficult cases
│   └── test_data/
│
├── topology_suite/             # Euler operator verification
│   ├── mod.rs
│   ├── euler_operators.rs
│   ├── manifold_tests.rs
│   └── validation_tests.rs
│
├── performance/                # Benchmark regression tests
│   ├── criterion_benches.rs
│   ├── baseline_values.json    # Expected performance baseline
│   └── regression_check.rs
│
└── integration/                # End-to-end feature modeling
    ├── mod.rs
    ├── sketch_to_solid.rs
    ├── feature_chain.rs
    ├── incremental_regen.rs    # DD integration tests
    └── event_replay.rs         # Event sourcing tests
```

### Test Categories for AI Agents

#### Category A: Unit Tests (Fast, Isolated)
- Run in < 100ms each
- No I/O dependencies
- Test single functions/methods
- Required for every code change

#### Category B: Integration Tests (Medium, Component)
- Run in < 5s each
- Test component interactions
- Run after passing Category A

#### Category C: Performance Tests (Slow, Benchmark)
- Run with `criterion` for statistical significance
- Compare against baseline
- Run before merging to main branch

#### Category D: Conformance Tests (Slow, External Data)
- NIST reference datasets
- STEP conformance suites
- Run nightly or on release

---

## Integration Points

### With `solverang` (Constraint Solving)

```rust
// Sketch constraints flow to geometry through solved positions
impl SketchToGeometry {
    /// Convert solved sketch to profile geometry
    pub fn extract_profile(
        solved_sketch: &SolvedSketch,
        solver_output: &SolverOutput,
    ) -> Result<Profile, ProfileError> {
        // Get solved point positions from constraint solver
        let positions: HashMap<VariableId, Point2D> = solver_output.solutions();

        // Extract closed loops
        let loops = solved_sketch.find_closed_loops(&positions)?;

        // Convert to NURBS curves
        let nurbs_loops: Vec<NurbsCurve> = loops.iter()
            .map(|loop_| self.loop_to_nurbs(loop_, &positions))
            .collect::<Result<_, _>>()?;

        Ok(Profile {
            outer_boundary: nurbs_loops[0].clone(),
            holes: nurbs_loops[1..].to_vec(),
            plane: solved_sketch.plane(),
        })
    }
}
```

### With `ecad_solver` (Differential Dataflow)

```rust
// Feature dependencies tracked via DD
pub struct FeatureDDIntegration {
    // DD worker for feature tree
    worker: Worker<Allocator>,

    // Feature definition collection
    features: InputSession<(FeatureId, Feature)>,

    // Dependency collection (auto-computed by DD)
    dependencies: InputSession<(FeatureId, FeatureId)>,

    // Computed B-rep results
    brep_probe: ProbeHandle<(FeatureId, Solid)>,
}

impl FeatureDDIntegration {
    /// When sketch dimension changes, DD determines affected features
    pub fn on_sketch_changed(&mut self, sketch_id: SketchId) {
        // DD propagates change through feature graph
        // Only features depending on this sketch are recomputed
        self.worker.step();

        // Query which features were affected
        let affected = self.brep_probe.changed_keys();
    }
}
```

### With `solverang_store` (Event Sourcing)

```rust
// Geometric operations as domain events
impl GeometryAggregate {
    /// Handle extrude command, emit events
    pub fn handle_extrude(&mut self, cmd: ExtrudeCommand) -> Result<Vec<GeometryEvent>, Error> {
        // Validate inputs
        let sketch = self.get_sketch(cmd.sketch_id)?;
        let profile = self.extract_profile(&sketch)?;

        // Perform extrusion
        let solid = self.extrude_profile(&profile, cmd.depth, cmd.direction)?;

        // Emit events (will be persisted with HLC timestamp)
        Ok(vec![
            GeometryEvent::FeatureCreated {
                id: cmd.feature_id,
                feature: Feature::Extrude(ExtrudeFeature {
                    sketch_id: cmd.sketch_id,
                    depth: cmd.depth,
                    direction: cmd.direction,
                    // ...
                }),
            },
            GeometryEvent::SolidCreated {
                id: solid.id,
                shell: solid.outer_shell,
            },
        ])
    }
}
```

---

## Implementation Phases

### Phase 1: NURBS Foundation (4 weeks)

**Goal**: Solid NURBS curve and surface implementation with evaluation and derivatives.

**Week 1-2**: NURBS Curves
- [ ] Implement `NurbsCurve` struct with knot vector validation
- [ ] De Boor's algorithm for point evaluation
- [ ] Derivative computation (1st and 2nd order)
- [ ] Curve splitting at parameter
- [ ] NIST StRD tests for numerical accuracy

**Week 3-4**: NURBS Surfaces
- [ ] Implement `NurbsSurface` with tensor-product structure
- [ ] Point and normal evaluation
- [ ] Partial derivative computation
- [ ] Surface trimming representation
- [ ] Performance benchmarks (< 10 μs per point)

**Deliverable**: `solverang_geometry` crate passing all NIST numerical tests.

### Phase 2: B-rep Topology (4 weeks)

**Goal**: Robust topology data structure with Euler operators.

**Week 1-2**: Core Data Structures
- [ ] Arena-based storage for topology elements
- [ ] Vertex, Edge, Loop, Face, Shell, Solid types
- [ ] Basic traversal methods
- [ ] Serialization (rkyv for zero-copy)

**Week 3-4**: Euler Operators
- [ ] Implement all 6 Euler operators
- [ ] Topology validation (Euler-Poincare)
- [ ] Manifold checking
- [ ] Orientation consistency
- [ ] Test suite for topology invariants

**Deliverable**: `solverang_topology` crate with 100% Euler operator coverage.

### Phase 3: Boolean Operations (6 weeks)

**Goal**: Robust boolean operations on B-rep solids.

**Week 1-2**: Surface-Surface Intersection
- [ ] Plane-plane intersection (analytical)
- [ ] Plane-cylinder intersection (analytical)
- [ ] NURBS-NURBS intersection (marching + subdivision)
- [ ] Edge-face intersection
- [ ] Intersection curve representation

**Week 3-4**: Face Classification
- [ ] Ray casting for point-in-solid
- [ ] Face classification (in/out/on)
- [ ] Handling tangent cases
- [ ] Coplanar face detection

**Week 5-6**: Topology Reconstruction
- [ ] Face selection by boolean type
- [ ] Shell reconstruction
- [ ] Edge sewing
- [ ] Result validation
- [ ] OpenCASCADE boolean test suite

**Deliverable**: `solverang_boolean` crate passing 90%+ of OCCT boolean tests.

### Phase 4: Feature Modeling (4 weeks)

**Goal**: Parametric features integrated with constraint solver.

**Week 1-2**: Profile Operations
- [ ] Profile extraction from solved sketches
- [ ] Closed loop detection
- [ ] Loop hierarchy (outer vs holes)
- [ ] 2D to 3D coordinate transform

**Week 3-4**: Basic Features
- [ ] Extrude (blind, through-all, mid-plane)
- [ ] Revolve (full revolution, partial)
- [ ] Feature boolean operations
- [ ] DD-based feature tree integration

**Deliverable**: `solverang_features` crate with working extrude/revolve.

### Phase 5: Advanced Features (4 weeks)

**Goal**: Sweep, loft, fillet, chamfer operations.

**Week 1-2**: Sweep and Loft
- [ ] Sweep along path curve
- [ ] Loft between profiles
- [ ] Guide curve support
- [ ] Twist and scale options

**Week 3-4**: Fillet and Chamfer
- [ ] Edge fillet with constant radius
- [ ] Variable radius fillet
- [ ] Chamfer (distance × distance, distance × angle)
- [ ] Chain selection for fillets

**Deliverable**: `solverang_features` crate with full feature palette.

**Note**: Fillet is a known hard problem. Start with simple cases (tangent blend surfaces) and handle complex vertex blends incrementally.

### Phase 6: STEP I/O & Polish (4 weeks)

**Goal**: Production-ready STEP import/export.

**Week 1-2**: STEP Export
- [ ] AP203 entity generation
- [ ] Geometry section (curves, surfaces)
- [ ] Topology section (B-rep)
- [ ] Product structure for assemblies

**Week 3-4**: STEP Import
- [ ] AP203 parser
- [ ] Geometry reconstruction
- [ ] Topology reconstruction
- [ ] Healing (gap closing, face stitching)
- [ ] NIST STEP conformance tests

**Deliverable**: `solverang_step` crate with roundtrip validation.

---

## Crate Structure

```
crates/
├── solverang_geometry/              # Layer 1: NURBS, primitives
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── point.rs              # Point2, Point3, Vector2, Vector3
│       ├── transform.rs          # Affine transforms
│       ├── plane.rs              # Plane representation
│       ├── nurbs/
│       │   ├── mod.rs
│       │   ├── curve.rs          # NurbsCurve
│       │   ├── surface.rs        # NurbsSurface
│       │   ├── knot_vector.rs    # Knot vector utilities
│       │   └── de_boor.rs        # De Boor's algorithm
│       ├── intersection/
│       │   ├── mod.rs
│       │   ├── curve_curve.rs
│       │   └── curve_surface.rs
│       └── tolerance.rs          # Global tolerance handling
│
├── solverang_topology/              # Layer 2: B-rep data structures
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── arena.rs              # Arena allocator for topology
│       ├── ids.rs                # VertexId, EdgeId, etc.
│       ├── vertex.rs
│       ├── edge.rs
│       ├── loop_.rs              # Loop (avoiding keyword)
│       ├── face.rs
│       ├── shell.rs
│       ├── solid.rs
│       ├── euler/
│       │   ├── mod.rs
│       │   ├── mvfs.rs           # Make Vertex, Face, Shell
│       │   ├── mev.rs            # Make Edge, Vertex
│       │   └── ...
│       ├── validation.rs         # Topology validation
│       └── traversal.rs          # Traversal utilities
│
├── solverang_boolean/               # Layer 3: Boolean operations
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── ssi/                  # Surface-Surface Intersection
│       │   ├── mod.rs
│       │   ├── plane_plane.rs
│       │   ├── plane_nurbs.rs
│       │   ├── nurbs_nurbs.rs
│       │   └── robustness.rs     # Interval arithmetic, tolerancing
│       ├── classify.rs           # Face classification
│       ├── select.rs             # Face selection by boolean type
│       ├── reconstruct.rs        # Topology reconstruction
│       └── boolean_op.rs         # Union, subtract, intersect API
│
├── solverang_features/              # Layer 4: Feature modeling
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── profile.rs            # Profile extraction from sketches
│       ├── extrude.rs
│       ├── revolve.rs
│       ├── sweep.rs
│       ├── loft.rs
│       ├── fillet.rs
│       ├── chamfer.rs
│       ├── pattern.rs            # Linear/circular patterns
│       ├── feature_tree.rs       # DD-based dependency tracking
│       └── parameters.rs         # Feature parameter expressions
│
├── solverang_query/                 # Layer 5: Spatial queries
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── bvh.rs                # Bounding Volume Hierarchy
│       ├── point_in_solid.rs
│       ├── ray_cast.rs
│       ├── closest_point.rs
│       ├── distance.rs
│       ├── mass_properties.rs
│       └── aabb.rs               # Axis-Aligned Bounding Box
│
├── solverang_step/                  # Layer 6: STEP I/O
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── parser/
│       │   ├── mod.rs
│       │   ├── lexer.rs
│       │   └── ap203.rs
│       ├── exporter/
│       │   ├── mod.rs
│       │   ├── geometry.rs
│       │   └── topology.rs
│       ├── importer/
│       │   ├── mod.rs
│       │   └── healing.rs
│       └── entities.rs           # STEP entity definitions
│
└── solverang_kernel/                # Unified API facade
    ├── Cargo.toml
    └── src/
        ├── lib.rs                # Re-exports from all layers
        ├── document.rs           # Part document with features
        ├── session.rs            # Editing session
        └── events.rs             # Event sourcing integration
```

---

## Risk Mitigation

| Risk                            | Likelihood | Impact | Mitigation Strategy                                                                                                        |
| ------------------------------- | ---------- | ------ | -------------------------------------------------------------------------------------------------------------------------- |
| SSI robustness issues           | High       | High   | Start with simpler surface types, add extensive edge case tests, use interval arithmetic for critical decisions            |
| Fillet complexity               | High       | Medium | Defer complex vertex blends, implement constant-radius edge fillet first, many real workflows work without complex fillets |
| Performance not meeting targets | Medium     | Medium | Profile early and often, use spatial indexing (BVH), cache intermediate results, parallelize where possible                |
| STEP compatibility issues       | Medium     | Medium | Test against real-world files early, implement healing for common issues, don't aim for 100% compatibility initially       |
| DD integration complexity       | Medium     | High   | Prototype DD integration early in Phase 4, keep feature dependencies simple initially                                      |
| Numerical precision errors      | Medium     | High   | Use consistent tolerance throughout, implement robust predicates for geometric decisions                                   |

### Fallback Strategies

1. **If SSI is too fragile**: Use tessellation-based boolean operations as fallback (mesh the surfaces, perform mesh booleans, fit surfaces to result)

2. **If fillet is too hard**: Defer to later phase, provide "shell" operation as alternative (offset surfaces)

3. **If STEP import fails**: Provide manual healing tools, allow users to fix imported geometry

---

## Success Criteria

### Must Have (Phase 6 complete)

1. **NIST StRD benchmarks**: Pass all numerical solver benchmarks with certified accuracy (6 significant digits)
2. **STEP roundtrip**: Export model, import back, volume difference < 0.1%
3. **Boolean operations**: Pass 85%+ of OpenCASCADE boolean test suite
4. **Feature regeneration**: < 200ms for 5-feature dependency chain
5. **Sketch integration**: Correctly extract profiles from constraint solver output

### Should Have

1. **ABC dataset import**: 90%+ success rate on simple subset (100K models)
2. **Performance targets**: Meet all benchmark targets in table above
3. **Fillet/chamfer**: Working constant-radius edge fillet

### Nice to Have

1. **Complex fillet**: Variable radius fillet, vertex blends
2. **Sweep/loft**: Advanced guide curve options
3. **STEP AP214**: Full assembly support

---

## AI Agent Development Guidelines

### Red-Green-TDD Loop

```
1. AI Agent reads test specification
2. Agent writes failing test (RED)
3. Agent implements minimal code to pass (GREEN)
4. Agent refactors while keeping tests green
5. Repeat
```

### Test-First Requirements

Every function must have tests before implementation:

```rust
// FIRST: Write the test
#[test]
fn nurbs_curve_evaluate_at_midpoint() {
    let curve = NurbsCurve::new(
        2,  // degree
        vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 0.0), Point3::new(2.0, 0.0, 0.0)],
        vec![1.0, 1.0, 1.0],  // uniform weights
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],  // clamped knot vector
    );

    let midpoint = curve.evaluate(0.5);

    // Expected: midpoint of parabola through these points
    assert_relative_eq!(midpoint.x, 1.0, epsilon = 1e-10);
    assert_relative_eq!(midpoint.y, 0.5, epsilon = 1e-10);  // Parabola peak
    assert_relative_eq!(midpoint.z, 0.0, epsilon = 1e-10);
}

// THEN: Implement the function
impl NurbsCurve {
    pub fn evaluate(&self, t: f64) -> Point3 {
        // ... implementation ...
    }
}
```

### Commit Granularity

- Each test + implementation = one commit
- Commit message format: `feat(solverang_geometry): implement NURBS curve evaluation [PASS: nurbs_curve_evaluate_at_midpoint]`
- Never commit with failing tests

### Performance Regression Prevention

```rust
// In CI, compare against baseline
#[test]
fn benchmark_regression_check() {
    let baseline = load_baseline("nurbs_curve_eval");
    let current = run_benchmark("nurbs_curve_eval");

    // Allow 10% regression
    assert!(current <= baseline * 1.1,
        "Performance regression: {} -> {} ({}% slower)",
        baseline, current, (current / baseline - 1.0) * 100.0
    );
}
```

---

## References

### Standards and Specifications
- **ISO 10303** (STEP): https://www.iso.org/standard/72562.html
- **NURBS Book**: Piegl & Tiller, "The NURBS Book", Springer 1997

### NIST Resources
- **StRD (Statistical Reference Data)**: https://www.itl.nist.gov/div898/strd/
- **STEP Conformance**: https://www.nist.gov/services-resources/software/step-file-analyzer-and-viewer

### Rust Crates
- **curvo** (NURBS): https://lib.rs/crates/curvo
- **nalgebra** (Linear algebra): https://nalgebra.org/
- **rstar** (R-tree): https://docs.rs/rstar/
- **criterion** (Benchmarks): https://docs.rs/criterion/

### Academic Datasets
- **ABC Dataset**: https://deep-geometry.github.io/abc-dataset/
- **Thingi10K**: https://ten-thousand-models.appspot.com/

### Reference Implementations
- **OpenCASCADE**: https://dev.opencascade.org/
- **SolveSpace**: https://solvespace.com/
- **truck** (Rust B-rep): https://github.com/ricosjp/truck

### Cadatomic Architecture
- Constraint Solver: [./CAD/00-overview.md](./CAD/00-overview.md)
- Differential Dataflow: [./CAD/01-differential-dataflow.md](./CAD/01-differential-dataflow.md)
- Event Sourcing: [../01-storage-layer.md](../01-storage-layer.md)
