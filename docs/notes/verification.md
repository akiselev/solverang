Differential testing against known-good implementations is exactly how
you'd validate a new geometric kernel at scale. The agents can iterate rapidly if they have a clear
oracle to test against.

Architecture: Reference Oracle Harness

┌─────────────────────────────────────────────────────────────────┐
│                     Test Harness (Rust)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ Test Case    │    │   Your New   │    │  Reference   │      │
│  │ Generator    │───▶│  Rust Kernel │    │   Oracles    │      │
│  │              │    └──────┬───────┘    ├──────────────┤      │
│  │ • Random     │           │            │ • OCCT (FFI) │      │
│  │ • Fuzzing    │           │            │ • CGAL (FFI) │      │
│  │ • Edge cases │           ▼            │ • truck      │      │
│  │ • STEP files │    ┌──────────────┐    │ • libigl     │      │
│  └──────────────┘    │   Result A   │    └──────┬───────┘      │
│                      └──────┬───────┘           │              │
│                             │                   ▼              │
│                             │            ┌──────────────┐      │
│                             │            │   Result B   │      │
│                             │            └──────┬───────┘      │
│                             │                   │              │
│                             ▼                   ▼              │
│                      ┌─────────────────────────────────┐       │
│                      │         Comparator              │       │
│                      │  • Topology equivalence         │       │
│                      │  • Geometry within tolerance    │       │
│                      │  • Mesh similarity (Hausdorff)  │       │
│                      └─────────────────────────────────┘       │
│                                      │                         │
│                                      ▼                         │
│                               PASS / FAIL                      │
│                          (with diagnostic diff)                │
└─────────────────────────────────────────────────────────────────┘

Reference Oracles by Operation
┌────────────────────────────────────┬────────────────┬───────────────┬──────────────────────────────┐
│             Operation              │ Primary Oracle │ Backup Oracle │            Notes             │
├────────────────────────────────────┼────────────────┼───────────────┼──────────────────────────────┤
│ Boolean (union/subtract/intersect) │ CGAL exact     │ OCCT          │ CGAL's exact is ground truth │
├────────────────────────────────────┼────────────────┼───────────────┼──────────────────────────────┤
│ Fillet/Chamfer                     │ OCCT           │ -             │ OCCT is best here            │
├────────────────────────────────────┼────────────────┼───────────────┼──────────────────────────────┤
│ NURBS evaluation                   │ OCCT           │ tinynurbs     │ Well-defined math            │
├────────────────────────────────────┼────────────────┼───────────────┼──────────────────────────────┤
│ Curve-curve intersection           │ CGAL           │ OCCT          │ CGAL has exact               │
├────────────────────────────────────┼────────────────┼───────────────┼──────────────────────────────┤
│ Surface-surface intersection       │ OCCT           │ CGAL          │ Hardest problem              │
├────────────────────────────────────┼────────────────┼───────────────┼──────────────────────────────┤
│ Triangulation/Meshing              │ CGAL           │ libigl        │ Both excellent               │
├────────────────────────────────────┼────────────────┼───────────────┼──────────────────────────────┤
│ Point-in-solid                     │ CGAL exact     │ OCCT          │ Exact predicates matter      │
├────────────────────────────────────┼────────────────┼───────────────┼──────────────────────────────┤
│ Convex hull                        │ CGAL           │ parry3d       │ Simple, well-solved          │
├────────────────────────────────────┼────────────────┼───────────────┼──────────────────────────────┤
│ STEP import                        │ OCCT           │ -             │ OCCT is the standard         │
└────────────────────────────────────┴────────────────┴───────────────┴──────────────────────────────┘
FFI Wrappers Needed

1. OCCT Wrapper (Primary Reference)

// crates/occt_oracle/src/lib.rs
use autocxx::prelude::*;

include_cpp! {
    #include "BRepAlgoAPI_Fuse.hxx"
    #include "BRepAlgoAPI_Cut.hxx"
    #include "BRepFilletAPI_MakeFillet.hxx"
    safety!(unsafe)
    generate!("BRepAlgoAPI_Fuse")
    generate!("BRepAlgoAPI_Cut")
}

pub struct OcctOracle;

impl BooleanOracle for OcctOracle {
    fn union(&self, a: &TestShape, b: &TestShape) -> OracleResult {
        // Convert TestShape → TopoDS_Shape
        // Run OCCT boolean
        // Convert back → OracleResult
    }
}

Existing crates to build on:
- opencascade-sys - Low-level bindings
- opencascade - Higher-level wrapper (incomplete)
- Or use autocxx / cxx for custom bindings

2. CGAL Wrapper (Exact Reference)

// crates/cgal_oracle/src/lib.rs

// CGAL's Nef_polyhedron_3 with exact kernel is the gold standard
// for boolean correctness

#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("cgal_wrapper.hpp");

        fn cgal_boolean_union(
            mesh_a: &[f64], faces_a: &[u32],
            mesh_b: &[f64], faces_b: &[u32],
        ) -> Vec<f64>;

        fn cgal_exact_boolean_union(
            mesh_a: &str,  // Exact rational coordinates as strings
            mesh_b: &str,
        ) -> String;
    }
}

3. libigl Wrapper (Mesh Operations)

// libigl is header-only C++, easy to wrap
// Excellent for mesh booleans (uses CGAL or Cork internally)

fn igl_mesh_boolean(
    va: &[[f64; 3]], fa: &[[usize; 3]],
    vb: &[[f64; 3]], fb: &[[usize; 3]],
    op: BooleanOp,
) -> (Vec<[f64; 3]>, Vec<[usize; 3]>);

Test Case Generation

1. Primitive Combinations

// Systematic: all primitive pairs
fn generate_primitive_pairs() -> impl Iterator<Item = TestCase> {
    let primitives = [
        Primitive::Box { size: [1.0, 1.0, 1.0] },
        Primitive::Sphere { radius: 1.0 },
        Primitive::Cylinder { radius: 0.5, height: 2.0 },
        Primitive::Cone { radius: 0.5, height: 1.5 },
        Primitive::Torus { major: 1.0, minor: 0.3 },
    ];

    let transforms = [
        Transform::identity(),
        Transform::translate(0.5, 0.0, 0.0),      // Partial overlap
        Transform::translate(0.01, 0.0, 0.0),     // Near-tangent (hard!)
        Transform::translate(2.0, 0.0, 0.0),      // No overlap
        Transform::rotate_z(0.1),                  // Rotated
        Transform::rotate_z(PI / 4.0),            // 45 degrees
    ];

    iproduct!(primitives, primitives, transforms, BooleanOp::all())
        .map(|(a, b, t, op)| TestCase::boolean(a, b.transform(t), op))
}

2. Fuzz Testing

// Random geometry within constraints
fn fuzz_boolean_ops(iterations: usize) {
    let mut rng = SmallRng::seed_from_u64(42);

    for _ in 0..iterations {
        let a = random_convex_polyhedron(&mut rng, 4..20);
        let b = random_convex_polyhedron(&mut rng, 4..20);
        let transform = random_transform(&mut rng);
        let op = rng.gen::<BooleanOp>();

        compare_with_oracle(a, b.transform(transform), op);
    }
}

3. Known Hard Cases

fn edge_cases() -> Vec<TestCase> {
    vec![
        // Coincident faces (notorious failure mode)
        TestCase::boolean(
            box_at_origin(1.0),
            box_at(1.0, 0.0, 0.0, 1.0),  // Shares a face exactly
            BooleanOp::Union,
        ),

        // Tangent spheres
        TestCase::boolean(
            sphere_at(0.0, 0.0, 0.0, 1.0),
            sphere_at(2.0, 0.0, 0.0, 1.0),  // Kiss at one point
            BooleanOp::Union,
        ),

        // Edge-on-face
        TestCase::boolean(
            box_at_origin(1.0),
            box_at(0.5, 0.5, 0.0, 1.0),  // Edge touches face interior
            BooleanOp::Subtract,
        ),

        // Vertex-on-vertex
        // ... many more
    ]
}

4. Import Real-World STEP Files

fn step_file_tests() -> Vec<TestCase> {
    // Grab STEP files from:
    // - GrabCAD (thousands of free models)
    // - NIST CAD test suite
    // - FreeCAD test files
    // - Your own collection

    glob("test_data/step/*.stp")
        .map(|path| TestCase::import_step(path))
        .collect()
}

Comparison Logic

Topology Comparison

fn compare_topology(result: &BrepSolid, oracle: &OracleResult) -> TopoComparison {
    TopoComparison {
        // Count-based (fast sanity check)
        vertex_count_match: result.vertices().len() == oracle.vertex_count,
        edge_count_match: result.edges().len() == oracle.edge_count,
        face_count_match: result.faces().len() == oracle.face_count,
        shell_count_match: result.shells().len() == oracle.shell_count,

        // Euler characteristic (topological invariant)
        // V - E + F = 2 for simple closed solid
        euler_matches: result.euler_characteristic() == oracle.euler,

        // Genus (number of holes/handles)
        genus_matches: result.genus() == oracle.genus,
    }
}

Geometry Comparison

fn compare_geometry(
    result: &BrepSolid,
    oracle: &OracleMesh,
    tolerance: f64,
) -> GeomComparison {
    // Mesh both for comparison
    let result_mesh = result.tessellate(tolerance / 10.0);

    // Hausdorff distance (max distance from any point to other surface)
    let hausdorff = symmetric_hausdorff_distance(&result_mesh, oracle);

    // Volume comparison
    let volume_diff = (result.volume() - oracle.volume).abs() / oracle.volume;

    // Surface area comparison
    let area_diff = (result.surface_area() - oracle.area).abs() / oracle.area;

    GeomComparison {
        hausdorff_distance: hausdorff,
        hausdorff_ok: hausdorff < tolerance,
        volume_relative_error: volume_diff,
        volume_ok: volume_diff < 1e-6,
        area_relative_error: area_diff,
        area_ok: area_diff < 1e-6,
    }
}

Point Sampling Comparison

fn compare_point_classification(
    result: &BrepSolid,
    oracle: &impl PointInSolidOracle,
    sample_count: usize,
) -> PointClassComparison {
    let bbox = result.bounding_box().expanded(0.1);
    let mut mismatches = Vec::new();

    for point in bbox.random_points(sample_count) {
        let ours = result.classify_point(point);
        let theirs = oracle.classify_point(point);

        if ours != theirs {
            mismatches.push((point, ours, theirs));
        }
    }

    PointClassComparison {
        samples: sample_count,
        mismatches,
        pass: mismatches.is_empty(),
    }
}

Harness Runner

// crates/kernel_harness/src/lib.rs

pub struct TestHarness {
    occt: OcctOracle,
    cgal: CgalOracle,
    our_kernel: RustKernel,
}

impl TestHarness {
    pub fn run_test(&self, test: &TestCase) -> TestResult {
        // Run our implementation
        let start = Instant::now();
        let our_result = match &test.operation {
            Op::Boolean { a, b, op } => self.our_kernel.boolean(a, b, *op),
            Op::Fillet { solid, edges, radius } => self.our_kernel.fillet(solid, edges, *radius),
            // ...
        };
        let our_time = start.elapsed();

        // Run oracle
        let oracle_result = match &test.operation {
            Op::Boolean { a, b, op } => self.occt.boolean(a, b, *op),
            // ...
        };

        // Compare
        let comparison = match (&our_result, &oracle_result) {
            (Ok(ours), Ok(oracle)) => {
                let topo = compare_topology(ours, oracle);
                let geom = compare_geometry(ours, oracle, test.tolerance);
                Comparison::Both { topo, geom }
            }
            (Err(e), Ok(_)) => Comparison::WeShouldntFail(e.clone()),
            (Ok(_), Err(e)) => Comparison::WeSucceededOracleFailed(e.clone()),
            (Err(e1), Err(e2)) => Comparison::BothFailed(e1.clone(), e2.clone()),
        };

        TestResult {
            test: test.clone(),
            our_time,
            comparison,
            pass: comparison.is_pass(),
        }
    }

    pub fn run_all(&self, tests: &[TestCase]) -> HarnessReport {
        tests.par_iter()  // Parallel!
            .map(|t| self.run_test(t))
            .collect()
    }
}

CI Integration

# .github/workflows/kernel-oracle.yml
name: Kernel Oracle Tests

on: [push, pull_request]

jobs:
oracle-tests:
    runs-on: ubuntu-latest
    container:
    image: ghcr.io/your-org/kernel-test:latest  # Has OCCT, CGAL installed

    steps:
    - uses: actions/checkout@v4

    - name: Run oracle comparison
        run: |
        cargo test -p kernel_harness --release -- --nocapture

    - name: Fuzz test (30 min)
        run: |
        cargo run -p kernel_harness --release -- fuzz --duration 30m

    - name: Upload failure cases
        if: failure()
        uses: actions/upload-artifact@v4
        with:
        name: failing-cases
        path: target/failing_cases/

Docker Image for Testing

# Dockerfile.kernel-test
FROM ubuntu:24.04

# OCCT
RUN apt-get update && apt-get install -y \
    libocct-*-dev \
    occt-misc

# CGAL
RUN apt-get install -y \
    libcgal-dev \
    libgmp-dev \
    libmpfr-dev

# libigl (header-only, just clone)
RUN git clone --depth 1 https://github.com/libigl/libigl /opt/libigl

# Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

ENV OCCT_ROOT=/usr
ENV CGAL_ROOT=/usr
ENV LIBIGL_ROOT=/opt/libigl

Summary: What You Need to Build

1. occt_oracle - FFI wrapper for OCCT operations
2. cgal_oracle - FFI wrapper for CGAL exact operations
3. kernel_harness - Test runner, comparison logic, reporting
4. test_generator - Systematic + fuzz test case generation
5. Docker image - Reproducible environment with all dependencies

With this harness, your Claude agents can:
1. Implement an operation
2. Run cargo test -p kernel_harness
3. See exactly which cases fail and how
  4. Iterate until green

  The oracle never lies. If OCCT and CGAL agree, and you don't match, you have a bug. This is the kind of
   tight feedback loop that makes agent-driven development actually work.

  Want me to start scaffolding the occt_oracle crate or the test harness structure?Differential testing against known-good implementations is exactly how
  you'd validate a new geometric kernel at scale. The agents can iterate rapidly if they have a clear
  oracle to test against.

  Architecture: Reference Oracle Harness

  ┌─────────────────────────────────────────────────────────────────┐
  │                     Test Harness (Rust)                         │
  ├─────────────────────────────────────────────────────────────────┤
  │                                                                 │
  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
  │  │ Test Case    │    │   Your New   │    │  Reference   │      │
  │  │ Generator    │───▶│  Rust Kernel │    │   Oracles    │      │
  │  │              │    └──────┬───────┘    ├──────────────┤      │
  │  │ • Random     │           │            │ • OCCT (FFI) │      │
  │  │ • Fuzzing    │           │            │ • CGAL (FFI) │      │
  │  │ • Edge cases │           ▼            │ • truck      │      │
  │  │ • STEP files │    ┌──────────────┐    │ • libigl     │      │
  │  └──────────────┘    │   Result A   │    └──────┬───────┘      │
  │                      └──────┬───────┘           │              │
  │                             │                   ▼              │
  │                             │            ┌──────────────┐      │
  │                             │            │   Result B   │      │
  │                             │            └──────┬───────┘      │
  │                             │                   │              │
  │                             ▼                   ▼              │
  │                      ┌─────────────────────────────────┐       │
  │                      │         Comparator              │       │
  │                      │  • Topology equivalence         │       │
  │                      │  • Geometry within tolerance    │       │
  │                      │  • Mesh similarity (Hausdorff)  │       │
  │                      └─────────────────────────────────┘       │
  │                                      │                         │
  │                                      ▼                         │
  │                               PASS / FAIL                      │
  │                          (with diagnostic diff)                │
  └─────────────────────────────────────────────────────────────────┘

  Reference Oracles by Operation
  ┌────────────────────────────────────┬────────────────┬───────────────┬──────────────────────────────┐
  │             Operation              │ Primary Oracle │ Backup Oracle │            Notes             │
  ├────────────────────────────────────┼────────────────┼───────────────┼──────────────────────────────┤
  │ Boolean (union/subtract/intersect) │ CGAL exact     │ OCCT          │ CGAL's exact is ground truth │
  ├────────────────────────────────────┼────────────────┼───────────────┼──────────────────────────────┤
  │ Fillet/Chamfer                     │ OCCT           │ -             │ OCCT is best here            │
  ├────────────────────────────────────┼────────────────┼───────────────┼──────────────────────────────┤
  │ NURBS evaluation                   │ OCCT           │ tinynurbs     │ Well-defined math            │
  ├────────────────────────────────────┼────────────────┼───────────────┼──────────────────────────────┤
  │ Curve-curve intersection           │ CGAL           │ OCCT          │ CGAL has exact               │
  ├────────────────────────────────────┼────────────────┼───────────────┼──────────────────────────────┤
  │ Surface-surface intersection       │ OCCT           │ CGAL          │ Hardest problem              │
  ├────────────────────────────────────┼────────────────┼───────────────┼──────────────────────────────┤
  │ Triangulation/Meshing              │ CGAL           │ libigl        │ Both excellent               │
  ├────────────────────────────────────┼────────────────┼───────────────┼──────────────────────────────┤
  │ Point-in-solid                     │ CGAL exact     │ OCCT          │ Exact predicates matter      │
  ├────────────────────────────────────┼────────────────┼───────────────┼──────────────────────────────┤
  │ Convex hull                        │ CGAL           │ parry3d       │ Simple, well-solved          │
  ├────────────────────────────────────┼────────────────┼───────────────┼──────────────────────────────┤
  │ STEP import                        │ OCCT           │ -             │ OCCT is the standard         │
  └────────────────────────────────────┴────────────────┴───────────────┴──────────────────────────────┘
  FFI Wrappers Needed

  1. OCCT Wrapper (Primary Reference)

  // crates/occt_oracle/src/lib.rs
  use autocxx::prelude::*;

  include_cpp! {
      #include "BRepAlgoAPI_Fuse.hxx"
      #include "BRepAlgoAPI_Cut.hxx"
      #include "BRepFilletAPI_MakeFillet.hxx"
      safety!(unsafe)
      generate!("BRepAlgoAPI_Fuse")
      generate!("BRepAlgoAPI_Cut")
  }

  pub struct OcctOracle;

  impl BooleanOracle for OcctOracle {
      fn union(&self, a: &TestShape, b: &TestShape) -> OracleResult {
          // Convert TestShape → TopoDS_Shape
          // Run OCCT boolean
          // Convert back → OracleResult
      }
  }

  Existing crates to build on:
  - opencascade-sys - Low-level bindings
  - opencascade - Higher-level wrapper (incomplete)
  - Or use autocxx / cxx for custom bindings

  2. CGAL Wrapper (Exact Reference)

  // crates/cgal_oracle/src/lib.rs

  // CGAL's Nef_polyhedron_3 with exact kernel is the gold standard
  // for boolean correctness

  #[cxx::bridge]
  mod ffi {
      unsafe extern "C++" {
          include!("cgal_wrapper.hpp");

          fn cgal_boolean_union(
              mesh_a: &[f64], faces_a: &[u32],
              mesh_b: &[f64], faces_b: &[u32],
          ) -> Vec<f64>;

          fn cgal_exact_boolean_union(
              mesh_a: &str,  // Exact rational coordinates as strings
              mesh_b: &str,
          ) -> String;
      }
  }

  3. libigl Wrapper (Mesh Operations)

  // libigl is header-only C++, easy to wrap
  // Excellent for mesh booleans (uses CGAL or Cork internally)

  fn igl_mesh_boolean(
      va: &[[f64; 3]], fa: &[[usize; 3]],
      vb: &[[f64; 3]], fb: &[[usize; 3]],
      op: BooleanOp,
  ) -> (Vec<[f64; 3]>, Vec<[usize; 3]>);

  Test Case Generation

  1. Primitive Combinations

  // Systematic: all primitive pairs
  fn generate_primitive_pairs() -> impl Iterator<Item = TestCase> {
      let primitives = [
          Primitive::Box { size: [1.0, 1.0, 1.0] },
          Primitive::Sphere { radius: 1.0 },
          Primitive::Cylinder { radius: 0.5, height: 2.0 },
          Primitive::Cone { radius: 0.5, height: 1.5 },
          Primitive::Torus { major: 1.0, minor: 0.3 },
      ];

      let transforms = [
          Transform::identity(),
          Transform::translate(0.5, 0.0, 0.0),      // Partial overlap
          Transform::translate(0.01, 0.0, 0.0),     // Near-tangent (hard!)
          Transform::translate(2.0, 0.0, 0.0),      // No overlap
          Transform::rotate_z(0.1),                  // Rotated
          Transform::rotate_z(PI / 4.0),            // 45 degrees
      ];

      iproduct!(primitives, primitives, transforms, BooleanOp::all())
          .map(|(a, b, t, op)| TestCase::boolean(a, b.transform(t), op))
  }

  2. Fuzz Testing

  // Random geometry within constraints
  fn fuzz_boolean_ops(iterations: usize) {
      let mut rng = SmallRng::seed_from_u64(42);

      for _ in 0..iterations {
          let a = random_convex_polyhedron(&mut rng, 4..20);
          let b = random_convex_polyhedron(&mut rng, 4..20);
          let transform = random_transform(&mut rng);
          let op = rng.gen::<BooleanOp>();

          compare_with_oracle(a, b.transform(transform), op);
      }
  }

  3. Known Hard Cases

  fn edge_cases() -> Vec<TestCase> {
      vec![
          // Coincident faces (notorious failure mode)
          TestCase::boolean(
              box_at_origin(1.0),
              box_at(1.0, 0.0, 0.0, 1.0),  // Shares a face exactly
              BooleanOp::Union,
          ),

          // Tangent spheres
          TestCase::boolean(
              sphere_at(0.0, 0.0, 0.0, 1.0),
              sphere_at(2.0, 0.0, 0.0, 1.0),  // Kiss at one point
              BooleanOp::Union,
          ),

          // Edge-on-face
          TestCase::boolean(
              box_at_origin(1.0),
              box_at(0.5, 0.5, 0.0, 1.0),  // Edge touches face interior
              BooleanOp::Subtract,
          ),

          // Vertex-on-vertex
          // ... many more
      ]
  }

  4. Import Real-World STEP Files

  fn step_file_tests() -> Vec<TestCase> {
      // Grab STEP files from:
      // - GrabCAD (thousands of free models)
      // - NIST CAD test suite
      // - FreeCAD test files
      // - Your own collection

      glob("test_data/step/*.stp")
          .map(|path| TestCase::import_step(path))
          .collect()
  }

  Comparison Logic

  Topology Comparison

  fn compare_topology(result: &BrepSolid, oracle: &OracleResult) -> TopoComparison {
      TopoComparison {
          // Count-based (fast sanity check)
          vertex_count_match: result.vertices().len() == oracle.vertex_count,
          edge_count_match: result.edges().len() == oracle.edge_count,
          face_count_match: result.faces().len() == oracle.face_count,
          shell_count_match: result.shells().len() == oracle.shell_count,

          // Euler characteristic (topological invariant)
          // V - E + F = 2 for simple closed solid
          euler_matches: result.euler_characteristic() == oracle.euler,

          // Genus (number of holes/handles)
          genus_matches: result.genus() == oracle.genus,
      }
  }

  Geometry Comparison

  fn compare_geometry(
      result: &BrepSolid,
      oracle: &OracleMesh,
      tolerance: f64,
  ) -> GeomComparison {
      // Mesh both for comparison
      let result_mesh = result.tessellate(tolerance / 10.0);

      // Hausdorff distance (max distance from any point to other surface)
      let hausdorff = symmetric_hausdorff_distance(&result_mesh, oracle);

      // Volume comparison
      let volume_diff = (result.volume() - oracle.volume).abs() / oracle.volume;

      // Surface area comparison
      let area_diff = (result.surface_area() - oracle.area).abs() / oracle.area;

      GeomComparison {
          hausdorff_distance: hausdorff,
          hausdorff_ok: hausdorff < tolerance,
          volume_relative_error: volume_diff,
          volume_ok: volume_diff < 1e-6,
          area_relative_error: area_diff,
          area_ok: area_diff < 1e-6,
      }
  }

  Point Sampling Comparison

  fn compare_point_classification(
      result: &BrepSolid,
      oracle: &impl PointInSolidOracle,
      sample_count: usize,
  ) -> PointClassComparison {
      let bbox = result.bounding_box().expanded(0.1);
      let mut mismatches = Vec::new();

      for point in bbox.random_points(sample_count) {
          let ours = result.classify_point(point);
          let theirs = oracle.classify_point(point);

          if ours != theirs {
              mismatches.push((point, ours, theirs));
          }
      }

      PointClassComparison {
          samples: sample_count,
          mismatches,
          pass: mismatches.is_empty(),
      }
  }

  Harness Runner

  // crates/kernel_harness/src/lib.rs

  pub struct TestHarness {
      occt: OcctOracle,
      cgal: CgalOracle,
      our_kernel: RustKernel,
  }

  impl TestHarness {
      pub fn run_test(&self, test: &TestCase) -> TestResult {
          // Run our implementation
          let start = Instant::now();
          let our_result = match &test.operation {
              Op::Boolean { a, b, op } => self.our_kernel.boolean(a, b, *op),
              Op::Fillet { solid, edges, radius } => self.our_kernel.fillet(solid, edges, *radius),
              // ...
          };
          let our_time = start.elapsed();

          // Run oracle
          let oracle_result = match &test.operation {
              Op::Boolean { a, b, op } => self.occt.boolean(a, b, *op),
              // ...
          };

          // Compare
          let comparison = match (&our_result, &oracle_result) {
              (Ok(ours), Ok(oracle)) => {
                  let topo = compare_topology(ours, oracle);
                  let geom = compare_geometry(ours, oracle, test.tolerance);
                  Comparison::Both { topo, geom }
              }
              (Err(e), Ok(_)) => Comparison::WeShouldntFail(e.clone()),
              (Ok(_), Err(e)) => Comparison::WeSucceededOracleFailed(e.clone()),
              (Err(e1), Err(e2)) => Comparison::BothFailed(e1.clone(), e2.clone()),
          };

          TestResult {
              test: test.clone(),
              our_time,
              comparison,
              pass: comparison.is_pass(),
          }
      }

      pub fn run_all(&self, tests: &[TestCase]) -> HarnessReport {
          tests.par_iter()  // Parallel!
              .map(|t| self.run_test(t))
              .collect()
      }
  }

  CI Integration

  # .github/workflows/kernel-oracle.yml
  name: Kernel Oracle Tests

  on: [push, pull_request]

  jobs:
    oracle-tests:
      runs-on: ubuntu-latest
      container:
        image: ghcr.io/your-org/kernel-test:latest  # Has OCCT, CGAL installed

      steps:
        - uses: actions/checkout@v4

        - name: Run oracle comparison
          run: |
            cargo test -p kernel_harness --release -- --nocapture

        - name: Fuzz test (30 min)
          run: |
            cargo run -p kernel_harness --release -- fuzz --duration 30m

        - name: Upload failure cases
          if: failure()
          uses: actions/upload-artifact@v4
          with:
            name: failing-cases
            path: target/failing_cases/

  Docker Image for Testing

  # Dockerfile.kernel-test
  FROM ubuntu:24.04

  # OCCT
  RUN apt-get update && apt-get install -y \
      libocct-*-dev \
      occt-misc

  # CGAL
  RUN apt-get install -y \
      libcgal-dev \
      libgmp-dev \
      libmpfr-dev

  # libigl (header-only, just clone)
  RUN git clone --depth 1 https://github.com/libigl/libigl /opt/libigl

  # Rust
  RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

  ENV OCCT_ROOT=/usr
  ENV CGAL_ROOT=/usr
  ENV LIBIGL_ROOT=/opt/libigl

  Summary: What You Need to Build

  1. occt_oracle - FFI wrapper for OCCT operations
  2. cgal_oracle - FFI wrapper for CGAL exact operations
  3. kernel_harness - Test runner, comparison logic, reporting
  4. test_generator - Systematic + fuzz test case generation
  5. Docker image - Reproducible environment with all dependencies

  With this harness, your Claude agents can:
  1. Implement an operation
  2. Run cargo test -p kernel_harness
  3. See exactly which cases fail and how
  4. Iterate until green

  The oracle never lies. If OCCT and CGAL agree, and you don't match, you have a bug. This is the kind of
   tight feedback loop that makes agent-driven development actually work.

  Want me to start scaffolding the occt_oracle crate or the test harness structure?