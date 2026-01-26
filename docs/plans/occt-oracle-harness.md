# OCCT Oracle Test Harness Plan

## Overview

This plan implements an OCCT oracle test harness for agent-driven geometric kernel development. The system consists of two crates: `occt_oracle` (FFI wrapper for Open CASCADE) and `kernel_harness` (test runner with comparison logic). The harness enables Claude Code agents to implement geometric operations by providing a red-green testing loop that compares results against OCCT as ground truth.

**Chosen approach**: Option C (Hybrid) - cxx FFI bindings for type safety + declarative test DSL macro for agent-friendly test creation. OCCT is required for tests (fail if unavailable).

**Location**: `/crates/` in the solverang repository.

## Planning Context

### Decision Log

| Decision | Reasoning Chain |
|----------|-----------------|
| cxx over autocxx/bindgen | cxx provides strong type safety at FFI boundary -> agents won't accidentally create memory bugs -> reduced debugging overhead outweighs manual wrapper code cost |
| Hybrid architecture (cxx + DSL) | Agents need to add tests rapidly without FFI knowledge -> DSL macro abstracts oracle calls -> agents focus on test cases, not plumbing -> fastest iteration speed |
| Required OCCT for tests | Oracle is ground truth for correctness -> graceful skip would allow bugs to pass CI -> fail-fast ensures no untested code ships |
| Primitives + Booleans MVP | Booleans are the hardest operation (coincident faces, tangent intersections) -> solving booleans first proves architecture handles hard cases -> primitives are prerequisites for boolean inputs |
| crates/ location | Solverang is now standalone repo -> crates/ is standard Rust workspace layout -> consistent with atomic_solver structure |
| Volume + topology comparison | Volume catches gross geometric errors -> topology (face/edge/vertex counts) catches topological bugs -> Hausdorff deferred to M3 as optional enhancement |
| Test case serialization with rkyv | Zero-copy deserialization matches solverang patterns (see atomic_solver) -> faster test loading than JSON -> binary format prevents accidental test case mutation |
| Tolerance 1e-6 for volume | OCCT uses double precision internally -> 1e-6 relative error catches meaningful bugs without false positives from floating-point noise -> matches solver_comparison.rs patterns |

### Rejected Alternatives

| Alternative | Why Rejected |
|-------------|--------------|
| bindgen raw bindings | Unsafe everywhere, no type safety at FFI boundary, agents would create memory bugs |
| autocxx | Learning curve + magic behavior, harder to debug FFI issues, cxx provides sufficient coverage |
| Feature-flagged optional OCCT | Would allow untested code paths, defeats purpose of oracle-based development |
| JSON test serialization | Slower than rkyv, human-editable (mutation risk), doesn't match codebase patterns |
| Exact numeric comparison | Impossible across FFI boundary due to floating-point representation differences |

### Constraints & Assumptions

- **Technical**: OCCT 7.7+ required (Ubuntu 24.04 has 7.6, may need PPA or source build)
- **Technical**: cxx requires C++17, OCCT headers are C++11 compatible
- **Technical**: All OCCT operations return TopoDS_Shape, need conversion to comparable format
- **Organizational**: Agents will write tests, humans review PRs
- **Dependencies**: cxx 1.0+, cxx-build for build.rs, OCCT development headers

### Known Risks

| Risk | Mitigation | Anchor |
|------|------------|--------|
| OCCT not installed in CI | Docker image with OCCT pre-installed, fail build if headers missing | build.rs:L1-20 (will add) |
| FFI memory leaks | cxx UniquePtr wraps all OCCT objects, Drop handles cleanup | types.rs design |
| OCCT API changes between versions | Pin to OCCT 7.7, version check in build.rs | build.rs:L25-35 (will add) |
| Comparison false positives | Start with loose tolerance (1e-6), tighten based on empirical data | metrics.rs design |
| Complex OCCT build setup | Provide Dockerfile and setup script | docs/SETUP.md (M4) |

## Invisible Knowledge

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        kernel_harness                           │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │ Test DSL    │───▶│ HarnessRunner│───▶│ ComparisonReport   │ │
│  │ oracle_test!│    │             │    │ (pass/fail + diff) │ │
│  └─────────────┘    └──────┬──────┘    └─────────────────────┘ │
│                            │                                    │
│         ┌──────────────────┼──────────────────┐                │
│         ▼                  ▼                  ▼                │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │ Your Kernel │    │ OCCT Oracle │    │ Comparator  │        │
│  │ (impl later)│    │ (occt_oracle)│   │ (metrics.rs)│        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ FFI (cxx)
                    ┌─────────────────────┐
                    │   Open CASCADE      │
                    │   (system library)  │
                    └─────────────────────┘
```

### Data Flow

```
Test Case (DSL)
      │
      ▼
┌─────────────────┐
│ Parse to Op enum│  (Boolean::Union, Primitive::Box, etc.)
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│Kernel │ │Oracle │  (parallel evaluation)
└───┬───┘ └───┬───┘
    │         │
    ▼         ▼
┌─────────────────┐
│   Comparator    │  (volume, face count, Hausdorff)
└────────┬────────┘
         │
         ▼
   PASS / FAIL
   (with diagnostic diff if fail)
```

### Why This Structure

- **Two crates**: occt_oracle is reusable for other oracles (CGAL later), kernel_harness is test-specific
- **DSL macro**: Agents add tests without understanding FFI, just declare inputs and expected comparisons
- **Trait-based oracle**: GeometricOracle trait allows swapping OCCT for other reference implementations
- **Comparison configurability**: Different operations need different metrics (booleans need volume, NURBS need point samples)

### Invariants

1. Oracle is always called, even if kernel panics (wrap kernel call in catch_unwind)
2. Test case serialization is deterministic (same input -> same bytes)
3. Comparison metrics are commutative (compare(a,b) == compare(b,a))
4. All OCCT objects are freed before test completes (no memory leaks between tests)

### Tradeoffs

- **Manual cxx wrappers**: More initial code, but safer and easier to debug than auto-generated
- **Required OCCT**: Higher barrier to run tests, but ensures correctness is always verified
- **DSL complexity**: Macro adds learning curve, but agents can add tests 10x faster
- **rkyv serialization**: Binary format not human-readable, but matches codebase and is fast

## Milestones

### Milestone 1: occt_oracle Core Types and Build Setup

**Files**:
- `crates/occt_oracle/Cargo.toml`
- `crates/occt_oracle/build.rs`
- `crates/occt_oracle/src/lib.rs`
- `crates/occt_oracle/src/types.rs`
- `crates/occt_oracle/src/ffi.rs`
- `crates/occt_oracle/cpp/wrapper.hpp`
- `crates/occt_oracle/cpp/wrapper.cpp`

**Flags**: `error-handling`, `security` (FFI boundary)

**Requirements**:
- build.rs detects OCCT installation, fails with clear error if missing
- build.rs checks OCCT version >= 7.7
- cxx bridge defines Shape, Solid, Point3D types
- C++ wrapper provides safe accessors for OCCT TopoDS_Shape
- Rust types wrap cxx::UniquePtr for automatic cleanup

**Acceptance Criteria**:
- `cargo build -p occt_oracle` succeeds with OCCT installed
- `cargo build -p occt_oracle` fails with clear message without OCCT
- Shape can be created and dropped without memory leak (valgrind clean)

**Tests**:
- **Test files**: `crates/occt_oracle/tests/build_test.rs`
- **Test type**: unit
- **Backing**: default-derived
- **Scenarios**:
  - Normal: Create Shape, drop it, no crash
  - Edge: Multiple shapes created/dropped in sequence
  - Error: (N/A - build failure is the error case)

**Code Intent**:
- `Cargo.toml`: Define crate with cxx, cxx-build dependencies, link to OCCT libs
- `build.rs`: Use pkg-config or cmake to find OCCT, emit cargo:rustc-link-lib directives, version check
- `ffi.rs`: cxx::bridge module defining Shape, Point3D, basic accessors
- `wrapper.hpp/cpp`: C++ side of cxx bridge, includes OCCT headers, provides safe wrapper functions
- `types.rs`: Rust-side wrapper structs with Drop impl, conversion traits
- `lib.rs`: Re-export public API

**Code Changes**:

```diff
--- /dev/null
+++ b/crates/occt_oracle/Cargo.toml
@@ -0,0 +1,15 @@
+[package]
+name = "occt_oracle"
+version = "0.1.0"
+edition = "2021"
+
+[lib]
+path = "src/lib.rs"
+
+[dependencies]
+cxx = "1.0"
+anyhow = "1.0"
+
+[build-dependencies]
+cxx-build = "1.0"
+pkg-config = "0.3"
```

```diff
--- /dev/null
+++ b/crates/occt_oracle/build.rs
@@ -0,0 +1,77 @@
+use std::env;
+use std::path::PathBuf;
+
+/// Extracted to provide clear error context at each parse site
+fn parse_version_component(component: &str) -> Result<u32, String> {
+    component.parse::<u32>()
+        .map_err(|_| format!("Invalid version component: '{}'", component))
+}
+
+fn main() {
+    // Try to find OCCT via pkg-config
+    let occt = pkg_config::Config::new()
+        .atleast_version("7.7.0")
+        .probe("opencascade")
+        .unwrap_or_else(|e| {
+            eprintln!("ERROR: Open CASCADE Technology (OCCT) >= 7.7.0 not found");
+            eprintln!("       {}", e);
+            eprintln!();
+            eprintln!("Install OCCT development headers:");
+            eprintln!("  Ubuntu/Debian: apt install libocct-*-dev");
+            eprintln!("  Fedora:        dnf install opencascade-devel");
+            eprintln!("  macOS:         brew install opencascade");
+            eprintln!();
+            eprintln!("Or build from source: https://dev.opencascade.org/");
+            std::process::exit(1);
+        });
+
+    // Verify version with explicit error handling
+    let version_parts = occt.version.split('.').collect::<Vec<_>>();
+
+    if version_parts.is_empty() {
+        eprintln!("ERROR: Invalid OCCT version format: '{}'", occt.version);
+        eprintln!("       Expected format: MAJOR.MINOR.PATCH (e.g., 7.7.0)");
+        std::process::exit(1);
+    }
+
+    let major = parse_version_component(version_parts[0])
+        .unwrap_or_else(|e| {
+            eprintln!("ERROR: {}", e);
+            eprintln!("       Full version string: '{}'", occt.version);
+            std::process::exit(1);
+        });
+
+    let minor = version_parts.get(1)
+        .map(|v| parse_version_component(v))
+        .transpose()
+        .unwrap_or_else(|e| {
+            eprintln!("ERROR: {}", e);
+            eprintln!("       Full version string: '{}'", occt.version);
+            std::process::exit(1);
+        })
+        .unwrap_or(0);
+
+    if major < 7 || (major == 7 && minor < 7) {
+        eprintln!("ERROR: OCCT version {}.{} found, but >= 7.7 required", major, minor);
+        std::process::exit(1);
+    }
+
+    println!("cargo:rerun-if-changed=src/ffi.rs");
+    println!("cargo:rerun-if-changed=cpp/wrapper.hpp");
+    println!("cargo:rerun-if-changed=cpp/wrapper.cpp");
+
+    // Build C++ bridge
+    cxx_build::bridge("src/ffi.rs")
+        .file("cpp/wrapper.cpp")
+        .flag_if_supported("-std=c++17")
+        .includes(&occt.include_paths)
+        .compile("occt_oracle_bridge");
+
+    // Link OCCT libraries required for geometric operations
+    // TKernel: Core data structures, TKMath: Mathematical utilities
+    // TKG3d: 3D geometry primitives, TKBRep: Boundary representation
+    // TKTopAlgo: Topological algorithms, TKPrim: Primitive shapes
+    // TKBO: Boolean operations, TKGeomBase/TKGeomAlgo: Geometric computations
+    for lib in &[
+        "TKernel", "TKMath", "TKG3d", "TKBRep", "TKTopAlgo",
+        "TKPrim", "TKBO", "TKGeomBase", "TKGeomAlgo",
+    ] {
+        println!("cargo:rustc-link-lib={}", lib);
+    }
+
+    for path in &occt.link_paths {
+        println!("cargo:rustc-link-search=native={}", path.display());
+    }
+}
+
```

```diff
--- /dev/null
+++ b/crates/occt_oracle/src/lib.rs
@@ -0,0 +1,12 @@
+//! FFI wrapper for Open CASCADE Technology (OCCT) geometric kernel.
+//!
+//! This crate provides safe Rust bindings to OCCT using cxx for type-safe FFI.
+//! All OCCT objects are wrapped in UniquePtr for automatic memory management.
+
+mod ffi;
+mod types;
+
+pub use types::{Shape, Point3D, OcctError};
+
+#[cfg(test)]
+mod tests;
```

```diff
--- /dev/null
+++ b/crates/occt_oracle/src/ffi.rs
@@ -0,0 +1,32 @@
+#[cxx::bridge]
+pub mod ffi {
+    unsafe extern "C++" {
+        include!("occt_oracle/cpp/wrapper.hpp");
+
+        // Core types
+        type Shape;
+        type Point3D;
+
+        // Shape constructors
+        fn make_null_shape() -> Result<UniquePtr<Shape>>;
+
+        // Shape properties
+        fn is_null(shape: &Shape) -> bool;
+        fn shape_type(shape: &Shape) -> u8;
+
+        // Point3D constructor
+        fn make_point(x: f64, y: f64, z: f64) -> Result<UniquePtr<Point3D>>;
+        fn point_x(pt: &Point3D) -> f64;
+        fn point_y(pt: &Point3D) -> f64;
+        fn point_z(pt: &Point3D) -> f64;
+    }
+}
+
+pub use ffi::{
+    make_null_shape,
+    is_null,
+    shape_type,
+    make_point,
+    point_x,
+    point_y,
+    point_z,
+};
```

```diff
--- /dev/null
+++ b/crates/occt_oracle/src/types.rs
@@ -0,0 +1,77 @@
+use cxx::UniquePtr;
+use std::fmt;
+
+/// Error type for OCCT operations
+#[derive(Debug, Clone)]
+pub enum OcctError {
+    InvalidOperation(String),
+    NullShape,
+    FfiError(String),
+}
+
+impl fmt::Display for OcctError {
+    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
+        match self {
+            OcctError::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
+            OcctError::NullShape => write!(f, "Null shape encountered"),
+            OcctError::FfiError(msg) => write!(f, "FFI error: {}", msg),
+        }
+    }
+}
+
+impl std::error::Error for OcctError {}
+
+/// Wrapper for OCCT TopoDS_Shape with automatic memory management
+pub struct Shape {
+    inner: UniquePtr<crate::ffi::ffi::Shape>,
+}
+
+impl Shape {
+    pub fn null() -> Result<Self, OcctError> {
+        let inner = crate::ffi::make_null_shape()
+            .map_err(|e| OcctError::FfiError(e.to_string()))?;
+        Ok(Shape { inner })
+    }
+
+    pub fn is_null(&self) -> bool {
+        crate::ffi::is_null(&self.inner)
+    }
+
+    pub fn shape_type(&self) -> ShapeType {
+        let type_code = crate::ffi::shape_type(&self.inner);
+        ShapeType::from_code(type_code)
+    }
+}
+
+#[derive(Debug, Clone, Copy, PartialEq, Eq)]
+pub enum ShapeType {
+    Compound,
+    CompSolid,
+    Solid,
+    Shell,
+    Face,
+    Wire,
+    Edge,
+    Vertex,
+    Shape,
+}
+
+impl ShapeType {
+    fn from_code(code: u8) -> Self {
+        match code {
+            0 => ShapeType::Compound,
+            1 => ShapeType::CompSolid,
+            2 => ShapeType::Solid,
+            3 => ShapeType::Shell,
+            4 => ShapeType::Face,
+            5 => ShapeType::Wire,
+            6 => ShapeType::Edge,
+            7 => ShapeType::Vertex,
+            // OCCT TopoDS_ShapeEnum values map 0-7, invalid/null shapes get generic type
+            _ => ShapeType::Shape,
+        }
+    }
+}
+
+/// 3D point wrapper
+pub struct Point3D {
+    inner: UniquePtr<crate::ffi::ffi::Point3D>,
+}
+
+impl Point3D {
+    pub fn new(x: f64, y: f64, z: f64) -> Result<Self, OcctError> {
+        let inner = crate::ffi::make_point(x, y, z)
+            .map_err(|e| OcctError::FfiError(e.to_string()))?;
+        Ok(Point3D { inner })
+    }
+
+    pub fn x(&self) -> f64 {
+        crate::ffi::point_x(&self.inner)
+    }
+
+    pub fn y(&self) -> f64 {
+        crate::ffi::point_y(&self.inner)
+    }
+
+    pub fn z(&self) -> f64 {
+        crate::ffi::point_z(&self.inner)
+    }
+
+    pub fn origin() -> Result<Self, OcctError> {
+        Self::new(0.0, 0.0, 0.0)
+    }
+}
+```

```diff
--- /dev/null
+++ b/crates/occt_oracle/cpp/wrapper.hpp
@@ -0,0 +1,39 @@
+#pragma once
+
+#include <memory>
+#include <TopoDS_Shape.hxx>
+#include <gp_Pnt.hxx>
+#include <BRep_Builder.hxx>
+#include <TopoDS_Compound.hxx>
+
+namespace occt_oracle {
+
+// Forward declarations for cxx
+struct Shape {
+    TopoDS_Shape inner;
+
+    // Default constructor creates null shape per OCCT convention
+    Shape() : inner() {}
+    explicit Shape(const TopoDS_Shape& s) : inner(s) {}
+};
+
+struct Point3D {
+    gp_Pnt inner;
+
+    Point3D(double x, double y, double z) : inner(x, y, z) {}
+    explicit Point3D(const gp_Pnt& p) : inner(p) {}
+};
+
+// Shape operations
+std::unique_ptr<Shape> make_null_shape();
+bool is_null(const Shape& shape);
+uint8_t shape_type(const Shape& shape);
+
+// Point3D operations
+std::unique_ptr<Point3D> make_point(double x, double y, double z);
+double point_x(const Point3D& pt);
+double point_y(const Point3D& pt);
+double point_z(const Point3D& pt);
+
+} // namespace occt_oracle
+
+
```

```diff
--- /dev/null
+++ b/crates/occt_oracle/cpp/wrapper.cpp
@@ -0,0 +1,66 @@
+#include "occt_oracle/cpp/wrapper.hpp"
+#include <TopoDS.hxx>
+#include <stdexcept>
+#include <string>
+
+namespace occt_oracle {
+
+std::unique_ptr<Shape> make_null_shape()
+try {
+    return std::make_unique<Shape>();
+} catch (const std::exception& e) {
+    throw std::runtime_error(std::string("Failed to create null shape: ") + e.what());
+} catch (...) {
+    // Catch all exceptions to prevent unwinding across FFI boundary (undefined behavior)
+    throw std::runtime_error("Failed to create null shape: unknown exception");
+}
+
+bool is_null(const Shape& shape)
+try {
+    return shape.inner.IsNull();
+} catch (const std::exception& e) {
+    throw std::runtime_error(std::string("Failed to check if shape is null: ") + e.what());
+} catch (...) {
+    // Catch all exceptions to prevent unwinding across FFI boundary (undefined behavior)
+    throw std::runtime_error("Failed to check if shape is null: unknown exception");
+}
+
+uint8_t shape_type(const Shape& shape)
+try {
+    if (shape.inner.IsNull()) {
+        // Null shapes have no valid type, return out-of-range value that maps to ShapeType::Shape in Rust
+        return 8;
+    }
+    return static_cast<uint8_t>(shape.inner.ShapeType());
+} catch (const std::exception& e) {
+    throw std::runtime_error(std::string("Failed to get shape type: ") + e.what());
+} catch (...) {
+    // Catch all exceptions to prevent unwinding across FFI boundary (undefined behavior)
+    throw std::runtime_error("Failed to get shape type: unknown exception");
+}
+
+std::unique_ptr<Point3D> make_point(double x, double y, double z)
+try {
+    return std::make_unique<Point3D>(x, y, z);
+} catch (const std::exception& e) {
+    throw std::runtime_error(std::string("Failed to create point: ") + e.what());
+} catch (...) {
+    // Catch all exceptions to prevent unwinding across FFI boundary (undefined behavior)
+    throw std::runtime_error("Failed to create point: unknown exception");
+}
+
+double point_x(const Point3D& pt)
+try {
+    return pt.inner.X();
+} catch (const std::exception& e) {
+    throw std::runtime_error(std::string("Failed to get point X coordinate: ") + e.what());
+} catch (...) {
+    // Catch all exceptions to prevent unwinding across FFI boundary (undefined behavior)
+    throw std::runtime_error("Failed to get point X coordinate: unknown exception");
+}
+
+double point_y(const Point3D& pt)
+try {
+    return pt.inner.Y();
+} catch (const std::exception& e) {
+    throw std::runtime_error(std::string("Failed to get point Y coordinate: ") + e.what());
+} catch (...) {
+    // Catch all exceptions to prevent unwinding across FFI boundary (undefined behavior)
+    throw std::runtime_error("Failed to get point Y coordinate: unknown exception");
+}
+
+double point_z(const Point3D& pt)
+try {
+    return pt.inner.Z();
+} catch (const std::exception& e) {
+    throw std::runtime_error(std::string("Failed to get point Z coordinate: ") + e.what());
+} catch (...) {
+    // Catch all exceptions to prevent unwinding across FFI boundary (undefined behavior)
+    throw std::runtime_error("Failed to get point Z coordinate: unknown exception");
+}
+
+} // namespace occt_oracle
+
+
```

```diff
--- /dev/null
+++ b/crates/occt_oracle/tests/build_test.rs
@@ -0,0 +1,31 @@
+use occt_oracle::{Shape, Point3D};
+
+#[test]
+fn test_create_null_shape() {
+    let shape = Shape::null().expect("Failed to create null shape");
+    assert!(shape.is_null());
+}
+
+#[test]
+fn test_multiple_shapes_no_leak() {
+    for _ in 0..100 {
+        let shape = Shape::null().expect("Failed to create null shape");
+        assert!(shape.is_null());
+        drop(shape);
+    }
+}
+
+#[test]
+fn test_point3d_creation() {
+    let pt = Point3D::new(1.0, 2.0, 3.0).expect("Failed to create point");
+    assert_eq!(pt.x(), 1.0);
+    assert_eq!(pt.y(), 2.0);
+    assert_eq!(pt.z(), 3.0);
+}
+
+#[test]
+fn test_point3d_origin() {
+    let origin = Point3D::origin().expect("Failed to create origin point");
+    assert_eq!(origin.x(), 0.0);
+    assert_eq!(origin.y(), 0.0);
+    assert_eq!(origin.z(), 0.0);
+}
```

---

### Milestone 2: Primitive and Boolean Operations

**Files**:
- `crates/occt_oracle/src/primitives.rs`
- `crates/occt_oracle/src/boolean.rs`
- `crates/occt_oracle/cpp/primitives.cpp`
- `crates/occt_oracle/cpp/boolean.cpp`

**Flags**: `needs-rationale` (operation choices)

**Requirements**:
- Box primitive: create from origin + dimensions
- Sphere primitive: create from center + radius
- Cylinder primitive: create from axis + radius + height
- Boolean union: combine two shapes
- Boolean subtract: cut shape B from shape A
- Boolean intersect: common volume of two shapes
- All operations return Result<Shape, OcctError>

**Acceptance Criteria**:
- `create_box(1.0, 1.0, 1.0)` returns valid shape with volume ~1.0
- `create_sphere(Point::origin(), 1.0)` returns valid shape with volume ~4.19
- `boolean_union(box, sphere)` returns combined shape
- `boolean_subtract(box, sphere)` returns difference shape
- `boolean_intersect(box, sphere)` returns intersection shape
- Invalid inputs (negative dimensions) return Err

**Tests**:
- **Test files**: `crates/occt_oracle/tests/primitives_test.rs`, `crates/occt_oracle/tests/boolean_test.rs`
- **Test type**: property-based + example-based (user-specified: both)
- **Backing**: user-specified
- **Scenarios**:
  - Normal: Box-box union, sphere-sphere intersect
  - Edge: Tangent spheres (kiss at one point), coincident faces
  - Error: Zero-dimension box, negative radius sphere

**Code Intent**:
- `primitives.rs`: Functions create_box, create_sphere, create_cylinder calling C++ via cxx
- `boolean.rs`: Functions boolean_union, boolean_subtract, boolean_intersect
- `primitives.cpp`: Wrap BRepPrimAPI_MakeBox, BRepPrimAPI_MakeSphere, BRepPrimAPI_MakeCylinder
- `boolean.cpp`: Wrap BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut, BRepAlgoAPI_Common
- Error handling: Catch OCCT exceptions, convert to Rust Result

---

### Milestone 3: Comparison Metrics

**Files**:
- `crates/kernel_harness/Cargo.toml`
- `crates/kernel_harness/src/lib.rs`
- `crates/kernel_harness/src/metrics.rs`
- `crates/kernel_harness/src/comparison.rs`
- `crates/kernel_harness/src/oracle.rs`

**Flags**: `complex-algorithm` (Hausdorff distance)

**Requirements**:
- Volume comparison: compute relative error between two shapes
- Topology comparison: compare face, edge, vertex counts
- Bounding box comparison: compare axis-aligned bounding boxes
- Hausdorff distance: maximum point-to-surface distance (optional, can defer)
- GeometricOracle trait: abstract interface for reference implementations
- OcctOracle: implements GeometricOracle using occt_oracle crate
- ComparisonResult: struct with pass/fail and detailed diagnostics

**Acceptance Criteria**:
- `volume_diff(a, b)` returns relative error as f64
- `topology_matches(a, b)` returns true if face/edge/vertex counts match
- `ComparisonResult::is_pass()` returns true when all metrics within tolerance
- `ComparisonResult::diagnostics()` returns human-readable diff on failure

**Tests**:
- **Test files**: `crates/kernel_harness/tests/metrics_test.rs`
- **Test type**: property-based + example-based
- **Backing**: user-specified
- **Scenarios**:
  - Normal: Identical shapes compare as equal
  - Edge: Shapes with same volume but different topology
  - Error: NaN in shape data handled gracefully

**Code Intent**:
- `Cargo.toml`: Depend on occt_oracle, rkyv for serialization
- `metrics.rs`: volume_diff, topology_compare, bbox_compare functions
- `comparison.rs`: ComparisonResult struct, ComparisonConfig (tolerances)
- `oracle.rs`: GeometricOracle trait, OcctOracle struct implementing it
- `lib.rs`: Re-export public API

---

### Milestone 4: Test DSL Macro

**Files**:
- `crates/kernel_harness/src/dsl.rs`
- `crates/kernel_harness/src/runner.rs`
- `crates/kernel_harness/src/testcase.rs`

**Flags**: `complex-algorithm` (macro implementation)

**Requirements**:
- `oracle_test!` macro for declarative test definition
- TestCase struct representing a single comparison test
- HarnessRunner: executes test cases, collects results
- Support for primitive inputs, boolean operations
- Configurable comparison metrics per test

**Acceptance Criteria**:
- DSL compiles: `oracle_test! { name: my_test, op: Union, inputs: [box(1,1,1), sphere(0.5,0.5,0.5,0.5)], compare: [Volume(1e-6)] }`
- HarnessRunner executes test and returns ComparisonResult
- Multiple tests can be defined and run in sequence
- Test failures include diagnostic information

**Tests**:
- **Test files**: `crates/kernel_harness/tests/dsl_test.rs`
- **Test type**: integration
- **Backing**: default-derived
- **Scenarios**:
  - Normal: Simple box-box union test passes
  - Edge: Empty inputs handled
  - Error: Invalid operation combination rejected at compile time

**Code Intent**:
- `dsl.rs`: oracle_test! macro parsing inputs, operations, comparison config
- `testcase.rs`: TestCase, Operation, Primitive enums
- `runner.rs`: HarnessRunner struct, run_test(), run_all() methods
- Macro generates #[test] function that calls HarnessRunner

---

### Milestone 5: Test Case Generation and STEP Import

**Files**:
- `crates/kernel_harness/src/generators.rs`
- `crates/kernel_harness/src/step.rs`
- `crates/kernel_harness/tests/fixtures/` (directory)

**Flags**: `conformance`

**Requirements**:
- Random primitive generator (proptest integration)
- Random boolean operation generator
- STEP file import for real-world test cases
- Fixture loading from tests/fixtures/

**Acceptance Criteria**:
- `random_box(&mut rng)` returns valid box with dimensions in [0.1, 10.0]
- `random_boolean(&mut rng, a, b)` returns random operation on shapes
- `load_step("fixtures/part.stp")` returns Shape
- proptest can generate thousands of random test cases

**Tests**:
- **Test files**: `crates/kernel_harness/tests/generation_test.rs`
- **Test type**: property-based
- **Backing**: user-specified
- **Scenarios**:
  - Normal: 1000 random primitives all valid
  - Edge: Generated shapes with extreme aspect ratios
  - Error: Invalid STEP file returns Err

**Code Intent**:
- `generators.rs`: Proptest Arbitrary impls for Primitive, Operation
- `step.rs`: load_step() using OCCT's STEPControl_Reader
- `fixtures/`: Sample STEP files for integration tests

---

### Milestone 6: CI Integration and Documentation

**Delegated to**: @agent-technical-writer (mode: post-implementation)

**Source**: `## Invisible Knowledge` section of this plan

**Files**:
- `crates/occt_oracle/README.md`
- `crates/kernel_harness/README.md`
- `crates/occt_oracle/CLAUDE.md`
- `crates/kernel_harness/CLAUDE.md`
- `.github/workflows/oracle-tests.yml`
- `docker/Dockerfile.occt-test`
- `docs/SETUP.md`

**Requirements**:
- README.md for each crate with usage examples
- CLAUDE.md index for agent navigation
- GitHub Actions workflow running oracle tests
- Dockerfile with OCCT pre-installed
- Setup guide for local development

**Acceptance Criteria**:
- `cargo test -p kernel_harness` passes in CI
- Docker image builds successfully
- README examples are copy-pasteable and work
- CLAUDE.md follows tabular format

## Milestone Dependencies

```
M1 (Core Types) ──┬──> M2 (Operations)
                  │
                  └──> M3 (Comparison) ──> M4 (DSL) ──> M5 (Generation)
                                                              │
                                                              ▼
                                                       M6 (Documentation)
```

**Parallel opportunities**:
- M2 and M3 can start in parallel after M1 completes
- M4 requires both M2 and M3
- M5 requires M4
- M6 runs after all implementation milestones

**Wave analysis**:
- Wave 1: M1
- Wave 2: M2 + M3 (parallel)
- Wave 3: M4
- Wave 4: M5
- Wave 5: M6
