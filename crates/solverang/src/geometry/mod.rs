//! Geometric constraint library for 2D and 3D constraint solving (v2).
//!
//! This version uses a flat parameter store model where all geometric entities
//! (points, circles, lines, arcs, beziers, etc.) are represented as contiguous
//! ranges in a single parameter vector. This enables first-class support for
//! all geometric primitives, not just points.
//!
//! # Overview
//!
//! The v2 geometry module provides:
//!
//! - **Parameter Store**: Flat vector storage for all entity parameters
//! - **Entity Types**: [`EntityKind`] enum covering all 2D/3D primitives
//! - **Constraint Trait**: [`Constraint`] operating on the flat parameter vector
//! - **Constraint System**: [`ConstraintSystem`] wrapping ParameterStore + constraints
//!
//! # Key Design Changes from v1
//!
//! - No more `Point<D>` with const generic D
//! - All entities are parameter ranges in a flat `Vec<f64>`
//! - Constraints operate on `&[f64]` (full parameter vector), not `&[Point<D>]`
//! - Jacobian columns are direct parameter indices (no more PARAM_COL_BASE sentinel)
//! - Mixing 2D and 3D entities in one system is supported
//!
//! # Building a Constraint System
//!
//! ```rust
//! use solverang::geometry::{ConstraintSystem, EntityKind};
//! use solverang::{LMSolver, LMConfig, SolveResult};
//!
//! let mut system = ConstraintSystem::with_name("Example");
//!
//! // Add entities (returns typed handles)
//! let p1 = system.add_point_2d(0.0, 0.0);
//! let p2 = system.add_point_2d(10.0, 0.0);
//! let circle = system.add_circle_2d(5.0, 5.0, 3.0);
//!
//! // Fix first point
//! system.fix_entity(&p1);
//!
//! // Add constraints (constraint types TBD - being ported)
//! // system.add_constraint(Box::new(Distance::new(id, &p1, &p2, 10.0)));
//!
//! // Check degrees of freedom
//! println!("DOF: {}", system.degrees_of_freedom());
//!
//! // Solve using the Problem trait
//! let solver = LMSolver::new(LMConfig::default());
//! let initial = system.current_values();
//! let result = solver.solve(&system, &initial);
//!
//! if let SolveResult::Converged { solution, .. } = result {
//!     system.set_values(&solution);
//! }
//! ```
//!
//! # Entity Types
//!
//! All entity types are defined in [`EntityKind`]:
//!
//! ## 2D Primitives
//! - `Point2D` - [x, y] (2 params)
//! - `Line2D` - [x1, y1, x2, y2] (4 params)
//! - `Circle2D` - [cx, cy, r] (3 params)
//! - `Arc2D` - [cx, cy, r, start_angle, end_angle] (5 params)
//! - `Ellipse2D` - [cx, cy, rx, ry, rotation] (5 params)
//! - `CubicBezier2D` - [x0,y0, x1,y1, x2,y2, x3,y3] (8 params)
//! - And more...
//!
//! ## 3D Primitives
//! - `Point3D` - [x, y, z] (3 params)
//! - `Line3D` - [x1,y1,z1, x2,y2,z2] (6 params)
//! - `Circle3D` - [cx,cy,cz, nx,ny,nz, r] (7 params)
//! - `Sphere` - [cx,cy,cz, r] (4 params)
//! - `Cylinder`, `Cone`, `Torus`, `Plane`, and more...
//!
//! ## Auxiliary
//! - `Scalar` - [value] (1 param) - for auxiliary parameters like t, angle, etc.

pub mod params;
pub mod entity;
pub mod constraint;
pub mod constraints;
pub mod system;
pub mod builder;
pub mod entities;

// Re-export main types
pub use params::{EntityId, ConstraintId, ParamRange, EntityHandle, ParameterStore};
pub use entity::EntityKind;
pub use constraint::{Constraint, Nonlinearity, MIN_EPSILON};
pub use system::ConstraintSystem;
pub use builder::ConstraintSystemBuilder;
