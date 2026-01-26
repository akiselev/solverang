//! Geometric constraint library for 2D and 3D constraint solving.
//!
//! This module provides a domain-specific layer on top of the generic solver core,
//! offering geometric primitives and constraint types commonly used in CAD applications.
//!
//! # Overview
//!
//! The geometry module provides:
//!
//! - **Primitives**: [`Point`], [`Vector`], [`Line`], [`Circle`], [`Sphere`]
//! - **Constraints**: Distance, coincident, parallel, perpendicular, tangent, etc.
//! - **Constraint System**: A builder-based API for constructing solvable systems
//!
//! # Dimension-Agnostic Design
//!
//! Most types use const generics to support both 2D and 3D:
//!
//! ```rust
//! use solverang::geometry::{Point, Point2D, Point3D};
//!
//! // Using type aliases
//! let p2d = Point2D::new(1.0, 2.0);
//! let p3d = Point3D::new(1.0, 2.0, 3.0);
//!
//! // Or using const generics directly
//! let p: Point<2> = Point::from_coords([1.0, 2.0]);
//! ```
//!
//! # Building a Constraint System
//!
//! ```rust
//! use solverang::geometry::{ConstraintSystemBuilder, Point2D};
//! use solverang::{LMSolver, LMConfig, SolveResult};
//!
//! // Create a triangle with fixed side lengths
//! let mut system = ConstraintSystemBuilder::<2>::new()
//!     .name("EquilateralTriangle")
//!     .point(Point2D::new(0.0, 0.0))       // p0 - will be fixed
//!     .point(Point2D::new(10.0, 0.0))      // p1
//!     .point(Point2D::new(5.0, 5.0))       // p2 - initial guess
//!     .fix(0)                              // Fix p0 at origin
//!     .horizontal(0, 1)                    // p0-p1 is horizontal
//!     .distance(0, 1, 10.0)                // |p0-p1| = 10
//!     .distance(1, 2, 10.0)                // |p1-p2| = 10
//!     .distance(2, 0, 10.0)                // |p2-p0| = 10
//!     .build();
//!
//! // Solve using Levenberg-Marquardt
//! let solver = LMSolver::new(LMConfig::default());
//! let initial = system.current_values();
//! let result = solver.solve(&system, &initial);
//!
//! if let SolveResult::Converged { solution, .. } = result {
//!     system.set_values(&solution);
//!     // p2 is now at the apex of the equilateral triangle
//! }
//! ```
//!
//! # Available Constraints
//!
//! | Constraint | 2D | 3D | Description |
//! |------------|----|----|-------------|
//! | `DistanceConstraint` | Yes | Yes | Point-to-point distance |
//! | `CoincidentConstraint` | Yes | Yes | Points at same location |
//! | `FixedConstraint` | Yes | Yes | Point at fixed position |
//! | `HorizontalConstraint` | Yes | - | Same y-coordinate |
//! | `VerticalConstraint` | Yes | - | Same x-coordinate |
//! | `AngleConstraint` | Yes | - | Line angle from horizontal |
//! | `ParallelConstraint` | Yes | Yes | Parallel lines |
//! | `PerpendicularConstraint` | Yes | Yes | Perpendicular lines |
//! | `MidpointConstraint` | Yes | Yes | Point at line midpoint |
//! | `PointOnLineConstraint` | Yes | Yes | Point lies on line |
//! | `PointOnCircleConstraint` | Yes | Yes | Point on circle/sphere |
//! | `LineTangentConstraint` | Yes | - | Line tangent to circle |
//! | `CircleTangentConstraint` | Yes | Yes | Circle/sphere tangency |
//! | `SymmetricConstraint` | Yes | Yes | Point symmetry |
//! | `SymmetricAboutLineConstraint` | Yes | - | Line symmetry |
//! | `CollinearConstraint` | Yes | Yes | Collinear segments |
//! | `EqualLengthConstraint` | Yes | Yes | Equal line lengths |

pub mod point;
pub mod vector;
pub mod line;
pub mod circle;
pub mod constraints;
pub mod system;
pub mod builder;

// Re-export main types
pub use point::{Point, Point2D, Point3D, MIN_EPSILON};
pub use vector::{Vector, Vector2D, Vector3D};
pub use line::{Line, Line2D, Line3D};
pub use circle::{Circle, Sphere, TangentType};
pub use system::ConstraintSystem;
pub use builder::ConstraintSystemBuilder;

// Re-export constraint trait
pub use constraints::GeometricConstraint;
