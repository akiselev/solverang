//! Geometric constraint implementations.
//!
//! This module provides constraint types for 2D and 3D geometric constraint solving.
//! All constraints implement the [`GeometricConstraint`] trait, which provides
//! residuals and Jacobian information needed by the solver.
//!
//! # Available Constraints
//!
//! | Constraint | Equations | Description |
//! |------------|-----------|-------------|
//! | [`DistanceConstraint`] | 1 | Point-to-point distance |
//! | [`CoincidentConstraint`] | D | Points must coincide |
//! | [`FixedConstraint`] | D | Point at fixed position |
//! | [`AngleConstraint`] | 1 | Angle between lines |
//! | [`ParallelConstraint`] | 1 (2D) / 2 (3D) | Parallel lines |
//! | [`PerpendicularConstraint`] | 1 (2D) / 1 (3D) | Perpendicular lines |
//! | [`MidpointConstraint`] | D | Point at midpoint |
//! | [`HorizontalConstraint`] | 1 | Same y-coordinate (2D) |
//! | [`VerticalConstraint`] | 1 | Same x-coordinate (2D) |
//! | [`PointOnLineConstraint`] | 1 (2D) / 2 (3D) | Point on line |
//! | [`PointOnCircleConstraint`] | 1 | Point on circle |
//! | [`TangentConstraint`] | 1 | Line/circle tangency |
//! | [`SymmetricConstraint`] | D | Point symmetry |
//! | [`CollinearConstraint`] | 2 (2D) / 4 (3D) | Collinear segments |
//! | [`EqualLengthConstraint`] | 1 | Equal line lengths |

mod angle;
mod coincident;
mod collinear;
mod distance;
mod equal_length;
mod fixed;
mod horizontal;
mod midpoint;
mod parallel;
mod perpendicular;
mod point_on_circle;
mod point_on_line;
mod symmetric;
mod tangent;
mod vertical;

pub use angle::AngleConstraint;
pub use coincident::CoincidentConstraint;
pub use collinear::CollinearConstraint;
pub use distance::DistanceConstraint;
pub use equal_length::EqualLengthConstraint;
pub use fixed::FixedConstraint;
pub use horizontal::HorizontalConstraint;
pub use midpoint::MidpointConstraint;
pub use parallel::ParallelConstraint;
pub use perpendicular::PerpendicularConstraint;
pub use point_on_circle::{PointOnCircleConstraint, PointOnCircleVariableRadiusConstraint};
pub use point_on_line::PointOnLineConstraint;
pub use symmetric::{SymmetricConstraint, SymmetricAboutLineConstraint};
pub use tangent::{CircleTangentConstraint, LineTangentConstraint};
pub use vertical::VerticalConstraint;

use crate::geometry::Point;

/// Trait for geometric constraints that can be solved numerically.
///
/// A geometric constraint defines relationships between points that must be satisfied.
/// Each constraint produces residual equations (which should be zero when satisfied)
/// and Jacobian entries (partial derivatives for the solver).
///
/// # Generic Parameter
///
/// * `D` - Dimension of the constraint (2 for 2D, 3 for 3D)
///
/// # Example
///
/// ```rust
/// use solverang::geometry::{Point2D, ConstraintSystem};
/// use solverang::geometry::constraints::DistanceConstraint;
///
/// let mut system = ConstraintSystem::<2>::new();
/// let p1 = system.add_point(Point2D::new(0.0, 0.0));
/// let p2 = system.add_point(Point2D::new(5.0, 0.0));
///
/// // Constrain distance between points to be 10 units
/// system.add_constraint(Box::new(DistanceConstraint::<2>::new(p1, p2, 10.0)));
/// ```
pub trait GeometricConstraint<const D: usize>: Send + Sync {
    /// Number of scalar equations this constraint generates.
    ///
    /// For example:
    /// - Distance constraint: 1 equation
    /// - Coincident constraint: D equations (one per dimension)
    /// - Parallel constraint in 2D: 1 equation
    fn equation_count(&self) -> usize;

    /// Evaluate the residual vector.
    ///
    /// Returns residuals that should be zero when the constraint is satisfied.
    /// The length of the returned vector must equal `equation_count()`.
    ///
    /// # Arguments
    ///
    /// * `points` - All points in the constraint system, indexed by variable index
    fn residuals(&self, points: &[Point<D>]) -> Vec<f64>;

    /// Compute sparse Jacobian entries: (row, col, value).
    ///
    /// Row indices are 0-indexed relative to this constraint's equations.
    /// Column indices are absolute variable indices (point_index * D + coordinate).
    ///
    /// # Arguments
    ///
    /// * `points` - All points in the constraint system
    fn jacobian(&self, points: &[Point<D>]) -> Vec<(usize, usize, f64)>;

    /// Get the indices of variables this constraint depends on.
    ///
    /// Used for problem decomposition and graph analysis.
    fn variable_indices(&self) -> Vec<usize>;

    /// Weight for this constraint in weighted least squares.
    ///
    /// Higher weights mean stronger enforcement. Default is 1.0.
    fn weight(&self) -> f64 {
        1.0
    }

    /// Whether this is a soft constraint.
    ///
    /// Hard constraints must be exactly satisfied.
    /// Soft constraints contribute to an objective function and may be relaxed.
    fn is_soft(&self) -> bool {
        false
    }

    /// Human-readable name for debugging.
    fn name(&self) -> &'static str;
}

/// Helper to compute variable column index from point index and coordinate.
#[inline]
pub(crate) fn var_col<const D: usize>(point_idx: usize, coord: usize) -> usize {
    point_idx * D + coord
}

/// Get a point safely, returning origin if index is out of bounds.
#[inline]
pub(crate) fn get_point<const D: usize>(points: &[Point<D>], idx: usize) -> Point<D> {
    points.get(idx).copied().unwrap_or_default()
}
