//! Geometric constraint implementations for the v2 rearchitecture.
//!
//! Each constraint operates on a flat parameter vector and references entities
//! via ParamRange handles. Constraints are dimension-agnostic where possible.

// Basic point constraints
mod distance;
mod coincident;
mod fixed;
mod horizontal;
mod vertical;
mod midpoint;
mod symmetric;

// Line/angle constraints
mod parallel;
mod perpendicular;
mod collinear;
mod equal_length;
mod angle;
mod point_on_line;

// Circle/arc/tangent constraints
mod point_on_circle;
mod tangent;
mod equal_radius;
mod concentric;
mod arc_constraints;

// Curve/ellipse/meta constraints
mod point_on_ellipse;
mod bezier_continuity;
mod point_on_curve;
mod meta_constraints;

// Re-export all constraint types
pub use distance::DistanceConstraint;
pub use coincident::CoincidentConstraint;
pub use fixed::FixedConstraint;
pub use horizontal::HorizontalConstraint;
pub use vertical::VerticalConstraint;
pub use midpoint::MidpointConstraint;
pub use symmetric::{SymmetricConstraint, SymmetricAboutLineConstraint};
pub use parallel::ParallelConstraint;
pub use perpendicular::PerpendicularConstraint;
pub use collinear::CollinearConstraint;
pub use equal_length::EqualLengthConstraint;
pub use angle::AngleConstraint;
pub use point_on_line::PointOnLineConstraint;
pub use point_on_circle::PointOnCircleConstraint;
pub use point_on_ellipse::PointOnEllipseConstraint;
pub use tangent::{LineTangentCircleConstraint, CircleTangentConstraint};
pub use equal_radius::EqualRadiusConstraint;
pub use concentric::ConcentricConstraint;
pub use arc_constraints::{ArcEndpointConstraint, ArcSweepConstraint, PointOnArcConstraint};
pub use bezier_continuity::{G0ContinuityConstraint, G1ContinuityConstraint, G2ContinuityConstraint};
pub use point_on_curve::PointOnBezierConstraint;
pub use meta_constraints::{FixedParamConstraint, EqualParamConstraint, ParamRangeConstraint, RatioParamConstraint};
