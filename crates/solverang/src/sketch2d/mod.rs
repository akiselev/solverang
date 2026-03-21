//! 2D sketch geometry plugin.
//!
//! Provides entity and constraint types for 2D sketching (points, lines,
//! circles, arcs) with squared formulations for smooth Jacobians.

pub mod builder;
pub mod constraints;
pub mod entities;

pub use builder::Sketch2DBuilder;
pub use constraints::{
    Angle, Coincident, Collinear, DistancePtLine, DistancePtPt, EqualLength, EqualRadius, Fixed,
    Horizontal, Midpoint, Parallel, Perpendicular, PointOnCircle, Symmetric, TangentCircleCircle,
    TangentLineCircle, Vertical,
};
pub use entities::{Arc2D, Circle2D, Ellipse2D, InfiniteLine2D, LineSegment2D, Point2D, Spline2D};
