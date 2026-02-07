//! 2D sketch geometry plugin.
//!
//! Provides entity and constraint types for 2D sketching (points, lines,
//! circles, arcs) with squared formulations for smooth Jacobians.

pub mod entities;
pub mod constraints;
pub mod builder;

pub use entities::{Point2D, LineSegment2D, Circle2D, Arc2D, InfiniteLine2D};
pub use constraints::{
    DistancePtPt, Coincident, Fixed, Horizontal, Vertical,
    Parallel, Perpendicular, Angle, Midpoint, Symmetric,
    EqualLength, PointOnCircle, TangentLineCircle, TangentCircleCircle,
    DistancePtLine,
};
pub use builder::Sketch2DBuilder;
