//! 3D sketch entities and constraints.
//!
//! This module provides geometric primitives and constraints for 3D sketch solving:
//!
//! - **Entities**: [`Point3D`], [`LineSegment3D`], [`Plane`], [`Axis3D`]
//! - **Constraints**: [`Distance3D`], [`Coincident3D`], [`Fixed3D`], [`PointOnPlane`],
//!   [`Coplanar`], [`Parallel3D`], [`Perpendicular3D`], [`Coaxial`]

pub mod entities;
pub mod constraints;

pub use entities::{Point3D, LineSegment3D, Plane, Axis3D};
pub use constraints::{
    Distance3D, Coincident3D, Fixed3D, PointOnPlane, Coplanar,
    Parallel3D, Perpendicular3D, Coaxial,
};
