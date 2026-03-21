//! 3D sketch entities and constraints.
//!
//! This module provides geometric primitives and constraints for 3D sketch solving:
//!
//! - **Entities**: [`Point3D`], [`LineSegment3D`], [`Plane`], [`Axis3D`]
//! - **Constraints**: [`Distance3D`], [`Coincident3D`], [`Fixed3D`], [`PointOnPlane`],
//!   [`Coplanar`], [`Parallel3D`], [`Perpendicular3D`], [`Coaxial`]

pub mod constraints;
pub mod entities;

pub use constraints::{
    Coaxial, Coincident3D, Coplanar, Distance3D, Fixed3D, Parallel3D, Perpendicular3D, PointOnPlane,
};
pub use entities::{Axis3D, LineSegment3D, Plane, Point3D};
