//! Assembly entities and constraints for rigid body systems.
//!
//! This module provides types for modeling assemblies of rigid bodies connected
//! by geometric constraints:
//!
//! - **Entities**: [`RigidBody`] (position + quaternion orientation)
//! - **Internal constraints**: [`UnitQuaternion`] (normalization)
//! - **Assembly constraints**: [`Mate`], [`CoaxialAssembly`], [`Insert`], [`Gear`]

pub mod entities;
pub mod constraints;

pub use entities::{RigidBody, UnitQuaternion};
pub use constraints::{Mate, CoaxialAssembly, Insert, Gear};
