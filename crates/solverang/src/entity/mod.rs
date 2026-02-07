//! Entity trait for the constraint system.
//!
//! An entity is a named group of parameters. The solver treats all entities
//! uniformly — it only cares about their [`ParamId`](crate::id::ParamId)s.
//! The geometry layer (if any) provides rich entity types that implement
//! this trait.
//!
//! # Implementing for a geometric kernel
//!
//! ```ignore
//! // A 2D point entity: two parameters (x, y).
//! struct Point2D {
//!     id: EntityId,
//!     x: ParamId,
//!     y: ParamId,
//!     params: [ParamId; 2],
//! }
//!
//! impl Entity for Point2D {
//!     fn id(&self) -> EntityId { self.id }
//!     fn params(&self) -> &[ParamId] { &self.params }
//!     fn name(&self) -> &str { "Point2D" }
//! }
//! ```

use crate::id::{EntityId, ParamId};

/// A solvable entity: a named group of parameters.
///
/// Entities represent geometric objects (points, circles, curves), physical
/// objects (rigid bodies, springs), or any other domain object with solvable
/// parameters. The solver treats all entities uniformly as parameter groups.
///
/// # What's NOT on this trait
///
/// - No `EntityKind` — that's for the geometry layer to define.
/// - No `evaluate()` — the solver doesn't evaluate geometry.
/// - No `dof()` — DOF is computed by the solver from the constraint graph.
/// - No dimension generic `<const D: usize>` — the solver is dimension-agnostic.
pub trait Entity: Send + Sync {
    /// Unique identifier for this entity.
    fn id(&self) -> EntityId;

    /// The parameter IDs owned by this entity.
    ///
    /// For a 2D point, this returns `[x, y]`.
    /// For a circle, this might return `[cx, cy, r]`.
    /// For a NURBS curve, this returns all control point coordinates and weights.
    fn params(&self) -> &[ParamId];

    /// Human-readable name for diagnostics and debugging.
    fn name(&self) -> &str;
}
