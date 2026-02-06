//! Entity evaluators for geometric primitives.
//!
//! These modules provide evaluation functions for each entity type: position at
//! parameter t, tangent vectors, curvature, bounding boxes, etc.

pub mod point;
pub mod line;
pub mod circle;
pub mod arc;
pub mod ellipse;
pub mod bezier;
pub mod plane;
pub mod sphere;
pub mod cylinder;
pub mod cone;
pub mod torus;

use crate::geometry::entity::EntityKind;

/// Evaluate a position on an entity at parameter t.
/// For points: returns the point itself (t ignored).
/// For curves: returns position at t ∈ [0, 1].
/// For surfaces: not supported here (use evaluate_surface).
/// `entity_params` is the slice of parameters for this entity.
pub fn evaluate_at(kind: EntityKind, entity_params: &[f64], t: f64) -> Option<Vec<f64>> {
    match kind {
        EntityKind::Point2D => Some(entity_params.to_vec()),
        EntityKind::Line2D => Some(line::evaluate(entity_params, t)),
        EntityKind::Circle2D => Some(circle::evaluate_2d(entity_params, t)),
        EntityKind::Arc2D => Some(arc::evaluate_2d(entity_params, t)),
        EntityKind::Ellipse2D => Some(ellipse::evaluate_2d(entity_params, t)),
        EntityKind::CubicBezier2D => Some(bezier::evaluate_cubic_2d(entity_params, t)),
        EntityKind::QuadBezier2D => Some(bezier::evaluate_quad_2d(entity_params, t)),
        EntityKind::Point3D => Some(entity_params.to_vec()),
        EntityKind::Line3D => Some(line::evaluate(entity_params, t)),
        EntityKind::Circle3D => Some(circle::evaluate_3d(entity_params, t)),
        EntityKind::CubicBezier3D => Some(bezier::evaluate_cubic_3d(entity_params, t)),
        _ => None, // Surfaces, Scalar, etc.
    }
}

/// Evaluate tangent vector at parameter t.
pub fn tangent_at(kind: EntityKind, entity_params: &[f64], t: f64) -> Option<Vec<f64>> {
    match kind {
        EntityKind::Line2D | EntityKind::Line3D => Some(line::tangent(entity_params)),
        EntityKind::Circle2D => Some(circle::tangent_2d(entity_params, t)),
        EntityKind::Arc2D => Some(arc::tangent_2d(entity_params, t)),
        EntityKind::Ellipse2D => Some(ellipse::tangent_2d(entity_params, t)),
        EntityKind::CubicBezier2D => Some(bezier::tangent_cubic_2d(entity_params, t)),
        EntityKind::QuadBezier2D => Some(bezier::tangent_quad_2d(entity_params, t)),
        EntityKind::CubicBezier3D => Some(bezier::tangent_cubic_3d(entity_params, t)),
        _ => None,
    }
}
