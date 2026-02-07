//! Constraint trait for the constraint system.
//!
//! A constraint produces residuals (equations that should be zero when satisfied)
//! and Jacobians (partial derivatives of residuals with respect to parameters).
//! The solver uses these to iteratively find parameter values that satisfy all
//! constraints simultaneously.
//!
//! # Key Design Decisions
//!
//! - **Jacobian returns `(row, ParamId, value)`, not `(row, col, value)`.** The
//!   constraint doesn't need to know the column ordering. The solver's
//!   [`SolverMapping`](crate::param::SolverMapping) handles it.
//!
//! - **Constraints read from [`ParamStore`](crate::param::ParamStore)**, not from
//!   point arrays. This allows constraints over any combination of parameters.
//!
//! - **No geometry types** — the solver never sees `Point2D`, `Circle`, etc.

use crate::id::{ConstraintId, EntityId, ParamId};
use crate::param::ParamStore;

/// A constraint: a set of equations over parameters.
///
/// Constraints produce residuals (which should be zero when satisfied) and
/// Jacobians (partial derivatives of residuals w.r.t. parameters). The solver
/// uses these to iteratively find parameter values that satisfy all constraints.
///
/// # What's NOT on this trait
///
/// - No `<const D: usize>` — constraints work in any dimension.
/// - No `points: &[Point<D>]` parameter — constraints read from `ParamStore`.
/// - No geometry types — the solver never sees `Point2D`, `Circle`, etc.
/// - Jacobian returns `ParamId`, not column indices — the system does the mapping.
pub trait Constraint: Send + Sync {
    /// Unique identifier for this constraint.
    fn id(&self) -> ConstraintId;

    /// Human-readable name for diagnostics and debugging.
    fn name(&self) -> &str;

    /// Which entities this constraint binds.
    fn entity_ids(&self) -> &[EntityId];

    /// Which parameters this constraint depends on (for graph building).
    fn param_ids(&self) -> &[ParamId];

    /// Number of scalar equations this constraint produces.
    fn equation_count(&self) -> usize;

    /// Evaluate residuals. Each element should be zero when satisfied.
    fn residuals(&self, store: &ParamStore) -> Vec<f64>;

    /// Sparse Jacobian: `(equation_row, param_id, partial_derivative)`.
    ///
    /// Only non-zero entries need to be returned. The system maps `ParamId` to
    /// column indices via [`SolverMapping`](crate::param::SolverMapping).
    fn jacobian(&self, store: &ParamStore) -> Vec<(usize, ParamId, f64)>;

    /// Weight for soft constraints (default 1.0).
    fn weight(&self) -> f64 {
        1.0
    }

    /// Is this a soft constraint that can be relaxed?
    fn is_soft(&self) -> bool {
        false
    }
}
