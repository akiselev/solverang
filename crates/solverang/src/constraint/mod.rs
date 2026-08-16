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
//!   [`crate::param::SolverMapping`] handles it.
//!
//! - **Constraints read from [`ParamStore`]**, not from
//!   point arrays. This allows constraints over any combination of parameters.
//!
//! - **No geometry types** — the solver never sees `Point2D`, `Circle`, etc.
//!
//! - **Symbolic export is optional** — a constraint may emit the same residual
//!   equations through the dependency-neutral [`symbolic::SymbolicSink`]. Existing
//!   constraints and consumers remain purely numerical unless they opt in.

pub mod symbolic;

use crate::id::{ConstraintId, EntityId, ParamId};
use crate::param::ParamStore;
use symbolic::{SymbolicNode, SymbolicSink};

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
/// - No mandatory CAS dependency — optional symbolic residuals are emitted into
///   a caller-supplied sink.
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
    /// column indices via [`crate::param::SolverMapping`].
    fn jacobian(&self, store: &ParamStore) -> Vec<(usize, ParamId, f64)>;

    /// Emit this constraint's residual equations into a caller-owned symbolic
    /// representation.
    ///
    /// `None` is a supported outcome: the constraint either has no exact representation
    /// through this protocol or has not implemented one yet. Numerical solving remains
    /// unchanged. The default preserves source compatibility for all existing constraints.
    fn symbolic_residuals(&self, _sink: &mut dyn SymbolicSink) -> Option<Vec<SymbolicNode>> {
        None
    }

    /// Weight for soft constraints (default 1.0).
    ///
    /// Reserved: no solver currently consumes weights — soft-constraint
    /// support is planned but not yet implemented.
    fn weight(&self) -> f64 {
        1.0
    }

    /// Is this a soft constraint that can be relaxed?
    ///
    /// Reserved: no solver currently relaxes soft constraints — this flag is
    /// recorded but not yet acted on.
    fn is_soft(&self) -> bool {
        false
    }
}
