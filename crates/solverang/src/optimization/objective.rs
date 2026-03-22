//! Objective function traits for optimization.
//!
//! An objective is a scalar-valued function to be minimized. It provides
//! its value and gradient (sparse, ParamId-based). An optional extension
//! trait [`ObjectiveHessian`] provides second derivatives for methods
//! that can exploit them (SQP, Newton).
//!
//! BFGS/L-BFGS only needs [`Objective`] (gradient). When [`ObjectiveHessian`]
//! is not implemented, solvers use quasi-Newton approximation automatically.

use crate::id::{EntityId, ParamId};
use crate::param::ParamStore;

use super::ObjectiveId;

/// A scalar objective function to be minimized.
///
/// Parallels [`crate::constraint::Constraint`] but returns a scalar value
/// and a gradient vector instead of residual vector and Jacobian matrix.
///
/// # Gradient Convention
///
/// `gradient()` returns sparse entries `(ParamId, df/dp)`. Only non-zero
/// entries should be returned — this affects sparsity structure used by
/// the solver.
pub trait Objective: Send + Sync {
    /// Unique identifier for this objective.
    fn id(&self) -> ObjectiveId;

    /// Human-readable name for diagnostics.
    fn name(&self) -> &str;

    /// Which entities this objective depends on (for graph building).
    fn entity_ids(&self) -> &[EntityId] {
        &[]
    }

    /// Which parameters this objective depends on.
    fn param_ids(&self) -> &[ParamId];

    /// Evaluate f(x) — the scalar value to minimize.
    fn value(&self, store: &ParamStore) -> f64;

    /// Sparse gradient: `(param_id, ∂f/∂p)` pairs.
    ///
    /// Only non-zero entries need be returned. The solver maps `ParamId`
    /// to column indices via [`crate::param::SolverMapping`].
    fn gradient(&self, store: &ParamStore) -> Vec<(ParamId, f64)>;
}

/// Optional Hessian extension for second-order optimization methods.
///
/// When this trait is NOT implemented, solvers fall back to BFGS/L-BFGS
/// quasi-Newton approximation. Implement this when exact Hessians are
/// available and the problem is small enough (N ≤ 30) for them to help.
///
/// # Convention
///
/// Returns lower-triangle sparse entries `(ParamId_i, ParamId_j, ∂²f/∂p_i∂p_j)`
/// where the row index ≥ column index. Symmetric entries (i,j) and (j,i) should
/// only appear once (as the lower-triangle entry).
pub trait ObjectiveHessian: Objective {
    /// Sparse Hessian: `(param_i, param_j, ∂²f/∂p_i∂p_j)` lower-triangle entries.
    fn hessian_entries(&self, store: &ParamStore) -> Vec<(ParamId, ParamId, f64)>;
}
