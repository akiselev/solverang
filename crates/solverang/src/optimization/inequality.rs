//! Inequality constraint trait for optimization.
//!
//! An inequality constraint represents `h(x) ≤ 0`. It parallels the
//! [`crate::constraint::Constraint`] trait (which represents `g(x) = 0`)
//! but uses a different sign convention: values should be ≤ 0 when satisfied.
//!
//! # Naming
//!
//! Named `InequalityFn` to avoid collision with the existing
//! `InequalityConstraint` in `crate::constraints::inequality` which uses
//! an array-based `&[f64]` interface (the "problem-level" API).

use crate::id::{ConstraintId, EntityId, ParamId};
use crate::param::ParamStore;

/// System-level inequality constraint: `h(x) ≤ 0`.
///
/// Parallels [`crate::constraint::Constraint`] (equality, `g(x) = 0`) but for
/// inequality constraints. Uses `ParamStore`-based interface with sparse Jacobian.
///
/// # Sign Convention
///
/// `values()` returns values that should be **≤ 0 when the constraint is satisfied**.
/// Positive values indicate violation.
///
/// This is the standard optimization convention (`h(x) ≤ 0`), opposite to the
/// existing `InequalityConstraint` in `crate::constraints` which uses `g(x) ≥ 0`.
pub trait InequalityFn: Send + Sync {
    /// Unique identifier (shares ID space with Constraint for pipeline integration).
    fn id(&self) -> ConstraintId;

    /// Human-readable name for diagnostics.
    fn name(&self) -> &str;

    /// Which entities this constraint binds.
    fn entity_ids(&self) -> &[EntityId];

    /// Which parameters this constraint depends on.
    fn param_ids(&self) -> &[ParamId];

    /// Number of scalar inequality equations.
    fn inequality_count(&self) -> usize;

    /// Evaluate h(x). Each element should be ≤ 0 when satisfied.
    fn values(&self, store: &ParamStore) -> Vec<f64>;

    /// Sparse Jacobian: `(inequality_row, param_id, ∂h_i/∂p)`.
    ///
    /// Only non-zero entries need be returned.
    fn jacobian(&self, store: &ParamStore) -> Vec<(usize, ParamId, f64)>;
}
