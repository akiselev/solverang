//! Adapters between Problem-level and Objective-level APIs.
//!
//! The primary adapter is [`LeastSquaresObjective`] which wraps any `Problem`
//! implementation as an `Objective` by minimizing `0.5 * ||F(x)||^2`.

use crate::id::ParamId;
use crate::param::ParamStore;
use crate::problem::Problem;

use super::{Objective, ObjectiveId};

/// Wraps a `Problem` as an `Objective` by minimizing `0.5 * ||F(x)||^2`.
///
/// This enables using any existing MINPACK test problem (or other `Problem`
/// implementation) through the optimization path.
///
/// # Gradient
///
/// The gradient of `0.5 ||F(x)||^2` is `J^T * r`, where `J` is the Jacobian
/// and `r` is the residual vector. This is computed from the `Problem`'s
/// `residuals()` and `jacobian()` methods.
pub struct LeastSquaresObjective<P> {
    /// The underlying Problem.
    pub problem: P,
    /// ParamIds corresponding to variable columns 0..n.
    pub param_ids: Vec<ParamId>,
    /// Objective ID.
    pub id: ObjectiveId,
}

impl<P: Problem> LeastSquaresObjective<P> {
    /// Create a new least-squares objective wrapping the given problem.
    ///
    /// `param_ids` must have length equal to `problem.variable_count()` and
    /// map solver column indices to ParamIds.
    pub fn new(problem: P, param_ids: Vec<ParamId>, id: ObjectiveId) -> Self {
        debug_assert_eq!(param_ids.len(), problem.variable_count());
        Self {
            problem,
            param_ids,
            id,
        }
    }

    /// Extract the current parameter values as a solver vector.
    fn x_from_store(&self, store: &ParamStore) -> Vec<f64> {
        self.param_ids.iter().map(|&pid| store.get(pid)).collect()
    }
}

impl<P: Problem> Objective for LeastSquaresObjective<P> {
    fn id(&self) -> ObjectiveId {
        self.id
    }

    fn name(&self) -> &str {
        self.problem.name()
    }

    fn param_ids(&self) -> &[ParamId] {
        &self.param_ids
    }

    fn value(&self, store: &ParamStore) -> f64 {
        let x = self.x_from_store(store);
        let r = self.problem.residuals(&x);
        0.5 * r.iter().map(|ri| ri * ri).sum::<f64>()
    }

    fn gradient(&self, store: &ParamStore) -> Vec<(ParamId, f64)> {
        let x = self.x_from_store(store);
        let r = self.problem.residuals(&x);
        let jac = self.problem.jacobian(&x);
        let n = self.param_ids.len();

        // Compute J^T * r
        let mut grad = vec![0.0; n];
        for (row, col, val) in &jac {
            grad[*col] += val * r[*row];
        }

        // Return sparse (non-zero entries)
        grad.into_iter()
            .enumerate()
            .filter(|(_, v)| v.abs() > 1e-30)
            .map(|(i, v)| (self.param_ids[i], v))
            .collect()
    }
}
