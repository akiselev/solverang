//! Ephemeral storage for Lagrange multipliers (dual variables).
//!
//! Multipliers are separate from [`ParamStore`] because they lack entity
//! ownership, fixed/free semantics, and change-tracking. They are recomputed
//! on each `optimize()` call.

use crate::constraint::Constraint;
use crate::id::ConstraintId;
use std::collections::HashMap;
use std::fmt;

use super::InequalityFn;

/// Identifies a specific Lagrange multiplier by its constraint and equation row.
///
/// Uses semantic addressing (constraint + row) rather than a generational index
/// because multipliers are ephemeral — recomputed each solve, not stored across
/// iterations.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct MultiplierId {
    /// The constraint this multiplier belongs to.
    pub constraint_id: ConstraintId,
    /// Which equation row within the constraint (0-based).
    pub equation_row: usize,
}

impl fmt::Debug for MultiplierId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Multiplier({:?}, row={})",
            self.constraint_id, self.equation_row
        )
    }
}

impl MultiplierId {
    /// Create a new multiplier identifier.
    pub fn new(constraint_id: ConstraintId, equation_row: usize) -> Self {
        Self {
            constraint_id,
            equation_row,
        }
    }
}

/// Ephemeral storage for Lagrange multipliers.
///
/// Cleared before each `optimize()` call. Populated by the solver after
/// convergence. Indexed by `MultiplierId` (constraint + equation row).
///
/// # Ordering Contract
///
/// Multipliers for a constraint are stored in equation-row order (0, 1, 2, ...).
/// `lambda_for_constraint()` returns them in this order, matching the residual
/// row ordering of `Constraint::residuals()`.
#[derive(Debug, Default)]
pub struct MultiplierStore {
    multipliers: HashMap<MultiplierId, f64>,
}

impl MultiplierStore {
    /// Create an empty multiplier store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Clear all stored multipliers.
    pub fn clear(&mut self) {
        self.multipliers.clear();
    }

    /// Set the multiplier for a specific constraint equation.
    pub fn set(&mut self, id: MultiplierId, value: f64) {
        self.multipliers.insert(id, value);
    }

    /// Get the multiplier for a specific constraint equation.
    pub fn get(&self, id: MultiplierId) -> Option<f64> {
        self.multipliers.get(&id).copied()
    }

    /// Get all multipliers for a constraint, ordered by equation row.
    ///
    /// Returns `None` if no multipliers are stored for this constraint.
    pub fn lambda_for_constraint(&self, constraint_id: ConstraintId) -> Option<Vec<f64>> {
        let mut entries: Vec<(usize, f64)> = self
            .multipliers
            .iter()
            .filter(|(id, _)| id.constraint_id == constraint_id)
            .map(|(id, &val)| (id.equation_row, val))
            .collect();

        if entries.is_empty() {
            return None;
        }

        entries.sort_by_key(|(row, _)| *row);
        Some(entries.into_iter().map(|(_, val)| val).collect())
    }

    /// Number of stored multipliers.
    pub fn len(&self) -> usize {
        self.multipliers.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.multipliers.is_empty()
    }

    /// Iterate over all stored multipliers.
    pub fn iter(&self) -> impl Iterator<Item = (MultiplierId, f64)> + '_ {
        self.multipliers.iter().map(|(&id, &val)| (id, val))
    }

    /// Extract equality multipliers as a flat vec in constraint iteration order.
    ///
    /// Returns 0.0 for any constraint not found in this store.
    pub fn extract_equality_vec(&self, constraints: &[&dyn Constraint]) -> Vec<f64> {
        let mut result = Vec::new();
        for c in constraints {
            let n = c.equation_count();
            match self.lambda_for_constraint(c.id()) {
                Some(vals) => {
                    for i in 0..n {
                        result.push(vals.get(i).copied().unwrap_or(0.0));
                    }
                }
                None => {
                    result.extend(std::iter::repeat(0.0).take(n));
                }
            }
        }
        result
    }

    /// Extract inequality multipliers as a flat vec in inequality iteration order.
    ///
    /// Values are clamped to >= 0 for dual feasibility. Returns 0.0 for any
    /// inequality not found in this store.
    pub fn extract_inequality_vec(&self, inequalities: &[&dyn InequalityFn]) -> Vec<f64> {
        let mut result = Vec::new();
        for h in inequalities {
            let n = h.inequality_count();
            match self.lambda_for_constraint(h.id()) {
                Some(vals) => {
                    for i in 0..n {
                        result.push(vals.get(i).copied().unwrap_or(0.0).max(0.0));
                    }
                }
                None => {
                    result.extend(std::iter::repeat(0.0).take(n));
                }
            }
        }
        result
    }
}
