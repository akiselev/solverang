//! Ephemeral storage for Lagrange multipliers (dual variables).
//!
//! Multipliers are separate from [`ParamStore`] because they lack entity
//! ownership, fixed/free semantics, and change-tracking. They are recomputed
//! on each `optimize()` call.

use crate::id::ConstraintId;
use std::collections::HashMap;
use std::fmt;

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
}
