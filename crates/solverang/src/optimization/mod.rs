//! Optimization support for the constraint system.
//!
//! This module adds constrained optimization (`min f(x) s.t. g(x)=0, h(x)≤0`)
//! alongside the existing constraint-satisfaction (`F(x)=0`) infrastructure.
//!
//! # Core Traits
//!
//! - [`Objective`] — scalar function to minimize (value + gradient)
//! - [`ObjectiveHessian`] — optional second derivatives for Newton-like methods
//! - [`InequalityFn`] — inequality constraints `h(x) ≤ 0`
//!
//! # Architecture
//!
//! Optimization adds an outer layer around existing solvers:
//! - **BFGS/L-BFGS** for unconstrained problems (gradient-only)
//! - **Augmented Lagrangian Method (ALM)** for constrained problems,
//!   reusing existing NR/LM as the inner solver
//!
//! Existing `Constraint` trait objects serve as equality constraints (`g(x) = 0`)
//! in the optimization formulation with zero modifications.

pub mod adapters;
pub mod config;
pub mod inequality;
pub mod multiplier_store;
pub mod objective;
pub mod result;

pub use config::{MultiplierInitStrategy, OptimizationAlgorithm, OptimizationConfig};
pub use inequality::InequalityFn;
pub use multiplier_store::{MultiplierId, MultiplierStore};
pub use objective::{Objective, ObjectiveHessian};
pub use result::{KktResidual, OptimizationResult, OptimizationStatus};

use crate::id::Generation;
use std::fmt;

/// A generational index for an objective function.
///
/// Follows the same pattern as [`crate::id::EntityId`] and
/// [`crate::id::ConstraintId`] — type-safe, generation-checked.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObjectiveId {
    pub(crate) index: u32,
    pub(crate) generation: Generation,
}

impl fmt::Debug for ObjectiveId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Objective({}g{})", self.index, self.generation)
    }
}

impl ObjectiveId {
    /// Create a new ObjectiveId.
    pub fn new(index: u32, generation: Generation) -> Self {
        Self { index, generation }
    }

    /// Raw index (for internal use).
    pub fn raw_index(self) -> u32 {
        self.index
    }
}
