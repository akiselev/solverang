//! Domain-neutral constraint graphs, solve orchestration, and diagnostics.
//!
//! Consumers own entity meaning and authoritative acceptance. Solverang owns
//! variables, residual relations, graph evaluation, candidate solving, and
//! diagnostics. Numerical algorithms are supplied by Methodus.

#![forbid(unsafe_code)]

mod error;
mod model;
mod solve;

pub use error::ConstraintError;
pub use model::{
    Constraint, ConstraintId, ConstraintModel, ConstraintOutput, ModelEvaluation, Relation,
    VariableId, VariableValues,
};
pub use solve::{
    ConstraintSolveConfig, ConstraintSolveReport, ConstraintStatus, solve_constraints,
    verify_constraint_jacobian,
};
