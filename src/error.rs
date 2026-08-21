use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConstraintError {
    #[error("unknown variable {0}")]
    UnknownVariable(usize),
    #[error("{operation}: expected length {expected}, got {actual}")]
    DimensionMismatch {
        operation: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("constraint definition is invalid: {0}")]
    InvalidDefinition(String),
    #[error("constraint evaluation produced a non-finite value")]
    NonFinite,
    #[error(transparent)]
    Numerical(#[from] methodus::SolveError),
}
