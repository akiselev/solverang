//! Jacobian computation and verification utilities.
//!
//! This module provides:
//! - `SparseJacobian`: A COO (coordinate) format sparse Jacobian representation
//! - `CsrMatrix`: A CSR (compressed sparse row) format for efficient operations
//! - `SparsityPattern`: Structural sparsity pattern for caching
//! - `verify_jacobian`: Finite-difference verification of analytical Jacobians
//! - `finite_difference_jacobian`: Compute Jacobian numerically

mod numeric;
mod pattern;
mod sparse;
mod verification;

pub use numeric::NumericJacobian;
pub use pattern::SparsityPattern;
pub use sparse::{CsrMatrix, SparseJacobian};
pub use verification::{finite_difference_jacobian, verify_jacobian, JacobianVerification};
