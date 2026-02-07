//! Parameter storage for the constraint system.
//!
//! The [`ParamStore`] is the single source of truth for all solvable parameter
//! values. Entities own parameters. Constraints read parameters. The solver
//! writes parameters.

mod store;

pub use store::{ParamStore, SolverMapping};
