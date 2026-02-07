//! Solve module: bridges trait-based constraints to the existing `Problem` trait.
//!
//! This module provides [`ReducedSubProblem`], the key adapter that allows
//! existing solvers (Newton-Raphson, Levenberg-Marquardt, etc.) to work
//! unchanged with the new entity/constraint/param system.
//!
//! # Architecture
//!
//! The constraint system decomposes into independent clusters of coupled
//! constraints. Each cluster becomes a [`ReducedSubProblem`] that implements
//! the [`Problem`](crate::problem::Problem) trait. The solver sees a standard
//! nonlinear system — it never knows about `ParamId`, `Entity`, or
//! `Constraint` directly.
//!
//! ```text
//! ConstraintSystem
//!   -> decompose into clusters
//!     -> for each cluster:
//!        ReducedSubProblem (implements Problem)
//!          -> LMSolver / AutoSolver / etc.
//!            -> solution written back to ParamStore
//! ```

mod sub_problem;
pub mod branch;
pub mod closed_form;
pub mod drag;

pub use sub_problem::ReducedSubProblem;
