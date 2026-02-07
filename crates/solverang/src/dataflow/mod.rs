//! Incremental dataflow tracking for the constraint solver.
//!
//! This module provides change tracking and solution caching to enable
//! incremental re-solving. Instead of re-solving the entire constraint system
//! when a single parameter changes, the solver can:
//!
//! 1. Identify which clusters are affected by the change ([`ChangeTracker`]).
//! 2. Re-solve only the dirty clusters.
//! 3. Use cached solutions as warm starts ([`SolutionCache`]).
//!
//! When structural changes occur (entities or constraints added/removed),
//! the system triggers a full re-decomposition before solving.

mod cache;
mod tracker;

pub use cache::{ClusterCache, SolutionCache};
pub use tracker::ChangeTracker;
