//! Phase traits for the pluggable solve pipeline.
//!
//! Each trait represents one phase of the solve pipeline. Users can swap any
//! phase independently by providing a custom implementation.
//!
//! The pipeline flows:
//! ```text
//! Decompose → Analyze → Reduce → Solve → PostProcess
//! ```

use crate::constraint::Constraint;
use crate::entity::Entity;
use crate::param::ParamStore;
use crate::system::{ClusterResult, SystemConfig};

use super::types::{ClusterAnalysis, ClusterData, ClusterSolution, ReducedCluster};

// ---------------------------------------------------------------------------
// Phase 1: Decompose
// ---------------------------------------------------------------------------

/// Decompose the full constraint system into independent clusters.
///
/// Two constraints belong to the same cluster if they share parameters
/// (directly or transitively through other constraints).
pub trait Decompose: Send + Sync {
    /// Partition constraints into independent clusters.
    fn decompose(
        &self,
        constraints: &[Option<Box<dyn Constraint>>],
        entities: &[Option<Box<dyn Entity>>],
        store: &ParamStore,
    ) -> Vec<ClusterData>;
}

// ---------------------------------------------------------------------------
// Phase 2: Analyze
// ---------------------------------------------------------------------------

/// Structural analysis of a single cluster: DOF, redundancy, patterns.
///
/// Analysis results are advisory — they inform the Solve and PostProcess
/// phases but do not modify the constraint system.
pub trait Analyze: Send + Sync {
    /// Analyze a single cluster.
    fn analyze(
        &self,
        cluster: &ClusterData,
        constraints: &[Option<Box<dyn Constraint>>],
        entities: &[Option<Box<dyn Entity>>],
        store: &ParamStore,
    ) -> ClusterAnalysis;
}

// ---------------------------------------------------------------------------
// Phase 3: Reduce
// ---------------------------------------------------------------------------

/// Symbolic reduction of a cluster before numerical solving.
///
/// Reduction passes eliminate fixed parameters, merge coincident parameters,
/// and solve trivial single-variable constraints analytically.
///
/// The store is passed mutably so that stages can write determined values
/// and temporarily mark parameters as fixed, enabling cascading reductions
/// (e.g. eliminating `p1` lets a subsequent constraint with `p1` and `p2`
/// become single-free-param and therefore also eliminable).
pub trait Reduce: Send + Sync {
    /// Reduce a cluster, returning a simplified version.
    fn reduce(
        &self,
        cluster: &ClusterData,
        constraints: &[Option<Box<dyn Constraint>>],
        store: &mut ParamStore,
    ) -> ReducedCluster;
}

// ---------------------------------------------------------------------------
// Phase 4: Solve (per-cluster)
// ---------------------------------------------------------------------------

/// Solve a single reduced cluster, producing parameter values.
///
/// The default implementation tries closed-form solvers for matched patterns,
/// then falls back to numerical solving (LM) for the remaining constraints.
pub trait SolveCluster: Send + Sync {
    /// Solve a single cluster.
    fn solve_cluster(
        &self,
        reduced: &ReducedCluster,
        analysis: &ClusterAnalysis,
        constraints: &[Option<Box<dyn Constraint>>],
        store: &ParamStore,
        warm_start: Option<&[f64]>,
        config: &SystemConfig,
    ) -> ClusterSolution;
}

// ---------------------------------------------------------------------------
// Phase 5: PostProcess
// ---------------------------------------------------------------------------

/// Convert a cluster solution into a final result with diagnostics.
pub trait PostProcess: Send + Sync {
    /// Post-process a cluster solution.
    fn post_process(
        &self,
        solution: &ClusterSolution,
        analysis: &ClusterAnalysis,
        cluster: &ClusterData,
    ) -> ClusterResult;
}
