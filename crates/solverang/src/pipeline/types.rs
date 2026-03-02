//! Intermediate data types that flow between pipeline phases.
//!
//! These are owned value types (`Clone + Debug`) passed between the
//! Decompose → Analyze → Reduce → Solve → PostProcess phases.

use std::collections::HashMap;

use crate::graph::dof::DofAnalysis;
use crate::graph::pattern::MatchedPattern;
use crate::graph::redundancy::RedundancyAnalysis;
use crate::id::{ClusterId, EntityId, ParamId};
use crate::param::SolverMapping;
use crate::system::{ClusterSolveStatus, DiagnosticIssue};

// ---------------------------------------------------------------------------
// Decompose output
// ---------------------------------------------------------------------------

/// An independent cluster of coupled constraints, produced by decomposition.
#[derive(Clone, Debug)]
pub struct ClusterData {
    /// Cluster identifier (dense index assigned during decomposition).
    pub id: ClusterId,
    /// Indices into the system's `constraints` vec.
    pub constraint_indices: Vec<usize>,
    /// All distinct `ParamId`s touched by constraints in this cluster.
    pub param_ids: Vec<ParamId>,
    /// All distinct `EntityId`s referenced by constraints in this cluster.
    pub entity_ids: Vec<EntityId>,
}

// ---------------------------------------------------------------------------
// Analyze output
// ---------------------------------------------------------------------------

/// Structural analysis of a single cluster.
#[derive(Clone, Debug, Default)]
pub struct ClusterAnalysis {
    /// Which cluster this analysis belongs to.
    pub cluster_id: ClusterId,
    /// Per-entity DOF breakdown (if computed).
    pub dof: Option<DofAnalysis>,
    /// Redundancy / conflict analysis (if computed).
    pub redundancy: Option<RedundancyAnalysis>,
    /// Solvable patterns detected in this cluster.
    pub patterns: Vec<MatchedPattern>,
    /// Diagnostic issues detected during analysis.
    pub diagnostics: Vec<DiagnosticIssue>,
}

// ---------------------------------------------------------------------------
// Reduce output
// ---------------------------------------------------------------------------

/// A cluster after symbolic reduction passes have been applied.
#[derive(Clone, Debug)]
pub struct ReducedCluster {
    /// Original cluster identifier.
    pub cluster_id: ClusterId,
    /// Constraint indices that remain active after reduction.
    pub active_constraint_indices: Vec<usize>,
    /// Parameter IDs that are still free after reduction.
    pub active_param_ids: Vec<ParamId>,
    /// Parameters whose values were analytically determined by elimination.
    pub eliminated_params: Vec<(ParamId, f64)>,
    /// Constraint indices removed by reduction (trivially satisfied or eliminated).
    pub removed_constraints: Vec<usize>,
    /// Parameter substitution map from coincident-param merging.
    /// Key = source (removed), Value = target (canonical).
    pub merge_map: HashMap<ParamId, ParamId>,
    /// Constraint indices that are trivially violated (cannot be satisfied).
    pub trivially_violated: Vec<usize>,
}

impl ReducedCluster {
    /// Create a "no reduction" passthrough from a `ClusterData`.
    pub fn passthrough(cluster: &ClusterData) -> Self {
        Self {
            cluster_id: cluster.id,
            active_constraint_indices: cluster.constraint_indices.clone(),
            active_param_ids: cluster.param_ids.clone(),
            eliminated_params: Vec::new(),
            removed_constraints: Vec::new(),
            merge_map: HashMap::new(),
            trivially_violated: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Solve output
// ---------------------------------------------------------------------------

/// Solution for a single cluster, combining closed-form and numerical results.
#[derive(Clone, Debug)]
pub struct ClusterSolution {
    /// Which cluster this solution belongs to.
    pub cluster_id: ClusterId,
    /// Solve status.
    pub status: ClusterSolveStatus,
    /// All determined parameter values (closed-form + numerical).
    pub param_values: Vec<(ParamId, f64)>,
    /// Solver mapping used for the numerical solve (if any).
    pub mapping: Option<SolverMapping>,
    /// Raw numerical solution in column order (if a numerical solve ran).
    pub numerical_solution: Option<Vec<f64>>,
    /// Total solver iterations (0 for pure closed-form or skipped clusters).
    pub iterations: usize,
    /// Final residual L2 norm.
    pub residual_norm: f64,
}
