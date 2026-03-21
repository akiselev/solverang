//! Post-processing phase: convert [`ClusterSolution`] into [`ClusterResult`].
//!
//! The default post-processor performs a straightforward conversion.
//! A diagnostic-aware variant and a helper for collecting diagnostics
//! from [`ClusterAnalysis`] are also provided.

use crate::system::{ClusterResult, DiagnosticIssue};

use super::traits::PostProcess;
use super::types::{ClusterAnalysis, ClusterData, ClusterSolution};

// ---------------------------------------------------------------------------
// DefaultPostProcess
// ---------------------------------------------------------------------------

/// Straightforward conversion from [`ClusterSolution`] to [`ClusterResult`].
pub struct DefaultPostProcess;

impl PostProcess for DefaultPostProcess {
    fn post_process(
        &self,
        solution: &ClusterSolution,
        _analysis: &ClusterAnalysis,
        cluster: &ClusterData,
    ) -> ClusterResult {
        ClusterResult {
            cluster_id: cluster.id,
            status: solution.status,
            iterations: solution.iterations,
            residual_norm: solution.residual_norm,
        }
    }
}

// ---------------------------------------------------------------------------
// DiagnosticPostProcess
// ---------------------------------------------------------------------------

/// A post-processor that can be extended to incorporate diagnostics.
///
/// Currently performs the same conversion as [`DefaultPostProcess`].
/// Diagnostics from [`ClusterAnalysis`] are collected separately via
/// [`collect_diagnostics`] at the pipeline orchestrator level.
pub struct DiagnosticPostProcess;

impl PostProcess for DiagnosticPostProcess {
    fn post_process(
        &self,
        solution: &ClusterSolution,
        _analysis: &ClusterAnalysis,
        cluster: &ClusterData,
    ) -> ClusterResult {
        ClusterResult {
            cluster_id: cluster.id,
            status: solution.status,
            iterations: solution.iterations,
            residual_norm: solution.residual_norm,
        }
    }
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

/// Extract diagnostics from a cluster analysis.
///
/// This is a convenience function for the pipeline orchestrator to gather
/// diagnostics from all clusters' analyses.
pub fn collect_diagnostics(analysis: &ClusterAnalysis) -> Vec<DiagnosticIssue> {
    analysis.diagnostics.clone()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::{ClusterId, ConstraintId, EntityId};
    use crate::system::ClusterSolveStatus;

    /// Build a minimal [`ClusterData`] for testing.
    fn test_cluster() -> ClusterData {
        ClusterData {
            id: ClusterId(42),
            constraint_indices: vec![0, 1],
            param_ids: Vec::new(),
            entity_ids: Vec::new(),
        }
    }

    /// Build a minimal [`ClusterAnalysis`] with no diagnostics.
    fn empty_analysis() -> ClusterAnalysis {
        ClusterAnalysis {
            cluster_id: ClusterId(42),
            dof: None,
            redundancy: None,
            patterns: Vec::new(),
            diagnostics: Vec::new(),
        }
    }

    /// Build a [`ClusterSolution`] with the given status, iterations, and residual.
    fn make_solution(
        status: ClusterSolveStatus,
        iterations: usize,
        residual_norm: f64,
    ) -> ClusterSolution {
        ClusterSolution {
            cluster_id: ClusterId(42),
            status,
            param_values: Vec::new(),
            mapping: None,
            numerical_solution: None,
            iterations,
            residual_norm,
        }
    }

    #[test]
    fn default_post_process_converged() {
        let pp = DefaultPostProcess;
        let solution = make_solution(ClusterSolveStatus::Converged, 10, 1e-12);
        let analysis = empty_analysis();
        let cluster = test_cluster();

        let result = pp.post_process(&solution, &analysis, &cluster);

        assert_eq!(result.cluster_id, ClusterId(42));
        assert_eq!(result.status, ClusterSolveStatus::Converged);
        assert_eq!(result.iterations, 10);
        assert!(result.residual_norm < 1e-10);
    }

    #[test]
    fn default_post_process_not_converged() {
        let pp = DefaultPostProcess;
        let solution = make_solution(ClusterSolveStatus::NotConverged, 100, 0.5);
        let analysis = empty_analysis();
        let cluster = test_cluster();

        let result = pp.post_process(&solution, &analysis, &cluster);

        assert_eq!(result.cluster_id, ClusterId(42));
        assert_eq!(result.status, ClusterSolveStatus::NotConverged);
        assert_eq!(result.iterations, 100);
        assert!((result.residual_norm - 0.5).abs() < 1e-15);
    }

    #[test]
    fn default_post_process_skipped() {
        let pp = DefaultPostProcess;
        let solution = make_solution(ClusterSolveStatus::Skipped, 0, 0.0);
        let analysis = empty_analysis();
        let cluster = test_cluster();

        let result = pp.post_process(&solution, &analysis, &cluster);

        assert_eq!(result.cluster_id, ClusterId(42));
        assert_eq!(result.status, ClusterSolveStatus::Skipped);
        assert_eq!(result.iterations, 0);
        assert!((result.residual_norm).abs() < 1e-15);
    }

    #[test]
    fn collect_diagnostics_returns_analysis_diagnostics() {
        let analysis = ClusterAnalysis {
            cluster_id: ClusterId(0),
            dof: None,
            redundancy: None,
            patterns: Vec::new(),
            diagnostics: vec![
                DiagnosticIssue::UnderConstrained {
                    entity: EntityId::new(0, 0),
                    free_directions: 2,
                },
                DiagnosticIssue::ConflictingConstraints {
                    constraints: vec![ConstraintId::new(0, 0), ConstraintId::new(1, 0)],
                },
            ],
        };

        let diags = collect_diagnostics(&analysis);
        assert_eq!(diags.len(), 2);
    }
}
