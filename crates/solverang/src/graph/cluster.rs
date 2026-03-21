//! Rigid clusters — groups of coupled constraints solved together.
//!
//! A [`RigidCluster`] is a connected component of the constraint graph: a
//! set of constraints that share parameters (directly or transitively) and
//! therefore must be solved simultaneously. Independent clusters can be
//! solved in parallel.

use crate::id::{ClusterId, EntityId, ParamId};
use crate::param::ParamStore;

/// Lifecycle status of a rigid cluster.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClusterStatus {
    /// Never solved — newly created or rebuilt after decomposition.
    Fresh,
    /// At least one parameter changed since the last successful solve.
    Dirty,
    /// Solved and up-to-date.
    Solved,
    /// The most recent solve attempt failed.
    Failed,
}

/// A group of coupled constraints that must be solved together.
///
/// Clusters are produced by [`decompose_clusters`](super::decompose::decompose_clusters)
/// and consumed by the solver. Each cluster carries enough bookkeeping for
/// warm-starting (cached solution) and incremental re-solve (status tracking).
#[derive(Clone, Debug)]
pub struct RigidCluster {
    /// Unique identifier for this cluster.
    pub id: ClusterId,
    /// Indices into the system's constraint vec.
    pub constraint_indices: Vec<usize>,
    /// All parameter IDs involved in this cluster.
    pub param_ids: Vec<ParamId>,
    /// All entity IDs involved in this cluster.
    pub entity_ids: Vec<EntityId>,
    /// Current lifecycle status.
    pub status: ClusterStatus,
    /// Warm-start values from the last successful solve.
    pub cached_solution: Option<Vec<f64>>,
    /// Residual norm from the last successful solve.
    pub last_residual_norm: Option<f64>,
}

impl RigidCluster {
    /// Create a new cluster in [`Fresh`](ClusterStatus::Fresh) status.
    pub fn new(
        id: ClusterId,
        constraint_indices: Vec<usize>,
        param_ids: Vec<ParamId>,
        entity_ids: Vec<EntityId>,
    ) -> Self {
        Self {
            id,
            constraint_indices,
            param_ids,
            entity_ids,
            status: ClusterStatus::Fresh,
            cached_solution: None,
            last_residual_norm: None,
        }
    }

    /// Number of free (non-fixed) parameters in this cluster.
    pub fn free_param_count(&self, store: &ParamStore) -> usize {
        self.param_ids
            .iter()
            .filter(|&&pid| !store.is_fixed(pid))
            .count()
    }

    /// Mark this cluster as needing a re-solve.
    pub fn mark_dirty(&mut self) {
        self.status = ClusterStatus::Dirty;
    }

    /// Record a successful solve with the given residual norm.
    pub fn mark_solved(&mut self, residual_norm: f64) {
        self.status = ClusterStatus::Solved;
        self.last_residual_norm = Some(residual_norm);
    }

    /// Record a failed solve attempt.
    pub fn mark_failed(&mut self) {
        self.status = ClusterStatus::Failed;
    }

    /// Store a solution snapshot for warm-starting the next solve.
    pub fn cache_solution(&mut self, solution: Vec<f64>) {
        self.cached_solution = Some(solution);
    }

    /// Return the cached solution for warm-starting, if available.
    pub fn warm_start(&self) -> Option<&[f64]> {
        self.cached_solution.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::{ClusterId, EntityId, ParamId};

    fn sample_cluster() -> RigidCluster {
        RigidCluster::new(
            ClusterId(0),
            vec![0, 1],
            vec![ParamId::new(0, 0), ParamId::new(1, 0)],
            vec![EntityId::new(0, 0)],
        )
    }

    #[test]
    fn test_new_cluster_is_fresh() {
        let c = sample_cluster();
        assert_eq!(c.status, ClusterStatus::Fresh);
        assert!(c.cached_solution.is_none());
        assert!(c.last_residual_norm.is_none());
    }

    #[test]
    fn test_status_transitions() {
        let mut c = sample_cluster();

        c.mark_dirty();
        assert_eq!(c.status, ClusterStatus::Dirty);

        c.mark_solved(1e-12);
        assert_eq!(c.status, ClusterStatus::Solved);
        assert!((c.last_residual_norm.unwrap() - 1e-12).abs() < f64::EPSILON);

        c.mark_failed();
        assert_eq!(c.status, ClusterStatus::Failed);
    }

    #[test]
    fn test_cache_and_warm_start() {
        let mut c = sample_cluster();
        assert!(c.warm_start().is_none());

        c.cache_solution(vec![1.0, 2.0]);
        assert_eq!(c.warm_start(), Some(&[1.0, 2.0][..]));
    }

    #[test]
    fn test_free_param_count() {
        let eid = EntityId::new(0, 0);
        let mut store = ParamStore::new();
        let p0 = store.alloc(0.0, eid);
        let p1 = store.alloc(0.0, eid);
        let p2 = store.alloc(0.0, eid);

        store.fix(p1);

        let c = RigidCluster::new(ClusterId(0), vec![0], vec![p0, p1, p2], vec![eid]);
        assert_eq!(c.free_param_count(&store), 2);
    }

    #[test]
    fn test_cluster_clone() {
        let mut c = sample_cluster();
        c.cache_solution(vec![3.0, 4.0]);
        c.mark_solved(0.001);

        let c2 = c.clone();
        assert_eq!(c2.status, ClusterStatus::Solved);
        assert_eq!(c2.warm_start(), Some(&[3.0, 4.0][..]));
        assert_eq!(c2.last_residual_norm, Some(0.001));
    }
}
