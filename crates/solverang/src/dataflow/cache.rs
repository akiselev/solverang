//! Per-cluster solution caching and warm-start support.
//!
//! After a successful solve, each cluster's solution is stored in a
//! [`SolutionCache`]. On the next solve cycle, the cached solution can serve
//! as a warm start, often reducing the number of iterations required.
//!
//! When clusters are invalidated (due to parameter changes) or removed
//! (due to re-decomposition), their cache entries are discarded.

use std::collections::HashMap;

use crate::id::ClusterId;

/// Cached solution state for a single cluster.
///
/// Stores the parameter values (in solver-column order), the residual norm
/// at those values, and the number of solver iterations that were used.
#[derive(Clone, Debug)]
pub struct ClusterCache {
    /// Cached parameter values in solver-column order.
    pub solution: Vec<f64>,
    /// Residual norm (L2) at the cached solution.
    pub residual_norm: f64,
    /// Number of solver iterations used to reach this solution.
    pub iterations: usize,
}

/// Solution cache for all clusters.
///
/// After a successful solve, each cluster's solution is cached. On the next
/// solve, the cached solution is used as a warm start if the cluster's
/// parameters haven't changed too much.
///
/// # Example
///
/// ```ignore
/// let mut cache = SolutionCache::new();
///
/// // After solving cluster 0:
/// cache.store(ClusterId(0), vec![1.0, 2.0, 3.0], 1e-12, 5);
///
/// // On next solve, retrieve the warm start:
/// if let Some(cached) = cache.get(&ClusterId(0)) {
///     // Use cached.solution as the initial guess.
/// }
/// ```
#[derive(Clone, Debug, Default)]
pub struct SolutionCache {
    clusters: HashMap<ClusterId, ClusterCache>,
}

impl SolutionCache {
    /// Create an empty solution cache.
    pub fn new() -> Self {
        Self {
            clusters: HashMap::new(),
        }
    }

    /// Store a solution for a cluster.
    ///
    /// Overwrites any previously cached solution for the same cluster.
    pub fn store(
        &mut self,
        cluster_id: ClusterId,
        solution: Vec<f64>,
        residual_norm: f64,
        iterations: usize,
    ) {
        self.clusters.insert(
            cluster_id,
            ClusterCache {
                solution,
                residual_norm,
                iterations,
            },
        );
    }

    /// Get the cached solution for a cluster, if one exists.
    ///
    /// Returns `None` if the cluster has no cached solution (either it was
    /// never solved or its cache was invalidated).
    pub fn get(&self, cluster_id: &ClusterId) -> Option<&ClusterCache> {
        self.clusters.get(cluster_id)
    }

    /// Invalidate (remove) the cached solution for a single cluster.
    ///
    /// Call this when a cluster's parameters have changed enough that
    /// the cached solution is no longer a useful warm start.
    pub fn invalidate(&mut self, cluster_id: &ClusterId) {
        self.clusters.remove(cluster_id);
    }

    /// Invalidate all cached solutions.
    ///
    /// Typically called after a full re-decomposition, since cluster IDs
    /// may have been reassigned.
    pub fn invalidate_all(&mut self) {
        self.clusters.clear();
    }

    /// Remove entries for clusters that no longer exist.
    ///
    /// After re-decomposition, old cluster IDs may be stale. This method
    /// retains only the entries whose IDs appear in `valid_ids`.
    pub fn retain_clusters(&mut self, valid_ids: &[ClusterId]) {
        let valid_set: std::collections::HashSet<&ClusterId> = valid_ids.iter().collect();
        self.clusters.retain(|id, _| valid_set.contains(id));
    }

    /// Returns the number of cached cluster solutions.
    pub fn len(&self) -> usize {
        self.clusters.len()
    }

    /// Returns `true` if the cache contains no entries.
    pub fn is_empty(&self) -> bool {
        self.clusters.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::ClusterId;

    #[test]
    fn new_cache_is_empty() {
        let cache = SolutionCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn store_and_get() {
        let mut cache = SolutionCache::new();
        let id = ClusterId(0);

        cache.store(id, vec![1.0, 2.0, 3.0], 1e-10, 7);

        let entry = cache.get(&id).expect("should have cached entry");
        assert_eq!(entry.solution, vec![1.0, 2.0, 3.0]);
        assert!((entry.residual_norm - 1e-10).abs() < 1e-20);
        assert_eq!(entry.iterations, 7);
    }

    #[test]
    fn store_overwrites_previous() {
        let mut cache = SolutionCache::new();
        let id = ClusterId(0);

        cache.store(id, vec![1.0], 0.1, 10);
        cache.store(id, vec![2.0], 0.01, 5);

        let entry = cache.get(&id).unwrap();
        assert_eq!(entry.solution, vec![2.0]);
        assert!((entry.residual_norm - 0.01).abs() < 1e-15);
        assert_eq!(entry.iterations, 5);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn get_nonexistent_returns_none() {
        let cache = SolutionCache::new();
        assert!(cache.get(&ClusterId(42)).is_none());
    }

    #[test]
    fn invalidate_single_cluster() {
        let mut cache = SolutionCache::new();
        cache.store(ClusterId(0), vec![1.0], 0.0, 1);
        cache.store(ClusterId(1), vec![2.0], 0.0, 2);

        cache.invalidate(&ClusterId(0));

        assert!(cache.get(&ClusterId(0)).is_none());
        assert!(cache.get(&ClusterId(1)).is_some());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn invalidate_nonexistent_is_noop() {
        let mut cache = SolutionCache::new();
        cache.store(ClusterId(0), vec![1.0], 0.0, 1);

        cache.invalidate(&ClusterId(99));

        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn invalidate_all_clears_cache() {
        let mut cache = SolutionCache::new();
        cache.store(ClusterId(0), vec![1.0], 0.0, 1);
        cache.store(ClusterId(1), vec![2.0], 0.0, 2);
        cache.store(ClusterId(2), vec![3.0], 0.0, 3);

        cache.invalidate_all();

        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn retain_clusters_keeps_valid_ids() {
        let mut cache = SolutionCache::new();
        cache.store(ClusterId(0), vec![1.0], 0.0, 1);
        cache.store(ClusterId(1), vec![2.0], 0.0, 2);
        cache.store(ClusterId(2), vec![3.0], 0.0, 3);
        cache.store(ClusterId(3), vec![4.0], 0.0, 4);

        // After re-decomposition, only clusters 1 and 3 still exist.
        cache.retain_clusters(&[ClusterId(1), ClusterId(3)]);

        assert_eq!(cache.len(), 2);
        assert!(cache.get(&ClusterId(0)).is_none());
        assert!(cache.get(&ClusterId(1)).is_some());
        assert!(cache.get(&ClusterId(2)).is_none());
        assert!(cache.get(&ClusterId(3)).is_some());
    }

    #[test]
    fn retain_clusters_with_empty_valid_ids() {
        let mut cache = SolutionCache::new();
        cache.store(ClusterId(0), vec![1.0], 0.0, 1);

        cache.retain_clusters(&[]);

        assert!(cache.is_empty());
    }

    #[test]
    fn default_is_empty() {
        let cache = SolutionCache::default();
        assert!(cache.is_empty());
    }

    #[test]
    fn clone_is_independent() {
        let mut cache = SolutionCache::new();
        cache.store(ClusterId(0), vec![1.0, 2.0], 1e-8, 3);

        let mut cloned = cache.clone();
        cloned.invalidate(&ClusterId(0));

        // Original should be unaffected.
        assert!(cache.get(&ClusterId(0)).is_some());
        assert!(cloned.get(&ClusterId(0)).is_none());
    }
}
