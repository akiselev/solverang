//! Default decomposition implementation using union-find.
//!
//! Partitions constraints into independent clusters by grouping constraints
//! that share parameters (directly or transitively). Uses union-find with
//! path splitting and union by rank for efficient connected component detection.

use std::collections::HashMap;

use crate::constraint::Constraint;
use crate::entity::Entity;
use crate::id::{ClusterId, EntityId, ParamId};
use crate::param::ParamStore;

use super::traits::Decompose;
use super::types::ClusterData;

// ---------------------------------------------------------------------------
// Union-Find
// ---------------------------------------------------------------------------

/// Union-Find (disjoint set) with path splitting and union by rank.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    /// Find the root of `x` with path splitting (each node on the path
    /// points to its grandparent, flattening the tree incrementally).
    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    /// Union the sets containing `a` and `b` by rank.
    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb;
        } else if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra;
        } else {
            self.parent[rb] = ra;
            self.rank[ra] += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// DefaultDecompose
// ---------------------------------------------------------------------------

/// Default decomposition strategy using union-find over shared parameters.
///
/// Two constraints belong to the same cluster if they share any parameter
/// (directly or transitively through other constraints). The resulting
/// clusters are sorted deterministically by first constraint index.
pub struct DefaultDecompose;

impl Decompose for DefaultDecompose {
    fn decompose(
        &self,
        constraints: &[Option<Box<dyn Constraint>>],
        _entities: &[Option<Box<dyn Entity>>],
        _store: &ParamStore,
    ) -> Vec<ClusterData> {
        // Collect indices of alive (non-None) constraints.
        let alive: Vec<usize> = constraints
            .iter()
            .enumerate()
            .filter_map(|(i, c)| c.as_ref().map(|_| i))
            .collect();

        if alive.is_empty() {
            return Vec::new();
        }

        // Build adjacency: ParamId -> list of alive constraint indices that use it.
        let mut param_to_constraints: HashMap<ParamId, Vec<usize>> = HashMap::new();
        for &idx in &alive {
            let constraint = constraints[idx].as_ref().unwrap();
            for &pid in constraint.param_ids() {
                param_to_constraints.entry(pid).or_default().push(idx);
            }
        }

        // Map alive constraint indices to dense [0..alive.len()) for union-find.
        let mut idx_to_dense: HashMap<usize, usize> = HashMap::new();
        for (dense, &idx) in alive.iter().enumerate() {
            idx_to_dense.insert(idx, dense);
        }

        let mut uf = UnionFind::new(alive.len());

        // Union constraints that share a parameter.
        for indices in param_to_constraints.values() {
            if indices.len() > 1 {
                let first = idx_to_dense[&indices[0]];
                for &ci in &indices[1..] {
                    uf.union(first, idx_to_dense[&ci]);
                }
            }
        }

        // Group by union-find root.
        let mut root_to_group: HashMap<usize, Vec<usize>> = HashMap::new();
        for (dense, &idx) in alive.iter().enumerate() {
            let root = uf.find(dense);
            root_to_group.entry(root).or_default().push(idx);
        }

        // Build ClusterData structs.
        let mut clusters: Vec<ClusterData> = root_to_group
            .into_values()
            .map(|mut constraint_indices| {
                constraint_indices.sort_unstable();

                // Collect all unique ParamIds.
                let mut param_ids: Vec<ParamId> = Vec::new();
                let mut param_seen: std::collections::HashSet<ParamId> =
                    std::collections::HashSet::new();

                // Collect all unique EntityIds.
                let mut entity_ids: Vec<EntityId> = Vec::new();
                let mut entity_seen: std::collections::HashSet<EntityId> =
                    std::collections::HashSet::new();

                for &ci in &constraint_indices {
                    let constraint = constraints[ci].as_ref().unwrap();
                    for &pid in constraint.param_ids() {
                        if param_seen.insert(pid) {
                            param_ids.push(pid);
                        }
                    }
                    for &eid in constraint.entity_ids() {
                        if entity_seen.insert(eid) {
                            entity_ids.push(eid);
                        }
                    }
                }

                ClusterData {
                    id: ClusterId(0), // placeholder, assigned after sorting
                    constraint_indices,
                    param_ids,
                    entity_ids,
                }
            })
            .collect();

        // Deterministic ordering by first constraint index.
        clusters.sort_by_key(|c| c.constraint_indices.first().copied().unwrap_or(usize::MAX));

        // Assign ClusterId based on sorted position.
        for (i, cluster) in clusters.iter_mut().enumerate() {
            cluster.id = ClusterId(i);
        }

        clusters
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::{ConstraintId, EntityId, ParamId};
    use crate::param::ParamStore;

    // -----------------------------------------------------------------------
    // Test constraint: fixes a single parameter to a target value.
    // -----------------------------------------------------------------------

    struct FixValueConstraint {
        id: ConstraintId,
        entity_ids: Vec<EntityId>,
        param: ParamId,
        target: f64,
    }

    impl Constraint for FixValueConstraint {
        fn id(&self) -> ConstraintId {
            self.id
        }
        fn name(&self) -> &str {
            "FixValue"
        }
        fn entity_ids(&self) -> &[EntityId] {
            &self.entity_ids
        }
        fn param_ids(&self) -> &[ParamId] {
            std::slice::from_ref(&self.param)
        }
        fn equation_count(&self) -> usize {
            1
        }
        fn residuals(&self, store: &ParamStore) -> Vec<f64> {
            vec![store.get(self.param) - self.target]
        }
        fn jacobian(&self, _store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
            vec![(0, self.param, 1.0)]
        }
    }

    // -----------------------------------------------------------------------
    // Test constraint: sum of parameters equals a target.
    // -----------------------------------------------------------------------

    struct SumConstraint {
        id: ConstraintId,
        entity_ids: Vec<EntityId>,
        params: Vec<ParamId>,
        target: f64,
    }

    impl Constraint for SumConstraint {
        fn id(&self) -> ConstraintId {
            self.id
        }
        fn name(&self) -> &str {
            "Sum"
        }
        fn entity_ids(&self) -> &[EntityId] {
            &self.entity_ids
        }
        fn param_ids(&self) -> &[ParamId] {
            &self.params
        }
        fn equation_count(&self) -> usize {
            1
        }
        fn residuals(&self, store: &ParamStore) -> Vec<f64> {
            let sum: f64 = self.params.iter().map(|&p| store.get(p)).sum();
            vec![sum - self.target]
        }
        fn jacobian(&self, _store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
            self.params.iter().map(|&p| (0, p, 1.0)).collect()
        }
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    #[test]
    fn empty_constraints_returns_empty_clusters() {
        let decomposer = DefaultDecompose;
        let constraints: Vec<Option<Box<dyn Constraint>>> = Vec::new();
        let entities: Vec<Option<Box<dyn Entity>>> = Vec::new();
        let store = ParamStore::new();

        let clusters = decomposer.decompose(&constraints, &entities, &store);
        assert!(clusters.is_empty());
    }

    #[test]
    fn two_independent_constraints_yield_two_clusters() {
        let decomposer = DefaultDecompose;
        let mut store = ParamStore::new();
        let owner = EntityId::new(0, 0);
        let p1 = store.alloc(1.0, owner);
        let p2 = store.alloc(2.0, owner);

        let c1: Box<dyn Constraint> = Box::new(FixValueConstraint {
            id: ConstraintId::new(0, 0),
            entity_ids: vec![owner],
            param: p1,
            target: 5.0,
        });
        let c2: Box<dyn Constraint> = Box::new(FixValueConstraint {
            id: ConstraintId::new(1, 0),
            entity_ids: vec![owner],
            param: p2,
            target: 10.0,
        });

        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![Some(c1), Some(c2)];
        let entities: Vec<Option<Box<dyn Entity>>> = Vec::new();
        let clusters = decomposer.decompose(&constraints, &entities, &store);

        assert_eq!(clusters.len(), 2, "Independent constraints -> 2 clusters");
        assert_eq!(clusters[0].id, ClusterId(0));
        assert_eq!(clusters[1].id, ClusterId(1));
        assert_eq!(clusters[0].constraint_indices, vec![0]);
        assert_eq!(clusters[1].constraint_indices, vec![1]);
    }

    #[test]
    fn two_coupled_constraints_yield_one_cluster() {
        let decomposer = DefaultDecompose;
        let mut store = ParamStore::new();
        let owner = EntityId::new(0, 0);
        let p1 = store.alloc(1.0, owner);
        let p2 = store.alloc(2.0, owner);

        // c1 uses p1; c2 uses p1 and p2 -> they share p1 -> same cluster
        let c1: Box<dyn Constraint> = Box::new(FixValueConstraint {
            id: ConstraintId::new(0, 0),
            entity_ids: vec![owner],
            param: p1,
            target: 5.0,
        });
        let c2: Box<dyn Constraint> = Box::new(SumConstraint {
            id: ConstraintId::new(1, 0),
            entity_ids: vec![owner],
            params: vec![p1, p2],
            target: 10.0,
        });

        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![Some(c1), Some(c2)];
        let entities: Vec<Option<Box<dyn Entity>>> = Vec::new();
        let clusters = decomposer.decompose(&constraints, &entities, &store);

        assert_eq!(clusters.len(), 1, "Coupled constraints -> 1 cluster");
        assert_eq!(clusters[0].id, ClusterId(0));
        assert_eq!(clusters[0].constraint_indices.len(), 2);
        assert_eq!(clusters[0].constraint_indices, vec![0, 1]);
        // Both params should be present
        assert_eq!(clusters[0].param_ids.len(), 2);
        assert!(clusters[0].param_ids.contains(&p1));
        assert!(clusters[0].param_ids.contains(&p2));
    }

    #[test]
    fn none_entries_are_ignored() {
        let decomposer = DefaultDecompose;
        let mut store = ParamStore::new();
        let owner = EntityId::new(0, 0);
        let p1 = store.alloc(1.0, owner);
        let p2 = store.alloc(2.0, owner);

        let c1: Box<dyn Constraint> = Box::new(FixValueConstraint {
            id: ConstraintId::new(0, 0),
            entity_ids: vec![owner],
            param: p1,
            target: 5.0,
        });
        let c3: Box<dyn Constraint> = Box::new(FixValueConstraint {
            id: ConstraintId::new(2, 0),
            entity_ids: vec![owner],
            param: p2,
            target: 10.0,
        });

        // Index 1 is None (removed constraint).
        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![Some(c1), None, Some(c3)];
        let entities: Vec<Option<Box<dyn Entity>>> = Vec::new();
        let clusters = decomposer.decompose(&constraints, &entities, &store);

        assert_eq!(clusters.len(), 2, "Two alive constraints -> 2 clusters");
        // Constraint indices should be 0 and 2, skipping the None at 1.
        assert_eq!(clusters[0].constraint_indices, vec![0]);
        assert_eq!(clusters[1].constraint_indices, vec![2]);
        // ClusterIds assigned by sorted position.
        assert_eq!(clusters[0].id, ClusterId(0));
        assert_eq!(clusters[1].id, ClusterId(1));
    }

    #[test]
    fn entity_ids_collected_from_constraints() {
        let decomposer = DefaultDecompose;
        let mut store = ParamStore::new();
        let entity_a = EntityId::new(0, 0);
        let entity_b = EntityId::new(1, 0);
        let p1 = store.alloc(1.0, entity_a);
        let p2 = store.alloc(2.0, entity_b);

        // c1 references entity_a, c2 references entity_b.
        // They share p1 so they end up in the same cluster.
        let c1: Box<dyn Constraint> = Box::new(FixValueConstraint {
            id: ConstraintId::new(0, 0),
            entity_ids: vec![entity_a],
            param: p1,
            target: 5.0,
        });
        let c2: Box<dyn Constraint> = Box::new(SumConstraint {
            id: ConstraintId::new(1, 0),
            entity_ids: vec![entity_b],
            params: vec![p1, p2],
            target: 10.0,
        });

        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![Some(c1), Some(c2)];
        let entities: Vec<Option<Box<dyn Entity>>> = Vec::new();
        let clusters = decomposer.decompose(&constraints, &entities, &store);

        assert_eq!(clusters.len(), 1);
        // Both entities should be collected.
        assert_eq!(clusters[0].entity_ids.len(), 2);
        assert!(clusters[0].entity_ids.contains(&entity_a));
        assert!(clusters[0].entity_ids.contains(&entity_b));
    }

    #[test]
    fn entity_ids_deduplicated() {
        let decomposer = DefaultDecompose;
        let mut store = ParamStore::new();
        let entity = EntityId::new(0, 0);
        let p1 = store.alloc(1.0, entity);
        let p2 = store.alloc(2.0, entity);

        // Both constraints reference the same entity.
        let c1: Box<dyn Constraint> = Box::new(FixValueConstraint {
            id: ConstraintId::new(0, 0),
            entity_ids: vec![entity],
            param: p1,
            target: 5.0,
        });
        let c2: Box<dyn Constraint> = Box::new(SumConstraint {
            id: ConstraintId::new(1, 0),
            entity_ids: vec![entity],
            params: vec![p1, p2],
            target: 10.0,
        });

        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![Some(c1), Some(c2)];
        let entities: Vec<Option<Box<dyn Entity>>> = Vec::new();
        let clusters = decomposer.decompose(&constraints, &entities, &store);

        assert_eq!(clusters.len(), 1);
        // Entity should appear only once despite being referenced by both constraints.
        assert_eq!(clusters[0].entity_ids.len(), 1);
        assert_eq!(clusters[0].entity_ids[0], entity);
    }
}
