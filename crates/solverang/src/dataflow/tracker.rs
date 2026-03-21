//! Change tracking for incremental constraint solving.
//!
//! [`ChangeTracker`] records which parameters, clusters, entities, and constraints
//! have been modified since the last solve. The solver inspects the tracker to
//! decide whether a full re-decomposition is needed (structural changes) or
//! whether only certain clusters require re-solving (parameter changes).

use std::collections::{HashMap, HashSet};

use crate::id::{ClusterId, ConstraintId, EntityId, ParamId};

/// Tracks changes to the constraint system for incremental solving.
///
/// When parameters change, only the clusters containing those parameters
/// need to be re-solved. When entities or constraints are added or removed,
/// the system needs a full re-decomposition before solving.
///
/// # Usage
///
/// ```ignore
/// let mut tracker = ChangeTracker::new();
///
/// // User drags a point, changing its x and y parameters.
/// tracker.mark_param_dirty(point_x);
/// tracker.mark_param_dirty(point_y);
///
/// // Determine which clusters to re-solve.
/// let dirty = tracker.compute_dirty_clusters(&param_to_cluster);
///
/// // After solving, clear the tracker for the next edit.
/// tracker.clear();
/// ```
#[derive(Clone, Debug)]
pub struct ChangeTracker {
    /// Parameters whose values have changed since the last solve.
    dirty_params: HashSet<ParamId>,
    /// Clusters that are known to need re-solving.
    dirty_clusters: HashSet<ClusterId>,
    /// Whether entities or constraints have been added or removed,
    /// requiring a full re-decomposition.
    structural_change: bool,
    /// Entity IDs added since the last solve.
    added_entities: Vec<EntityId>,
    /// Entity IDs removed since the last solve.
    removed_entities: Vec<EntityId>,
    /// Constraint IDs added since the last solve.
    added_constraints: Vec<ConstraintId>,
    /// Constraint IDs removed since the last solve.
    removed_constraints: Vec<ConstraintId>,
}

impl ChangeTracker {
    /// Create a new, empty change tracker with no pending changes.
    pub fn new() -> Self {
        Self {
            dirty_params: HashSet::new(),
            dirty_clusters: HashSet::new(),
            structural_change: false,
            added_entities: Vec::new(),
            removed_entities: Vec::new(),
            added_constraints: Vec::new(),
            removed_constraints: Vec::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Mark changes
    // -----------------------------------------------------------------------

    /// Record that a parameter value has changed.
    ///
    /// The cluster containing this parameter will be re-solved on the next
    /// incremental solve pass.
    pub fn mark_param_dirty(&mut self, id: ParamId) {
        self.dirty_params.insert(id);
    }

    /// Explicitly mark a cluster as dirty so it will be re-solved.
    ///
    /// This is useful when external logic determines a cluster needs
    /// re-evaluation independent of parameter changes (e.g., a constraint
    /// weight was modified).
    pub fn mark_cluster_dirty(&mut self, id: ClusterId) {
        self.dirty_clusters.insert(id);
    }

    /// Record that an entity was added to the system.
    ///
    /// Adding an entity is a structural change that requires re-decomposition.
    pub fn mark_entity_added(&mut self, id: EntityId) {
        self.structural_change = true;
        self.added_entities.push(id);
    }

    /// Record that an entity was removed from the system.
    ///
    /// Removing an entity is a structural change that requires re-decomposition.
    pub fn mark_entity_removed(&mut self, id: EntityId) {
        self.structural_change = true;
        self.removed_entities.push(id);
    }

    /// Record that a constraint was added to the system.
    ///
    /// Adding a constraint is a structural change that requires re-decomposition.
    pub fn mark_constraint_added(&mut self, id: ConstraintId) {
        self.structural_change = true;
        self.added_constraints.push(id);
    }

    /// Record that a constraint was removed from the system.
    ///
    /// Removing a constraint is a structural change that requires re-decomposition.
    pub fn mark_constraint_removed(&mut self, id: ConstraintId) {
        self.structural_change = true;
        self.removed_constraints.push(id);
    }

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------

    /// Returns `true` if entities or constraints were added or removed.
    pub fn has_structural_changes(&self) -> bool {
        self.structural_change
    }

    /// Returns `true` if any changes have been recorded (structural or parametric).
    pub fn has_any_changes(&self) -> bool {
        self.structural_change || !self.dirty_params.is_empty() || !self.dirty_clusters.is_empty()
    }

    /// The set of parameters that have changed since the last solve.
    pub fn dirty_params(&self) -> &HashSet<ParamId> {
        &self.dirty_params
    }

    /// The set of clusters that have been explicitly marked dirty.
    pub fn dirty_clusters(&self) -> &HashSet<ClusterId> {
        &self.dirty_clusters
    }

    /// The entity IDs that were added since the last solve.
    pub fn added_entities(&self) -> &[EntityId] {
        &self.added_entities
    }

    /// The entity IDs that were removed since the last solve.
    pub fn removed_entities(&self) -> &[EntityId] {
        &self.removed_entities
    }

    /// The constraint IDs that were added since the last solve.
    pub fn added_constraints(&self) -> &[ConstraintId] {
        &self.added_constraints
    }

    /// The constraint IDs that were removed since the last solve.
    pub fn removed_constraints(&self) -> &[ConstraintId] {
        &self.removed_constraints
    }

    /// Returns `true` if the constraint graph needs to be re-decomposed.
    ///
    /// Re-decomposition is required when structural changes have occurred
    /// (entities or constraints added/removed). Pure parameter value changes
    /// do not require re-decomposition.
    pub fn needs_redecompose(&self) -> bool {
        self.structural_change
    }

    // -----------------------------------------------------------------------
    // Reset
    // -----------------------------------------------------------------------

    /// Clear all tracked changes after a solve completes.
    ///
    /// This resets the tracker to its initial empty state, ready to accumulate
    /// changes for the next solve cycle.
    pub fn clear(&mut self) {
        self.dirty_params.clear();
        self.dirty_clusters.clear();
        self.structural_change = false;
        self.added_entities.clear();
        self.removed_entities.clear();
        self.added_constraints.clear();
        self.removed_constraints.clear();
    }

    // -----------------------------------------------------------------------
    // Cluster dirtying from parameter changes
    // -----------------------------------------------------------------------

    /// Determine which clusters need re-solving based on dirty parameters.
    ///
    /// Looks up each dirty parameter in the provided mapping and collects the
    /// corresponding cluster IDs. The result is merged with any clusters that
    /// were explicitly marked dirty via [`mark_cluster_dirty`](Self::mark_cluster_dirty).
    ///
    /// Parameters not found in the mapping are silently ignored (they may
    /// belong to fixed parameters or entities not yet assigned to a cluster).
    pub fn compute_dirty_clusters(
        &self,
        param_to_cluster: &HashMap<ParamId, ClusterId>,
    ) -> HashSet<ClusterId> {
        let mut result = self.dirty_clusters.clone();

        for param_id in &self.dirty_params {
            if let Some(&cluster_id) = param_to_cluster.get(param_id) {
                result.insert(cluster_id);
            }
        }

        result
    }
}

impl Default for ChangeTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::{ClusterId, ConstraintId, EntityId, ParamId};

    fn param(index: u32) -> ParamId {
        ParamId::new(index, 0)
    }

    fn entity(index: u32) -> EntityId {
        EntityId::new(index, 0)
    }

    fn constraint(index: u32) -> ConstraintId {
        ConstraintId::new(index, 0)
    }

    #[test]
    fn new_tracker_has_no_changes() {
        let tracker = ChangeTracker::new();
        assert!(!tracker.has_any_changes());
        assert!(!tracker.has_structural_changes());
        assert!(!tracker.needs_redecompose());
        assert!(tracker.dirty_params().is_empty());
        assert!(tracker.dirty_clusters().is_empty());
    }

    #[test]
    fn mark_param_dirty() {
        let mut tracker = ChangeTracker::new();
        let p = param(0);

        tracker.mark_param_dirty(p);

        assert!(tracker.has_any_changes());
        assert!(!tracker.has_structural_changes());
        assert!(!tracker.needs_redecompose());
        assert!(tracker.dirty_params().contains(&p));
    }

    #[test]
    fn mark_param_dirty_is_idempotent() {
        let mut tracker = ChangeTracker::new();
        let p = param(0);

        tracker.mark_param_dirty(p);
        tracker.mark_param_dirty(p);

        assert_eq!(tracker.dirty_params().len(), 1);
    }

    #[test]
    fn mark_cluster_dirty() {
        let mut tracker = ChangeTracker::new();
        let c = ClusterId(0);

        tracker.mark_cluster_dirty(c);

        assert!(tracker.has_any_changes());
        assert!(!tracker.has_structural_changes());
        assert!(tracker.dirty_clusters().contains(&c));
    }

    #[test]
    fn mark_entity_added_sets_structural_change() {
        let mut tracker = ChangeTracker::new();
        let e = entity(0);

        tracker.mark_entity_added(e);

        assert!(tracker.has_any_changes());
        assert!(tracker.has_structural_changes());
        assert!(tracker.needs_redecompose());
        assert_eq!(tracker.added_entities(), &[e]);
    }

    #[test]
    fn mark_entity_removed_sets_structural_change() {
        let mut tracker = ChangeTracker::new();
        let e = entity(1);

        tracker.mark_entity_removed(e);

        assert!(tracker.has_structural_changes());
        assert!(tracker.needs_redecompose());
        assert_eq!(tracker.removed_entities(), &[e]);
    }

    #[test]
    fn mark_constraint_added_sets_structural_change() {
        let mut tracker = ChangeTracker::new();
        let c = constraint(0);

        tracker.mark_constraint_added(c);

        assert!(tracker.has_structural_changes());
        assert!(tracker.needs_redecompose());
        assert_eq!(tracker.added_constraints(), &[c]);
    }

    #[test]
    fn mark_constraint_removed_sets_structural_change() {
        let mut tracker = ChangeTracker::new();
        let c = constraint(2);

        tracker.mark_constraint_removed(c);

        assert!(tracker.has_structural_changes());
        assert!(tracker.needs_redecompose());
        assert_eq!(tracker.removed_constraints(), &[c]);
    }

    #[test]
    fn clear_resets_everything() {
        let mut tracker = ChangeTracker::new();

        tracker.mark_param_dirty(param(0));
        tracker.mark_param_dirty(param(1));
        tracker.mark_cluster_dirty(ClusterId(0));
        tracker.mark_entity_added(entity(0));
        tracker.mark_entity_removed(entity(1));
        tracker.mark_constraint_added(constraint(0));
        tracker.mark_constraint_removed(constraint(1));

        assert!(tracker.has_any_changes());

        tracker.clear();

        assert!(!tracker.has_any_changes());
        assert!(!tracker.has_structural_changes());
        assert!(!tracker.needs_redecompose());
        assert!(tracker.dirty_params().is_empty());
        assert!(tracker.dirty_clusters().is_empty());
        assert!(tracker.added_entities().is_empty());
        assert!(tracker.removed_entities().is_empty());
        assert!(tracker.added_constraints().is_empty());
        assert!(tracker.removed_constraints().is_empty());
    }

    #[test]
    fn compute_dirty_clusters_from_params() {
        let mut tracker = ChangeTracker::new();
        tracker.mark_param_dirty(param(0));
        tracker.mark_param_dirty(param(1));
        tracker.mark_param_dirty(param(2));

        let mut mapping = HashMap::new();
        mapping.insert(param(0), ClusterId(0));
        mapping.insert(param(1), ClusterId(0)); // same cluster as param 0
        mapping.insert(param(2), ClusterId(1));
        // param(3) is not dirty, should not appear

        let dirty = tracker.compute_dirty_clusters(&mapping);

        assert_eq!(dirty.len(), 2);
        assert!(dirty.contains(&ClusterId(0)));
        assert!(dirty.contains(&ClusterId(1)));
    }

    #[test]
    fn compute_dirty_clusters_merges_explicit_clusters() {
        let mut tracker = ChangeTracker::new();
        tracker.mark_param_dirty(param(0));
        tracker.mark_cluster_dirty(ClusterId(5));

        let mut mapping = HashMap::new();
        mapping.insert(param(0), ClusterId(2));

        let dirty = tracker.compute_dirty_clusters(&mapping);

        assert_eq!(dirty.len(), 2);
        assert!(dirty.contains(&ClusterId(2))); // from param
        assert!(dirty.contains(&ClusterId(5))); // from explicit mark
    }

    #[test]
    fn compute_dirty_clusters_ignores_unknown_params() {
        let mut tracker = ChangeTracker::new();
        tracker.mark_param_dirty(param(99)); // not in the mapping

        let mapping = HashMap::new();
        let dirty = tracker.compute_dirty_clusters(&mapping);

        assert!(dirty.is_empty());
    }

    #[test]
    fn default_is_empty() {
        let tracker = ChangeTracker::default();
        assert!(!tracker.has_any_changes());
    }

    #[test]
    fn multiple_structural_changes_accumulate() {
        let mut tracker = ChangeTracker::new();

        tracker.mark_entity_added(entity(0));
        tracker.mark_entity_added(entity(1));
        tracker.mark_constraint_removed(constraint(0));

        assert_eq!(tracker.added_entities().len(), 2);
        assert_eq!(tracker.removed_constraints().len(), 1);
        assert!(tracker.has_structural_changes());
    }
}
