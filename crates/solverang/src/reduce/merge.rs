//! Coincident parameter merging.
//!
//! When a constraint enforces that two parameters are equal (`a == b`), one
//! parameter can be replaced by the other everywhere, reducing the variable
//! count by one per merge. This module detects such equality constraints by
//! inspecting Jacobian structure, and produces a substitution map using a
//! union-find algorithm to handle transitive chains (`a == b`, `b == c`
//! implies `a`, `b`, `c` all share the same canonical representative).

use std::collections::HashMap;

use crate::constraint::Constraint;
use crate::id::ParamId;
use crate::param::ParamStore;

/// A parameter merge: replace `source` with `target` everywhere.
#[derive(Clone, Debug)]
pub struct ParamMerge {
    /// The parameter to be removed (merged away).
    pub source: ParamId,
    /// The parameter that replaces `source`.
    pub target: ParamId,
}

/// Result of merge analysis.
#[derive(Clone, Debug)]
pub struct MergeResult {
    /// The list of pairwise merges detected.
    pub merges: Vec<ParamMerge>,
    /// Number of equality constraints that can be removed from the solve
    /// (one per merge).
    pub constraints_removed: usize,
}

/// Tolerance for checking whether a Jacobian entry matches +1 or -1.
const JACOBIAN_TOLERANCE: f64 = 1e-10;

/// Detect coincident constraints (constraints that enforce `param_a == param_b`)
/// and produce a list of parameter merges.
///
/// A constraint is considered an equality constraint if:
/// - It has exactly 1 equation.
/// - It depends on exactly 2 parameters.
/// - Its Jacobian has entries `[+1, -1]` or `[-1, +1]` for those two
///   parameters (indicating a residual of the form `a - b`).
///
/// This is a structural heuristic that covers the most common case: a simple
/// coincident/equality constraint between two scalar parameters.
pub fn detect_merges(
    constraints: &[&dyn Constraint],
    store: &ParamStore,
) -> MergeResult {
    let mut merges = Vec::new();
    let mut constraints_removed = 0;

    for c in constraints {
        // Only consider single-equation constraints.
        if c.equation_count() != 1 {
            continue;
        }

        let params = c.param_ids();
        if params.len() != 2 {
            continue;
        }

        // Both params must be free for a merge to be useful.
        // (If one is fixed, the eliminate pass handles it instead.)
        if store.is_fixed(params[0]) || store.is_fixed(params[1]) {
            continue;
        }

        // Check the Jacobian structure.
        let jac = c.jacobian(store);
        if !is_equality_jacobian(&jac, params[0], params[1]) {
            continue;
        }

        // Convention: the param with the higher raw index is the source
        // (will be merged away), the lower one is the target (canonical).
        let (source, target) = if params[0].raw_index() > params[1].raw_index() {
            (params[0], params[1])
        } else {
            (params[1], params[0])
        };

        merges.push(ParamMerge { source, target });
        constraints_removed += 1;
    }

    MergeResult {
        merges,
        constraints_removed,
    }
}

/// Check whether a Jacobian represents a simple equality `a - b = 0`.
///
/// We look for exactly two entries in row 0, one being +1 and the other -1
/// (or vice versa), matching the two parameter IDs.
fn is_equality_jacobian(
    jac: &[(usize, ParamId, f64)],
    param_a: ParamId,
    param_b: ParamId,
) -> bool {
    // Collect entries for row 0 that match our two params.
    let mut val_a: Option<f64> = None;
    let mut val_b: Option<f64> = None;

    for &(row, pid, value) in jac {
        if row != 0 {
            continue;
        }
        if pid == param_a {
            val_a = Some(value);
        } else if pid == param_b {
            val_b = Some(value);
        }
    }

    match (val_a, val_b) {
        (Some(a), Some(b)) => {
            // Check for (+1, -1) or (-1, +1).
            ((a - 1.0).abs() < JACOBIAN_TOLERANCE && (b + 1.0).abs() < JACOBIAN_TOLERANCE)
                || ((a + 1.0).abs() < JACOBIAN_TOLERANCE
                    && (b - 1.0).abs() < JACOBIAN_TOLERANCE)
        }
        _ => false,
    }
}

/// Build a substitution map: for each merged parameter, find its canonical
/// representative.
///
/// Uses a union-find algorithm to handle transitive merges:
/// if `a = b` and `b = c`, then `a`, `b`, and `c` all map to the same
/// canonical representative.
pub fn build_substitution_map(merges: &[ParamMerge]) -> HashMap<ParamId, ParamId> {
    // Parent map for union-find. Each param points to its parent.
    let mut parent: HashMap<ParamId, ParamId> = HashMap::new();

    // Find with path compression.
    fn find(parent: &mut HashMap<ParamId, ParamId>, x: ParamId) -> ParamId {
        let p = match parent.get(&x) {
            Some(&p) if p != x => p,
            _ => return x,
        };
        let root = find(parent, p);
        parent.insert(x, root);
        root
    }

    // Union: merge source into target's set.
    fn union(parent: &mut HashMap<ParamId, ParamId>, source: ParamId, target: ParamId) {
        let root_s = find(parent, source);
        let root_t = find(parent, target);
        if root_s != root_t {
            parent.insert(root_s, root_t);
        }
    }

    // Initialize: ensure both source and target are in the map.
    for m in merges {
        parent.entry(m.source).or_insert(m.source);
        parent.entry(m.target).or_insert(m.target);
    }

    // Apply unions.
    for m in merges {
        union(&mut parent, m.source, m.target);
    }

    // Build the final substitution map: every non-root param maps to its root.
    let all_params: Vec<ParamId> = parent.keys().copied().collect();
    let mut substitution = HashMap::new();
    for p in all_params {
        let root = find(&mut parent, p);
        if root != p {
            substitution.insert(p, root);
        }
    }

    substitution
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::{ConstraintId, EntityId, ParamId};
    use crate::param::ParamStore;

    /// A constraint that enforces param_a == param_b.
    /// Residual: a - b. Jacobian: [(0, a, +1), (0, b, -1)].
    struct EqualityConstraint {
        id: ConstraintId,
        params: [ParamId; 2],
    }

    impl Constraint for EqualityConstraint {
        fn id(&self) -> ConstraintId {
            self.id
        }
        fn name(&self) -> &str {
            "equality"
        }
        fn entity_ids(&self) -> &[EntityId] {
            &[]
        }
        fn param_ids(&self) -> &[ParamId] {
            &self.params
        }
        fn equation_count(&self) -> usize {
            1
        }
        fn residuals(&self, store: &ParamStore) -> Vec<f64> {
            vec![store.get(self.params[0]) - store.get(self.params[1])]
        }
        fn jacobian(&self, _store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
            vec![(0, self.params[0], 1.0), (0, self.params[1], -1.0)]
        }
    }

    /// A non-equality constraint (e.g., distance).
    struct DistanceConstraint {
        id: ConstraintId,
        params: [ParamId; 2],
    }

    impl Constraint for DistanceConstraint {
        fn id(&self) -> ConstraintId {
            self.id
        }
        fn name(&self) -> &str {
            "distance"
        }
        fn entity_ids(&self) -> &[EntityId] {
            &[]
        }
        fn param_ids(&self) -> &[ParamId] {
            &self.params
        }
        fn equation_count(&self) -> usize {
            1
        }
        fn residuals(&self, store: &ParamStore) -> Vec<f64> {
            let a = store.get(self.params[0]);
            let b = store.get(self.params[1]);
            vec![(a - b).powi(2) - 1.0]
        }
        fn jacobian(&self, store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
            let a = store.get(self.params[0]);
            let b = store.get(self.params[1]);
            vec![
                (0, self.params[0], 2.0 * (a - b)),
                (0, self.params[1], -2.0 * (a - b)),
            ]
        }
    }

    fn dummy_owner() -> EntityId {
        EntityId::new(0, 0)
    }

    #[test]
    fn test_detect_simple_merge() {
        let mut store = ParamStore::new();
        let owner = dummy_owner();
        let a = store.alloc(5.0, owner);
        let b = store.alloc(5.0, owner);

        let c = EqualityConstraint {
            id: ConstraintId::new(0, 0),
            params: [a, b],
        };

        let constraints: Vec<&dyn Constraint> = vec![&c];
        let result = detect_merges(&constraints, &store);

        assert_eq!(result.merges.len(), 1);
        assert_eq!(result.constraints_removed, 1);
        // Higher raw index is the source.
        assert_eq!(result.merges[0].source, b);
        assert_eq!(result.merges[0].target, a);
    }

    #[test]
    fn test_no_merge_for_non_equality() {
        let mut store = ParamStore::new();
        let owner = dummy_owner();
        let a = store.alloc(1.0, owner);
        let b = store.alloc(2.0, owner);

        let c = DistanceConstraint {
            id: ConstraintId::new(0, 0),
            params: [a, b],
        };

        let constraints: Vec<&dyn Constraint> = vec![&c];
        let result = detect_merges(&constraints, &store);

        assert_eq!(result.merges.len(), 0);
        assert_eq!(result.constraints_removed, 0);
    }

    #[test]
    fn test_no_merge_when_param_fixed() {
        let mut store = ParamStore::new();
        let owner = dummy_owner();
        let a = store.alloc(5.0, owner);
        let b = store.alloc(5.0, owner);
        store.fix(a);

        let c = EqualityConstraint {
            id: ConstraintId::new(0, 0),
            params: [a, b],
        };

        let constraints: Vec<&dyn Constraint> = vec![&c];
        let result = detect_merges(&constraints, &store);

        // Should not merge because one param is fixed.
        assert_eq!(result.merges.len(), 0);
    }

    #[test]
    fn test_transitive_substitution_map() {
        // a = b, b = c  =>  a -> c, b -> c  (or a,b -> some canonical)
        let a = ParamId::new(0, 0);
        let b = ParamId::new(1, 0);
        let c = ParamId::new(2, 0);

        let merges = vec![
            ParamMerge {
                source: b,
                target: a,
            },
            ParamMerge {
                source: c,
                target: b,
            },
        ];

        let map = build_substitution_map(&merges);

        // Both b and c should map to a (the transitive root).
        let root_b = map.get(&b).copied().unwrap_or(b);
        let root_c = map.get(&c).copied().unwrap_or(c);
        assert_eq!(root_b, root_c);
        // a should be the canonical (not in the map, or mapping to itself).
        assert!(!map.contains_key(&a) || map[&a] == a);
    }

    #[test]
    fn test_substitution_map_single() {
        let a = ParamId::new(0, 0);
        let b = ParamId::new(1, 0);

        let merges = vec![ParamMerge {
            source: b,
            target: a,
        }];

        let map = build_substitution_map(&merges);

        assert_eq!(map.get(&b), Some(&a));
        assert!(!map.contains_key(&a));
    }

    #[test]
    fn test_substitution_map_empty() {
        let map = build_substitution_map(&[]);
        assert!(map.is_empty());
    }
}
