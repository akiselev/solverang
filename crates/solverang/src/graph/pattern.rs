//! Solvable pattern detection: match subgraphs to known closed-form templates.
//!
//! Not all constraint sub-systems need an iterative solver. Many common
//! configurations (a single scalar equation, two distances on a point,
//! horizontal + vertical, distance + angle) admit closed-form solutions
//! that are faster and more robust than Newton-Raphson or LM.
//!
//! This module scans the entity-constraint neighbourhood for each entity
//! and matches it against a catalogue of known patterns. Matched patterns
//! are passed to the closed-form solvers in [`crate::solve::closed_form`].
//!
//! # Supported Patterns
//!
//! | Pattern | Description | DOF consumed |
//! |---------|-------------|-------------|
//! | `ScalarSolve` | 1 equation, 1 free param | 1 |
//! | `TwoDistances` | 2 distance equations on a 2-param entity | 2 |
//! | `HorizontalVertical` | horizontal + vertical on a 2-param entity | 2 |
//! | `DistanceAngle` | distance + angle on a 2-param entity | 2 |

use std::collections::{HashMap, HashSet};

use crate::constraint::Constraint;
use crate::entity::Entity;
use crate::id::{EntityId, ParamId};
use crate::param::ParamStore;

/// Known solvable patterns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PatternKind {
    /// Single constraint with a single free param (scalar solve).
    ScalarSolve,
    /// Two distance constraints on a point (circle-circle intersection).
    TwoDistances,
    /// Point constrained by horizontal + vertical (direct assignment).
    HorizontalVertical,
    /// Point constrained by distance + angle (polar coordinates).
    DistanceAngle,
}

/// A matched pattern in the constraint graph.
#[derive(Clone, Debug)]
pub struct MatchedPattern {
    /// What kind of pattern was detected.
    pub kind: PatternKind,
    /// Entity IDs involved in this pattern.
    pub entity_ids: Vec<EntityId>,
    /// Indices into the system's constraint vec for the constraints that
    /// form this pattern.
    pub constraint_indices: Vec<usize>,
    /// Parameter IDs solved by this pattern.
    pub param_ids: Vec<ParamId>,
}

/// Classify a constraint by name into a category for pattern matching.
///
/// This is intentionally broad: concrete constraint types in the geometry
/// layer use well-known names. We match on the name string so that the
/// pattern detector does not depend on concrete geometry types.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ConstraintCategory {
    Distance,
    Horizontal,
    Vertical,
    Angle,
    Other,
}

fn categorize(name: &str) -> ConstraintCategory {
    let lower = name.to_ascii_lowercase();
    if lower.contains("distance") {
        ConstraintCategory::Distance
    } else if lower.contains("horizontal") {
        ConstraintCategory::Horizontal
    } else if lower.contains("vertical") {
        ConstraintCategory::Vertical
    } else if lower.contains("angle") {
        ConstraintCategory::Angle
    } else {
        ConstraintCategory::Other
    }
}

/// Build a map from `EntityId` to the list of constraint indices that
/// reference that entity.
fn build_entity_constraint_map(
    constraints: &[(usize, &dyn Constraint)],
) -> HashMap<EntityId, Vec<usize>> {
    let mut map: HashMap<EntityId, Vec<usize>> = HashMap::new();
    for &(idx, c) in constraints {
        for &eid in c.entity_ids() {
            map.entry(eid).or_default().push(idx);
        }
    }
    map
}

/// Count the free (non-fixed) parameters for an entity.
fn free_params_for_entity(entity: &dyn Entity, store: &ParamStore) -> Vec<ParamId> {
    entity
        .params()
        .iter()
        .copied()
        .filter(|&pid| !store.is_fixed(pid))
        .collect()
}

/// Try to match a scalar-solve pattern: a single constraint acting on a single
/// free parameter. This is the simplest possible pattern.
fn try_scalar_solve(
    entity: &dyn Entity,
    free_params: &[ParamId],
    constraint_indices: &[usize],
    constraints: &[(usize, &dyn Constraint)],
    store: &ParamStore,
) -> Option<MatchedPattern> {
    if free_params.len() != 1 || constraint_indices.len() != 1 {
        return None;
    }

    let cidx = constraint_indices[0];
    let c = constraints.iter().find(|&&(i, _)| i == cidx)?.1;

    // The constraint must produce exactly one equation.
    if c.equation_count() != 1 {
        return None;
    }

    // The constraint must depend on exactly this one free parameter
    // (it may also depend on fixed parameters, which is fine).
    let c_free_params: Vec<_> = c
        .param_ids()
        .iter()
        .copied()
        .filter(|pid| !store.is_fixed(*pid))
        .collect();

    if c_free_params.len() != 1 || c_free_params[0] != free_params[0] {
        return None;
    }

    Some(MatchedPattern {
        kind: PatternKind::ScalarSolve,
        entity_ids: vec![entity.id()],
        constraint_indices: vec![cidx],
        param_ids: free_params.to_vec(),
    })
}

/// Try to match a two-distances pattern: an entity with exactly 2 free
/// parameters constrained by exactly 2 distance constraints.
fn try_two_distances(
    entity: &dyn Entity,
    free_params: &[ParamId],
    constraint_indices: &[usize],
    constraints: &[(usize, &dyn Constraint)],
) -> Option<MatchedPattern> {
    if free_params.len() != 2 || constraint_indices.len() != 2 {
        return None;
    }

    let mut distance_indices = Vec::new();
    for &cidx in constraint_indices {
        let c = constraints.iter().find(|&&(i, _)| i == cidx)?.1;
        if categorize(c.name()) == ConstraintCategory::Distance && c.equation_count() == 1 {
            distance_indices.push(cidx);
        }
    }

    if distance_indices.len() != 2 {
        return None;
    }

    Some(MatchedPattern {
        kind: PatternKind::TwoDistances,
        entity_ids: vec![entity.id()],
        constraint_indices: distance_indices,
        param_ids: free_params.to_vec(),
    })
}

/// Try to match a horizontal + vertical pattern: an entity with exactly 2
/// free parameters constrained by one horizontal and one vertical constraint.
fn try_horizontal_vertical(
    entity: &dyn Entity,
    free_params: &[ParamId],
    constraint_indices: &[usize],
    constraints: &[(usize, &dyn Constraint)],
) -> Option<MatchedPattern> {
    if free_params.len() != 2 || constraint_indices.len() != 2 {
        return None;
    }

    let mut h_idx = None;
    let mut v_idx = None;

    for &cidx in constraint_indices {
        let c = constraints.iter().find(|&&(i, _)| i == cidx)?.1;
        match categorize(c.name()) {
            ConstraintCategory::Horizontal if c.equation_count() == 1 => {
                h_idx = Some(cidx);
            }
            ConstraintCategory::Vertical if c.equation_count() == 1 => {
                v_idx = Some(cidx);
            }
            _ => {}
        }
    }

    match (h_idx, v_idx) {
        (Some(h), Some(v)) => Some(MatchedPattern {
            kind: PatternKind::HorizontalVertical,
            entity_ids: vec![entity.id()],
            constraint_indices: vec![h, v],
            param_ids: free_params.to_vec(),
        }),
        _ => None,
    }
}

/// Try to match a distance + angle pattern: an entity with exactly 2 free
/// parameters constrained by one distance and one angle constraint.
fn try_distance_angle(
    entity: &dyn Entity,
    free_params: &[ParamId],
    constraint_indices: &[usize],
    constraints: &[(usize, &dyn Constraint)],
) -> Option<MatchedPattern> {
    if free_params.len() != 2 || constraint_indices.len() != 2 {
        return None;
    }

    let mut dist_idx = None;
    let mut angle_idx = None;

    for &cidx in constraint_indices {
        let c = constraints.iter().find(|&&(i, _)| i == cidx)?.1;
        match categorize(c.name()) {
            ConstraintCategory::Distance if c.equation_count() == 1 => {
                dist_idx = Some(cidx);
            }
            ConstraintCategory::Angle if c.equation_count() == 1 => {
                angle_idx = Some(cidx);
            }
            _ => {}
        }
    }

    match (dist_idx, angle_idx) {
        (Some(d), Some(a)) => Some(MatchedPattern {
            kind: PatternKind::DistanceAngle,
            entity_ids: vec![entity.id()],
            constraint_indices: vec![d, a],
            param_ids: free_params.to_vec(),
        }),
        _ => None,
    }
}

/// Scan constraints for known solvable patterns.
///
/// Pattern matching works by examining the local structure around each entity:
/// how many free parameters it has, how many constraints affect it, and what
/// types those constraints are.
///
/// # Arguments
///
/// * `entities` - All entities in the system.
/// * `constraints` - Pairs of `(constraint_index, constraint_ref)` for the
///   active constraints.
/// * `store` - The parameter store (to determine which params are free).
///
/// # Returns
///
/// A vector of matched patterns. Each pattern includes the entity, constraint
/// indices, and parameter IDs involved. A given entity appears in at most one
/// pattern.
pub fn detect_patterns(
    entities: &[&dyn Entity],
    constraints: &[(usize, &dyn Constraint)],
    store: &ParamStore,
) -> Vec<MatchedPattern> {
    let entity_constraint_map = build_entity_constraint_map(constraints);
    let mut patterns = Vec::new();
    let mut claimed_constraints: HashSet<usize> = HashSet::new();

    for &entity in entities {
        let eid = entity.id();
        let free_params = free_params_for_entity(entity, store);

        if free_params.is_empty() {
            continue;
        }

        // Get constraint indices that reference this entity and are not yet
        // claimed by another pattern.
        let constraint_indices: Vec<usize> = entity_constraint_map
            .get(&eid)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
            .iter()
            .copied()
            .filter(|idx| !claimed_constraints.contains(idx))
            .collect();

        if constraint_indices.is_empty() {
            continue;
        }

        // Try patterns in order of specificity (most specific first).
        let matched = try_horizontal_vertical(
            entity,
            &free_params,
            &constraint_indices,
            constraints,
        )
        .or_else(|| {
            try_two_distances(entity, &free_params, &constraint_indices, constraints)
        })
        .or_else(|| {
            try_distance_angle(entity, &free_params, &constraint_indices, constraints)
        })
        .or_else(|| {
            try_scalar_solve(
                entity,
                &free_params,
                &constraint_indices,
                constraints,
                store,
            )
        });

        if let Some(pattern) = matched {
            // Mark the constraints as claimed so they are not double-matched.
            for &cidx in &pattern.constraint_indices {
                claimed_constraints.insert(cidx);
            }
            patterns.push(pattern);
        }
    }

    patterns
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::{ConstraintId, EntityId, ParamId};
    use crate::param::ParamStore;

    // --- Stub entity ---

    struct StubEntity {
        id: EntityId,
        params: Vec<ParamId>,
        label: &'static str,
    }

    impl Entity for StubEntity {
        fn id(&self) -> EntityId {
            self.id
        }
        fn params(&self) -> &[ParamId] {
            &self.params
        }
        fn name(&self) -> &str {
            self.label
        }
    }

    // --- Stub constraint ---

    struct StubConstraint {
        id: ConstraintId,
        entities: Vec<EntityId>,
        params: Vec<ParamId>,
        label: &'static str,
        eq_count: usize,
    }

    impl Constraint for StubConstraint {
        fn id(&self) -> ConstraintId {
            self.id
        }
        fn name(&self) -> &str {
            self.label
        }
        fn entity_ids(&self) -> &[EntityId] {
            &self.entities
        }
        fn param_ids(&self) -> &[ParamId] {
            &self.params
        }
        fn equation_count(&self) -> usize {
            self.eq_count
        }
        fn residuals(&self, _store: &ParamStore) -> Vec<f64> {
            vec![0.0; self.eq_count]
        }
        fn jacobian(&self, _store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
            vec![]
        }
    }

    #[test]
    fn test_detect_scalar_solve() {
        let eid = EntityId::new(0, 0);
        let mut store = ParamStore::new();
        let px = store.alloc(1.0, eid);

        let entity = StubEntity {
            id: eid,
            params: vec![px],
            label: "point",
        };
        let c = StubConstraint {
            id: ConstraintId::new(0, 0),
            entities: vec![eid],
            params: vec![px],
            label: "fix_x",
            eq_count: 1,
        };

        let entities: Vec<&dyn Entity> = vec![&entity];
        let constraints: Vec<(usize, &dyn Constraint)> = vec![(0, &c)];

        let patterns = detect_patterns(&entities, &constraints, &store);
        assert_eq!(patterns.len(), 1);
        assert_eq!(patterns[0].kind, PatternKind::ScalarSolve);
        assert_eq!(patterns[0].param_ids, vec![px]);
    }

    #[test]
    fn test_detect_two_distances() {
        let eid = EntityId::new(0, 0);
        let mut store = ParamStore::new();
        let px = store.alloc(0.0, eid);
        let py = store.alloc(0.0, eid);

        let entity = StubEntity {
            id: eid,
            params: vec![px, py],
            label: "point",
        };
        let c0 = StubConstraint {
            id: ConstraintId::new(0, 0),
            entities: vec![eid],
            params: vec![px, py],
            label: "Distance",
            eq_count: 1,
        };
        let c1 = StubConstraint {
            id: ConstraintId::new(1, 0),
            entities: vec![eid],
            params: vec![px, py],
            label: "Distance",
            eq_count: 1,
        };

        let entities: Vec<&dyn Entity> = vec![&entity];
        let constraints: Vec<(usize, &dyn Constraint)> = vec![(0, &c0), (1, &c1)];

        let patterns = detect_patterns(&entities, &constraints, &store);
        assert_eq!(patterns.len(), 1);
        assert_eq!(patterns[0].kind, PatternKind::TwoDistances);
    }

    #[test]
    fn test_detect_horizontal_vertical() {
        let eid = EntityId::new(0, 0);
        let mut store = ParamStore::new();
        let px = store.alloc(0.0, eid);
        let py = store.alloc(0.0, eid);

        let entity = StubEntity {
            id: eid,
            params: vec![px, py],
            label: "point",
        };
        let c0 = StubConstraint {
            id: ConstraintId::new(0, 0),
            entities: vec![eid],
            params: vec![px],
            label: "Horizontal",
            eq_count: 1,
        };
        let c1 = StubConstraint {
            id: ConstraintId::new(1, 0),
            entities: vec![eid],
            params: vec![py],
            label: "Vertical",
            eq_count: 1,
        };

        let entities: Vec<&dyn Entity> = vec![&entity];
        let constraints: Vec<(usize, &dyn Constraint)> = vec![(0, &c0), (1, &c1)];

        let patterns = detect_patterns(&entities, &constraints, &store);
        assert_eq!(patterns.len(), 1);
        assert_eq!(patterns[0].kind, PatternKind::HorizontalVertical);
    }

    #[test]
    fn test_detect_distance_angle() {
        let eid = EntityId::new(0, 0);
        let mut store = ParamStore::new();
        let px = store.alloc(0.0, eid);
        let py = store.alloc(0.0, eid);

        let entity = StubEntity {
            id: eid,
            params: vec![px, py],
            label: "point",
        };
        let c0 = StubConstraint {
            id: ConstraintId::new(0, 0),
            entities: vec![eid],
            params: vec![px, py],
            label: "Distance",
            eq_count: 1,
        };
        let c1 = StubConstraint {
            id: ConstraintId::new(1, 0),
            entities: vec![eid],
            params: vec![px, py],
            label: "Angle",
            eq_count: 1,
        };

        let entities: Vec<&dyn Entity> = vec![&entity];
        let constraints: Vec<(usize, &dyn Constraint)> = vec![(0, &c0), (1, &c1)];

        let patterns = detect_patterns(&entities, &constraints, &store);
        assert_eq!(patterns.len(), 1);
        assert_eq!(patterns[0].kind, PatternKind::DistanceAngle);
    }

    #[test]
    fn test_no_patterns_when_too_many_constraints() {
        let eid = EntityId::new(0, 0);
        let mut store = ParamStore::new();
        let px = store.alloc(0.0, eid);
        let py = store.alloc(0.0, eid);

        let entity = StubEntity {
            id: eid,
            params: vec![px, py],
            label: "point",
        };
        // Three constraints on a 2-param entity -- over-constrained, no simple
        // pattern matches (our patterns expect exactly 2 constraints for 2-param).
        let c0 = StubConstraint {
            id: ConstraintId::new(0, 0),
            entities: vec![eid],
            params: vec![px, py],
            label: "Distance",
            eq_count: 1,
        };
        let c1 = StubConstraint {
            id: ConstraintId::new(1, 0),
            entities: vec![eid],
            params: vec![px, py],
            label: "Angle",
            eq_count: 1,
        };
        let c2 = StubConstraint {
            id: ConstraintId::new(2, 0),
            entities: vec![eid],
            params: vec![px, py],
            label: "Distance",
            eq_count: 1,
        };

        let entities: Vec<&dyn Entity> = vec![&entity];
        let constraints: Vec<(usize, &dyn Constraint)> = vec![(0, &c0), (1, &c1), (2, &c2)];

        let patterns = detect_patterns(&entities, &constraints, &store);
        assert!(patterns.is_empty());
    }

    #[test]
    fn test_fixed_params_affect_detection() {
        let eid = EntityId::new(0, 0);
        let mut store = ParamStore::new();
        let px = store.alloc(0.0, eid);
        let py = store.alloc(0.0, eid);

        // Fix y, leaving only 1 free param.
        store.fix(py);

        let entity = StubEntity {
            id: eid,
            params: vec![px, py],
            label: "point",
        };
        let c0 = StubConstraint {
            id: ConstraintId::new(0, 0),
            entities: vec![eid],
            params: vec![px],
            label: "fix_x",
            eq_count: 1,
        };

        let entities: Vec<&dyn Entity> = vec![&entity];
        let constraints: Vec<(usize, &dyn Constraint)> = vec![(0, &c0)];

        let patterns = detect_patterns(&entities, &constraints, &store);
        assert_eq!(patterns.len(), 1);
        assert_eq!(patterns[0].kind, PatternKind::ScalarSolve);
    }

    #[test]
    fn test_multiple_entities_independent_patterns() {
        let e0 = EntityId::new(0, 0);
        let e1 = EntityId::new(1, 0);
        let mut store = ParamStore::new();
        let p0 = store.alloc(0.0, e0);
        let p1 = store.alloc(0.0, e1);

        let ent0 = StubEntity {
            id: e0,
            params: vec![p0],
            label: "point_a",
        };
        let ent1 = StubEntity {
            id: e1,
            params: vec![p1],
            label: "point_b",
        };
        let c0 = StubConstraint {
            id: ConstraintId::new(0, 0),
            entities: vec![e0],
            params: vec![p0],
            label: "fix_x",
            eq_count: 1,
        };
        let c1 = StubConstraint {
            id: ConstraintId::new(1, 0),
            entities: vec![e1],
            params: vec![p1],
            label: "fix_y",
            eq_count: 1,
        };

        let entities: Vec<&dyn Entity> = vec![&ent0, &ent1];
        let constraints: Vec<(usize, &dyn Constraint)> = vec![(0, &c0), (1, &c1)];

        let patterns = detect_patterns(&entities, &constraints, &store);
        assert_eq!(patterns.len(), 2);
        assert!(patterns.iter().all(|p| p.kind == PatternKind::ScalarSolve));
    }

    #[test]
    fn test_empty_input() {
        let store = ParamStore::new();
        let entities: Vec<&dyn Entity> = vec![];
        let constraints: Vec<(usize, &dyn Constraint)> = vec![];

        let patterns = detect_patterns(&entities, &constraints, &store);
        assert!(patterns.is_empty());
    }

    #[test]
    fn test_categorize() {
        assert_eq!(categorize("Distance"), ConstraintCategory::Distance);
        assert_eq!(categorize("distance_to_point"), ConstraintCategory::Distance);
        assert_eq!(categorize("Horizontal"), ConstraintCategory::Horizontal);
        assert_eq!(categorize("Vertical"), ConstraintCategory::Vertical);
        assert_eq!(categorize("Angle"), ConstraintCategory::Angle);
        assert_eq!(categorize("Coincident"), ConstraintCategory::Other);
    }
}
